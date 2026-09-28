"""Saved-result verification without optimizers or raw EEG/manifest access.

Separate NumPy forward, scalar correlations and covariance eigensystems verify
the saved training choices, transformed-source decoder lineage and endpoints.
Raw preprocessing and every training update are not independently reproduced.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from cfeg.analysis.dan_runtime import sha_file
from cfeg.analysis.dan_signal import filter_prefix


def _state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        x = state[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((x.dtype.str, x.shape)).encode())
        digest.update(x.tobytes())
    return digest.hexdigest()


def numpy_forward(state, x):
    p = {key: value.cpu().numpy().astype(np.float64) for key, value in state.items()}
    shape = (len(x), x.shape[-1], x.shape[-2])
    z = np.asarray(x, dtype=np.float32).astype(float).transpose(0, 2, 1) @ p["spatial.weight"].T
    z = z.reshape(len(x), -1)
    z = (z - p["normalization.running_mean"]) / np.sqrt(p["normalization.running_var"] + 1e-5)
    z = z * p["normalization.weight"] + p["normalization.bias"]
    z = np.tanh(z.reshape(shape) @ p["hidden.weight"].T + p["hidden.bias"])
    return (z @ p["output.weight"].T + p["output.bias"]).transpose(0, 2, 1)


def _scale(values, observed):
    mean, sd = [], []
    for column in range(values.shape[-1]):
        sample = np.asarray(values[:, column][observed[:, column]], dtype=float)
        mean.append(float(sample.mean()) if len(sample) else 0.)
        deviation = float(sample.std(ddof=0)) if len(sample) else 0.
        sd.append(deviation if deviation >= 1e-8 else 1.)
    return np.array(mean), np.array(sd)


def _metadata(x):
    return np.log1p(np.where(np.isfinite(x), x, 0)), np.isfinite(x)


def _weight(cache, fit_indices, row, donor, e, b, k, arm):
    qtrain = np.ascontiguousarray(cache["q"][fit_indices, e, b]).reshape(-1, cache["q"].shape[-1])
    mean, sd = _scale(qtrain, np.ones_like(qtrain, dtype=bool))
    q = (cache["q"][row, e, b, :k] - mean) / sd
    if arm == "U":
        return np.full_like(q, 1 / k)
    auxiliary, observed = np.zeros_like(q), np.ones_like(q, dtype=bool)
    if arm == "Q2":
        train = cache["q2"][fit_indices, e, b].reshape(-1, q.shape[-1])
        mean, sd = _scale(train, np.ones_like(train, dtype=bool))
        auxiliary = (cache["q2"][row, e, b, :k] - mean) / sd
    elif arm in ("QM", "SHAM"):
        train, flags = _metadata(cache["impedance"][fit_indices, e].reshape(-1, q.shape[-1]))
        mean, sd = _scale(train, flags)
        values, observed = _metadata(cache["impedance"][donor if arm == "SHAM" else row, e, :k])
        auxiliary = np.where(observed, (values - mean) / sd, 0)
    centered = np.zeros_like(q)
    for channel in range(q.shape[-1]):
        mask = observed[:, channel]
        if mask.any():
            v = auxiliary[mask, channel]
            v = v - v[0]
            centered[mask, channel] = v - v.mean()
    q = q - q[:1]
    logits = np.clip(-(q - q.mean(0)) - .5 * centered, -3, 3)
    soft = np.exp(logits - logits.max(0))
    return .5 / k + .5 * soft / soft.sum(0)


def _trca_scores(filters, templates, query):
    result = np.zeros((query.shape[1], len(templates[0])))
    for band in range(3):
        left = np.array([filters[band].T @ row for row in query[band]])
        right = np.array([filters[band].T @ row for row in templates[band]])
        left, right = left.reshape(len(left), -1), right.reshape(len(right), -1)
        left -= left.mean(1, keepdims=True)
        right -= right.mean(1, keepdims=True)
        for label in range(len(right)):
            rho = (left * right[label]).sum(1) / (
                np.sqrt((left * left).sum(1)) * np.linalg.norm(right[label]))
            result[:, label] += ((band + 1) ** -1.25 + .25) * rho * abs(rho)
    return result


def _cca_scores(query, config):
    result = np.zeros((query.shape[1], len(config["frequencies"])))
    t = np.arange(query.shape[-1]) / config["sfreq"]
    for band in range(3):
        for trial, eeg in enumerate(query[band]):
            x = eeg - eeg.mean(-1, keepdims=True)
            xx = x @ x.T
            xx += np.eye(len(x)) * 1e-8 * np.trace(xx) / len(x)
            for label, f in enumerate(config["frequencies"]):
                angle = 2 * np.pi * f * np.arange(1, 4)[:, None] * t
                y = np.concatenate([np.sin(angle), np.cos(angle)])
                y -= y.mean(-1, keepdims=True)
                yy = y @ y.T
                yy += np.eye(6) * 1e-8 * np.trace(yy) / 6
                xy = x @ y.T
                values = np.linalg.eigvals(np.linalg.solve(xx, xy @ np.linalg.solve(yy, xy.T)))
                result[trial, label] += ((band + 1) ** -1.25 + .25) * np.clip(values.real.max(), 0, 1)
    return result


def _decoder_lineage(calibration, filters, templates):
    x = np.asarray(calibration, dtype=float)
    x -= x.mean(-1, keepdims=True)
    np.testing.assert_allclose(templates, x.mean(1), rtol=5e-5, atol=5e-6)
    for band in range(3):
        for label in range(x.shape[2]):
            trials = x[band, :, label]
            within = sum(trial @ trial.T for trial in trials)
            total = trials.sum(0)
            between = total @ total.T - within
            within += np.eye(len(within)) * 1e-8 * np.trace(within) / len(within)
            largest = linalg.eigvalsh((between + between.T) / 2, within)[-1]
            vector = filters[band, :, label]
            value = float(vector @ between @ vector / (vector @ within @ vector))
            np.testing.assert_allclose(value, largest, rtol=2e-3, atol=2e-4)


def _verify_endpoints(report, config):
    rows = report["rows"]
    ids = sorted({r["target"] for r in rows})

    def avg(arm, ks, person=None, seed=None):
        subset = [r for r in rows if r["arm"] == arm and r["k"] in ks
                  and (person is None or person == r["target"])
                  and (seed is None or seed == r["seed"])]
        assert subset
        assert len({r["trials"] for r in subset}) == 1
        return sum(r["correct"] for r in subset) / (len(subset) * subset[0]["trials"])

    for arm, curve in report["curves"].items():
        for k, value in curve.items():
            np.testing.assert_allclose(value, avg(arm, [int(k)]), atol=1e-12, rtol=0)
    primary = config["evaluation"]["primary_k"]
    deltas = {}
    for arm in ("Q", "Q2", "SHAM"):
        delta = np.array([avg("QM", primary, s) - avg(arm, primary, s) for s in ids])
        deltas[arm] = (avg("QM", primary) - avg(arm, primary)) * 100
        recorded = report["primary_qm_minus"][arm]
        np.testing.assert_allclose(recorded["mean_pp"], deltas[arm], atol=1e-12, rtol=0)
        np.testing.assert_allclose([recorded["participant_delta_pp"][str(s)] for s in ids],
                                   delta * 100, atol=1e-12, rtol=0)
        rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
        draws = rng.choice(delta, (config["evaluation"]["bootstrap_draws"], len(ids))).mean(1)
        np.testing.assert_allclose(recorded["descriptive_ci95_pp"],
                                   np.quantile(draws, [.025, .975]) * 100, atol=1e-12, rtol=0)
    seed_deltas = {str(s): (avg("QM", primary, seed=s) - avg("Q", primary, seed=s)) * 100
                   for s in config["training"]["seeds"]}
    for seed, value in seed_deltas.items():
        np.testing.assert_allclose(report["primary_qm_minus_q_by_seed_pp"][seed], value, atol=1e-12)
    harm = sum(avg("QM", primary, s) - avg("Q", primary, s) < -.05 for s in ids) / len(ids)
    qm3, q5, cca = avg("QM", [3]), avg("Q", [5]), avg("CCA", [0])
    gates = {"qm_q_2pp": deltas["Q"] >= 2, "qm_q2_positive": deltas["Q2"] > 0,
             "qm_sham_1pp": deltas["SHAM"] >= 1,
             "each_seed_positive": all(v > 0 for v in seed_deltas.values()),
             "qm3_at_least_80pct": qm3 >= .8, "qm3_noninferior_q5": qm3 - q5 >= -.01,
             "qm3_noninferior_cca": qm3 - cca >= -.01, "harm_fraction_at_most_20pct": harm <= .2}
    assert gates == report["promotion_gates"]
    assert all(gates.values()) == report["producer_all_gates_pass"]
    np.testing.assert_allclose(report["harm_fraction"], harm, atol=1e-12)
    np.testing.assert_allclose(report["cca_accuracy"], cca, atol=1e-12)
    np.testing.assert_allclose(report["baseline_u_minus_trca_primary_pp"],
                               (avg("U", primary) - avg("TRCA", primary)) * 100, atol=1e-12)
    cost = report["fixed_cost_contrast"]
    assert (cost["qm_trials"], cost["q_trials"], cost["nominal_trial_reduction"]) == (36, 60, .4)
    np.testing.assert_allclose([cost["qm3_accuracy"], cost["q5_accuracy"]], [qm3, q5], atol=1e-12)
    assert cost["performance_criteria_met"] == (qm3 >= .8 and qm3 - q5 >= -.01 and qm3 - cca >= -.01)
    for row in report["attainment_descriptive_only"]:
        passing = [r["k"] * len(config["frequencies"]) for r in rows
                   if all(row[key] == r[key] for key in ("target", "interface", "seed", "arm"))
                   and r["correct"] / r["trials"] >= .8]
        assert row["first_grid_80pct_trials"] == (min(passing) if passing else None)


def audit_saved(output: Path) -> dict:
    started = time.monotonic()
    cfg = json.loads((output / "config.json").read_text())
    report = json.loads((output / "result.json").read_text())
    frozen = json.loads((output / "freeze.json").read_text())

    def check_time():
        if time.monotonic() - started > cfg["limits"]["saved_audit_seconds"]:
            raise TimeoutError("Saved audit budget exhausted; no automatic retry.")

    assert sha_file(output / "freeze.json") == report["freeze_sha256"]
    assert sha_file(output / "scores.npz") == report["scores_sha256"]
    assert sha_file(output / "cache.npz") == frozen["cache_sha256"]
    assert sha_file(output / "config.json") == frozen["config_sha256"]
    for name, digest in frozen["artifacts"].items():
        path = output / name
        assert path.parent == output and not path.is_symlink()
        assert sha_file(path) == digest
        check_time()
    with np.load(output / "cache.npz", allow_pickle=False) as packet:
        cache = {key: packet[key] for key in packet.files}
    ids = cache["ids"].tolist()
    if not frozen["generated"]:
        assert frozen["targets"] == cfg["source_subject_ids"] == ids
        assert frozen["interfaces"] == [0, 1]
    expected = set(itertools.product(frozen["targets"], frozen["interfaces"], cfg["budgets"],
                                     cfg["arms"], cfg["training"]["seeds"], range(3)))
    assert len(frozen["models"]) == len(expected)
    assert {tuple(m["key"]) for m in frozen["models"]} == expected
    # Recompute source role scalers and donor identities independently.
    for fold in range(3):
        target = [s for i, s in enumerate(ids) if i % 3 == fold]
        source = [s for i, s in enumerate(ids) if i % 3 != fold][:5]
        role = frozen["roles"][str(fold)]
        assert role["source"] == source and role["fit"] == source[:4] and role["target"] == target
        generator = np.random.default_rng(cfg["splits"]["sham_seed"] + fold)
        donors = {}
        for order in (0, 1):
            members = sorted(s for s in target if cache["orders"][ids.index(s)] == order)
            if members:
                assert len(members) >= 2
                permutation = generator.permutation(members).tolist()
                donors.update(zip(map(str, permutation), permutation[1:] + permutation[:1]))
        assert role["donors"] == donors
        fit_indices = [ids.index(s) for s in source[:4]]
        for e in range(2):
            values, flags = _metadata(cache["impedance"][fit_indices, e].reshape(-1, len(cfg["channels"])))
            pairs = [((e, "M"), values, flags)]
            for band in range(3):
                for name in ("q", "q2"):
                    v = cache[name][fit_indices, e, band].reshape(-1, len(cfg["channels"]))
                    pairs.append(((e, band, name), v, np.ones_like(v, dtype=bool)))
            for key, values, flags in pairs:
                mean, sd = _scale(values, flags)
                np.testing.assert_allclose(role["scalers"][str(key)]["mean"], mean, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(role["scalers"][str(key)]["scale"], sd, rtol=1e-10, atol=1e-10)
    model_info = {m["file"]: m for m in frozen["models"]}
    model_checks, selected_checks, total_updates, max_score_error = 0, 0, 0, 0.
    paired = {}
    query = {(s, e): filter_prefix(cache["query_prefix"][ids.index(s), e], cfg).reshape(
        3, -1, len(cfg["channels"]), cfg["n_samples"])
        for s in frozen["targets"] for e in frozen["interfaces"]}
    with np.load(output / "scores.npz", allow_pickle=False) as packet:
        scores, labels = packet["scores"], packet["labels"]
    np.testing.assert_array_equal(labels, np.tile(np.arange(len(cfg["frequencies"])), 4))
    expected_rows = {(s, e, 0, "CCA", -1) for s in frozen["targets"] for e in frozen["interfaces"]}
    expected_rows.update((d["target"], d["interface"], d["k"], d["arm"], d["seed"]) for d in frozen["decoders"])
    assert len(report["rows"]) == len(expected_rows) == len(scores)
    assert {(r["target"], r["interface"], r["k"], r["arm"], r["seed"]) for r in report["rows"]} == expected_rows
    for score_index, row in enumerate(report["rows"]):
        check_time()
        assert row["score_index"] == score_index
        target, e, k, arm, seed = (row[key] for key in ("target", "interface", "k", "arm", "seed"))
        index = ids.index(target)
        if arm == "CCA":
            prediction_scores = _cca_scores(query[target, e], cfg)
        else:
            with np.load(output / row["file"], allow_pickle=False) as packet:
                filters, templates = packet["filters"], packet["templates"]
            support = cache["bands"][index, e, :, :k]
            calibration = support.copy()
            if arm != "TRCA":
                augmented = []
                for band, name in enumerate(row["alignment_files"]):
                    info = model_info[name]
                    assert info["key"] == [target, e, k, arm, seed, band]
                    record = torch.load(output / name, map_location="cpu", weights_only=True)
                    role = frozen["roles"][str(info["fold"])]
                    assert record["source_ids"] == role["source"] and record["fit_ids"] == role["fit"]
                    source_indices = [ids.index(s) for s in role["source"]]
                    source = cache["bands"][source_indices, e, band]
                    donor = ids.index(role["donors"][str(target)])
                    weight = _weight(cache, source_indices[:4], index, donor, e, band, k, arm)
                    np.testing.assert_allclose(record["weights"].numpy(), weight, rtol=1e-10, atol=1e-10)
                    teacher = sum(support[band, j].astype(np.float32) * weight[j].astype(np.float32)[:, None]
                                  for j in range(k))
                    np.testing.assert_allclose(record["teacher"].numpy(), teacher, rtol=2e-6, atol=2e-6)
                    transformed = []
                    signature = (record["initial_sha256"], tuple(r["batch_order_sha256"] for r in record["records"]))
                    pair = (target, e, k, seed, band)
                    if pair in paired:
                        assert paired[pair] == signature
                    paired[pair] = signature
                    for stage, (state, history) in enumerate(zip(record["states"], record["records"])):
                        epochs = cfg["training"]["pretrain_epochs" if stage == 0 else "fine_epochs"]
                        assert len(history["validation_mse"]) == epochs
                        assert history["best_epoch_zero_based"] == int(np.argmin(history["validation_mse"]))
                        assert _state_sha(state) == history["selected_sha256"]
                        train_rows, val_rows = (288, 72) if stage == 0 else (48, 24)
                        assert history["train_rows"] == train_rows and history["validation_rows"] == val_rows
                        updates = epochs * (3 if stage == 0 else 1)
                        assert history["optimizer_updates"] == updates
                        total_updates += updates
                        valid = source[4] if stage == 0 else source[stage - 1, 4:]
                        output_array = numpy_forward(state, valid.reshape(-1, *source.shape[-2:]))
                        val_teacher = np.tile(record["teacher"].numpy(), (len(valid), 1, 1))
                        mse = np.mean((output_array - val_teacher) ** 2)
                        np.testing.assert_allclose(mse, history["validation_mse"][history["best_epoch_zero_based"]],
                                                   rtol=5e-5, atol=5e-6)
                        if stage:
                            assert history["initial_sha256"] == record["records"][0]["selected_sha256"]
                            aligned = numpy_forward(state, source[stage - 1].reshape(-1, *source.shape[-2:]))
                            transformed.append(aligned.reshape(source.shape[1:]))
                        selected_checks += 1
                    augmented.append(np.concatenate((support[band], *transformed)))
                    model_checks += 1
                calibration = np.stack(augmented)
            _decoder_lineage(calibration, filters, templates)
            prediction_scores = _trca_scores(filters, templates, query[target, e])
        np.testing.assert_allclose(prediction_scores, scores[score_index], rtol=1e-5, atol=1e-5)
        max_score_error = max(max_score_error, float(np.max(abs(prediction_scores - scores[score_index]))))
        np.testing.assert_array_equal(prediction_scores.argmax(1), scores[score_index].argmax(1))
        correct = int(np.count_nonzero(prediction_scores.argmax(1) == labels))
        assert correct == row["correct"] and row["trials"] == len(labels)
        assert row["accuracy"] == correct / len(labels)
    assert model_checks == len(expected) and selected_checks == len(expected) * 6
    assert total_updates == frozen["counts"]["optimizer_updates"]
    assert frozen["counts"]["optimizer_fits"] == selected_checks
    assert report["meter"]["counts"]["outer_reveal_batches"] == 1
    _verify_endpoints(report, cfg)
    check_time()
    return {"status": "PASS_SAVED_MODELS_DECODERS_AND_ENDPOINTS", "generated": frozen["generated"],
            "seconds": time.monotonic() - started, "alignment_cells_checked": model_checks,
            "selected_states_checked": selected_checks, "trial_decisions_checked": int(scores.shape[0] * len(labels)),
            "max_score_error": max_score_error, "argmax_disagreements": 0,
            "raw_reads": 0, "manifest_reads": 0, "optimizer_updates": 0,
            "producer_all_gates_pass": report["producer_all_gates_pass"],
            "validated_metadata_calibration_reduction": report["producer_all_gates_pass"],
            "scope_limits": ["Saved-model/decoder/endpoint verification, not independent raw preprocessing or all training steps.",
                             "Model-selection traces are checked at selected checkpoints; unselected weights were not saved."]}
