"""Saved-artifact audit with a NumPy/SciPy forward, no fitting or raw EEG reads."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

from cfeg.analysis.joint_harmonic import ARMS
from cfeg.analysis.joint_harmonic_runner import file_sha, utc_now, write_json


def restore_matlab_context_layout(values: np.ndarray) -> np.ndarray:
    """Restore the generated-reproduced MAT/F extraction layout, without changing values.

    Logical axes are person/interface/block/class/channel/feature. Physical order
    after the original MAT/F extraction was person/class/block/interface/channel/feature.
    NPZ restores a C-contiguous array instead; float32 reduction order then differs.
    This fixed transform is selected before human re-audit, never fitted to its values.
    """
    if values.ndim != 6 or values.shape[-1] != 2 or values.dtype != np.float32:
        raise ValueError("Expected six-dimensional float32 Q/Q2 cache.")
    axes = (0, 3, 2, 1, 4, 5)
    return np.ascontiguousarray(values.transpose(axes)).transpose(axes)


def independent_contexts(cache: dict, people: list[int], arm: str, donors: dict) -> np.ndarray:
    """Rebuild prefix contexts without the production context/scaler functions."""
    ids = cache["ids"].tolist()
    output = []
    for subject in people:
        pos = ids.index(subject)
        by_interface = []
        for interface in range(2):
            by_k = []
            for k in (1, 3, 5):
                q = cache["q"][pos, interface, :k].mean(axis=(0, 1)).astype(float)
                flags = np.ones_like(q)
                if arm == "Q":
                    auxiliary = q
                elif arm == "Q2":
                    auxiliary = cache["q2"][pos, interface, :k].mean(axis=(0, 1)).astype(float)
                else:
                    z = cache["impedance"][ids.index(donors[subject]), interface, :k]
                    auxiliary = np.zeros_like(q)
                    flags = np.zeros_like(q)
                    for channel in range(z.shape[1]):
                        observed = z[:, channel][np.isfinite(z[:, channel])]
                        if len(observed):
                            logs = np.log1p(observed)
                            auxiliary[channel] = [logs.mean(), logs.std(ddof=0)]
                            flags[channel] = 1
                common = np.tile([interface, cache["orders"][pos],
                                  int(interface != cache["orders"][pos]), np.log(k)], (len(q), 1))
                by_k.append(np.concatenate((q, auxiliary, flags, common), axis=-1))
            by_interface.append(by_k)
        output.append(by_interface)
    return np.asarray(output)


def check_normalizer(cache: dict, record: dict) -> tuple[float, float]:
    donors = {int(k): v for k, v in record["sham_donors"].items()}
    raw = independent_contexts(cache, record["fit_subject_ids"], record["arm"], donors)
    raw = raw.reshape(-1, raw.shape[-2], 10)
    means, scales = np.zeros((raw.shape[1], 4)), np.ones((raw.shape[1], 4))
    for channel in range(raw.shape[1]):
        for feature in range(4):
            rows = raw[:, channel, feature]
            if feature >= 2:
                rows = rows[raw[:, channel, feature + 2] == 1]
            if len(rows):
                means[channel, feature] = rows.mean()
                deviation = rows.std(ddof=0)
                scales[channel, feature] = deviation if deviation >= 1e-8 else 1
    np.testing.assert_allclose(means, record["normalizer_mean"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(scales, record["normalizer_scale"], atol=1e-10, rtol=1e-10)
    return (float(np.abs(means - record["normalizer_mean"]).max()),
            float(np.abs(scales - record["normalizer_scale"]).max()))


def numpy_forward(support: np.ndarray, query: np.ndarray, context: np.ndarray,
                  weights: dict[str, np.ndarray], classes: int, config: dict) -> np.ndarray:
    def gelu(x):
        return x * 0.5 * (1 + erf(x / np.sqrt(2)))

    def linear(x, name):
        return x @ weights[f"{name}.weight"].T + weights[f"{name}.bias"]

    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    raw = linear(gelu(linear(context.astype(float), "conditioner.0")), "conditioner.2")
    scale, shift = np.split(raw, 2, axis=-1)
    bound = config["model"]["affine_bound"]

    def encode(x):
        h = gelu(linear(x.astype(float), "channel_encoder"))
        h = gelu((1 + bound * np.tanh(scale[:, None])) * h + bound * np.tanh(shift[:, None]))
        return unit(linear(h.reshape(*h.shape[:2], -1), "spatial_encoder"))

    support_z, query_z = encode(support), encode(query)
    # Complete-block ordering, with no dependence on query labels.
    prototype = unit(support_z.reshape(support_z.shape[0], -1, classes,
                                      support_z.shape[-1]).mean(axis=1))
    return config["model"]["temperature"] * (query_z @ prototype.swapaxes(-1, -2))


def audit_saved(output: Path, *, generated: bool = False, verification_r1: bool = False,
                matlab_layout: bool = False) -> dict:
    started = time.monotonic()
    prefix = "audit_r1" if verification_r1 else "audit"
    if not generated and verification_r1:
        if output.resolve() != Path("/home/whwovy/eeg-data/joint-harmonic-human-v1-0ioz9_2y"):
            raise ValueError("Verification extension is limited to the existing human result.")
        assert matlab_layout, "R1 human layout was fixed before the saved-result audit."
        assert file_sha(output / "audit.json") == "0ae4beeb1037345892da3b2d723bce63d0a8130ca1fc6bc6f6b8ed8305c7bf7f"
        assert file_sha(output / "result.json") == "a02ab289da364c10a2d2b6bf412237cc653439cc8dc7accce66d2a2b3066b664"
    if not generated and matlab_layout and not verification_r1:
        raise ValueError("Human layout repair requires the one-shot verification extension.")
    write_json(output / f"{prefix}_start.json", {
        "at": utc_now(), "generated": generated, "verification_r1": verification_r1,
        "matlab_layout": matlab_layout, "auditor_sha256": file_sha(Path(__file__)),
        "normalizer_atol": 1e-10, "normalizer_rtol": 1e-10,
        "forward_absolute_limit": 1e-4})
    report = {"status": "RUNNING", "generated": generated, "raw_reads": 0, "fits": 0,
              "feature_cache_loads": 0, "verification_r1": verification_r1,
              "matlab_layout": matlab_layout, "normalizers_checked": 0,
              "max_normalizer_mean_difference": 0.0, "max_normalizer_scale_difference": 0.0,
              "forward_models_checked": 0, "forward_trials_checked": 0,
              "max_context_difference": 0.0,
              "max_logit_error": 0.0, "numpy_argmax_differences": 0}
    try:
        config = json.loads((output / "config.json").read_text())
        result = json.loads((output / "result.json").read_text())
        freeze_path = output / "all_models_frozen.json"
        assert file_sha(freeze_path) == result["freeze_sha256"]
        assert file_sha(output / "all_query_scores.npz") == result["scores_sha256"]
        cache_receipt = json.loads((output / "cache_receipt.json").read_text())
        assert file_sha(output / "features.npz") == cache_receipt["sha256"] == result["cache_sha256"]
        with np.load(output / "features.npz", allow_pickle=False) as archive:
            report["feature_cache_loads"] += 1
            cache = {k: archive[k] for k in archive.files}
        if matlab_layout:
            for key in ("q", "q2"):
                cache[key] = restore_matlab_context_layout(cache[key])
        with np.load(output / "all_query_scores.npz", allow_pickle=False) as archive:
            scores = {k: archive[k] for k in archive.files}
        ids = cache["ids"].tolist()
        assert ids == config["source_subject_ids"] == scores["subject_ids"].tolist()
        freeze = json.loads(freeze_path.read_text())
        journal = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
        events = [row["event"] for row in journal]
        assert events.count("fit_complete") == events.count("fit_start") == 48
        assert events.count("all_models_frozen") == events.count("outer_reveal_start") == 1
        gate_index = events.index("all_models_frozen")
        assert max(i for i, e in enumerate(events) if e == "fit_complete") < gate_index
        assert gate_index < events.index("outer_reveal_start") < events.index("outer_scores_saved")
        classes = len(config["frequencies"])
        labels = np.tile(np.arange(classes), 5)
        all_records = []
        for fit_id in freeze["fit_records"]:
            assert file_sha(output / f"{fit_id}.json") == freeze["fit_record_sha256"][fit_id]
            record = json.loads((output / f"{fit_id}.json").read_text())
            all_records.append(record)
            fold = record["fold"]
            outer = sorted(ids)[fold::3]
            fit_ids = [s for s in sorted(ids) if s not in outer]
            if record["kind"] == "inner":
                fit_ids = fit_ids[::2]
            assert record["fit_subject_ids"] == record["normalizer_fit_subject_ids"] == fit_ids
            assert record["steps"] == config["training"]["steps"]
            assert file_sha(output / f"{fit_id}.pt") == record["checkpoint_sha256"]
            mean_error, scale_error = check_normalizer(cache, record)
            report["normalizers_checked"] += 1
            report["max_normalizer_mean_difference"] = max(report["max_normalizer_mean_difference"], mean_error)
            report["max_normalizer_scale_difference"] = max(report["max_normalizer_scale_difference"], scale_error)
            if record["arm"] == "SHAM":
                donors = {int(k): v for k, v in record["sham_donors"].items()}
                assert set(donors) == set(donors.values()) == set(fit_ids)
                for receiver, donor in donors.items():
                    assert receiver != donor
                    assert cache["orders"][ids.index(receiver)] == cache["orders"][ids.index(donor)]
        for kind in ("inner", "outer"):
            for fold in range(3):
                for seed in config["training"]["seeds"]:
                    group = [r for r in all_records if (r["kind"], r["fold"], r["seed"])
                             == (kind, fold, seed)]
                    if group:
                        assert len({r["initial_state_sha256"] for r in group}) == 1
                        assert len({r["episode_schedule_sha256"] for r in group}) == 1
        assert len({r["parameter_count"] for r in all_records}) == 1
        assert len(all_records) == 48 and len(freeze["final_models"]) == 24
        # Verify source-validation choices from existing saved predictions, without refitting.
        for fold in range(3):
            outer_ids = sorted(ids)[fold::3]
            validation_ids = [s for s in sorted(ids) if s not in outer_ids][1::2]
            with np.load(output / f"inner-cca-fold{fold}.npz", allow_pickle=False) as archive:
                assert archive["subject_ids"].tolist() == validation_ids
                cca_validation = float((archive["scores"].argmax(-1) == labels).mean())
            for arm in ARMS:
                rows = [r for r in all_records if (r["kind"], r["fold"], r["arm"])
                        == ("inner", fold, arm)]
                assert len(rows) == len(config["training"]["learning_rates"]) == 2
                curves = {}
                for row in rows:
                    with np.load(output / f"{row['fit_id']}-validation.npz", allow_pickle=False) as archive:
                        assert archive["subject_ids"].tolist() == validation_ids
                        curves[row["lr"]] = (archive["scores"].argmax(-1) == labels).mean(axis=(0, 1, 3))
                chosen = min(curves, key=lambda lr: (-float(curves[lr].mean()), lr))
                choice = freeze["choices"][str(fold)][arm]
                assert choice["lr"] == chosen
                for lr, curve in curves.items():
                    np.testing.assert_allclose(choice["lr_validation_means"][str(lr)], curve.mean(), atol=1e-14, rtol=0)
                np.testing.assert_allclose([choice["validation_curve"][str(k)] for k in (0, 1, 3, 5)],
                                           [cca_validation, *curves[chosen]], atol=1e-14, rtol=0)
        report["inner_choices_checked"] = 12
        for record in freeze["final_models"]:
            if time.monotonic() - started > config["limits"]["saved_audit_seconds"]:
                raise TimeoutError("Saved-artifact audit budget exceeded.")
            fit_id = record["fit_id"]
            weights = {k: v.numpy().astype(float) for k, v in torch.load(
                output / f"{fit_id}.pt", map_location="cpu", weights_only=True).items()}
            assert sum(w.size for w in weights.values()) == record["parameter_count"]
            with np.load(output / f"{fit_id}-query.npz", allow_pickle=False) as archive:
                saved, context, query_ids = archive["scores"], archive["contexts"], archive["subject_ids"]
            fold, ai = record["fold"], ARMS.index(record["arm"])
            si = config["training"]["seeds"].index(record["seed"])
            assert query_ids.tolist() == sorted(ids)[fold::3]
            assert not set(query_ids) & set(record["fit_subject_ids"])
            query_role = json.loads((output / f"{fit_id}-query-role.json").read_text())
            donors = {int(k): v for k, v in query_role["donors"].items()}
            assert set(donors) == set(donors.values()) == set(query_ids)
            if record["arm"] == "SHAM":
                assert all(s != d and cache["orders"][ids.index(s)] == cache["orders"][ids.index(d)]
                           for s, d in donors.items())
            else:
                assert all(s == d for s, d in donors.items())
            raw_context = independent_contexts(cache, query_ids.tolist(), record["arm"], donors)
            normalized = raw_context.copy()
            normalized[..., :4] = (raw_context[..., :4] - np.asarray(record["normalizer_mean"])) / np.asarray(record["normalizer_scale"])
            normalized[..., 2:4] = np.where(raw_context[..., 4:6] > 0, normalized[..., 2:4], 0)
            np.testing.assert_allclose(context, normalized, atol=1e-5, rtol=1e-6)
            report["max_context_difference"] = max(report["max_context_difference"],
                                                    float(np.abs(context - normalized).max()))
            positions = [ids.index(s) for s in query_ids]
            np.testing.assert_array_equal(saved, scores["learned"][ai, si, positions])
            for i, pos in enumerate(positions):
                features = cache["spectra"][pos]
                q = features[:, 5:].reshape(2, 5 * classes, *features.shape[-2:])
                for ki, k in enumerate(config["budgets"]):
                    support = features[:, :k].reshape(2, k * classes, *features.shape[-2:])
                    recalculated = numpy_forward(support, q, context[i, :, ki], weights, classes, config)
                    error = float(np.max(np.abs(recalculated - saved[i, :, ki])))
                    report["max_logit_error"] = max(report["max_logit_error"], error)
                    assert error <= 1e-4, (fit_id, error)
                    report["numpy_argmax_differences"] += int(
                        (recalculated.argmax(-1) != saved[i, :, ki].argmax(-1)).sum())
                    report["forward_trials_checked"] += int(np.prod(recalculated.shape[:-1]))
            report["forward_models_checked"] += 1
        assert report["forward_models_checked"] == 24
        assert report["numpy_argmax_differences"] == 0, "Independent predictions differ."
        # Independent loops recompute all displayed accuracy/cost summaries from sealed scores.
        learned_accuracy = (scores["learned"].argmax(-1) == labels).mean(-1)
        common_accuracy = (scores["common"].argmax(-1) == labels).mean(-1)
        independent_policy_acc, independent_policy_cost = [], []
        for ai, arm in enumerate(ARMS):
            claimed = result["arms"][arm]
            np.testing.assert_allclose(claimed["ba_by_k"], learned_accuracy[ai].mean(axis=(0, 1, 2)),
                                       atol=1e-14, rtol=0)
            np.testing.assert_allclose(claimed["low_k_mean_ba"], learned_accuracy[ai, ..., :2].mean(),
                                       atol=1e-14, rtol=0)
            np.testing.assert_allclose(claimed["low_k_seed_ba"], learned_accuracy[ai, ..., :2].mean(axis=(1, 2, 3)), atol=1e-14, rtol=0)
            np.testing.assert_allclose(claimed["ba_by_interface_k"], learned_accuracy[ai].mean(axis=(0, 1)), atol=1e-14, rtol=0)
            np.testing.assert_allclose(claimed["participant_interface_k_seed_mean"], learned_accuracy[ai].mean(axis=0), atol=1e-14, rtol=0)
            policy_values, policy_costs = [], []
            for p, subject in enumerate(ids):
                fold = sorted(ids).index(subject) % 3
                policy = freeze["choices"][str(fold)][arm]["policy"]
                curve = freeze["choices"][str(fold)][arm]["validation_curve"]
                passed = [k for k in (0, 1, 3, 5) if curve[str(k)] >= 0.8]
                assert policy["k"] == (passed[0] if passed else 5)
                assert policy["fallback"] == (not passed)
                k = policy["k"]
                policy_costs.append(classes * k)
                value = common_accuracy[p].mean() if k == 0 else learned_accuracy[
                    ai, :, p, :, (1, 3, 5).index(k)].mean()
                policy_values.append(value)
            np.testing.assert_allclose(claimed["policy_mean_ba"], np.mean(policy_values), atol=1e-14)
            np.testing.assert_allclose(claimed["policy_mean_acquired_trials"], np.mean(policy_costs), atol=0)
            independent_policy_acc.append(float(np.mean(policy_values)))
            independent_policy_cost.append(float(np.mean(policy_costs)))
        low = learned_accuracy[..., :2].mean(axis=(1, 3, 4))
        rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
        indices = rng.integers(len(ids), size=(config["evaluation"]["bootstrap_draws"], len(ids)))
        deltas = {}
        for ai, arm in enumerate(ARMS):
            if arm == "QM":
                continue
            d = (low[2] - low[ai]) * 100
            deltas[arm] = float(d.mean())
            claimed = result["low_k_contrasts"][f"QM_minus_{arm}"]
            np.testing.assert_allclose(claimed["mean_pp"], d.mean(), atol=1e-12)
            ci = np.quantile(d[indices].mean(axis=1), [0.025, 0.975])
            np.testing.assert_allclose(claimed["descriptive_paired_bootstrap_95_pp"], ci, atol=1e-12)
        q_cost = independent_policy_cost[0]
        saving = (q_cost - independent_policy_cost[2]) / q_cost if q_cost else None
        assert result["policy_trial_reduction_fraction"] == saving
        seed_gain = (learned_accuracy[2, ..., :2] - learned_accuracy[0, ..., :2]).mean(axis=(1, 2, 3)) * 100
        harm = float(((low[2] - low[0]) * 100 < -5).mean())
        policy_delta = (independent_policy_acc[2] - independent_policy_acc[0]) * 100
        np.testing.assert_allclose(result["qm_minus_q_by_seed_pp"], seed_gain, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result["harmed_participant_fraction"], harm, atol=1e-14, rtol=0)
        np.testing.assert_allclose(result["policy_qm_minus_q_pp"], policy_delta, atol=1e-12, rtol=0)
        p = config["promotion"]
        independent_gates = {
            "low_q_gain": deltas["Q"] >= p["qm_minus_q_pp"],
            "beats_q2": deltas["Q2"] > p["qm_minus_q2_pp_strict"],
            "beats_sham": deltas["SHAM"] >= p["qm_minus_sham_pp"],
            "both_seeds_positive": bool((seed_gain > 0).all()),
            "policy_target": independent_policy_acc[2] >= p["policy_ba"],
            "policy_noninferiority": policy_delta >= p["policy_noninferiority_pp"],
            "policy_trial_reduction": saving is not None and saving >= p["acquired_trial_reduction"],
            "harm_fraction": harm <= p["maximum_harmed_fraction"],
        }
        assert independent_gates == result["gates"]
        assert all(independent_gates.values()) == result["scientific_screen_passed"]
        observed = []
        for ai in (0, 2):
            costs = []
            for seed in range(2):
                for person in range(len(ids)):
                    for interface in range(2):
                        curve = [common_accuracy[person, interface],
                                 *learned_accuracy[ai, seed, person, interface].tolist()]
                        costs.append(next((classes * k for k, a in zip((0, 1, 3, 5), curve)
                                           if a >= 0.8), np.nan))
            observed.append(np.asarray(costs))
        joint = np.isfinite(observed[0]) & np.isfinite(observed[1])
        threshold = result["observed_threshold_comparison"]
        assert threshold["jointly_attained"] == int(joint.sum())
        assert threshold["qm_only_attained"] == int((~np.isfinite(observed[0]) & np.isfinite(observed[1])).sum())
        assert threshold["q_only_attained"] == int((np.isfinite(observed[0]) & ~np.isfinite(observed[1])).sum())
        assert threshold["neither_attained"] == int((~np.isfinite(observed[0]) & ~np.isfinite(observed[1])).sum())
        if joint.any():
            np.testing.assert_allclose(threshold["paired_mean_q_trials"], observed[0][joint].mean(), atol=1e-12)
            np.testing.assert_allclose(threshold["paired_mean_qm_trials"], observed[1][joint].mean(), atol=1e-12)
            np.testing.assert_allclose(threshold["paired_mean_trial_saving"],
                                       (observed[0] - observed[1])[joint].mean(), atol=1e-12)
        baseline_accuracy = (scores["baselines"].argmax(-1) == labels).mean(-1)
        baseline = result["baseline"]
        np.testing.assert_allclose(baseline["common_cca_ba"], common_accuracy.mean(), atol=1e-14, rtol=0)
        np.testing.assert_allclose(baseline["common_cca_interface_ba"], common_accuracy.mean(axis=0), atol=1e-14, rtol=0)
        for bi, name in enumerate(("b0_ba_by_k", "direct_m_ba_by_k")):
            np.testing.assert_allclose(baseline[name], baseline_accuracy[bi].mean(axis=(0, 1)), atol=1e-14, rtol=0)
        b0_low = baseline_accuracy[0, ..., :2].mean(axis=(1, 2))
        direct_low = baseline_accuracy[1, ..., :2].mean(axis=(1, 2))
        for name, difference in (("q_minus_b0_low_k", low[0] - b0_low),
                                 ("direct_m_minus_b0_low_k", direct_low - b0_low)):
            np.testing.assert_allclose(baseline[name]["mean_pp"], 100 * difference.mean(), atol=1e-12, rtol=0)
            np.testing.assert_allclose(baseline[name]["descriptive_paired_bootstrap_95_pp"],
                                       np.quantile(100 * difference[indices].mean(axis=1), [0.025, 0.975]), atol=1e-12, rtol=0)
        report.update(status="PASS_SAVED_FORWARD_ROLES_AND_METRICS", model_count=24,
                      result_sha256=file_sha(output / "result.json"),
                      scope_limit="Cached features/contexts and saved weights; not independent raw preprocessing or training reproduction")
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        write_json(output / f"{prefix}.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--verification-r1", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_saved(args.output, verification_r1=args.verification_r1,
                                matlab_layout=args.verification_r1), indent=2))
