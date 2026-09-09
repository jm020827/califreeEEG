"""Single-attempt generated-only orchestration; never a human-data reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import stat
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

REGIMES = ("coupled", "null")
SCHEMA = "n1-metadata-generated-efficacy-result-v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def descriptor(path):
    path = Path(path)
    if path.resolve() != path or path.is_symlink():
        raise ValueError("unaliased absolute path required")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("regular single-link artifact required")
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            hasher.update(block)
        digest = hasher.hexdigest()
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("artifact changed during read")
    return {"sha256": digest, "bytes": before.st_size}


def publish(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o400)


def save_arrays(path, **arrays):
    if any(np.asarray(a).dtype.kind not in "biuf" for a in arrays.values()):
        raise ValueError("numeric arrays only")
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o400)


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite paired group differences required")
    mean = float(values.mean())
    mcse = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(t.ppf(0.975, len(values) - 1)) * mcse
    return {
        "mean_pp": mean,
        "mcse_pp": mcse,
        "low_pp": mean - half,
        "high_pp": mean + half,
        "n_groups": len(values),
    }


def summarize(scores, oracle_scores, groups, arms, config, *, validity, actuation):
    """Engineering gates with descriptive conditional paired-group intervals."""
    scores, oracle_scores, groups = map(np.asarray, (scores, oracle_scores, groups))
    gcount = config["evaluation_groups"]
    if scores.shape != (2, 2 * gcount, 10, 48, 12):
        raise ValueError("complete frozen score grid required")
    if oracle_scores.shape != (2, 2 * gcount, 48, 12):
        raise ValueError("complete frozen oracle grid required")
    if groups.shape != (2 * gcount,) or not np.array_equal(groups, np.tile(np.arange(gcount), 2)):
        raise ValueError("complete member-major group grid required")
    all_scores = np.concatenate((scores, oracle_scores[:, :, None]), axis=2)
    if not np.isfinite(all_scores).all():
        raise ValueError("nonfinite scores")
    correct = all_scores.argmax(-1) == np.tile(np.arange(12), 4)
    accuracy = np.stack(
        [correct[:, groups == g].mean(axis=(1, 3)) * 100 for g in range(gcount)], axis=1
    )
    qi, q2i, mi, si = (arms.index(a) for a in ("Q", "Q2", "QM", "SHAM_REFIT"))
    gain, null = accuracy[0, :, mi] - accuracy[0, :, qi], accuracy[1, :, mi] - accuracy[1, :, qi]
    contrasts = {
        "coupled_QM_minus_Q": interval(gain),
        "coupled_QM_minus_Q2": interval(accuracy[0, :, mi] - accuracy[0, :, q2i]),
        "coupled_QM_minus_SHAM_REFIT": interval(accuracy[0, :, mi] - accuracy[0, :, si]),
        "coupled_ORACLE_minus_Q": interval(accuracy[0, :, -1] - accuracy[0, :, qi]),
        "null_QM_minus_Q": interval(null),
        "interaction": interval(gain - null),
    }
    checks = dict(validity)
    oracle, neg = contrasts["coupled_ORACLE_minus_Q"], contrasts["null_QM_minus_Q"]
    low, high = config["q_accuracy_range_percent"]
    checks["positive_headroom"] = bool(
        low <= accuracy[0, :, qi].mean() <= high
        and oracle["mean_pp"] >= config["oracle_min_gain_pp"]
        and oracle["low_pp"] > 0
    )
    checks["null_equivalence"] = bool(
        neg["low_pp"] >= -config["null_equivalence_pp"]
        and neg["high_pp"] <= config["null_equivalence_pp"]
    )
    checks["positive_improvement"] = all(
        contrasts[key]["mean_pp"] >= config["learner_min_gain_pp"] and contrasts[key]["low_pp"] > 0
        for key in (
            "coupled_QM_minus_Q",
            "coupled_QM_minus_Q2",
            "coupled_QM_minus_SHAM_REFIT",
            "interaction",
        )
    )
    checks["metadata_learned_and_actuated"] = all(
        actuation["coupled"][key] > 0
        for key in (
            "coefficient_max_abs",
            "gradient_max",
            "max_abs_R",
            "max_abs_score",
            "argmax_changed",
        )
    )
    if not all(validity.values()):
        terminal = "EFFICACY_NOT_EVALUATED"
    elif not checks["positive_headroom"]:
        terminal = "POSITIVE_CONTROL_NOT_QUALIFIED"
    elif not checks["null_equivalence"]:
        terminal = "NEGATIVE_CONTROL_NOT_QUALIFIED"
    elif not (checks["positive_improvement"] and checks["metadata_learned_and_actuated"]):
        terminal = "LEARNER_CAPACITY_NOT_ESTABLISHED"
    else:
        terminal = "GENERATED_M_CAPACITY_PASS"
    return {
        "terminal": terminal,
        "group_accuracy_percent": accuracy.tolist(),
        "contrasts": contrasts,
        "checks": checks,
    }


def primary(folder, config):
    import torch

    from cfeg.analysis import n1_metadata_generated_efficacy_fixture as fixture
    from cfeg.analysis import task_trca_n1_evaluation as evaluation
    from cfeg.analysis import task_trca_n1_learning as learn
    from cfeg.analysis import task_trca_n1_signfree as temporal
    from cfeg.analysis import task_trca_shape_operator as shape

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    if folder.resolve() != folder or folder.is_symlink() or list(folder.iterdir()):
        raise ValueError("new empty unaliased primary folder required")
    publish(
        folder / "primary_start.json",
        {
            "schema": "n1-metadata-generated-efficacy-primary-start-v1",
            "utc": utc(),
            "config": config,
            "registered_attempt": 1,
            "charged_updates": config["updates_total"],
            "prohibited": ["human_data", "held60", "old_models", "outreach", "paid", "gpu"],
        },
    )
    events = (folder / "events.jsonl").open("x")
    sequence = 0
    completed = 0

    def event(name, **fields):
        nonlocal sequence
        sequence += 1
        events.write(
            json.dumps(dict(sequence=sequence, event=name, utc=utc(), **fields), allow_nan=False)
            + "\n"
        )
        events.flush()
        os.fsync(events.fileno())
        print(json.dumps(dict(event=name, **fields)), flush=True)

    try:
        event("generation_started")
        manifest = fixture.write_fixture(folder, config)
        event("generation_completed", artifacts=manifest["artifacts"])
        for name, item in manifest["artifacts"].items():
            if descriptor(folder / name) != item:
                raise ValueError("generated fixture hash mismatch")
        frequencies = np.arange(6, 18) * config["sampling_rate"] / config["samples"]
        weights = np.asarray(config["weights"], dtype=np.float64)
        pipelines, feature_rows = {}, {key: [] for key in ("q", "m", "s", "c")}
        geometry_max = 0.0
        donors_good = True
        # Only support/source fitting arrays are decoded here. Query NPZ remains unopened.
        with np.load(folder / "fit.npz", allow_pickle=False) as fit:
            for regime in REGIMES:
                support, source, packets = fit["support"], fit["source"], fit["packet_" + regime]
                cases = tuple(
                    learn.make_task_case(
                        int(pid),
                        0,
                        0,
                        support[i],
                        packets[i],
                        frequencies,
                        source[i],
                        weights=weights,
                    )
                    for i, pid in enumerate(fit["ids"])
                )
                for key, feature_values in feature_rows.items():
                    feature_values.append(np.stack([getattr(c, key) for c in cases]))
                ideal_s = 6 * config["samples"] * np.ones((8, 8))
                ideal_c = 3 * config["samples"] * (np.ones((8, 8)) + 0.04 * np.eye(8))
                geometry_max = max(
                    geometry_max,
                    max(float((c.s - torch.tensor(ideal_s)).abs().max()) for c in cases),
                    max(float((c.c - torch.tensor(ideal_c)).abs().max()) for c in cases),
                )
                event(
                    "fit_started", regime=regime, charged_updates=800, completed_updates=completed
                )
                pipelines[regime] = learn.fit_pipeline(
                    cases, config["regularization"], backend="batch"
                )
                completed += 800
                id_groups = dict(zip(fit["ids"].tolist(), fit["groups"].tolist()))
                donors_good &= all(
                    id_groups[key[0]] != id_groups[donor]
                    for key, donor in pipelines[regime].donor_ids.items()
                )
                event(
                    "fit_completed",
                    regime=regime,
                    completed_updates=completed,
                    q_hash=pipelines[regime].q_hash,
                )
                del cases, support, source, packets
        save_arrays(
            folder / "fit_features.npz", **{k: np.stack(v) for k, v in feature_rows.items()}
        )
        del feature_rows
        publish(
            folder / "models.json",
            {
                "schema": "n1-metadata-generated-efficacy-models-v1",
                "pipelines": {k: p.record() for k, p in pipelines.items()},
                "oracle": {"coefficients": config["oracle_coefficients"], "trained": False},
            },
        )
        publish(
            folder / "freeze.json",
            {
                "schema": "n1-metadata-generated-efficacy-freeze-v1",
                "utc": utc(),
                "models_sha256": descriptor(folder / "models.json")["sha256"],
                "query_sha256": manifest["artifacts"]["query.npz"]["sha256"],
                "completed_updates": completed,
                "event": "both_models_frozen_before_query_decode",
            },
        )
        event(
            "both_models_frozen_before_query_decode",
            completed_updates=completed,
            models_sha256=descriptor(folder / "models.json")["sha256"],
        )
        results = {
            k: []
            for k in ("scores", "r", "projectors", "oracle_scores", "oracle_r", "oracle_projectors")
        }
        actuation = {}
        with (
            np.load(folder / "support.npz", allow_pickle=False) as support,
            np.load(folder / "query.npz", allow_pickle=False) as query_file,
        ):
            event("query_decode_started")
            query = query_file["query"]
            ids = tuple(map(int, support["ids"]))
            groups = support["groups"]
            if not np.array_equal(query_file["ids"], ids) or not np.array_equal(
                query_file["groups"], groups
            ):
                raise ValueError("query/support identity mismatch")
            x = support["support"]
            for regime in REGIMES:
                p = pipelines[regime]
                packets = support["packet_" + regime]
                states = tuple(
                    evaluation.support_state(pid, 0, 0, x[i], packets[i], frequencies, weights)
                    for i, pid in enumerate(ids)
                )
                partition = evaluation.EvaluationPartition(
                    states, ids, ((0, config["samples"], 3),)
                )
                id_groups = dict(zip(ids, groups.tolist()))
                donors_good &= all(
                    id_groups[ids[i]] != id_groups[ids[(i + 1) % len(ids)]] for i in range(len(ids))
                )
                rows = {key: [] for key in results}
                a = {
                    key: 0
                    for key in (
                        "max_abs_R",
                        "max_abs_F",
                        "max_abs_J",
                        "max_abs_score",
                        "argmax_changed",
                    )
                }
                a["gradient_max"] = max(v["gradient_norm"] for v in p.residuals["QM"].trace)
                a["coefficient_max_abs"] = float(np.abs(p.residuals["QM"].coefficients).max())
                for i, state in enumerate(states):
                    value = evaluation.evaluate(p, state, query[i], partition=partition)
                    for key in ("scores", "r", "projectors"):
                        rows[key].append(value[key])
                    with torch.no_grad():
                        qlog = learn._head(
                            evaluation._tensor(p.q_scaler.transform(state.q)),
                            evaluation._tensor(p.q.coefficients),
                            0.8,
                        )
                        mlog = learn._head(
                            evaluation._tensor(p.m_scaler.transform(state.m, state.available)),
                            evaluation._tensor(config["oracle_coefficients"]),
                            0.2,
                        )
                        r = shape.shape_prior(qlog + mlog * evaluation._tensor(state.available))
                        f = temporal.bounded_projectors(state.s, state.c, r[:, None, :])
                        stats = temporal.temporal_statistics(
                            evaluation._tensor(state.model.templates), evaluation._tensor(query[i])
                        )
                        oscores, _ = temporal.score_temporal_gram(f, stats, state.weights)
                    rows["oracle_scores"].append(oscores.numpy())
                    rows["oracle_r"].append(r.numpy())
                    rows["oracle_projectors"].append(f.numpy())
                    for key, difference in value["actuation_QM_minus_Q"].items():
                        a[key] = (
                            a[key] + difference
                            if key == "argmax_changed"
                            else max(a[key], difference)
                        )
                for key, result_values in results.items():
                    result_values.append(np.stack(rows[key]))
                actuation[regime] = a
                event("regime_evaluation_completed", regime=regime)
        arrays = {k: np.stack(v) for k, v in results.items()}
        save_arrays(folder / "evaluation.npz", **arrays)
        common_q = pipelines["coupled"].q_hash == pipelines["null"].q_hash
        validity = {
            "finite_arrays": all(bool(np.isfinite(v).all()) for v in arrays.values()),
            "common_q_hashes": common_q,
            "constant_support_geometry": geometry_max <= config["numeric_atol"],
            "donors_cross_group": bool(donors_good),
        }
        summary = summarize(
            arrays["scores"],
            arrays["oracle_scores"],
            groups,
            evaluation.ARMS,
            config,
            validity=validity,
            actuation=actuation,
        )
        event("evaluation_completed", terminal=summary["terminal"], completed_updates=completed)
        events.close()
        (folder / "events.jsonl").chmod(0o400)
        artifacts = {p.name: descriptor(p) for p in sorted(folder.iterdir()) if p.is_file()}
        result = dict(
            schema=SCHEMA,
            config=config,
            arms=list(evaluation.ARMS),
            **summary,
            actuation=actuation,
            pipeline_q_hashes={k: p.q_hash for k, p in pipelines.items()},
            updates_completed=completed,
            geometry_max_abs=geometry_max,
            artifacts=artifacts,
        )
        publish(folder / "result.json", result)
        return result
    except BaseException as exc:
        if not events.closed:
            event(
                "primary_failed",
                error_type=type(exc).__name__,
                message=str(exc),
                updates_completed_lower_bound=completed,
                charged_updates=config["updates_total"],
            )
            events.close()
            (folder / "events.jsonl").chmod(0o400)
        publish(
            folder / "failure.json",
            {
                "terminal": "EFFICACY_NOT_EVALUATED",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "updates_completed_lower_bound": completed,
                "charged_updates": config["updates_total"],
                "partial_update_count_known": False,
                "utc": utc(),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("primary", "audit"))
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    descriptor(args.config)
    config = json.loads(args.config.read_text())
    resource.setrlimit(resource.RLIMIT_AS, (config["address_limit_bytes"],) * 2)
    if args.mode == "primary":
        result = primary(args.folder, config)
        print(
            json.dumps({"terminal": result["terminal"], "updates": result["updates_completed"]}),
            flush=True,
        )
    else:
        from cfeg.analysis.n1_metadata_generated_efficacy_audit import audit

        result = audit(args.folder, config)
        publish(args.folder / "audit.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
        if result["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
