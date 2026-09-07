"""Pure source-crossfit subset routing. No filesystem, toolbox, or raw-data loaders."""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
from scipy.stats import t

PLAN_SHA256 = "ee9b758f755a18f7540c978e9faf18a5711e652948b94df595377bbfb83cb80a"
MODES = ("Q", "QM", "SHAM_REFIT")
KEYS = ("interface", "n_samples", "method", "k")


def projection_arrays(payload, plan):
    ids = plan["source_subject_ids"]
    spec = plan["source_projection"]
    if (
        set(payload)
        != {
            "manifest_sha256",
            "packets",
            "returned_rows",
            "returned_packets",
            "returned_subject_ids",
            "columns",
        }
        or payload["manifest_sha256"] != spec["manifest_sha256"]
        or payload["returned_rows"] != spec["returned_rows"]
        or payload["returned_packets"] != spec["packets"]
        or payload["returned_subject_ids"] != ids
        or payload["columns"] != spec["columns"]
        or len(payload["packets"]) != len(ids) * 20
    ):
        raise ValueError("Projection does not match source-only contract")
    z = np.full((len(ids), 2, 10, 8), np.nan, dtype=np.float64)
    order = np.full(len(ids), -1, dtype=np.int64)
    seen = set()
    for packet in payload["packets"]:
        if set(packet) != {
            "subject_id",
            "interface",
            "block_id",
            "impedance_kohm",
            "headband_order",
            "condition_period",
        }:
            raise ValueError("Unexpected metadata fields")
        if packet["subject_id"] not in ids or packet["interface"] not in ("dry", "wet"):
            raise ValueError("Forbidden metadata participant/interface")
        p, i, b = (
            ids.index(packet["subject_id"]),
            ("dry", "wet").index(packet["interface"]),
            packet["block_id"],
        )
        if type(b) is not int or b not in range(10) or (p, i, b) in seen:
            raise ValueError("Invalid or duplicated metadata block")
        if packet["headband_order"] not in ("dry", "wet"):
            raise ValueError("Invalid shared headband order")
        first = ("dry", "wet").index(packet["headband_order"])
        if order[p] not in (-1, first) or packet["condition_period"] != (
            "first" if first == i else "second"
        ):
            raise ValueError("Inconsistent acquisition order/period")
        values = np.asarray(packet["impedance_kohm"], dtype=np.float64)
        if (
            values.shape != (8,)
            or np.isinf(values).any()
            or np.any(values[np.isfinite(values)] < 0)
        ):
            raise ValueError("Invalid impedance vector")
        z[p, i, b], order[p] = values, first
        seen.add((p, i, b))
    if len(seen) != len(ids) * 20 or (order < 0).any():
        raise ValueError("Incomplete projection")
    return z, order


def validate_cache(cache, plan):
    if set(cache) != set(plan["cache"]):
        raise ValueError("Unexpected feature cache keys")
    for name, spec in plan["cache"].items():
        x = cache[name]
        if x.shape != tuple(spec["shape"]) or x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError("Invalid feature shape/dtype/value: " + name)
        if name.endswith("_r") and np.max(np.abs(x)) > 1 + 1e-10:
            raise ValueError("Invalid native correlation")
    if np.any(cache["expert_r"][:, :, :, 0, 4:]) or np.any(cache["q_features"][:, :, :, 0, 3:]):
        raise ValueError("Padded expert/omission slots must be zero")


def native_scores(cache, plan):
    outputs = {}
    for name, method in (("a0_r", "A0_author"), ("expert_r", "ETRCA")):
        r = cache[name]
        out = np.empty(r.shape[:-2] + (12,), dtype=np.float64)
        for i, interface in enumerate(plan["interfaces"]):
            w = np.asarray(plan["native_weights"][method][interface])[None, :]
            shape = r[:, i].shape[:-2] + (12,)
            out[:, i] = np.asarray([w @ x for x in r[:, i].reshape(-1, 5, 12)]).reshape(shape)
        outputs[name] = out
    return outputs


def fit_metadata_scaler(z, indices):
    median = np.zeros((2, 8))
    scale = np.full((2, 8), 0.1)
    available = np.zeros((2, 8), dtype=bool)
    for i in range(2):
        for c in range(8):
            values = z[indices, i, :, c].ravel()
            values = np.log1p(values[np.isfinite(values)])
            if len(values):
                median[i, c] = np.median(values)
                scale[i, c] = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 0.1)
                available[i, c] = True
    return {"median": median.tolist(), "scale": scale.tolist(), "available": available.tolist()}


def support_maps(prefix):
    """All products of mask-group cyclic rotations, excluding global identity."""
    groups = {}
    for j, row in enumerate(prefix):
        groups.setdefault(tuple(np.isfinite(row)), []).append(j)
    groups = list(groups.values())
    maps = set()
    for shifts in itertools.product(*(range(len(g)) for g in groups)):
        mapping = np.arange(len(prefix))
        for group, shift in zip(groups, shifts):
            mapping[group] = np.roll(group, -shift)
        item = tuple(int(v) for v in mapping)
        if item != tuple(range(len(prefix))):
            maps.add(item)
    return sorted(maps)


def sham_map(prefix, plan, fold, subject, interface, k):
    maps = support_maps(prefix)
    if not maps:
        return tuple(range(k))
    key = f"{plan['study_id']}|{fold}|{subject}|{interface}|{k}"
    index = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(maps)
    return maps[index]


def packet_changes(prefix, mapping):
    return sum(
        not np.array_equal(prefix[b], prefix[mapping[b]], equal_nan=True)
        for b in range(len(prefix))
    )


def feature_changes(original, changed):
    """Omission/query rows with a numeric M change beyond summation noise."""
    return int(np.sum(np.any(np.abs(original[..., 33:] - changed[..., 33:]) > 1e-12, axis=-1)))


def fit_sham_diagnostics(z, indices, plan, fold):
    records = []
    for p in indices:
        for i, interface in enumerate(plan["interfaces"]):
            for k in plan["budgets"]:
                prefix = z[p, i, :k]
                maps = support_maps(prefix)
                mapping = sham_map(prefix, plan, fold, plan["source_subject_ids"][p], interface, k)
                records.append(
                    {
                        "participant": plan["source_subject_ids"][p],
                        "interface": interface,
                        "k": k,
                        "admissible_maps": len(maps),
                        "control_available": bool(maps),
                        "mapping": list(mapping),
                        "packet_changes": packet_changes(prefix, mapping),
                    }
                )
    return {"support_groups": records, "feature_change_tolerance": 1e-12}


def metadata_features(prefix, query_packets, q, scaler, interface):
    """Return [k,60,36] extras and 60 query-local exact-Q fallback flags."""
    median = np.asarray(scaler["median"])[interface]
    scale = np.asarray(scaler["scale"])[interface]
    available = np.asarray(scaler["available"], dtype=bool)[interface]

    def transform(raw):
        valid = np.isfinite(raw) & available
        values = np.zeros_like(raw, dtype=np.float64)
        transformed = (np.log1p(np.where(valid, raw, 0.0)) - median) / scale
        values[valid] = np.clip(transformed[valid], -8, 8)
        return values, valid

    sp, sv = transform(prefix)
    qp, qv = transform(query_packets)
    k = len(prefix)
    result = np.empty((k, 60, 36), dtype=np.float64)
    fallback = np.repeat(~np.any(qv & np.any(sv, axis=0), axis=1), 12)
    for j in range(k):
        keep = [b for b in range(k) if b != j]
        rv = sv[keep]
        count = rv.sum(axis=0)
        retained = np.sum(sp[keep], axis=0) / np.maximum(count, 1)
        difference = np.where(sv[j] & (count > 0), sp[j] - retained, 0.0)
        for block in range(5):
            omit_sq = np.where(qv[block] & sv[j], (sp[j] - qp[block]) ** 2, 0.0)
            retained_sq = np.mean(
                np.where(rv & qv[block], (sp[keep] - qp[block]) ** 2, 0.0), axis=0
            )
            aggregates = np.array([omit_sq.mean(), retained_sq.mean(), np.abs(difference).mean()])
            base = np.concatenate(
                [
                    omit_sq - retained_sq,
                    difference,
                    qp[block],
                    [qv[block].mean(), sv[j].mean(), rv.mean()],
                    aggregates,
                ]
            )
            sl = slice(block * 12, (block + 1) * 12)
            result[j, sl, :30] = base
            result[j, sl, 30:33] = q[j, sl, 20, None] * aggregates
            result[j, sl, 33:36] = q[j, sl, 28, None] * aggregates
    return result, fallback


def inputs_for_cell(cache, z, plan, scaler, p, i, w, bi, mode, fold, mapping=None):
    k = plan["budgets"][bi]
    q = cache["q_features"][p, i, w, bi, :k].reshape(k, 60, 33)
    if mode == "Q":
        return np.concatenate([q, np.zeros((k, 60, 36))], axis=-1), np.zeros(60, dtype=bool)
    prefix = z[p, i, :k]
    query = z[p, i, 5:10]
    if mode == "SHAM_REFIT":
        mapping = sham_map(
            prefix, plan, fold, plan["source_subject_ids"][p], plan["interfaces"][i], k
        )
    if mapping is not None:
        prefix = prefix[np.asarray(mapping)]
    if mode == "M_STALE":
        query = np.where(np.isfinite(query), z[p, i, k - 1], np.nan)
    extra, fallback = metadata_features(prefix, query, q, scaler, i)
    return np.concatenate([q, extra], axis=-1), fallback


def training_examples(cache, z, scores, plan, scaler, indices, mode, fold):
    matrices, targets, weights = [], [], []
    truth = np.tile(np.arange(12), 5)
    for p in indices:
        for i in range(2):
            for w in range(4):
                for bi, k in enumerate(plan["budgets"]):
                    x, _ = inputs_for_cell(cache, z, plan, scaler, p, i, w, bi, mode, fold)
                    pred = (
                        scores["expert_r"][p, i, w, bi, : k + 1]
                        .reshape(k + 1, 60, 12)
                        .argmax(axis=-1)
                    )
                    y = (pred[1:] == truth).astype(float) - (pred[0] == truth).astype(float)
                    matrices.append(x.reshape(-1, 69))
                    targets.append(y.ravel())
                    weights.append(np.full(k * 60, 1 / (len(indices) * 8 * 2 * k * 60)))
    return np.concatenate(matrices), np.concatenate(targets), np.concatenate(weights)


def fit_router(x, y, weights, ridge=0.1):
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("Invalid ridge inputs")
    if abs(weights.sum() - 1) > 1e-12 or ridge != 0.1:
        raise ValueError("Unfrozen ridge mass/penalty")
    mean = np.sum(weights[:, None] * x, axis=0)
    sd = np.sqrt(np.sum(weights[:, None] * (x - mean) ** 2, axis=0))
    effective = int(np.sum(sd >= 1e-12))
    sd[sd < 1e-12] = 1
    design = np.column_stack([np.ones(len(x)), np.clip((x - mean) / sd, -10, 10)])
    penalty = np.eye(70) * ridge
    penalty[0, 0] = 0
    coef = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty, design.T @ (weights * y)
    )
    mse = float(np.sum(weights * (design @ coef - y) ** 2))
    regularizer = float(ridge * np.sum(coef[1:] ** 2))
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": sd.tolist(),
        "coef": coef.tolist(),
        "train_weighted_mse": mse,
        "ridge_penalty": regularizer,
        "objective": mse + regularizer,
        "train_rows": len(x),
        "effective_nonconstant_features": effective,
    }


def predict_gain(x, router):
    values = np.clip((x - router["feature_mean"]) / router["feature_scale"], -10, 10)
    coef = np.asarray(router["coef"])
    return np.round(values @ coef[1:] + coef[0], 10)


def choose(gains):
    return np.column_stack([np.zeros(gains.shape[1]), gains.T]).argmax(axis=-1)


def fit_all(cache, projection, plan):
    validate_cache(cache, plan)
    z, order = projection_arrays(projection, plan)
    scores = native_scores(cache, plan)
    ids = plan["source_subject_ids"]
    result = {"schema": "cfeg.native-subset-m.freezes.v1", "plan_sha256": PLAN_SHA256, "folds": []}
    for p in range(len(ids)):
        for i in range(2):
            for bi, k in enumerate((3, 5)):
                q = cache["q_features"][p, i, :, bi, :k]
                if not np.all(q[..., 6] == order[p]) or not np.all(q[..., 7] == int(order[p] != i)):
                    raise ValueError("Shared order features contradict projection")
    for fold in range(3):
        fit = [p for p in range(len(ids)) if p % 3 != fold]
        evaluation = [p for p in range(len(ids)) if p % 3 == fold]
        scaler = fit_metadata_scaler(z, fit)
        routers = {}
        sham_diagnostics = fit_sham_diagnostics(z, fit, plan, fold)
        for mode in MODES:
            x, y, weights = training_examples(cache, z, scores, plan, scaler, fit, mode, fold)
            routers[mode] = fit_router(x, y, weights)
            if mode == "QM":
                qm_inputs = x
            elif mode == "SHAM_REFIT":
                sham_diagnostics["omission_query_rows"] = len(x)
                sham_diagnostics["changed_feature_rows"] = feature_changes(qm_inputs, x)
        result["folds"].append(
            {
                "fold": fold,
                "fit_subject_ids": [ids[p] for p in fit],
                "evaluation_subject_ids": [ids[p] for p in evaluation],
                "metadata_scaler": scaler,
                "routers": routers,
                "sham_diagnostics": sham_diagnostics,
            }
        )
    return result


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    radius = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {
        "mean": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "n_participants": len(values),
    }


def reporting(rows, plan):
    ids = plan["source_subject_ids"]
    lookup = {tuple(row[k] for k in ("participant", *KEYS)): row["ba"] for row in rows}
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in KEYS), []).append(row["ba"])
    summary = []
    for key, values in sorted(groups.items()):
        stat = interval(values)
        summary.append({**dict(zip(KEYS, key)), "mean_ba": stat.pop("mean"), **stat})
    comparisons = []
    for k in (3, 5):
        for method in ("Q", "FULL", "A0_author", "SHAM_REFIT", "M_SHUFFLE", "M_STALE"):
            comparisons.append(
                (
                    f"QM{k}_minus_{method}{0 if method == 'A0_author' else k}",
                    "QM",
                    k,
                    method,
                    0 if method == "A0_author" else k,
                )
            )
        comparisons.append((f"Q{k}_minus_FULL{k}", "Q", k, "FULL", k))
    comparisons.extend(
        [
            ("Q5_minus_Q3", "Q", 5, "Q", 3),
            ("QM3_minus_Q5", "QM", 3, "Q", 5),
            ("QM5_minus_QM3", "QM", 5, "QM", 3),
        ]
    )
    contrasts, attainment = [], []
    for name, m1, k1, m2, k2 in comparisons:
        deltas = []
        for interface in ("dry", "wet"):
            for n in plan["sample_counts"]:
                delta = [
                    lookup[(p, interface, n, m1, k1)] - lookup[(p, interface, n, m2, k2)]
                    for p in ids
                ]
                deltas.append(delta)
                contrasts.append(
                    {"contrast": name, "interface": interface, "n_samples": n, **interval(delta)}
                )
        contrasts.append(
            {
                "contrast": name,
                "interface": "all8",
                "n_samples": None,
                **interval(np.mean(deltas, axis=0)),
            }
        )
    for p in ids:
        for interface in ("dry", "wet"):
            for n in plan["sample_counts"]:
                for method in ("FULL", "Q", "QM", "SHAM_REFIT"):
                    first = next(
                        (
                            k
                            for k, ba in [(0, lookup[(p, interface, n, "A0_author", 0)])]
                            + [(k, lookup[(p, interface, n, method, k)]) for k in (3, 5)]
                            if ba >= 0.8
                        ),
                        None,
                    )
                    attainment.append(
                        {
                            "participant": p,
                            "interface": interface,
                            "n_samples": n,
                            "method": method,
                            "first_observed_k": first,
                            "unreached_tested_grid": first is None,
                            "label_count": None if first is None else 12 * first,
                            "retained_seconds": None if first is None else 12 * first * n / 250,
                            "history_seconds": None if first is None else 12 * first * 0.14,
                        }
                    )
    return summary, contrasts, attainment


def evaluate_all(cache, projection, freezes, plan):
    validate_cache(cache, plan)
    z, _ = projection_arrays(projection, plan)
    scores = native_scores(cache, plan)
    rows, diagnostic = [], []
    truth = np.tile(np.arange(12), 5)
    ids = plan["source_subject_ids"]
    if len(freezes["folds"]) != 3 or freezes["plan_sha256"] != PLAN_SHA256:
        raise ValueError("Invalid fit freeze")
    for p, participant in enumerate(ids):
        fold = p % 3
        freeze = freezes["folds"][fold]
        if (
            freeze["fold"] != fold
            or freeze["fit_subject_ids"] != [v for j, v in enumerate(ids) if j % 3 != fold]
            or freeze["evaluation_subject_ids"] != [v for j, v in enumerate(ids) if j % 3 == fold]
        ):
            raise ValueError("Fit/evaluation participant leakage or drift")
        routers, scaler = freeze["routers"], freeze["metadata_scaler"]
        for i, interface in enumerate(plan["interfaces"]):
            for w, n in enumerate(plan["sample_counts"]):

                def emit(
                    method,
                    k,
                    ba,
                    interventions=1,
                    available=True,
                    participant=participant,
                    interface=interface,
                    n=n,
                ):
                    rows.append(
                        {
                            "participant": participant,
                            "interface": interface,
                            "n_samples": n,
                            "method": method,
                            "k": k,
                            "ba": float(ba),
                            "query_count": 60,
                            "label_count": 12 * k,
                            "intervention_count": interventions,
                            "control_available": bool(available),
                        }
                    )

                a0 = scores["a0_r"][p, i, w].reshape(60, 12).argmax(axis=-1)
                emit("A0_author", 0, np.mean(a0 == truth))
                for bi, k in enumerate((3, 5)):
                    pred = (
                        scores["expert_r"][p, i, w, bi, : k + 1]
                        .reshape(k + 1, 60, 12)
                        .argmax(axis=-1)
                    )
                    actions, decisions = {}, {"FULL": pred[0]}
                    fallbacks, mode_inputs = {}, {}
                    for mode in ("Q", "QM", "SHAM_REFIT", "M_STALE"):
                        x, fallback = inputs_for_cell(
                            cache, z, plan, scaler, p, i, w, bi, mode, fold
                        )
                        router = routers["QM" if mode == "M_STALE" else mode]
                        action = choose(predict_gain(x, router))
                        if mode != "Q":
                            action[fallback] = actions["Q"][fallback]
                        actions[mode] = action
                        decisions[mode] = pred[action, np.arange(60)]
                        fallbacks[mode] = fallback
                        mode_inputs[mode] = x
                    maps = support_maps(z[p, i, :k])
                    for mode in ("FULL", "Q", "QM", "SHAM_REFIT", "M_STALE"):
                        available = bool(maps) if mode == "SHAM_REFIT" else True
                        emit(mode, k, np.mean(decisions[mode] == truth), int(available), available)
                    emit("M_MISSING", k, np.mean(decisions["Q"] == truth), 0)
                    shuffle_records = []
                    for mapping in maps or [tuple(range(k))]:
                        x, fallback = inputs_for_cell(
                            cache, z, plan, scaler, p, i, w, bi, "QM", fold, mapping
                        )
                        action = choose(predict_gain(x, routers["QM"]))
                        action[fallback] = actions["Q"][fallback]
                        prediction = pred[action, np.arange(60)]
                        prefix = z[p, i, :k]
                        shuffle_records.append(
                            {
                                "mapping": list(mapping),
                                "control_available": bool(maps),
                                "packet_changes": packet_changes(prefix, mapping),
                                "feature_changes": feature_changes(mode_inputs["QM"], x),
                                "selector_changes": int(np.sum(action != actions["QM"])),
                                "prediction_changes": int(np.sum(prediction != decisions["QM"])),
                                "ba": float(np.mean(prediction == truth)),
                                "actions": action.tolist(),
                            }
                        )
                    emit(
                        "M_SHUFFLE",
                        k,
                        np.mean([v["ba"] for v in shuffle_records]),
                        len(maps),
                        bool(maps),
                    )
                    qright, mright = decisions["Q"] == truth, decisions["QM"] == truth
                    diagnostic.append(
                        {
                            "participant": participant,
                            "interface": interface,
                            "n_samples": n,
                            "k": k,
                            "expert_disagreement_queries": int(
                                np.sum(np.any(pred[1:] != pred[0], axis=0))
                            ),
                            "qm_vs_q_selector_changes": int(np.sum(actions["QM"] != actions["Q"])),
                            "qm_vs_q_prediction_changes": int(
                                np.sum(decisions["QM"] != decisions["Q"])
                            ),
                            "qm_repaired_q": int(np.sum(mright & ~qright)),
                            "qm_damaged_q": int(np.sum(qright & ~mright)),
                            "qm_missing_fallback_queries": int(fallbacks["QM"].sum()),
                            "admissible_shuffle_maps": len(maps),
                            "mean_shuffled_packet_changes": float(
                                np.mean([v["packet_changes"] for v in shuffle_records])
                            ),
                            "mean_shuffle_feature_changes": float(
                                np.mean([v["feature_changes"] for v in shuffle_records])
                            ),
                            "mean_shuffle_selector_changes": float(
                                np.mean([v["selector_changes"] for v in shuffle_records])
                            ),
                            "mean_shuffle_prediction_changes": float(
                                np.mean([v["prediction_changes"] for v in shuffle_records])
                            ),
                            "sham_feature_changes": feature_changes(
                                mode_inputs["QM"], mode_inputs["SHAM_REFIT"]
                            ),
                            "stale_feature_changes": feature_changes(
                                mode_inputs["QM"], mode_inputs["M_STALE"]
                            ),
                            "feature_change_tolerance": 1e-12,
                            "shuffle_interventions": shuffle_records,
                            "actions": {name: value.tolist() for name, value in actions.items()},
                        }
                    )
    rows.sort(key=lambda row: tuple(row[k] for k in ("participant", *KEYS)))
    summary, contrasts, attainment = reporting(rows, plan)
    if len(rows) != 4680 or len(summary) != 120 or len(contrasts) != 153 or len(attainment) != 1248:
        raise ValueError("Incomplete development reporting grid")
    return {
        "rows": rows,
        "summary": summary,
        "contrasts": contrasts,
        "diagnostics": diagnostic,
        "attainment": attainment,
        "evidence_scope": "adaptively exposed development, descriptive intervals, no automatic promotion",
    }
