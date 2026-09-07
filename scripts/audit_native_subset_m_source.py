"""Independent artifact-only audit of the frozen native subset M experiment.

Never imports the producer/core/toolbox and never opens human raw EEG, the
original projection, the old baseline cache, or the full metadata manifest.
Signal extraction is outside this audit: it replays saved correlations and Q
features, independently checking every Q coordinate derivable from that cache.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import stat
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import t

PLAN_SHA256 = "ee9b758f755a18f7540c978e9faf18a5711e652948b94df595377bbfb83cb80a"
ROW_KEYS = ("participant", "interface", "n_samples", "method", "k")
CELL_KEYS = ROW_KEYS[1:]
ROUTERS = ("Q", "QM", "SHAM_REFIT")
FEATURE_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label, atol=1e-12):
    """Recursive structural comparison, including exact boolean/integer types."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ": keys")
        for key in expected:
            equal(actual[key], expected[key], label + "." + str(key), atol)
    elif isinstance(expected, (list, tuple)):
        require(
            isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ": length"
        )
        for index, value in enumerate(expected):
            equal(actual[index], value, f"{label}[{index}]", atol)
    elif isinstance(expected, (bool, int)) or expected is None:
        require(type(actual) is type(expected) and actual == expected, label + ": exact value/type")
    elif isinstance(expected, (float, np.floating)):
        require(
            isinstance(actual, (float, int))
            and not isinstance(actual, bool)
            and np.isfinite(actual)
            and abs(actual - expected) <= atol,
            label + ": numeric value",
        )
    else:
        require(actual == expected, label + ": value")


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.allclose(actual, expected, atol=atol, rtol=0),
        label + ": array mismatch",
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_plan(path):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == PLAN_SHA256, "Frozen plan hash")
    return json.loads(data)


def load_projection(wrapper, plan):
    spec = plan["source_projection"]
    require(
        set(wrapper) == {"schema", "input_path", "input_sha256", "projection"}, "Projection wrapper"
    )
    equal(wrapper["schema"], "cfeg.native-subset-m.projection.v1", "Projection schema")
    equal(wrapper["input_path"], spec["path"], "Projection origin path")
    equal(wrapper["input_sha256"], spec["sha256"], "Projection origin hash")
    data = wrapper["projection"]
    require(
        set(data)
        == {
            "manifest_sha256",
            "packets",
            "returned_rows",
            "returned_packets",
            "returned_subject_ids",
            "columns",
        },
        "Projection fields",
    )
    ids = plan["source_subject_ids"]
    for key, expected in (
        ("manifest_sha256", spec["manifest_sha256"]),
        ("returned_rows", spec["returned_rows"]),
        ("returned_packets", len(ids) * 20),
        ("returned_subject_ids", ids),
        ("columns", spec["columns"]),
    ):
        equal(data[key], expected, "Projection " + key)
    require(len(data["packets"]) == len(ids) * 20, "Projection packet count")
    impedance = np.full((len(ids), 2, 10, 8), np.nan)
    order = np.full(len(ids), -1, dtype=int)
    seen = set()
    for packet in data["packets"]:
        require(
            set(packet)
            == {
                "subject_id",
                "interface",
                "block_id",
                "impedance_kohm",
                "headband_order",
                "condition_period",
            },
            "Packet fields",
        )
        subject, interface, block = packet["subject_id"], packet["interface"], packet["block_id"]
        require(type(subject) is int and subject in ids, "Packet subject")
        require(
            interface in ("dry", "wet") and type(block) is int and 0 <= block < 10,
            "Packet interface/block",
        )
        key = subject, interface, block
        require(key not in seen, "Duplicate metadata packet")
        seen.add(key)
        p, i = ids.index(subject), ("dry", "wet").index(interface)
        require(packet["headband_order"] in ("dry", "wet"), "Headband order")
        first = int(packet["headband_order"] == "wet")
        require(order[p] in (-1, first), "Inconsistent order")
        equal(packet["condition_period"], "first" if i == first else "second", "Condition period")
        raw = packet["impedance_kohm"]
        require(isinstance(raw, list) and len(raw) == 8, "Impedance vector")
        require(
            all(
                value is None or (isinstance(value, (float, int)) and not isinstance(value, bool))
                for value in raw
            ),
            "Impedance numeric types",
        )
        vector = np.asarray(raw, dtype=np.float64)
        require(
            not np.isinf(vector).any() and not np.any(vector[np.isfinite(vector)] < 0),
            "Impedance domain",
        )
        impedance[p, i, block], order[p] = vector, first
    return impedance, order


def cache_scores(cache, plan):
    require(set(cache) == set(plan["cache"]), "Cache array keys")
    for name, array in cache.items():
        require(
            array.shape == tuple(plan["cache"][name]["shape"])
            and array.dtype == np.float64
            and np.isfinite(array).all(),
            "Cache " + name,
        )
        if name.endswith("_r"):
            require(np.max(np.abs(array)) <= 1 + 1e-10, "Correlation domain " + name)
    require(not cache["expert_r"][:, :, :, 0, 4:].any(), "Padded experts")
    require(not cache["q_features"][:, :, :, 0, 3:].any(), "Padded Q features")
    result = {}
    for name, method in (("a0_r", "A0_author"), ("expert_r", "ETRCA")):
        values = cache[name]
        scores = np.empty(values.shape[:-2] + (12,))
        for i, interface in enumerate(plan["interfaces"]):
            weight = np.asarray(plan["native_weights"][method][interface]).reshape(1, 5)
            # Broadcast native row @ matrix, retaining the declared dot order.
            scores[:, i] = np.matmul(weight, values[:, i]).squeeze(-2)
        result[name] = scores
    return result


def verify_q(cache, scores, order, plan):
    """Raw-free checks: protocol 0:8, confidence 8:21, deltas/interactions."""
    maximum = 0.0
    for p in range(len(order)):
        for i, interface in enumerate(plan["interfaces"]):
            aw = sum(plan["native_weights"]["A0_author"][interface])
            ew = sum(plan["native_weights"]["ETRCA"][interface])
            for wi, n in enumerate(plan["sample_counts"]):
                a0 = scores["a0_r"][p, i, wi].reshape(60, 12) / aw
                a0sort = np.sort(a0, axis=1)
                for bi, k in enumerate(plan["budgets"]):
                    native = scores["expert_r"][p, i, wi, bi, : k + 1].reshape(k + 1, 60, 12) / ew
                    full, fullsort = native[0], np.sort(native[0], axis=1)
                    q = cache["q_features"][p, i, wi, bi, :k].reshape(k, 60, 33)
                    for j in range(k):
                        candidate, ordered = native[j + 1], np.sort(native[j + 1], axis=1)
                        expected = np.empty((60, 21))
                        expected[:, 0] = i
                        expected[:, 1] = n / 500
                        expected[:, 2] = k / 5
                        expected[:, 3] = np.repeat(np.arange(5, 10), 12) / 9
                        expected[:, 4] = j / 4
                        expected[:, 5] = (np.repeat(np.arange(5, 10), 12) - j) / 9
                        expected[:, 6] = order[p]
                        expected[:, 7] = int(order[p] != i)
                        expected[:, 8] = fullsort[:, -1]
                        expected[:, 9] = fullsort[:, -1] - fullsort[:, -2]
                        expected[:, 10] = np.std(full, axis=1)
                        expected[:, 11] = a0sort[:, -1]
                        expected[:, 12] = a0sort[:, -1] - a0sort[:, -2]
                        expected[:, 13] = full.argmax(1) == a0.argmax(1)
                        expected[:, 14] = ordered[:, -1]
                        expected[:, 15] = ordered[:, -1] - ordered[:, -2]
                        expected[:, 16] = np.sqrt(np.mean((candidate - full) ** 2, axis=1))
                        expected[:, 17] = candidate.argmax(1) == full.argmax(1)
                        expected[:, 18] = candidate.argmax(1) == a0.argmax(1)
                        expected[:, 19] = expected[:, 14] - expected[:, 8]
                        expected[:, 20] = expected[:, 15] - expected[:, 9]
                        close(q[j, :, :21], expected, "Q protocol/confidence", 2e-12)
                        maximum = max(maximum, float(np.max(np.abs(q[j, :, :21] - expected))))
                        close(q[j, :, 27], q[j, :, 25] - q[j, :, 26], "Q distance delta")
                        close(q[j, :, 30], q[j, :, 20] ** 2, "Q margin square")
                        close(q[j, :, 31], q[j, :, 27] * q[j, :, 28], "Q covariance interaction")
                        close(
                            q[j, :, 32],
                            (q[j, :, 23] - q[j, :, 24]) * q[j, :, 20],
                            "Q support interaction",
                        )
    return maximum


def impedance_scaler(z, fit):
    median, scale, available = np.zeros((2, 8)), np.full((2, 8), 0.1), np.zeros((2, 8), bool)
    for i, c in itertools.product(range(2), range(8)):
        values = z[np.asarray(fit), i, :, c].reshape(-1)
        values = np.log1p(values[np.isfinite(values)])
        if values.size:
            quartiles = np.quantile(values, [0.25, 0.5, 0.75])
            median[i, c], scale[i, c], available[i, c] = (
                quartiles[1],
                max(0.1, quartiles[2] - quartiles[0]),
                True,
            )
    return {"median": median.tolist(), "scale": scale.tolist(), "available": available.tolist()}


def permutation_family(prefix):
    groups = defaultdict(list)
    for index, row in enumerate(prefix):
        groups[tuple(np.flatnonzero(np.isfinite(row)))].append(index)
    group_rotations = []
    for indices in groups.values():
        group_rotations.append([(indices, indices[s:] + indices[:s]) for s in range(len(indices))])
    output = []
    for selection in itertools.product(*group_rotations):
        assignment = list(range(len(prefix)))
        for original, rotated in selection:
            for left, right in zip(original, rotated):
                assignment[left] = right
        if assignment != list(range(len(prefix))):
            output.append(tuple(assignment))
    return sorted(output)


def select_sham(prefix, study_id, fold, participant, interface):
    maps = permutation_family(prefix)
    if not maps:
        return tuple(range(len(prefix)))
    text = "|".join(map(str, (study_id, fold, participant, interface, len(prefix))))
    return maps[int.from_bytes(hashlib.sha256(text.encode()).digest(), "big") % len(maps)]


def m_features(prefix, query, q, scaler, interface):
    """Independent broadcast construction; missing retained blocks count in k-1."""
    raw = np.concatenate((prefix, query), axis=0)
    allowed = np.asarray(scaler["available"], bool)[interface]
    valid = np.isfinite(raw) & allowed
    standardized = np.zeros(raw.shape)
    for channel in range(8):
        use = valid[:, channel]
        standardized[use, channel] = np.clip(
            (np.log1p(raw[use, channel]) - scaler["median"][interface][channel])
            / scaler["scale"][interface][channel],
            -8,
            8,
        )
    k = len(prefix)
    support, queries = standardized[:k], standardized[k:]
    smask, qmask = valid[:k], valid[k:]
    distances = (support[:, None, :] - queries[None, :, :]) ** 2
    distances *= smask[:, None, :] & qmask[None, :, :]
    fallback = np.repeat(~(qmask & smask.any(axis=0)).any(axis=1), 12)
    result = np.zeros((k, 60, 36))
    for omitted in range(k):
        keep = np.arange(k) != omitted
        retained_count = smask[keep].sum(axis=0)
        retained_mean = support[keep].sum(axis=0) / np.maximum(retained_count, 1)
        difference = np.where(
            smask[omitted] & (retained_count > 0), support[omitted] - retained_mean, 0
        )
        omitted_distance, retained_distance = (
            distances[omitted],
            distances[keep].sum(axis=0) / (k - 1),
        )
        base = np.zeros((5, 30))
        base[:, :8] = omitted_distance - retained_distance
        base[:, 8:16] = difference
        base[:, 16:24] = queries
        base[:, 24] = qmask.sum(axis=1) / 8
        base[:, 25] = smask[omitted].sum() / 8
        base[:, 26] = smask[keep].sum() / ((k - 1) * 8)
        base[:, 27] = omitted_distance.sum(axis=1) / 8
        base[:, 28] = retained_distance.sum(axis=1) / 8
        base[:, 29] = np.abs(difference).sum() / 8
        result[omitted, :, :30] = np.repeat(base, 12, axis=0)
        result[omitted, :, 30:33] = q[omitted, :, 20, None] * result[omitted, :, 27:30]
        result[omitted, :, 33:36] = q[omitted, :, 28, None] * result[omitted, :, 27:30]
    return result, fallback


def cell_inputs(cache, z, plan, scaler, p, i, w, bi, mode, fold, mapping=None):
    k = plan["budgets"][bi]
    q = cache["q_features"][p, i, w, bi, :k].reshape(k, 60, 33)
    x = np.zeros((k, 60, 69))
    x[..., :33] = q
    if mode == "Q":
        return x, np.zeros(60, bool)
    prefix = z[p, i, :k]
    query = z[p, i, 5:10].copy()
    if mode == "SHAM_REFIT":
        mapping = select_sham(
            prefix, plan["study_id"], fold, plan["source_subject_ids"][p], plan["interfaces"][i]
        )
    if mapping is not None:
        prefix = prefix[list(mapping)]
    if mode == "M_STALE":
        query[np.isfinite(query)] = np.broadcast_to(z[p, i, k - 1], query.shape)[np.isfinite(query)]
    x[..., 33:], fallback = m_features(prefix, query, q, scaler, i)
    return x, fallback


def training_data(cache, scores, z, plan, scaler, fold, mode):
    fit = [p for p in range(len(z)) if p % 3 != fold]
    matrices, outcomes, masses = [], [], []
    labels = np.tile(np.arange(12), 5)
    for p, i, w, bi in itertools.product(fit, range(2), range(4), range(2)):
        k = plan["budgets"][bi]
        features, _ = cell_inputs(cache, z, plan, scaler, p, i, w, bi, mode, fold)
        pred = scores["expert_r"][p, i, w, bi, : k + 1].reshape(k + 1, 60, 12).argmax(-1)
        correct = (pred == labels).astype(np.int8)
        matrices.append(features.reshape(k * 60, 69))
        outcomes.append((correct[1:] - correct[0]).reshape(-1))
        masses.append(np.full(k * 60, 1 / (len(fit) * 16 * k * 60)))
    return np.concatenate(matrices), np.concatenate(outcomes), np.concatenate(masses)


def ridge_replay(x, y, mass):
    require(
        np.isfinite(x).all()
        and np.isfinite(y).all()
        and np.isfinite(mass).all()
        and np.all(mass > 0)
        and abs(mass.sum() - 1) < 1e-12,
        "Ridge inputs/mass",
    )
    # Sum rather than np.average avoids introducing a second mass normalization.
    mean = np.einsum("r,rc->c", mass, x, optimize=False)
    variance = np.einsum("r,rc->c", mass, (x - mean) ** 2, optimize=False)
    sd = np.sqrt(variance)
    effective = int(np.count_nonzero(sd >= 1e-12))
    sd[sd < 1e-12] = 1
    design = np.empty((len(x), 70))
    design[:, 0] = 1
    design[:, 1:] = np.clip((x - mean) / sd, -10, 10)
    weighted = design * np.sqrt(mass[:, None])
    normal = weighted.T @ weighted
    normal.flat[::71] += np.r_[0.0, np.full(69, 0.1)]
    rhs = weighted.T @ (y * np.sqrt(mass))
    coefficients = cho_solve(cho_factor(normal, lower=True), rhs)
    mse = float(np.dot(mass, (y - design @ coefficients) ** 2))
    penalty = float(0.1 * np.dot(coefficients[1:], coefficients[1:]))
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": sd.tolist(),
        "coef": coefficients.tolist(),
        "train_weighted_mse": mse,
        "ridge_penalty": penalty,
        "objective": mse + penalty,
        "train_rows": len(x),
        "effective_nonconstant_features": effective,
    }


def changed_packets(prefix, mapping):
    return int(
        sum(
            not np.array_equal(prefix[j], prefix[other], equal_nan=True)
            for j, other in enumerate(mapping)
        )
    )


def changed_features(first, second):
    return int(
        np.count_nonzero(
            np.any(np.abs(first[..., 33:] - second[..., 33:]) > FEATURE_TOLERANCE, axis=-1)
        )
    )


def verify_fits(cache, scores, z, plan, freezes):
    require(
        freezes["schema"] == "cfeg.native-subset-m.freezes.v1"
        and freezes["plan_sha256"] == PLAN_SHA256
        and len(freezes["folds"]) == 3,
        "Freeze schema/fold count",
    )
    ids = plan["source_subject_ids"]
    maximum = 0.0
    for fold, item in enumerate(freezes["folds"]):
        equal(item["fold"], fold, "Fold number")
        fit = [p for p in range(len(z)) if p % 3 != fold]
        equal(item["fit_subject_ids"], [ids[p] for p in fit], "Fit identities")
        equal(
            item["evaluation_subject_ids"],
            [ids[p] for p in range(len(z)) if p % 3 == fold],
            "Evaluation identities",
        )
        scaler = impedance_scaler(z, fit)
        equal(item["metadata_scaler"], scaler, "Fit-only impedance scaler", 2e-12)
        require(set(item["routers"]) == set(ROUTERS), "Three fitted routers")
        groups = []
        for p, i, k in itertools.product(fit, range(2), (3, 5)):
            prefix = z[p, i, :k]
            maps = permutation_family(prefix)
            mapping = select_sham(prefix, plan["study_id"], fold, ids[p], plan["interfaces"][i])
            groups.append(
                {
                    "participant": ids[p],
                    "interface": plan["interfaces"][i],
                    "k": k,
                    "admissible_maps": len(maps),
                    "control_available": bool(maps),
                    "mapping": list(mapping),
                    "packet_changes": changed_packets(prefix, mapping),
                }
            )
        real_m = None
        for mode in ROUTERS:
            x, y, mass = training_data(cache, scores, z, plan, scaler, fold, mode)
            reconstructed = ridge_replay(x, y, mass)
            equal(item["routers"][mode], reconstructed, f"Ridge fold{fold}/{mode}", 2e-10)
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(np.asarray(item["routers"][mode]["coef"]) - reconstructed["coef"])
                    )
                ),
            )
            if mode == "QM":
                real_m = x[:, 33:].copy()
            if mode == "SHAM_REFIT":
                changed = int(
                    np.count_nonzero(np.any(np.abs(real_m - x[:, 33:]) > FEATURE_TOLERANCE, axis=1))
                )
                equal(
                    item["sham_diagnostics"],
                    {
                        "support_groups": groups,
                        "feature_change_tolerance": FEATURE_TOLERANCE,
                        "omission_query_rows": len(x),
                        "changed_feature_rows": changed,
                    },
                    "Fit sham coverage",
                )
    return maximum


def route(x, model, fallback=None, q_action=None):
    slopes = np.asarray(model["coef"])[1:]
    transformed = np.clip((x - model["feature_mean"]) / model["feature_scale"], -10, 10)
    gains = np.round(transformed @ slopes + model["coef"][0], 10)
    # Sequential strict improvement gives full/earliest omission ties explicitly.
    best = np.zeros(gains.shape[1])
    action = np.zeros(gains.shape[1], dtype=int)
    for omission in range(gains.shape[0]):
        improve = gains[omission] > best
        action[improve], best[improve] = omission + 1, gains[omission, improve]
    if fallback is not None:
        action[fallback] = q_action[fallback]
    return action


def replay_evaluation(cache, scores, z, plan, freezes):
    rows, diagnostics = [], []
    truth = np.tile(np.arange(12), 5)
    for p, participant in enumerate(plan["source_subject_ids"]):
        fold = p % 3
        saved = freezes["folds"][fold]
        for i, interface in enumerate(plan["interfaces"]):
            for w, n in enumerate(plan["sample_counts"]):
                cell = {"participant": participant, "interface": interface, "n_samples": n}
                a0 = scores["a0_r"][p, i, w].reshape(60, 12).argmax(1)
                rows.append(
                    {
                        **cell,
                        "method": "A0_author",
                        "k": 0,
                        "ba": float(np.mean(a0 == truth)),
                        "query_count": 60,
                        "label_count": 0,
                        "intervention_count": 1,
                        "control_available": True,
                    }
                )
                for bi, k in enumerate(plan["budgets"]):
                    pred = (
                        scores["expert_r"][p, i, w, bi, : k + 1].reshape(k + 1, 60, 12).argmax(-1)
                    )
                    maps = permutation_family(z[p, i, :k])
                    actions, matrices, fallback_masks = {}, {}, {}
                    decisions = {"FULL": pred[0]}
                    for mode in (*ROUTERS, "M_STALE"):
                        x, missing = cell_inputs(
                            cache, z, plan, saved["metadata_scaler"], p, i, w, bi, mode, fold
                        )
                        model = saved["routers"]["QM" if mode == "M_STALE" else mode]
                        actions[mode] = route(
                            x, model, missing if mode != "Q" else None, actions.get("Q")
                        )
                        matrices[mode], fallback_masks[mode] = x, missing
                        decisions[mode] = pred[actions[mode], np.arange(60)]
                    for mode in ("FULL", *ROUTERS, "M_STALE", "M_MISSING"):
                        available = bool(maps) if mode == "SHAM_REFIT" else True
                        interventions = 0 if mode == "M_MISSING" else int(available)
                        selected = decisions["Q" if mode == "M_MISSING" else mode]
                        rows.append(
                            {
                                **cell,
                                "method": mode,
                                "k": k,
                                "ba": float(np.mean(selected == truth)),
                                "query_count": 60,
                                "label_count": 12 * k,
                                "intervention_count": interventions,
                                "control_available": available,
                            }
                        )
                    interventions = []
                    for mapping in maps or [tuple(range(k))]:
                        x, missing = cell_inputs(
                            cache,
                            z,
                            plan,
                            saved["metadata_scaler"],
                            p,
                            i,
                            w,
                            bi,
                            "QM",
                            fold,
                            mapping,
                        )
                        choice = route(x, saved["routers"]["QM"], missing, actions["Q"])
                        selected = pred[choice, np.arange(60)]
                        interventions.append(
                            {
                                "mapping": list(mapping),
                                "control_available": bool(maps),
                                "packet_changes": changed_packets(z[p, i, :k], mapping),
                                "feature_changes": changed_features(matrices["QM"], x),
                                "selector_changes": int(np.count_nonzero(choice != actions["QM"])),
                                "prediction_changes": int(
                                    np.count_nonzero(selected != decisions["QM"])
                                ),
                                "ba": float(np.mean(selected == truth)),
                                "actions": choice.tolist(),
                            }
                        )
                    rows.append(
                        {
                            **cell,
                            "method": "M_SHUFFLE",
                            "k": k,
                            "ba": float(np.mean([r["ba"] for r in interventions])),
                            "query_count": 60,
                            "label_count": 12 * k,
                            "intervention_count": len(maps),
                            "control_available": bool(maps),
                        }
                    )
                    qr, mr = decisions["Q"] == truth, decisions["QM"] == truth
                    diagnostics.append(
                        {
                            **cell,
                            "k": k,
                            "expert_disagreement_queries": int(
                                np.any(pred[1:] != pred[0], axis=0).sum()
                            ),
                            "qm_vs_q_selector_changes": int(
                                np.count_nonzero(actions["QM"] != actions["Q"])
                            ),
                            "qm_vs_q_prediction_changes": int(
                                np.count_nonzero(decisions["QM"] != decisions["Q"])
                            ),
                            "qm_repaired_q": int(np.count_nonzero(mr & ~qr)),
                            "qm_damaged_q": int(np.count_nonzero(qr & ~mr)),
                            "qm_missing_fallback_queries": int(fallback_masks["QM"].sum()),
                            "admissible_shuffle_maps": len(maps),
                            "mean_shuffled_packet_changes": float(
                                np.mean([r["packet_changes"] for r in interventions])
                            ),
                            "mean_shuffle_feature_changes": float(
                                np.mean([r["feature_changes"] for r in interventions])
                            ),
                            "mean_shuffle_selector_changes": float(
                                np.mean([r["selector_changes"] for r in interventions])
                            ),
                            "mean_shuffle_prediction_changes": float(
                                np.mean([r["prediction_changes"] for r in interventions])
                            ),
                            "sham_feature_changes": changed_features(
                                matrices["QM"], matrices["SHAM_REFIT"]
                            ),
                            "stale_feature_changes": changed_features(
                                matrices["QM"], matrices["M_STALE"]
                            ),
                            "feature_change_tolerance": FEATURE_TOLERANCE,
                            "shuffle_interventions": interventions,
                            "actions": {mode: values.tolist() for mode, values in actions.items()},
                        }
                    )
    rows.sort(key=lambda row: tuple(row[key] for key in ROW_KEYS))
    return rows, diagnostics


def interval(values):
    x = np.asarray(values, float)
    mean = float(x.mean())
    radius = float(
        t.isf(0.025, len(x) - 1) * np.sqrt(np.sum((x - mean) ** 2) / (len(x) - 1) / len(x))
    )
    return {
        "mean": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "n_participants": len(x),
    }


def replay_reporting(rows, plan):
    lookup = {tuple(row[key] for key in ROW_KEYS): row["ba"] for row in rows}
    require(len(lookup) == len(rows), "Duplicated result row")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in CELL_KEYS)].append(row["ba"])
    summary = []
    for key, values in sorted(groups.items()):
        entry = interval(values)
        entry["mean_ba"] = entry.pop("mean")
        summary.append({**dict(zip(CELL_KEYS, key)), **entry})
    comparisons = []
    for k in (3, 5):
        comparisons.extend(
            ("QM", k, other, 0 if other == "A0_author" else k)
            for other in ("Q", "FULL", "A0_author", "SHAM_REFIT", "M_SHUFFLE", "M_STALE")
        )
        comparisons.append(("Q", k, "FULL", k))
    comparisons += [("Q", 5, "Q", 3), ("QM", 3, "Q", 5), ("QM", 5, "QM", 3)]
    contrasts, attainment = [], []
    for left, lk, right, rk in comparisons:
        name = f"{left}{lk}_minus_{right}{rk}"
        per_cell = []
        for interface, n in itertools.product(plan["interfaces"], plan["sample_counts"]):
            delta = [
                lookup[p, interface, n, left, lk] - lookup[p, interface, n, right, rk]
                for p in plan["source_subject_ids"]
            ]
            per_cell.append(delta)
            contrasts.append(
                {"contrast": name, "interface": interface, "n_samples": n, **interval(delta)}
            )
        contrasts.append(
            {
                "contrast": name,
                "interface": "all8",
                "n_samples": None,
                **interval(np.asarray(per_cell).mean(axis=0)),
            }
        )
    for p, interface, n, method in itertools.product(
        plan["source_subject_ids"],
        plan["interfaces"],
        plan["sample_counts"],
        ("FULL", "Q", "QM", "SHAM_REFIT"),
    ):
        hit = None
        for k in (0, 3, 5):
            if lookup[p, interface, n, "A0_author" if k == 0 else method, k] >= 0.8:
                hit = k
                break
        attainment.append(
            {
                "participant": p,
                "interface": interface,
                "n_samples": n,
                "method": method,
                "first_observed_k": hit,
                "unreached_tested_grid": hit is None,
                "label_count": None if hit is None else 12 * hit,
                "retained_seconds": None if hit is None else 12 * hit * n / 250,
                "history_seconds": None if hit is None else 12 * hit * 0.14,
            }
        )
    return summary, contrasts, attainment


def verify_provenance(root, plan, payloads, hashes, repository):
    equal(str(root), plan["execution"]["output_root"], "Frozen output destination")
    start, freeze, result = (
        payloads[name] for name in ("start.json", "fold-freezes.json", "result.json")
    )
    require(start["schema"] == "cfeg.native-subset-m.start.v1", "Start schema")
    require(
        result["schema"] == plan["reporting"]["schema"]
        and result["status"] == "DEVELOPMENT_ASSESSMENT_COMPLETE",
        "Completion schema/status",
    )
    common = {
        "plan_sha256": PLAN_SHA256,
        "study_id": plan["study_id"],
        "upstream_revision": plan["upstream"]["revision"],
    }
    for key, expected in common.items():
        for name, payload in (("start", start), ("freeze", freeze), ("result", result)):
            equal(payload[key], expected, name + "." + key)
    for key in ("source_commit", "source_tree", "started_at"):
        equal(freeze[key], start[key], "Freeze provenance " + key)
        equal(result[key], start[key], "Result provenance " + key)
    for key in ("source_commit", "source_tree"):
        require(re.fullmatch(r"[0-9a-f]{40}", start[key]) is not None, "Git hash format")
    tree = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", start["source_commit"] + "^{tree}"], text=True
    ).strip()
    equal(start["source_tree"], tree, "Committed source tree")
    for relative, expected in (
        ("configs/analysis/native_subset_m_source39_v1.json", PLAN_SHA256),
        ("scripts/native_subset_m_core.py", start["core_sha256"]),
    ):
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "show", start["source_commit"] + ":" + relative]
        )
        equal(hashlib.sha256(blob).hexdigest(), expected, "Committed " + relative)
    for relative, expected in plan["upstream"]["pins"].items():
        equal(
            start["imported_source_hashes"][relative], expected, "Imported native pin " + relative
        )
    equal(start["source_subject_ids"], plan["source_subject_ids"], "Source identities")
    equal(start["raw_root"], plan["raw_root"], "Raw root record")
    equal(start["output_root"], str(root), "Output root record")
    equal(start["python"], plan["execution"]["python_version"], "Python record")
    equal(start["dependencies"], plan["execution"]["dependencies"], "Dependency record")
    for flag in ("metadata_access", "source_projection_reuse", "baseline_cache_read"):
        equal(start[flag], True, "Input authority " + flag)
    for flag in ("held_access", "retired_access", "manifest_or_full_impedance_access"):
        equal(start[flag], False, "Protected authority " + flag)
    for payload in (freeze, result):
        equal(
            payload["source_projection_sha256"],
            hashes["source-projection.json"],
            "Projection binding",
        )
        equal(payload["features_sha256"], hashes["features.npz"], "Cache binding")
    equal(result["fold_freezes_sha256"], hashes["fold-freezes.json"], "Freeze binding")
    before, after = (
        datetime.fromisoformat(start["started_at"]),
        datetime.fromisoformat(result["completed_at"]),
    )
    require(
        before.tzinfo is not None and after.tzinfo is not None and after >= before,
        "Execution timestamps",
    )
    require(
        (after - before).total_seconds() <= plan["execution"]["expected_runtime_ceiling_seconds"],
        "Recorded runtime ceiling",
    )
    records = result["raw_files"]
    equal([row["subject"] for row in records], plan["source_subject_ids"], "Raw identity records")
    for row in records:
        require(
            set(row) == {"subject", "path", "sha256", "stored_dtype", "shape"}, "Raw record fields"
        )
        equal(
            row["path"],
            str(Path(plan["raw_root"]) / f"S{row['subject']:03d}.mat"),
            "Raw path record",
        )
        equal(row["shape"], plan["raw_shape"], "Raw shape record")
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "Raw digest format")
        require(np.dtype(row["stored_dtype"]).kind in "fiu", "Raw numeric dtype record")
    baseline = result["baseline_agreement"]
    equal(baseline["input_path"], plan["baseline_reference"]["path"], "Baseline path record")
    equal(baseline["input_sha256"], plan["baseline_reference"]["sha256"], "Baseline hash record")
    for key in ("a0_r", "full_expert_r"):
        require(
            0 <= baseline[key]["max_abs_error"] <= 1e-12 and baseline[key]["argmax_exact"] is True,
            "Recorded baseline comparison",
        )


def audit(output_root, plan_path, repository=None):
    root, plan_path = Path(output_root), Path(plan_path)
    plan = load_plan(plan_path)
    require(
        root.is_dir() and not root.is_symlink() and root.resolve() == root,
        "Output must be an absolute non-symlink directory",
    )
    names = plan["execution"]["artifacts"]
    require({p.name for p in root.iterdir()} == set(names), "Exact five-artifact inventory")
    times, hashes, payloads, size = [], {}, {}, 0
    for name in names:
        path = root / name
        info = path.lstat()
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_nlink == 1,
            "Immutable regular artifact " + name,
        )
        times.append(info.st_mtime_ns)
        size += info.st_size
        hashes[name] = digest(path)
        if name.endswith(".json"):
            payloads[name] = json.loads(path.read_bytes())
    require(times == sorted(times), "Publication order by artifact mtimes")
    require(size <= plan["execution"]["resource_budget_bytes"], "Artifact byte budget")
    repository = Path(repository) if repository is not None else Path(__file__).resolve().parents[1]
    verify_provenance(root, plan, payloads, hashes, repository)
    z, order = load_projection(payloads["source-projection.json"], plan)
    with np.load(root / "features.npz", allow_pickle=False) as stored:
        require(len(stored.files) == len(set(stored.files)), "Duplicate NPZ members")
        cache = {key: stored[key] for key in stored.files}
    scores = cache_scores(cache, plan)
    q_error = verify_q(cache, scores, order, plan)
    fit_error = verify_fits(cache, scores, z, plan, payloads["fold-freezes.json"])
    rows, diagnostics = replay_evaluation(cache, scores, z, plan, payloads["fold-freezes.json"])
    summary, contrasts, attainment = replay_reporting(rows, plan)
    require(
        (len(rows), len(summary), len(contrasts), len(attainment), len(diagnostics))
        == (4680, 120, 153, 1248, 624),
        "Complete frozen reporting grid",
    )
    result = payloads["result.json"]
    for key, expected in (
        ("rows", rows),
        ("summary", summary),
        ("contrasts", contrasts),
        ("attainment", attainment),
        ("diagnostics", diagnostics),
    ):
        equal(result[key], expected, key)
    equal(
        result["evidence_scope"],
        "adaptively exposed development, descriptive intervals, no automatic promotion",
        "Evidence scope",
    )
    for name, expected in hashes.items():
        equal(digest(root / name), expected, "Artifact stable during audit " + name)
    return {
        "status": "PASS",
        "study_id": plan["study_id"],
        "plan_sha256": PLAN_SHA256,
        "artifact_sha256": hashes,
        "artifact_bytes": size,
        "rows": len(rows),
        "summaries": len(summary),
        "contrasts": len(contrasts),
        "attainment": len(attainment),
        "diagnostics": len(diagnostics),
        "independently_replayed_routers": 9,
        "q_cache_derived_max_abs_error": q_error,
        "ridge_coef_max_abs_error": fit_error,
        "scope": "Artifact hashes/provenance, cache-derived Q, fit-only M scaling/features, weighted ridge, all routes, controls and reports.",
        "not_replayed": [
            "raw EEG preprocessing/native fits",
            "raw-derived Q support-quality/covariance/consistency",
            "original projection/raw impedance precision",
            "old baseline cache comparison",
            "runtime access or lifecycle ordering beyond recorded provenance/mtimes",
        ],
        "evidence_role": "adaptively_exposed_development_not_confirmation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.output_root, args.plan), sort_keys=True, allow_nan=False))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
