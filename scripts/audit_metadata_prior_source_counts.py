"""Exact-count reporting correction for immutable source39 scores, never a refit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

PLAN_SHA256 = "ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b"
RESULT_SHA256 = "f1d8158c0228cfefdf5bd762bdc1d960f7268f79f93c24268a519e46d6460ce6"
SCORES_SHA256 = "a8cc09c170d2e7396463eee2655a647262eabfdb4d66214dd5cc0037682c0c61"
AUDIT_SHA256 = "ce5e355674f940a6d256226c573df0727780676e927ccbbe3e0626e124616be8"
REPOSITORY = Path("/home/whwovy/califreeEEG")
STUDY = Path("/home/whwovy/metadata-prior-source39-v1")
ARMS = ("FULL", "ISO", "Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_counts(numerator, denominator):
    """Signs come from integer net-correct counts, never rounded accuracy means."""
    net = np.asarray(numerator)
    require(
        net.ndim == 1 and len(net) >= 2 and net.dtype.kind in "iu", "Integer participant counts"
    )
    require(type(denominator) is int and denominator > 0, "Positive integer denominator")
    require(np.all((net >= -denominator) & (net <= denominator)), "Net-correct count bounds")
    values = net.astype(np.float64) / denominator
    mean = float(values.mean())
    radius = float(t.ppf(0.975, len(net) - 1) * values.std(ddof=1) / np.sqrt(len(net)))
    return {
        "participant_net_correct": net.tolist(),
        "per_participant_denominator": denominator,
        "mean_delta": mean,
        "ci_low": mean - radius,
        "ci_high": mean + radius,
        "help": int((net > 0).sum()),
        "tie": int((net == 0).sum()),
        "harm": int((net < 0).sum()),
        "n": len(net),
    }


def classify_counts(scores, anchor_scores, plan):
    p, i, w = len(plan["source_subject_ids"]), len(plan["interfaces"]), len(plan["sample_counts"])
    require(plan["budgets"] == [3, 5] and tuple(plan["arms"]) == ARMS, "Frozen arm/budget order")
    require(
        plan["endpoints"]["query_count"] == 48
        and len(plan["frequencies"]) == 12
        and plan["query_blocks"] == [6, 7, 8, 9],
        "Frozen query geometry",
    )
    s, a = np.asarray(scores), np.asarray(anchor_scores)
    require(s.shape == (p, i, w, 2, 9, 48, 12) and a.shape == (p, i, w, 48, 12), "Score geometry")
    require(
        s.dtype.kind in "fiu"
        and a.dtype.kind in "fiu"
        and np.isfinite(s).all()
        and np.isfinite(a).all(),
        "Finite real scores",
    )
    labels = np.tile(np.arange(12), 4)
    return np.sum(s.argmax(-1) == labels, axis=-1, dtype=np.int64), np.sum(
        a.argmax(-1) == labels, axis=-1, dtype=np.int64
    )


def exact_contrasts(counts, anchor_counts, plan):
    c, a = np.asarray(counts), np.asarray(anchor_counts)
    p, i, w = len(plan["source_subject_ids"]), len(plan["interfaces"]), len(plan["sample_counts"])
    require(c.shape == (p, i, w, 2, 9) and a.shape == (p, i, w), "Correct-count geometry")
    require(
        c.dtype.kind in "iu"
        and a.dtype.kind in "iu"
        and np.all((c >= 0) & (c <= 48))
        and np.all((a >= 0) & (a <= 48)),
        "Integer correct-count domain",
    )
    scopes = [("ALL", None, None)] + [
        (f"{e}_{n}", ei, wi)
        for ei, e in enumerate(plan["interfaces"])
        for wi, n in enumerate(plan["sample_counts"])
    ]
    rows = []
    for scope, ei, wi in scopes:
        for bi, k in enumerate((3, 5)):
            actual = c[..., bi, ARMS.index("QM")]
            for comparator in ("Q", "Q2", "SHAM_REFIT", "FULL", "A0") + (("Q5",) if k == 3 else ()):
                baseline = (
                    a
                    if comparator == "A0"
                    else c[..., 1, ARMS.index("Q")]
                    if comparator == "Q5"
                    else c[..., bi, ARMS.index(comparator)]
                )
                delta = actual.astype(np.int64) - baseline.astype(np.int64)
                net = delta.sum(axis=(1, 2), dtype=np.int64) if scope == "ALL" else delta[:, ei, wi]
                denominator = 48 * i * w if scope == "ALL" else 48
                rows.append(
                    {
                        "scope": scope,
                        "k": k,
                        "comparator": comparator,
                        "participant_ids": list(plan["source_subject_ids"]),
                        **paired_counts(net, denominator),
                    }
                )
    return rows


def calibration_counts(counts, anchors, plan):
    require(
        plan["endpoints"]["attainment_threshold"] == 0.8
        and plan["endpoints"]["attainment_grid"] == [0, 3, 5],
        "Frozen attainment endpoint",
    )
    costs = {}
    for name in ("Q", "QM"):
        value = np.full(anchors.shape, -1, dtype=int)
        for bi, k in ((1, 5), (0, 3)):
            value[counts[..., bi, ARMS.index(name)] >= 39] = plan["endpoints"]["label_costs"][
                str(k)
            ]
        value[anchors >= 39] = 0
        costs[name] = value
    q, m = costs["Q"], costs["QM"]
    both, new, lost, neither = (
        (q >= 0) & (m >= 0),
        (q < 0) & (m >= 0),
        (q >= 0) & (m < 0),
        (q < 0) & (m < 0),
    )
    return {
        "both_reached": int(both.sum()),
        "new_reach": int(new.sum()),
        "lost_reach": int(lost.sum()),
        "neither_reached": int(neither.sum()),
        "mean_label_delta_both": float((m[both] - q[both]).mean()) if both.any() else None,
    }


def corrected_verdict(contrasts, counts, calibration, coverage, plan):
    lookup = {(row["k"], row["comparator"]): row for row in contrasts if row["scope"] == "ALL"}
    primary = lookup[3, "Q"]
    increment = bool(
        primary["mean_delta"] >= 0.01
        and primary["ci_low"] > 0
        and lookup[3, "Q2"]["ci_low"] > 0
        and lookup[3, "SHAM_REFIT"]["ci_low"] > 0
        and coverage >= 0.5
    )
    qm = float(counts[..., 0, ARMS.index("QM")].mean() / 48)
    saving = bool(
        increment
        and lookup[3, "Q5"]["ci_low"] > -0.01
        and qm >= 0.8
        and calibration["mean_label_delta_both"] is not None
        and calibration["mean_label_delta_both"] < 0
        and calibration["new_reach"] >= calibration["lost_reach"]
    )
    return {
        "status": "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE"
        if saving
        else "CLASSIFICATION_INCREMENT_ONLY"
        if increment
        else "METADATA_INCREMENT_NOT_ESTABLISHED",
        "metadata_increment": increment,
        "calibration_benefit": saving,
        "primary_mean_QM3": qm,
        "coverage_changed_fraction": coverage,
    }


def compare_reporting(scores, anchors, result, original_audit, plan):
    require(
        result["schema"] == "cfeg.metadata-prior-source.result.v1"
        and result["study_id"] == plan["study_id"],
        "Frozen result identity",
    )
    require(
        original_audit["status"] == "PASS" and original_audit["study_id"] == plan["study_id"],
        "Prior audit identity",
    )
    counts, anchor_counts = classify_counts(scores, anchors, plan)
    corrected = exact_contrasts(counts, anchor_counts, plan)
    old = {(x["scope"], x["k"], x["comparator"]): x for x in result["contrasts"]}
    require(
        len(old) == len(result["contrasts"]) == len(corrected), "Exact original contrast inventory"
    )
    changed, max_error = [], 0.0
    for row in corrected:
        key = row["scope"], row["k"], row["comparator"]
        require(key in old, "Original contrast key")
        previous = old[key]
        require(previous["n"] == row["n"], "Participant denominator unchanged")
        for field in ("mean_delta", "ci_low", "ci_high"):
            error = abs(previous[field] - row[field])
            require(np.isfinite(error) and error <= 1e-12, "Mean/CI changed beyond roundoff")
            max_error = max(max_error, error)
        before, after = (
            {field: previous[field] for field in ("help", "tie", "harm")},
            {field: row[field] for field in ("help", "tie", "harm")},
        )
        if before != after:
            changed.append(
                {
                    "scope": key[0],
                    "k": key[1],
                    "comparator": key[2],
                    "original": before,
                    "corrected": after,
                }
            )
    calibration = calibration_counts(counts, anchor_counts, plan)
    for field, value in calibration.items():
        require(result["calibration"][field] == value, "Calibration costs unchanged")
    verdict = corrected_verdict(
        corrected, counts, calibration, result["verdict"]["coverage_changed_fraction"], plan
    )
    for field, value in verdict.items():
        for previous in (result["verdict"], original_audit["verdict"]):
            if field == "primary_mean_QM3":
                require(abs(previous[field] - value) <= 1e-12, "Mean QM3 unchanged")
            else:
                require(previous[field] == value, "Original verdict unchanged")
    return {
        "status": "REPORTING_CORRECTION_VERIFIED",
        "contrast_count": len(corrected),
        "changed_help_tie_harm_count": len(changed),
        "changed_contrasts": changed,
        "contrasts": corrected,
        "max_mean_or_ci_absolute_change": max_error,
        "calibration": calibration,
        "verdict": verdict,
        "limits": [
            "Reporting correction from existing scores, not independent evaluation or a new fit",
            "M-feature coverage reused from frozen result/prior audit; no metadata inputs reopened",
            "Original model/result/audit files remain unchanged; only exact-count sign reporting is superseded",
        ],
    }


def read_regular(path, expected=None):
    path = Path(path).absolute()
    require(path.resolve() == path, "Exact non-symlink input")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Regular input")
        encoded = stream.read()
    digest = hashlib.sha256(encoded).hexdigest()
    require(expected is None or digest == expected, "Pinned input hash")
    return encoded, digest


def publish(path, payload):
    path = Path(path).absolute()
    require(path.parent.resolve() == path.parent and not path.is_symlink(), "Exact output location")
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def run():
    require(
        Path(__file__).resolve().parents[1] == REPOSITORY,
        "Only integrated canonical implementation",
    )
    output = STUDY / "reporting-audit/exact-counts.json"
    require(not output.exists() and not output.is_symlink(), "No reporting overwrite")
    paths = {
        "plan": REPOSITORY / "configs/analysis/metadata_prior_source39_v1.json",
        "result": STUDY / "analysis/result.json",
        "scores": STUDY / "analysis/scores.npz",
        "original_audit": STUDY / "analysis/audit.json",
    }
    pins = {
        "plan": PLAN_SHA256,
        "result": RESULT_SHA256,
        "scores": SCORES_SHA256,
        "original_audit": AUDIT_SHA256,
    }
    raw = {}
    hashes = {}
    for name, path in paths.items():
        raw[name], hashes[name] = read_regular(path, pins[name])
    plan, result, prior = (json.loads(raw[name]) for name in ("plan", "result", "original_audit"))
    require(
        prior["plan_sha256"] == PLAN_SHA256
        and prior["inputs"]["scores.npz"] == SCORES_SHA256
        and prior["inputs"]["result.json"] == RESULT_SHA256,
        "Prior audit immutable-input chain",
    )
    require(
        result["provenance"]["plan_sha256"] == PLAN_SHA256
        and result["provenance"]["scores_sha256"] == SCORES_SHA256,
        "Result immutable-score chain",
    )
    with np.load(io.BytesIO(raw["scores"]), allow_pickle=False) as archive:
        require(set(archive.files) == {"scores", "a0_scores"}, "Exact score inventory")
        correction = compare_reporting(archive["scores"], archive["a0_scores"], result, prior, plan)
    for name, path in paths.items():
        require(read_regular(path)[1] == hashes[name], "Inputs unchanged during correction")
    payload = {
        "schema": "cfeg.metadata-prior-source.exact-count-correction.v1",
        "study_id": plan["study_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_hashes": hashes,
        "source_sha256": read_regular(Path(__file__))[1],
        **correction,
    }
    require(
        output.parent.parent.resolve() == output.parent.parent and not output.parent.is_symlink(),
        "Exact reporting directory",
    )
    output.parent.mkdir(mode=0o700, exist_ok=True)
    publish(output, payload)
    return payload


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "changed_contrasts": result["changed_help_tie_harm_count"],
                "verdict": result["verdict"],
            }
        )
    )


if __name__ == "__main__":
    main()
