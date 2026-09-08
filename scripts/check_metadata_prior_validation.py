"""Deterministic engineering checks only; no RNG, M learner or EEG input files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import metadata_prior_validation as check
from cfeg.analysis import metadata_trca_prior as op


def actuation_fixture():
    """Analytic rank-separated 2-channel support; queries have NO efficacy labels."""
    basis = np.array(
        [
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
        ],
        dtype=float,
    ) / np.sqrt(8)
    u, v, p, q = basis
    b2 = np.array([0.03, 0.0325])
    a, b = np.sqrt(1 - 2 * b2 / 3), np.sqrt(b2)
    support = np.empty((3, 2, 1, 2, 8))
    for r, c in enumerate((1, -1, 0)):
        support[r, 0, 0] = (a[0] * u + b[0] * c * p, a[1] * v + b[1] * c * q)
        support[r, 1, 0] = (a[0] * v + b[0] * c * p, a[1] * u + b[1] * c * q)
    queries = np.array([[u, u], [v, v]])[:, None]
    return support, queries, np.diag(6 - 6 * b2), 3 * np.eye(2)


def run_checks():
    support, queries, expected_s, expected_c = actuation_fixture()
    errors = []
    for label in range(2):
        s, c = op.trca_matrices(support[:, label, 0])
        errors.extend((float(np.max(abs(s - expected_s))), float(np.max(abs(c - expected_c)))))
    base = np.array([[1.1, 0.9]])
    delta = np.array([[-0.2, 0.2]])
    changed = op.residual_prior(base, delta, np.ones_like(base, dtype=bool))
    baseline = op.fit_trca(support, base, 0.1)
    alternative = op.fit_trca(support, changed, 0.1)
    scores, _ = op.score_trca(baseline, queries, np.ones(1))
    other_scores, _ = op.score_trca(alternative, queries, np.ones(1))
    sensitivity = check.score_sensitivity(scores, other_scores)
    penalty_error = float(
        np.max(
            abs(
                np.array(baseline.diagnostics["penalty_trace"])
                - alternative.diagnostics["penalty_trace"]
            )
        )
    )
    if max(errors) > 1e-12 or penalty_error > 1e-12:
        raise AssertionError("Analytic S/C or fixed penalty trace mismatch")
    if scores.argmax(-1).tolist() != [1, 0] or other_scores.argmax(-1).tolist() != [0, 1]:
        raise AssertionError("Declared label-free decision switch missing")
    if not np.all(sensitivity["score_max_abs_change"] > 1e-8):
        raise AssertionError("No downstream score sensitivity")
    gamma0_base = op.fit_trca(support, base, 0.0)
    gamma0_alt = op.fit_trca(support, changed, 0.0)
    gamma0_error = float(np.max(abs(gamma0_base.filters - gamma0_alt.filters)))
    missing = op.residual_prior(base, delta, np.zeros_like(base, dtype=bool))
    uniform = op.residual_prior(base, np.full_like(base, 0.15), np.ones_like(base, dtype=bool))
    uniform_error = float(np.max(abs(uniform - base)))
    if gamma0_error > 1e-12 or uniform_error > 1e-12 or not np.array_equal(missing, base):
        raise AssertionError("Prior invariance or fallback mismatch")

    # A designed linear identity, not realistic proxy observations or a new M comparison.
    features = (np.arange(9)[:, None] - 4 + np.array([[-1.0, 1.0]]))[:, None, None, :, None]
    target = np.broadcast_to(np.array([-1.0, 1.0]), (9, 1, 1, 2)).copy()
    model, oof, receipt = check.crossfit_q(features, target, np.arange(100, 109))
    oof_error = float(np.max(abs(oof - target)))
    if oof_error > 1e-10 or model.representation != "context":
        raise AssertionError("Q context cannot recover the declared centered linear identity")
    return {
        "study_kind": "deterministic engineering verification; NOT an efficacy study",
        "status": "ENGINEERING_CHECKS_PASSED",
        "human_data_access": False,
        "metadata_fit": False,
        "calibration_savings_established": False,
        "actuation": {
            "S": expected_s.tolist(),
            "C": expected_c.tolist(),
            "max_matrix_error": max(errors),
            "penalty_trace_error": penalty_error,
            "base_prior": base.tolist(),
            "alternative_prior": changed.tolist(),
            "baseline_filters": baseline.filters.tolist(),
            "alternative_filters": alternative.filters.tolist(),
            "baseline_scores": scores.tolist(),
            "alternative_scores": other_scores.tolist(),
            "baseline_predictions": scores.argmax(-1).tolist(),
            "alternative_predictions": other_scores.argmax(-1).tolist(),
            "sensitivity": {k: v.tolist() for k, v in sensitivity.items()},
            "gamma0_filter_error": gamma0_error,
            "uniform_prior_error": uniform_error,
            "missing_exact": np.array_equal(missing, base),
            "accuracy": None,
            "query_labels": None,
        },
        "linear_identity": {
            "features": features.tolist(),
            "targets": target.tolist(),
            "oof_predictions": oof.tolist(),
            "max_oof_error": oof_error,
            "receipt": receipt,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = (
        "scripts/check_metadata_prior_validation.py",
        "src/cfeg/analysis/metadata_prior_validation.py",
        "src/cfeg/analysis/metadata_trca_prior.py",
        "docs/metadata_prior_validation_repair.md",
    )
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
    }
    # Exclusive creation refuses an existing receipt before any engineering calculation.
    with args.output.open("x", encoding="utf-8") as stream:
        result = run_checks()
        result["provenance"] = provenance
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
