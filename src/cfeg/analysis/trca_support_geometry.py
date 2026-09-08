"""Support-only diagnostics, not a new learner or an efficacy evaluation."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from cfeg.analysis.metadata_trca_prior import trca_matrices


def _leading(s, b):
    if linalg.eigvalsh(b)[0] <= 0:
        raise ValueError("Denominator must be positive definite; no ridge rescue")
    values, vectors = linalg.eig(s, b)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(vectors).all()
        or np.iscomplex(values).any()
        or np.iscomplex(vectors).any()
    ):
        raise ValueError("Finite real eigensystem required")
    order = np.argsort(values)[::-1]
    gap = float(np.real(values[order[0]] - values[order[1]]))
    if gap <= 1e-10 * max(1.0, float(np.abs(values).max())):
        raise ValueError("Degenerate leading eigenspace; no tie breaker")
    u = np.real(vectors[:, order[0]])
    return u / np.linalg.norm(u), gap


def matrix_geometry(s, c):
    """Fixed gamma=.1: return scalar/spectral diagnostics and two unit directions.

    The normalization factor holds the ISO direction fixed: it is not the
    FULL-to-ISO Euclidean norm ratio, nor literal signal attenuation.
    """
    s, c = np.asarray(s), np.asarray(c)
    if (
        s.ndim != 2
        or s.shape != c.shape
        or s.shape[0] != s.shape[1]
        or s.shape[0] < 2
        or s.dtype.kind not in "fiu"
        or c.dtype.kind not in "fiu"
        or not np.isfinite(s).all()
        or not np.isfinite(c).all()
        or not np.allclose(s, s.T, rtol=1e-12, atol=1e-12)
        or not np.allclose(c, c.T, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("Finite real symmetric square S,C required")
    s, c = s.astype(np.float64), c.astype(np.float64)
    eigenvalues = linalg.eigvalsh(c)
    if eigenvalues[0] <= 0:
        raise ValueError("Covariance must be positive definite")
    tau = float(0.1 * np.trace(c) / len(c))
    full, gap0 = _leading(s, c)
    iso, gap1 = _leading(s, c + tau * np.eye(len(c)))
    rho0 = float(tau * (full @ full) / (full @ c @ full))
    rho1 = float(tau * (iso @ iso) / (iso @ c @ iso))
    sine = min(1.0, float(0.5 * np.linalg.norm(full - iso) * np.linalg.norm(full + iso)))
    ratios = tau / eigenvalues
    metrics = {
        "tau": tau,
        "covariance_eigenvalues_over_mean": (eigenvalues / (np.trace(c) / len(c))).tolist(),
        "tau_over_covariance_eigenvalues": ratios.tolist(),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "effective_rank": float(
            (eigenvalues / eigenvalues[-1]).sum() ** 2
            / np.sum((eigenvalues / eigenvalues[-1]) ** 2)
        ),
        "full_direction_rho": rho0,
        "iso_direction_rho": rho1,
        "direction_sine": sine,
        "iso_c_to_b_norm_factor": float(1 / np.sqrt(1 + rho1)),
        "modes_tau_over_lambda_ge1": int(np.count_nonzero(ratios >= 1)),
        "modes_tau_over_lambda_ge10": int(np.count_nonzero(ratios >= 10)),
        "full_eigenvalue_gap": gap0,
        "iso_eigenvalue_gap": gap1,
    }
    for value in metrics.values():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite diagnostic arithmetic")
    return metrics, full, iso


def support_geometry(support, *, subject, interface, samples, k):
    """One participant/condition/budget, support[k,12,5,8,N] only."""
    x = np.asarray(support)
    if k not in (3, 5) or x.shape != (k, 12, 5, 8, samples):
        raise ValueError("Frozen support geometry required")
    rows, matrices, full_units, iso_units = [], [], [], []
    for label in range(12):
        for band in range(5):
            s, c = trca_matrices(x[:, label, band])
            metrics, full, iso = matrix_geometry(s, c)
            rows.append(
                {
                    "subject": subject,
                    "interface": interface,
                    "samples": samples,
                    "k": k,
                    "class": label,
                    "band": band,
                    **metrics,
                }
            )
            matrices.append((s, c))
            full_units.append(full)
            iso_units.append(iso)
    return rows, np.asarray(matrices), np.asarray(full_units), np.asarray(iso_units)


SCALARS = (
    "condition_number",
    "effective_rank",
    "full_direction_rho",
    "iso_direction_rho",
    "direction_sine",
    "iso_c_to_b_norm_factor",
    "modes_tau_over_lambda_ge1",
    "modes_tau_over_lambda_ge10",
)


def finite_mean(values):
    a = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(np.abs(a)))
    return float((a / maximum).mean() * maximum) if maximum else 0.0


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Nonempty finite one-dimensional summary input required")
    maximum = float(np.max(np.abs(a)))
    quantiles = (
        np.quantile(a / maximum, [0, 0.1, 0.5, 0.9, 1]) * maximum if maximum else np.zeros(5)
    )
    return {
        "n": len(a),
        "mean": finite_mean(a),
        **dict(zip(("min", "p10", "median", "p90", "max"), quantiles.tolist())),
    }


def summarize(rows, plan):
    """Descriptive repeated-filter and participant summaries, no inferential CI."""
    expected_keys = {
        (p, i, n, k, c, b)
        for p in plan["source_subject_ids"]
        for i in plan["interfaces"]
        for n in plan["sample_counts"]
        for k in (3, 5)
        for c in range(12)
        for b in range(5)
    }
    keys = [
        (r["subject"], r["interface"], r["samples"], r["k"], r["class"], r["band"]) for r in rows
    ]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("Missing, duplicate or out-of-contract geometry rows")
    bands = {}
    for row in rows:
        key = (row["subject"], row["interface"], row["samples"], row["k"], row["band"])
        bands.setdefault(key, []).append(row["iso_c_to_b_norm_factor"])
    band_rows = []
    for (p, i, n, k, b), factors in bands.items():
        a = np.asarray(factors)
        band_rows.append(
            {
                "subject": p,
                "interface": i,
                "samples": n,
                "k": k,
                "band": b,
                "class_norm_factor_min": float(a.min()),
                "class_norm_factor_max": float(a.max()),
                "class_norm_factor_cv": float(a.std(ddof=0) / a.mean()),
            }
        )
    summaries = []
    scopes = [("ALL", None, None)] + [
        (f"{i}_{n}", i, n) for i in plan["interfaces"] for n in plan["sample_counts"]
    ]
    for scope, interface, samples in scopes:
        for k in (3, 5):
            selected = [
                r
                for r in rows
                if r["k"] == k
                and (interface is None or (r["interface"], r["samples"]) == (interface, samples))
            ]
            selected_bands = [
                r
                for r in band_rows
                if r["k"] == k
                and (interface is None or (r["interface"], r["samples"]) == (interface, samples))
            ]
            participant_means = [
                {
                    "subject": p,
                    **{
                        metric: finite_mean([r[metric] for r in selected if r["subject"] == p])
                        for metric in SCALARS
                    },
                }
                for p in plan["source_subject_ids"]
            ]
            summaries.append(
                {
                    "scope": scope,
                    "k": k,
                    "filter_records": len(selected),
                    "participant_count": len(participant_means),
                    "pooled_filter_descriptive": {
                        m: distribution([r[m] for r in selected]) for m in SCALARS
                    },
                    "participant_mean_descriptive": {
                        m: distribution([r[m] for r in participant_means]) for m in SCALARS
                    },
                    "participant_means": participant_means,
                    "class_norm_factor_cv": distribution(
                        [r["class_norm_factor_cv"] for r in selected_bands]
                    ),
                    "direction_sine_ge01_count": sum(r["direction_sine"] >= 0.1 for r in selected),
                    "direction_sine_ge05_count": sum(r["direction_sine"] >= 0.5 for r in selected),
                }
            )
    return summaries, band_rows
