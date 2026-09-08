"""Retained independent S/C-derived replay; only four pinned diagnostic inputs.

Staged by read-only reviewer prior_sensitivity_review, then retained by root.
Run with the existing project Python, without -O. No producer imports or writes.
This does not independently reconstruct S/C from the native EEG prefixes.
"""

import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy import linalg

started = time.monotonic()
base = Path("/home/whwovy/trca-support-geometry-v1")
plan_path = Path("/home/whwovy/califreeEEG/configs/analysis/trca_support_geometry_v1.json")
paths = {
    "plan": plan_path,
    "start": base / "start.json",
    "result": base / "result.json",
    "geometry": base / "geometry.npz",
}
pins = {
    "plan": "e9bbf0d132b25355e2ae189b081b5ed8c40f366435b3c079566b3faec7119b86",
    "start": "af0d3b6517a5e6075cea2caeda286d298abe75b75f5d98c2785c69d607aff856",
    "result": "c7e1b173ca031d4a8f9afd92314aea3a6e73e0d431ae8f0b4c6ba079683a6fd6",
    "geometry": "619866fb742217a0a66c381553ec222d6b13075cb6e973dfda862ab79b9f66d5",
}
hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
assert hashes == pins
p = json.loads(paths["plan"].read_text())
start = json.loads(paths["start"].read_text())
result = json.loads(paths["result"].read_text())
assert start["plan_sha256"] == result["plan_sha256"] == pins["plan"]
assert result["start"] == {
    "filename": "start.json",
    "sha256": pins["start"],
    "bytes": paths["start"].stat().st_size,
}
assert result["geometry"] == {
    "filename": "geometry.npz",
    "sha256": pins["geometry"],
    "bytes": paths["geometry"].stat().st_size,
}
assert (
    start["source_commit"] == result["source_commit"] == "43a6f4bdac544ac93a9f95bc0b6a18a9ac85fdf3"
)
assert start["source_hashes"] == result["source_hashes"]
assert (
    start["native_manifest_sha256"]
    == result["native_manifest_sha256"]
    == p["native_manifest_sha256"]
)
assert start["pins"] == p["pins"]
assert start["query_scoring"] is False and start["numeric_metadata"] is False
assert all(
    result[k] is False
    for k in [
        "query_values_decoded",
        "proxy_values_decoded",
        "numeric_metadata_read",
        "classification_executed",
    ]
)
assert result["record_count"] == p["records_expected"] == 37440
with np.load(paths["geometry"], allow_pickle=False) as z:
    assert set(z.files) == set(p["sufficient_statistics"])
    arrays = {k: z[k] for k in z.files}
for k, x in arrays.items():
    assert list(x.shape) == p["array_shapes"][k] and x.dtype == np.float64 and np.isfinite(x).all()
S, C, full, iso = (arrays[k] for k in ("S", "C", "full_unit", "iso_unit"))
ids = list(
    product(
        p["source_subject_ids"],
        p["sample_counts"],
        p["interfaces"],
        p["budgets"],
        range(12),
        range(5),
    )
)
rows = result["rows"]
bands = result["band_groups"]
summaries = result["summaries"]
assert len(rows) == len(ids) == 37440
keys = ("subject", "samples", "interface", "k", "class", "band")
assert [tuple(row[k] for k in keys) for row in rows] == ids
assert len(set(ids)) == len(ids)
symS = np.linalg.norm(S - S.transpose(0, 2, 1), axis=(1, 2)) / np.linalg.norm(S, axis=(1, 2))
symC = np.linalg.norm(C - C.transpose(0, 2, 1), axis=(1, 2)) / np.linalg.norm(C, axis=(1, 2))
assert symS.max() < 1e-12 and symC.max() < 1e-12
trace = np.trace(C, axis1=1, axis2=2)
tau = p["gamma"] * trace / 8
spectrum = np.linalg.eigvalsh(C)
assert (spectrum > 0).all()
spectrum_ratio = spectrum / (trace[:, None] / 8)
regularization_ratio = tau[:, None] / spectrum
N = len(rows)
independent_full = np.empty_like(full)
independent_iso = np.empty_like(iso)
full_roots = np.empty((N, 8))
iso_roots = np.empty((N, 8))
for i in range(N):
    w, v = linalg.eigh(S[i], C[i], check_finite=False)
    full_roots[i] = w
    independent_full[i] = v[:, -1] / np.linalg.norm(v[:, -1])
    w, v = linalg.eigh(S[i], C[i] + tau[i] * np.eye(8), check_finite=False)
    iso_roots[i] = w
    independent_iso[i] = v[:, -1] / np.linalg.norm(v[:, -1])


def derived(u, v):
    fu = np.einsum("ni,nij,nj->n", u, C, u)
    iv = np.einsum("ni,nij,nj->n", v, C, v)
    rf = tau * np.einsum("ni,ni->n", u, u) / fu
    ri = tau * np.einsum("ni,ni->n", v, v) / iv
    cosine = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return {
        "tau": tau,
        "covariance_eigenvalues_over_mean": spectrum_ratio,
        "tau_over_covariance_eigenvalues": regularization_ratio,
        "condition_number": spectrum[:, -1] / spectrum[:, 0],
        "effective_rank": trace**2 / np.sum(C * C, axis=(1, 2)),
        "full_direction_rho": rf,
        "iso_direction_rho": ri,
        "direction_sine": np.sqrt(np.maximum(0, 1 - np.clip(cosine, -1, 1) ** 2)),
        "iso_c_to_b_norm_factor": 1 / np.sqrt(1 + ri),
        "modes_tau_over_lambda_ge1": np.sum(regularization_ratio >= 1, axis=1),
        "modes_tau_over_lambda_ge10": np.sum(regularization_ratio >= 10, axis=1),
        "full_eigenvalue_gap": full_roots[:, -1] - full_roots[:, -2],
        "iso_eigenvalue_gap": iso_roots[:, -1] - iso_roots[:, -2],
    }


metrics = derived(independent_full, independent_iso)
saved_direction_metrics = derived(full, iso)
errors = {}
for k, x in metrics.items():
    observed = np.array([r[k] for r in rows])
    absdiff = np.abs(x - observed)
    errors[k] = {
        "max_abs": float(absdiff.max()),
        "max_scaled": float(np.max(absdiff / np.maximum(1, np.abs(observed)))),
    }
    np.testing.assert_allclose(x, observed, rtol=1e-8, atol=1e-10, err_msg=k)
    if k.startswith("modes_"):
        assert np.array_equal(x, observed)


def sign_distance(a, b):
    return np.minimum(np.linalg.norm(a - b, axis=1), np.linalg.norm(a + b, axis=1))


d_full = sign_distance(full, independent_full)
d_iso = sign_distance(iso, independent_iso)
assert d_full.max() < 1e-7 and d_iso.max() < 1e-7
assert np.max(np.abs(np.linalg.norm(full, axis=1) - 1)) < 1e-12
assert np.max(np.abs(np.linalg.norm(iso, axis=1) - 1)) < 1e-12
residual = {}
for label, u, roots, penalty in [("FULL", full, full_roots, 0), ("ISO", iso, iso_roots, tau)]:
    B = C + np.asarray(penalty)[..., None, None] * np.eye(8) if label == "ISO" else C
    Su = np.einsum("nij,nj->ni", S, u)
    Bu = np.einsum("nij,nj->ni", B, u)
    rayleigh = np.einsum("ni,ni->n", u, Su) / np.einsum("ni,ni->n", u, Bu)
    res = np.linalg.norm(Su - roots[:, -1, None] * Bu, axis=1) / (
        np.linalg.norm(S, axis=(1, 2)) + np.abs(roots[:, -1]) * np.linalg.norm(B, axis=(1, 2))
    )
    gap = roots[:, -1] - roots[:, -2]
    gate = 1e-10 * np.maximum(1, np.max(np.abs(roots), axis=1))
    assert (gap > gate).all() and res.max() < 1e-10
    residual[label] = {
        "max_normalized_residual": float(res.max()),
        "max_rayleigh_vs_top_abs": float(np.max(np.abs(rayleigh - roots[:, -1]))),
        "min_top_gap": float(gap.min()),
        "min_gap_over_reject_threshold": float(np.min(gap / gate)),
    }
# Group independently by identity, one group per 12 classes.
group_keys = ("subject", "samples", "interface", "k", "band")
group_index = {}
for i, row in enumerate(rows):
    group_index.setdefault(tuple(row[k] for k in group_keys), []).append(i)
assert len(group_index) == 3120 == len(bands)
computed_groups = {}
for key, indices in group_index.items():
    assert len(indices) == 12 and {rows[i]["class"] for i in indices} == set(range(12))
    a = metrics["iso_c_to_b_norm_factor"][indices]
    computed_groups[key] = {
        "class_norm_factor_min": float(a.min()),
        "class_norm_factor_max": float(a.max()),
        "class_norm_factor_cv": float(a.std(ddof=0) / a.mean()),
    }
group_error = 0.0
assert len({tuple(g[k] for k in group_keys) for g in bands}) == 3120
for g in bands:
    expected = computed_groups[tuple(g[k] for k in group_keys)]
    for k, v in expected.items():
        group_error = max(group_error, abs(v - g[k]))
        np.testing.assert_allclose(v, g[k], rtol=1e-8, atol=1e-10)


# Independent summaries use own row metrics, NOT saved row scalars.
def describe(values):
    values = np.asarray(values, dtype=float)
    q = np.quantile(values, [0, 0.1, 0.5, 0.9, 1], method="linear")
    return dict(  # noqa: C408 -- retain the independently executed equation payload
        n=len(values),
        mean=float(values.mean()),
        min=float(q[0]),
        p10=float(q[1]),
        median=float(q[2]),
        p90=float(q[3]),
        max=float(q[4]),
    )


aggregate_error_abs = 0.0
aggregate_error_scaled = 0.0
values_checked = 0


def compare(actual, expected, label):
    global aggregate_error_abs, aggregate_error_scaled, values_checked
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (label, set(actual) ^ set(expected))
        for k in expected:
            compare(actual[k], expected[k], label + "." + k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), (label, len(actual), len(expected))
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, label + f"[{i}]")
    elif isinstance(expected, (int, str)):
        assert actual == expected, (label, actual, expected)
    else:
        d = abs(actual - expected)
        aggregate_error_abs = max(aggregate_error_abs, d)
        aggregate_error_scaled = max(aggregate_error_scaled, d / max(1, abs(actual)))
        values_checked += 1
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, err_msg=label)


scalar_keys = [
    "condition_number",
    "effective_rank",
    "full_direction_rho",
    "iso_direction_rho",
    "direction_sine",
    "iso_c_to_b_norm_factor",
    "modes_tau_over_lambda_ge1",
    "modes_tau_over_lambda_ge10",
]
summary_keys = {(s["scope"], s["k"]) for s in summaries}
expected_keys = {
    (scope, k)
    for scope in ["ALL"] + [f"{i}_{n}" for i in p["interfaces"] for n in p["sample_counts"]]
    for k in p["budgets"]
}
assert len(summaries) == 18 and summary_keys == expected_keys, (summary_keys, expected_keys)
for summary in summaries:
    scope = summary["scope"]
    k = summary["k"]
    selected = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["k"] == k and (scope == "ALL" or f"{r['interface']}_{r['samples']}" == scope)
        ]
    )
    assert len(selected) == (18720 if scope == "ALL" else 2340)
    participant = []
    for subject in p["source_subject_ids"]:
        sel = np.array([i for i in selected if rows[i]["subject"] == subject])
        assert len(sel) == (480 if scope == "ALL" else 60)
        participant.append(
            {"subject": subject, **{key: float(metrics[key][sel].mean()) for key in scalar_keys}}
        )
    cv = np.array(
        [
            g["class_norm_factor_cv"]
            for key, g in computed_groups.items()
            if key[3] == k and (scope == "ALL" or f"{key[2]}_{key[1]}" == scope)
        ]
    )
    assert len(cv) == (1560 if scope == "ALL" else 195)
    expected = {
        "scope": scope,
        "k": k,
        "filter_records": len(selected),
        "participant_count": 39,
        "pooled_filter_descriptive": {key: describe(metrics[key][selected]) for key in scalar_keys},
        "participant_mean_descriptive": {
            key: describe([r[key] for r in participant]) for key in scalar_keys
        },
        "participant_means": participant,
        "class_norm_factor_cv": describe(cv),
        "direction_sine_ge01_count": int(np.sum(metrics["direction_sine"][selected] >= 0.1)),
        "direction_sine_ge05_count": int(np.sum(metrics["direction_sine"][selected] >= 0.5)),
    }
    compare(summary, expected, f"summary:{scope}:{k}")
assert {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()} == hashes
print(
    json.dumps(
        {
            "audit": "PASS",
            "hashes": hashes,
            "paired_rows": N,
            "covariance_directions": spectrum.size,
            "class_cv_groups": len(group_index),
            "summary_scopes": len(summaries),
            "summary_float_values_checked": values_checked,
            "symmetry_relative_max_S": float(symS.max()),
            "symmetry_relative_max_C": float(symC.max()),
            "positive_spectrum_min": float(spectrum.min()),
            "max_unit_norm_error_full": float(np.max(np.abs(np.linalg.norm(full, axis=1) - 1))),
            "max_unit_norm_error_iso": float(np.max(np.abs(np.linalg.norm(iso, axis=1) - 1))),
            "max_sign_aligned_direction_L2_error_full": float(d_full.max()),
            "max_sign_aligned_direction_L2_error_iso": float(d_iso.max()),
            "eigenpair_checks": residual,
            "row_metric_errors": errors,
            "class_group_max_abs_error": group_error,
            "summary_max_abs_error": aggregate_error_abs,
            "summary_max_scaled_error": aggregate_error_scaled,
            "seconds": time.monotonic() - started,
        },
        indent=2,
    )
)
