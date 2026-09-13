"""Frozen S001 diagnostic algebra. No I/O, M extraction, gate fits or accuracy.

The timestamp and sample frequencies re-express the same event train. Neither is
an independently verified physical frequency or a permitted query predictor.
"""

import hashlib

import numpy as np
from scipy.linalg import eigvalsh

from cfeg.mamem_events_v1 import FREQUENCIES, _scalar
from cfeg.mamem_events_v2 import parse_main_trials
from cfeg.mamem_signal_v1 import _center_and_scale, _quadrature, analyze_window

SEED = 20260914
RANK_INDICES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SCOPE = {
    "subject": "S001",
    "development_only": True,
    "main_windows": 15,
    "eeg_rows": [0, 256],
    "window_samples": 500,
    "offset_samples": 250,
    "M_features_computed": False,
    "M_flag_meaning": "learned_two_feature_MAD_lag1_extractor_not_called",
    "descriptor_interpretation": False,
    "outside_selected_eeg_used": False,
    "raw_arrays_exported": False,
    "fits": 0,
    "accuracy_computed": False,
    "frequency_search": False,
    "independent_label_truth": False,
    "physical_timing_verified": False,
    "held60": 0,
    "source_cohort_reads": 0,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def fixed_permutation():
    return np.random.default_rng(SEED).permutation(500)


def permutation_hash(permutation):
    return hashlib.sha256(np.asarray(permutation, dtype="<i8").tobytes()).hexdigest()


def score_summary(scores, label):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.shape == (5,) and bool(np.isfinite(scores).all()), "five_finite_scores")
    require(type(label) is int and 0 <= label < 5, "label")
    ordered = np.sort(scores)
    return {
        "values": scores.tolist(),
        "top_gap": float(ordered[-1] - ordered[-2]),
        "inferred_label_margin": float(scores[label] - np.max(np.delete(scores, label))),
    }


def reference_diagnostics(x, frequencies):
    require(x.shape == (256, 500), "selected_shape")
    require(set(frequencies) == {"nominal", "time", "sample"}, "three_fixed_clocks")
    total = float(np.sum(x * x))
    bases, energy = {}, {}
    for clock, frequency in frequencies.items():
        require(np.isfinite(frequency) and 0 < frequency < 62.5, "reference_frequency_range")
        bases[clock] = [_quadrature((frequency * h,), 500) for h in (1, 2)]
        energy[clock] = [float(np.sum((x @ basis) ** 2) / total) for basis in bases[clock]]
    overlap = {}
    for a, b in (("nominal", "time"), ("nominal", "sample"), ("time", "sample")):
        pair = []
        for ua, ub in zip(bases[a], bases[b]):
            squared = np.linalg.svd(ua.T @ ub, compute_uv=False) ** 2
            require(bool(((squared >= -1e-10) & (squared <= 1 + 1e-10)).all()),
                    "reference_overlap_bound")
            pair.append({"mean_squared_overlap": float(squared.mean()),
                         "squared_singular_values": squared.tolist()})
        overlap[a + "_" + b] = pair
    return {"relative_energy_h1_h2": energy, "reference_overlap_h1_h2": overlap}


def covariance_summary(covariance):
    eigenvalues = eigvalsh(covariance, check_finite=False)
    trace = float(np.trace(covariance))
    require(float(eigenvalues[0]) >= -1e-10 * trace, "covariance_psd")
    raw_min = float(eigenvalues[0])
    eigenvalues = np.maximum(eigenvalues, 0)[::-1]
    p = eigenvalues / eigenvalues.sum()
    alpha = 1e-6 * trace / 256
    return {
        "trace": trace,
        "rank_indices": list(RANK_INDICES),
        "eigenvalue_fractions_at_indices": [float(p[i - 1]) for i in RANK_INDICES],
        "cumulative_energy_at_indices": [float(p[:i].sum()) for i in RANK_INDICES],
        "min_eigenvalue_over_trace": raw_min / trace,
        "entropy_effective_rank": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
        "cca_ridge_alpha": alpha,
        "regularized_condition_number": float((eigenvalues[0] + alpha) / (raw_min + alpha)),
        "ridge_effective_dimension": float(np.sum(eigenvalues / (eigenvalues + alpha))),
    }


def nominal_summary(features, label):
    energy = features["q"][..., 0]
    return {
        "cca": score_summary(features["zero_shot"], label),
        "weighted_energy": score_summary((energy[:, 0] + 0.5 * energy[:, 1]) / 1.5, label),
        "relative_energy_h1_h2": energy.tolist(),
        "neighbor_log_ratio_h1_h2": features["q"][..., 1].tolist(),
        "q2_channel_concentration_lag1": features["q2"][0].tolist(),
    }


def diagnose_window(selected, frequencies, label, permutation):
    """Selected raw rows only, with a poisoned excluded row in feature wrappers."""
    require(isinstance(selected, np.ndarray) and selected.shape == (256, 500)
            and selected.dtype == np.float64, "selected_float64")
    require(bool(np.isfinite(selected).all()), "selected_finite")
    require(np.asarray(permutation).shape == (500,) and np.asarray(permutation).dtype.kind in "iu"
            and np.array_equal(np.sort(permutation), np.arange(500)), "permutation")
    raw = np.full((257, 500), np.nan, dtype=np.float64)
    raw[:256] = selected
    original = analyze_window(raw, 0, 500)
    raw[:256] = selected[:, permutation]
    permuted = analyze_window(raw, 0, 500)
    c = original["covariance"]
    cov_error = float(np.linalg.norm(c - permuted["covariance"]) / np.linalg.norm(c))
    require(cov_error <= 1e-10, "permutation_covariance_changed")
    x = _center_and_scale(selected.copy())
    reference = reference_diagnostics(x, frequencies)
    require(np.allclose(reference["relative_energy_h1_h2"]["nominal"],
                        original["q"][label, :, 0], atol=1e-12, rtol=1e-10),
            "nominal_projection_mismatch")
    return {
        "frequencies_hz": frequencies,
        "reference": reference,
        "covariance": covariance_summary(c),
        "permutation_covariance_relative_error": cov_error,
        "original": nominal_summary(original, label),
        "permuted": nominal_summary(permuted, label),
    }


def diagnose(eeg, din, checkpoint=lambda: None):
    """Only parser-selected values are numerically inspected in the full EEG."""
    require(isinstance(eeg, np.ndarray) and eeg.ndim == 2 and eeg.shape[0] == 257
            and eeg.dtype == np.float64, "EEG_structure")
    trials = parse_main_trials(din, eeg.shape[1], include_metadata=False)
    stamps = np.array([float(_scalar(v)) for v in din[1]], dtype=np.float64)
    samples = np.array([int(_scalar(v)) for v in din[3]], dtype=np.int64)
    bounds = np.r_[0, np.flatnonzero(np.diff(stamps) > 2000) + 1, len(stamps)]
    permutation = fixed_permutation()
    rows = []
    for trial in trials:
        checkpoint()
        index = trial["group_index"]
        left, right = bounds[index:index + 2]
        dt, ds = np.diff(stamps[left:right]), np.diff(samples[left:right])
        mt, ms = float(dt.mean()), float(ds.mean())
        frequencies = {"nominal": float(FREQUENCIES[trial["label"]]),
                       "time": 1000 / (2 * mt), "sample": 250 / (2 * ms)}
        # First EEG-value operation is the permitted copy, not full-array QC.
        selected = np.array(eeg[:256, trial["start0"]:trial["end0"]], dtype=np.float64,
                            order="C", copy=True)
        row = {
            "group_index": index,
            "inferred_label": trial["label"],
            "start0": trial["start0"],
            "end0": trial["end0"],
            "event_count": int(right - left),
            "window_event_count": trial["event_count"],
            "mean_interval_ms": mt,
            "mean_interval_samples": ms,
            "mean_ms_per_sample": mt / ms,
            "max_interval_clock_residual_ms": float(np.max(np.abs(dt - 4 * ds))),
            **diagnose_window(selected, frequencies, trial["label"], permutation),
        }
        rows.append(row)
    checkpoint()
    return {"schema": "cfeg.mamem-signal-validity-v1", "scope": dict(SCOPE),
            "permutation_seed": SEED, "permutation": permutation.tolist(),
            "permutation_sha256": permutation_hash(permutation), "groups": rows}


def validate_report(result):
    """Parent-side schema and numerical bounds, not an independent raw replay."""
    require(result["schema"] == "cfeg.mamem-signal-validity-v1" and result["scope"] == SCOPE,
            "report_scope")
    permutation = fixed_permutation()
    require(result["permutation_seed"] == SEED
            and result["permutation"] == permutation.tolist()
            and result["permutation_sha256"] == permutation_hash(permutation), "report_permutation")
    require([r["group_index"] for r in result["groups"]] == list(range(8, 23)), "report_groups")
    labels = [r["inferred_label"] for r in result["groups"]]
    require(sorted(labels) == [j for j in range(5) for _ in range(3)], "report_labels")
    require(all(len(set(labels[i:i + 3])) == 1 for i in range(0, 15, 3)), "report_blocks")
    for row in result["groups"]:
        label = row["inferred_label"]
        require(type(label) is int and 0 <= label < 5
                and type(row["start0"]) is int and type(row["end0"]) is int
                and 0 <= row["start0"] < row["end0"] <= 117917
                and row["end0"] - row["start0"] == 500,
                "report_window")
        mt, ms = row["mean_interval_ms"], row["mean_interval_samples"]
        require(np.isfinite([mt, ms]).all() and min(mt, ms) > 0, "report_intervals")
        require(set(row["frequencies_hz"]) == {"nominal", "time", "sample"}
                and abs(row["frequencies_hz"]["time"] - 1000 / (2 * mt)) <= 1e-12
                and abs(row["frequencies_hz"]["sample"] - 250 / (2 * ms)) <= 1e-12
                and abs(row["mean_ms_per_sample"] - mt / ms) <= 1e-12, "report_clock")
        require(np.isfinite(row["max_interval_clock_residual_ms"])
                and row["max_interval_clock_residual_ms"] >= 0, "report_residual")
        require(row["frequencies_hz"]["nominal"] == float(FREQUENCIES[label]), "report_nominal")
        require(0 <= row["permutation_covariance_relative_error"] <= 1e-10, "report_covariance")
        covariance = row["covariance"]
        require(covariance["rank_indices"] == list(RANK_INDICES), "report_rank_indices")
        eig = np.asarray(covariance["eigenvalue_fractions_at_indices"])
        cumulative = np.asarray(covariance["cumulative_energy_at_indices"])
        require(eig.shape == cumulative.shape == (9,) and np.isfinite(eig).all()
                and np.isfinite(cumulative).all() and (eig >= 0).all() and (eig <= 1).all()
                and (np.diff(eig) <= 1e-12).all() and (np.diff(cumulative) >= -1e-12).all()
                and (cumulative >= 0).all() and (cumulative <= 1+1e-10).all()
                and abs(cumulative[-1]-1) <= 1e-10, "report_spectrum")
        require(abs(covariance["trace"] - 256) <= 1e-8
                and abs(covariance["cca_ridge_alpha"] - 1e-6*covariance["trace"]/256) <= 1e-14
                and 1-1e-10 <= covariance["entropy_effective_rank"] <= 256+1e-8
                and 0 <= covariance["ridge_effective_dimension"] <= 256+1e-8
                and np.isfinite(covariance["regularized_condition_number"])
                and covariance["regularized_condition_number"] >= 1-1e-10
                and -1e-10 <= covariance["min_eigenvalue_over_trace"] <= 1/256+1e-10,
                "report_covariance_summary")
        for variant in ("original", "permuted"):
            for metric in ("cca", "weighted_energy"):
                saved = row[variant][metric]
                values = np.asarray(saved["values"])
                require(values.shape == (5,) and bool(np.isfinite(values).all())
                        and bool(((values >= -1e-10) & (values <= 1 + 1e-10)).all()),
                        "report_score_bounds")
                expected = score_summary(values, label)
                require(abs(saved["top_gap"] - expected["top_gap"]) <= 1e-12
                        and abs(saved["inferred_label_margin"]
                                - expected["inferred_label_margin"]) <= 1e-12,
                        "report_margin")
            for key in ("relative_energy_h1_h2", "neighbor_log_ratio_h1_h2"):
                values = np.asarray(row[variant][key])
                require(values.shape == (5, 2) and np.isfinite(values).all(), "report_nominal_shape")
                if key == "relative_energy_h1_h2":
                    require((values >= 0).all() and (values <= 1+1e-10).all(), "report_nominal_energy")
                    expected = (values[:, 0] + .5 * values[:, 1]) / 1.5
                    require(np.allclose(expected, row[variant]["weighted_energy"]["values"],
                                        atol=1e-12, rtol=0), "report_weighted_energy")
            q2 = np.asarray(row[variant]["q2_channel_concentration_lag1"])
            require(q2.shape == (2,) and np.isfinite(q2).all() and q2[0] >= 0
                    and -1-1e-10 <= q2[1] <= 1+1e-10, "report_q2")
        for key in ("nominal", "time", "sample"):
            energy = np.asarray(row["reference"]["relative_energy_h1_h2"][key])
            require(energy.shape == (2,) and bool(np.isfinite(energy).all())
                        and bool(((energy >= 0) & (energy <= 1 + 1e-10)).all()), "report_energy")
        require(np.allclose(row["reference"]["relative_energy_h1_h2"]["nominal"],
                            row["original"]["relative_energy_h1_h2"][label], atol=1e-12, rtol=1e-10),
                "report_nominal_energy_agreement")
        overlaps = row["reference"]["reference_overlap_h1_h2"]
        require(set(overlaps) == {"nominal_time", "nominal_sample", "time_sample"}, "report_overlap_keys")
        for pair in overlaps.values():
            require(len(pair) == 2, "report_overlap_harmonics")
            for item in pair:
                squared = np.asarray(item["squared_singular_values"])
                require(squared.shape == (2,) and np.isfinite(squared).all()
                        and (squared >= 0).all() and (squared <= 1+1e-10).all()
                        and abs(item["mean_squared_overlap"] - squared.mean()) <= 1e-12,
                        "report_overlap")
    return True
