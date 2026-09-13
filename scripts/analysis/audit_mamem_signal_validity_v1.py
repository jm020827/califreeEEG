"""Independent saved-scalar/role audit; no MAT, feature producer, fit, or solve."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "docs/reports/mamem_signal_validity_v1_run"


def check(value, reason):
    if not value:
        raise ValueError(reason)


def read(name):
    path = RUN / name
    check(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128*1024, "file_bound")
    return json.loads(path.read_bytes())


def moments(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def main():
    terminal, manifest, result = read("terminal.json"), read("manifest.json"), read("diagnostic.json")
    check(terminal["status"] == "COMPLETE" and terminal["fits"] == 0
          and terminal["attempts"] == 1 and terminal["windows"] == 15, "terminal")
    digest = hashlib.sha256((RUN / "diagnostic.json").read_bytes()).hexdigest()
    check(digest == terminal["diagnostic_sha256"] == hashlib.sha256(
        (RUN / "diagnostic.json").read_bytes()).hexdigest(), "result_hash")
    expected_pins = {"scripts/analysis/diagnose_mamem_signal_v1.py",
                     "src/cfeg/mamem_signal_diagnostic_v1.py",
                     "docs/mamem_signal_validity_v1_contract.md",
                     "src/cfeg/mamem_events_v1.py", "src/cfeg/mamem_events_v2.py",
                     "src/cfeg/mamem_signal_v1.py"}
    check(set(manifest["code_sha256"]) == expected_pins, "six_pins")
    for name, expected in manifest["code_sha256"].items():
        check(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "code_pin")
    check(manifest["fits"] == 0 and manifest["attempt_budget"] == 1
          and manifest["subject"] == "S001"
          and result["mat_sha256"] == manifest["mat_sha256"]
          == "57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a", "input_binding")
    scope = result["scope"]
    check(scope["subject"] == "S001" and scope["development_only"] is True
          and scope["main_windows"] == 15 and scope["eeg_rows"] == [0, 256]
          and scope["window_samples"] == 500 and scope["offset_samples"] == 250, "scope")
    check(all(scope[k] is False for k in ("M_features_computed", "outside_selected_eeg_used",
          "accuracy_computed", "frequency_search", "independent_label_truth", "physical_timing_verified"))
          and all(scope[k] == 0 for k in ("fits", "held60", "source_cohort_reads")), "forbidden_scope")
    p = np.random.default_rng(20260914).permutation(500)
    check(result["permutation"] == p.tolist()
          and result["permutation_sha256"] == hashlib.sha256(p.astype("<i8").tobytes()).hexdigest(),
          "fixed_permutation")
    rows = result["groups"]
    check([r["group_index"] for r in rows] == list(range(8, 23)), "15_groups")
    check(sorted(r["inferred_label"] for r in rows) == [j for j in range(5) for _ in range(3)], "labels")
    errors = []
    for r in rows:
        label = r["inferred_label"]
        f = r["frequencies_hz"]
        check(f["nominal"] == [6.66, 7.5, 8.57, 10, 12][label], "nominal")
        errors += [abs(f["time"] - 500/r["mean_interval_ms"]),
                   abs(f["sample"] - 125/r["mean_interval_samples"]),
                   abs(f["time"]/f["sample"] - 4/r["mean_ms_per_sample"])]
        check(0 <= r["permutation_covariance_relative_error"] <= 1e-10, "covariance_invariant")
        check(0 <= r["start0"] < r["end0"] <= 117917 and r["end0"]-r["start0"] == 500, "window")
        for name in ("original", "permuted"):
            energy = np.asarray(r[name]["relative_energy_h1_h2"])
            check(energy.shape == (5, 2) and np.isfinite(energy).all(), "energy_shape")
            weighted = (energy[:, 0] + .5*energy[:, 1])/1.5
            errors.append(float(np.max(np.abs(weighted-r[name]["weighted_energy"]["values"]))))
            for metric in ("cca", "weighted_energy"):
                v = np.array(r[name][metric]["values"])
                check(v.shape == (5,) and np.isfinite(v).all(), "score_shape")
                ordered = sorted(v)
                errors += [abs(r[name][metric]["top_gap"] - (ordered[-1]-ordered[-2])),
                           abs(r[name][metric]["inferred_label_margin"]
                               - (v[label] - max(v[j] for j in range(5) if j != label)))]
    check(max(errors) <= 1e-12, "saved_scalar_arithmetic")
    classes = []
    for label in range(5):
        subset = [r for r in rows if r["inferred_label"] == label]
        classes.append({"inferred_label": label,
                        "frequency_mean_hz": {c: float(np.mean([r["frequencies_hz"][c] for r in subset]))
                                              for c in ("nominal", "time", "sample")},
                        "relative_energy_mean_h1_h2": {c: np.mean([
                            r["reference"]["relative_energy_h1_h2"][c] for r in subset], axis=0).tolist()
                            for c in ("nominal", "time", "sample")}})
    print(json.dumps({"status": "PASS_SAVED_SCALARS_AND_RECORDED_SCOPE", "windows": 15,
          "result_sha256": digest, "max_scalar_error": max(errors), "refits": 0, "raw_reads": 0,
          "classes": classes,
          "time_minus_sample_hz": moments([r["frequencies_hz"]["time"]-r["frequencies_hz"]["sample"] for r in rows]),
          "max_interval_clock_residual_ms": moments([r["max_interval_clock_residual_ms"] for r in rows]),
          "entropy_effective_rank": moments([r["covariance"]["entropy_effective_rank"] for r in rows]),
          "ridge_effective_dimension": moments([r["covariance"]["ridge_effective_dimension"] for r in rows]),
          "original_permuted": {name: {metric: {
              "top_score": moments([max(r[name][metric]["values"]) for r in rows]),
              "top_gap": moments([r[name][metric]["top_gap"] for r in rows]),
              "inferred_label_margin": moments([r[name][metric]["inferred_label_margin"] for r in rows])}
              for metric in ("cca", "weighted_energy")} for name in ("original", "permuted")},
          "limitations": "Saved scalar arithmetic only; no raw reconstruction, independent label truth, phase-only null, efficacy or p-value."}, allow_nan=False))


if __name__ == "__main__":
    main()
