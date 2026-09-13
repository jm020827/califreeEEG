"""Independent saved-event arithmetic audit. Never import producer or read raw MAT."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "docs/reports/mamem_reference_residual_v1_run"
SUBJECTS = [f"S{p:03d}" for p in range(2, 12)]
NAMES = [s+r for r in ("a", "b") for s in SUBJECTS]
SPEC = "configs/governance/mamem_reference_residual_v1_inputs.json"
CAP = 1024**2


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= CAP, "json_file")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_key")
            result[key] = value
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "nonfinite_json"))


def sha(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= CAP, "hash_file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.utcoffset().total_seconds() == 0, "utc")
    return result.timestamp()


def independent_summary(f):
    """Scalar loop formulas, not the producer's vectorized implementation."""
    means = [[[sum(f[p][r][c])/3 for c in range(5)] for r in range(2)] for p in range(10)]
    repeats = {}
    for r, run in enumerate(("a", "b")):
        within = sum(sum((v-means[p][r][c])**2 for v in f[p][r][c])/2
                     for p in range(10) for c in range(5))/50
        between = 0
        for c in range(5):
            mean = sum(means[p][r][c] for p in range(10))/10
            between += sum((means[p][r][c]-mean)**2 for p in range(10))/9/5
        excess = math.sqrt(max(between-within/3, 0))
        repeats[run] = {"within_variance_hz2": within, "between_mean_variance_hz2": between,
                        "repeat_mean_noise_proxy_hz2": within/3, "excess_rms_hz": excess,
                        "eligible": between >= 2*within/3 and excess >= 0.025}
    people = []
    for p, subject in enumerate(SUBJECTS):
        common = [[sum(math.log(f[o][r][c][j]) for o in range(10) if o != p
                       for j in range(3))/27 for c in range(5)] for r in range(2)]
        x = [math.log(f[p][0][c][0])-common[0][c] for c in range(5)]
        target = [math.exp(sum(math.log(v) for v in f[p][1][c])/3) for c in range(5)]
        baseline = [math.exp(v) for v in common[1]]
        direct = [math.exp(v+delta) for v, delta in zip(common[1], x)]
        errors = {"common": [(a-b)**2 for a, b in zip(baseline, target)],
                  "direct": [(a-b)**2 for a, b in zip(direct, target)]}
        mse = {key: sum(error)/5 for key, error in errors.items()}
        people.append({"subject": subject, "common_a_hz": [math.exp(v) for v in common[0]],
                       "common_b_hz": baseline, "target_b_hz": target, "direct_b_hz": direct,
                       "a_first_residual_log": x,
                       "b_mean_residual_log": [math.log(v)-b for v, b in zip(target, common[1])],
                       "squared_error_hz2": errors, "mse_hz2": mse,
                       "strict_improvement": mse["common"]-mse["direct"] > 1e-12})
    mse = {arm: sum(p["mse_hz2"][arm] for p in people)/10 for arm in ("common", "direct")}
    floor = mse["common"] <= 1e-24
    gain = 0.0 if floor else 1-mse["direct"]/mse["common"]
    improved = sum(p["strict_improvement"] for p in people)
    residuals = [p["a_first_residual_log"] for p in people]
    total = sum(v*v for row in residuals for v in row)
    factor = sum(5*(sum(row)/5)**2 for row in residuals)
    orthogonal = math.sqrt(sum((v-sum(row)/5)**2 for row in residuals for v in row)/50)
    return {"repeatability": repeats, "participants": people, "mse_hz2": mse,
            "common_error_floor": floor, "gain_fraction": gain, "improved_participants": improved,
            "factor_descriptive": {"log_all_ones_energy_fraction": factor/total
                                   if total > 1e-24 else 0.0,
                                   "zero_residual_energy": total <= 1e-24,
                                   "orthogonal_log_rms": orthogonal, "promotion_allowed": False},
            "decision": "ELIGIBLE_CLASSWISE_DIRECT_TRANSFER_ONLY" if
            all(v["eligible"] for v in repeats.values()) and not floor and gain >= .1
            and improved >= 7 else "RETIRE_FULL_RESIDUAL_ROUTE"}


def compare(actual, expected):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "dictionary_keys")
        return max((compare(actual[k], v) for k, v in expected.items()), default=0)
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "list_shape")
        return max((compare(a, b) for a, b in zip(actual, expected)), default=0)
    if isinstance(expected, (bool, str)) or expected is None:
        require(type(actual) is type(expected) and actual == expected, "exact_scalar")
        return 0
    if type(expected) is int:
        require(type(actual) is int and actual == expected, "exact_integer")
        return 0
    require(type(actual) in (int, float) and math.isfinite(actual) and
            math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), "numeric_scalar")
    return abs(actual-expected)


def audit(root=ROOT, run=RUN):
    manifest = read(run/"manifest.json")
    result = read(run/"result.json")
    start, claim, terminal = [read(run/(n+".json")) for n in ("started", "worker_claim", "terminal")]
    input_spec = read(root/SPEC)
    require(set(input_spec) == {"schema", "inputs"} and input_spec["schema"] ==
            "cfeg.mamem-reference-residual-v1.inputs", "input_schema")
    inputs = input_spec["inputs"]
    require(set(manifest) == {"schema", "created_utc", "deadline_utc", "attempts", "scope", "pins"}
            and manifest["schema"] == "cfeg.mamem-reference-residual-v1.manifest"
            and manifest["deadline_utc"] == "2026-09-13T15:30:00+00:00", "manifest_schema")
    require(set(result) == {"schema", "completed_utc", "scope", "record_sha256", "a_frozen_sha256",
                            "frequencies_hz", "summary", "costs"}
            and result["schema"] == "cfeg.mamem-reference-residual-v1.result", "result_schema")
    require(len(inputs) == 20 and [(r["subject"], r["run"]) for r in inputs] ==
            [(s, r) for s in SUBJECTS for r in ("a", "b")], "source_roles")
    expected_pins = {SPEC, "docs/mamem_reference_residual_v1_contract.md",
                     "scripts/analysis/run_mamem_reference_residual_v1.py",
                     "scripts/analysis/audit_mamem_reference_residual_v1.py",
                     "src/cfeg/mamem_reference_residual_v1.py", "src/cfeg/mamem_reference_probe_v1.py",
                     "src/cfeg/mamem_events_v1.py", "src/cfeg/mamem_events_v2.py"}
    require(set(manifest["pins"]) == expected_pins, "pin_keyset")
    require(all(sha(root/p) == h for p, h in manifest["pins"].items()), "code_hashes")
    expected_scope = {"mat_byte_hashes": 20, "mat_headers": 20, "mat_din_loads": 20,
                      "a_diagnostic_event_windows": 150, "b_diagnostic_event_windows": 150,
                      "b_predictor_metadata_used": 0, "eeg_arrays_decoded": 0,
                      "eeg_numeric_windows": 0, "fits": 0, "eeg_predictions": 0,
                      "held60_openings": 0}
    compare(manifest["scope"], expected_scope)
    compare(result["scope"], expected_scope)
    require(all(type(v) is int for v in
                (manifest["attempts"], start["attempt"], claim["attempt"], terminal["attempts"])),
            "attempt_integer")
    require(manifest["attempts"] == start["attempt"] == claim["attempt"] == terminal["attempts"] == 1
            and start["parent_pid"] == claim["parent_pid"] and start["status"] == "STARTED"
            and terminal["status"] == "COMPLETE" and terminal["fits"] == 0, "lifetime")
    require(all(d["manifest_sha256"] == sha(run/"manifest.json") for d in (start, claim, terminal))
            and terminal["started_sha256"] == sha(run/"started.json")
            and terminal["worker_claim_sha256"] == sha(run/"worker_claim.json")
            and terminal["result_sha256"] == sha(run/"result.json"), "lifetime_binding")
    require(stamp(manifest["created_utc"]) <= stamp(start["started_utc"]) <= stamp(claim["started_utc"])
            <= stamp(result["completed_utc"]) <= stamp(terminal["ended_utc"])
            <= stamp(manifest["deadline_utc"])
            and stamp(terminal["ended_utc"])-stamp(start["started_utc"]) <= 180, "lifetime_times")
    seal = read(run/"a_frozen.json")
    require(result["a_frozen_sha256"] == sha(run/"a_frozen.json") and seal["event_windows"] == 150
            and set(result["record_sha256"]) == set(NAMES)
            and set(seal["record_sha256"]) == set(NAMES[:10]), "a_seal_keys")
    values = np.zeros((10, 2, 5, 3))
    costs = {}
    latest = stamp(claim["started_utc"])
    for name in NAMES:
        row = next(r for r in inputs if r["subject"]+r["run"] == name)
        record = read(run/("record_"+name+".json"))
        attempt = read(run/("attempt_"+name+".json"))
        require(result["record_sha256"][name] == sha(run/("record_"+name+".json")), "record_hash")
        require(record["mat_sha256"] == attempt["mat_sha256"] == row["sha256"]
                and record["receipt_sha256"] == row["receipt_sha256"] and attempt["name"] == name
                and record["subject"] == row["subject"] and record["run"] == row["run"]
                and record["loaded_variables"] == attempt["variable_names"] == ["DIN_1", "samplingRate"],
                "record_role")
        require(latest <= stamp(attempt["started_utc"]) <= stamp(record["completed_utc"]), "read_order")
        latest = stamp(record["completed_utc"])
        if row["run"] == "a":
            require(seal["record_sha256"][name] == result["record_sha256"][name]
                    and latest <= stamp(seal["created_utc"]), "a_seal")
        else:
            require(stamp(seal["created_utc"]) <= stamp(attempt["started_utc"]), "b_after_a_seal")
        require(type(record["total_samples"]) is int and 0 < record["total_samples"] <= 200000,
                "total_samples")
        trials = record["records"]
        require(len(trials) == 15 and [t["group_index"] for t in trials] == list(range(8, 23)),
                "main_groups")
        counts = [0]*5
        previous_end = 0
        for trial in trials:
            label, repeat = trial["label"], trial["repeat"]
            require(type(label) is int and 0 <= label < 5 and type(repeat) is int
                    and counts[label] == repeat < 3, "trial_role")
            counts[label] += 1
            samples = trial["event_samples"]
            require(all(type(trial[k]) is int for k in
                        ("group_index", "start0", "end0", "trial_end0")), "window_integer")
            require(4 <= len(samples) <= 500 and all(type(v) is int for v in samples)
                    and all(a < b for a, b in pairwise(samples))
                    and all(1 <= v <= record["total_samples"] and
                            trial["start0"] <= v-1 < trial["end0"] for v in samples), "sample_bounds")
            require(trial["start0"] >= previous_end and trial["end0"]-trial["start0"] == 500
                    and trial["trial_end0"]-trial["start0"] == 1000
                    and trial["trial_end0"] <= record["total_samples"], "window_bounds")
            previous_end = trial["end0"]
            f = 250*(len(samples)-1)/(2*(samples[-1]-samples[0]))
            compare(trial["frequency_hz"], f)
            values[SUBJECTS.index(row["subject"]), ("a", "b").index(row["run"]), label, repeat] = f
        require(counts == [3]*5 and all(len({t["label"] for t in trials[k:k+3]}) == 1
                                      for k in range(0, 15, 3)), "class_coverage")
        if row["run"] == "a":
            costs[row["subject"]] = {"selected_trials": 5, "selected_stimulus_seconds": 25,
                                      "analyzed_din_seconds": 10,
                                      "selected_elapsed_prefix_seconds":
                                      max(t["trial_end0"] for t in trials if t["repeat"] == 0)/250,
                                      "full_a_record_seconds": record["total_samples"]/250,
                                      "query_ready_elapsed": "UNKNOWN", "setup_seconds": "UNKNOWN"}
    require(latest <= stamp(result["completed_utc"]), "result_after_reads")
    require(bool((values > 0).all() and (values < 62.5).all() and
                 (np.diff(values, axis=2) > 0).all()), "frequency_bank_order")
    errors = [compare(result["frequencies_hz"], values.tolist()),
              compare(seal["frequencies_hz"], values[:, 0].tolist()),
              compare(result["costs"], costs),
              compare(result["summary"], independent_summary(values.tolist()))]
    expected_files = {"manifest.json", "started.json", "worker_claim.json", "a_frozen.json",
                      "result.json", "terminal.json", "stdout.txt", "stderr.txt"}
    expected_files.update(prefix+n+".json" for n in NAMES for prefix in ("attempt_", "record_"))
    require({p.name for p in run.iterdir()} == expected_files and
            all(p.is_file() and not p.is_symlink() and p.stat().st_size <= CAP for p in run.iterdir())
            and sum(p.stat().st_size for p in run.iterdir()) <= 8*CAP, "artifact_budget")
    return {"status": "PASS_SAVED_EVENTS_AND_RECORDED_SCOPE", "actual_audit_attempts": 1,
            "event_windows": 300, "participants": 10, "raw_reads": 0, "fits": 0,
            "max_scalar_error": max(errors), "result_sha256": sha(run/"result.json"),
            "decision": result["summary"]["decision"],
            "limitation": "No independent raw-DIN segmentation, physical-clock or OS-I/O audit",
            "completed_utc": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        try:
            result = audit()
        except Exception as error:  # noqa: BLE001 -- retain terminal failure, never retry
            result = {"status": "AUDIT_FAIL_NO_RETRY", "reason": str(error)[:4096],
                      "error_type": type(error).__name__}
        raw = json.dumps(result, allow_nan=False, indent=2)
        require(len(raw.encode()) <= 64*1024, "audit_output_cap")
        stream.write(raw)
        print(json.dumps(result))
    if result["status"] != "PASS_SAVED_EVENTS_AND_RECORDED_SCOPE":
        raise SystemExit(1)
