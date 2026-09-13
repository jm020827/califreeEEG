"""Independent, read-only saved-projection audit; no producer imports or raw IO.

The main entry point reads only five bounded run JSON files and six pinned
repository source files. A passing audit does NOT reconstruct EEG -> projection,
prove support/query provenance, or establish independent stimulus ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import resource
import signal
import stat
import sys
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_RELATIVE = "docs/reports/mamem_reference_probe_development_v1_run"
CODE_PATHS = frozenset({
    "docs/mamem_reference_probe_development_v1_contract.md",
    "scripts/analysis/run_mamem_reference_probe_v1.py",
    "scripts/analysis/audit_mamem_reference_probe_v1.py",
    "src/cfeg/mamem_reference_probe_v1.py",
    "src/cfeg/mamem_events_v1.py",
    "src/cfeg/mamem_events_v2.py",
})
ARTIFACT_NAMES = frozenset({
    "manifest.json", "started.json", "worker_claim.json", "terminal.json", "result.json",
})
MAT_SHA256 = {
    "a": "57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a",
    "b": "3beaafa44f7717c690915e7ed85dfa720d96e7da89c7af59113305686c7c9a03",
}
ROLE_SHA256 = {
    "a": "dc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18",
    "b": "4d9f3ab103dd5fba24dd6bb3653c4bb2aaa6f7bd4f791e1a5b01bfa5ee380b59",
}
OPERATOR_SHA256 = "a147f8ac18016d521832f33dc9f45f956ac0f8163aa56df5a5da9f38f32eb7e3"
ARMS = ("NOMINAL", "SAMPLE_SUPPORT")
NOMINAL = (6.66, 7.5, 8.57, 10.0, 12.0)
DEADLINE = datetime(2026, 9, 13, 15, tzinfo=timezone.utc)
FILE_CAP = 128 * 1024
RUN_CAP = 1024 * 1024
FLOAT_TOL = 1e-10
BOUND_TOL = 1e-12
LIMITATION = (
    "Saved projection/scalar arithmetic and recorded scope only: no independent "
    "EEG-to-projection reconstruction, source-file rehash, support/query provenance "
    "proof, independent label truth, physical clock validation, efficacy or savings claim."
)


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _keys(value, expected, name):
    _require(type(value) is dict and set(value) == set(expected), name + "_keys")


def _integer(value, name, *, expected=None, lower=0, upper=None):
    _require(type(value) is int and value >= lower, name + "_integer")
    _require(upper is None or value <= upper, name + "_upper_bound")
    _require(expected is None or value == expected, name + "_value")
    return value


def _number(value, name):
    _require(type(value) in (int, float), name + "_number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + "_finite") from error
    _require(math.isfinite(number), name + "_finite")
    return number


def _same(actual, expected, name):
    """JSON structural equality that does not silently equate bool and int."""
    if type(expected) is dict:
        _keys(actual, expected, name)
        for key, value in expected.items():
            _same(actual[key], value, name + "." + key)
    elif type(expected) is list:
        _require(type(actual) is list and len(actual) == len(expected), name + "_list")
        for index, value in enumerate(expected):
            _same(actual[index], value, name + f"[{index}]")
    else:
        _require(type(actual) is type(expected) and actual == expected, name + "_value")


def _digest(value, name):
    _require(type(value) is str and len(value) == 64
             and all(character in "0123456789abcdef" for character in value), name + "_sha256")
    return value


def _utc(value, name):
    _require(type(value) is str, name + "_utc")
    matched = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?(?:Z|\+00:00)", value,
    )
    _require(matched is not None, name + "_utc")
    # Python 3.10 only accepts 3/6 fractional digits in fromisoformat. Normalize
    # valid 1..6-digit UTC input explicitly, without truncating timestamp precision.
    normalized = matched.group(1)
    if matched.group(2) is not None:
        normalized += "." + matched.group(2).ljust(6, "0")
    try:
        parsed = datetime.fromisoformat(normalized + "+00:00")
    except ValueError as error:
        raise ValueError(name + "_utc") from error
    _require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
             name + "_utc")
    return parsed


def _near(actual, expected, errors, name):
    error = abs(_number(actual, name) - expected)
    _require(error <= FLOAT_TOL, name + "_arithmetic")
    errors.append(error)


def _list(value, count, name):
    _require(type(value) is list and len(value) == count, name + "_shape")
    return value


def _window(record, maximum, name):
    start = _integer(record["start0"], name + "_start", lower=250, upper=maximum)
    end = _integer(record["end0"], name + "_end", upper=maximum)
    _require(end == start + 500, name + "_window_length")
    return start, end


def audit_objects(manifest, started, claim, terminal, result, *, artifact_sha256,
                  actual_code_sha256):
    """Validate in-memory JSON; supplied hashes must refer to those exact bytes.

    ``main`` computes those hashes during its single read of each file. This API
    is separately testable with synthetic objects and never reads any files.
    """
    _keys(manifest, {"schema", "created_utc", "deadline_utc", "subject", "attempt_budget",
                     "fits", "queries", "predictions", "mat_sha256", "role_sha256",
                     "code_sha256"}, "manifest")
    _same(manifest["schema"], "cfeg.mamem-reference-probe-dev-v1.manifest", "manifest_schema")
    _same(manifest["subject"], "S001", "manifest_subject")
    for key, value in {"attempt_budget": 1, "fits": 0, "queries": 15, "predictions": 30}.items():
        _integer(manifest[key], "manifest_" + key, expected=value)
    _same(manifest["mat_sha256"], MAT_SHA256, "manifest_mat")
    _same(manifest["role_sha256"], ROLE_SHA256, "manifest_role")
    _keys(manifest["code_sha256"], CODE_PATHS, "six_code_pins")
    _keys(actual_code_sha256, CODE_PATHS, "six_actual_code_pins")
    for name in CODE_PATHS:
        _require(_digest(manifest["code_sha256"][name], "code_pin")
                 == _digest(actual_code_sha256[name], "actual_code_pin"), "code_hash_binding")
    _same(manifest["code_sha256"]["src/cfeg/mamem_reference_probe_v1.py"],
          OPERATOR_SHA256, "unchanged_operator")
    _keys(artifact_sha256, ARTIFACT_NAMES, "artifact_hashes")
    for name, digest in artifact_sha256.items():
        if name not in {"result.json", "worker_claim.json"} or digest is not None:
            _digest(digest, name)
    created = _utc(manifest["created_utc"], "manifest_created")
    _require(_utc(manifest["deadline_utc"], "deadline") == DEADLINE, "frozen_deadline")

    _keys(started, {"status", "attempt", "parent_pid", "started_utc", "manifest_sha256"},
          "started")
    _same(started["status"], "STARTED", "started_status")
    _integer(started["attempt"], "started_attempt", expected=1)
    parent_pid = _integer(started["parent_pid"], "parent_pid", lower=1)
    start_time = _utc(started["started_utc"], "started_time")
    _require(created <= start_time < DEADLINE, "start_time_order")
    _same(started["manifest_sha256"], artifact_sha256["manifest.json"], "start_manifest_binding")
    if claim is not None:
        _keys(claim, {"attempt", "pid", "parent_pid", "started_utc", "manifest_sha256"}, "claim")
        _integer(claim["attempt"], "claim_attempt", expected=1)
        _integer(claim["parent_pid"], "claim_parent_pid", expected=parent_pid, lower=1)
        _require(_integer(claim["pid"], "claim_pid", lower=1) != parent_pid, "distinct_worker_pid")
        claim_time = _utc(claim["started_utc"], "claim_time")
        _require(start_time <= claim_time < DEADLINE, "claim_time_order")
        _same(claim["manifest_sha256"], artifact_sha256["manifest.json"], "claim_manifest_binding")
        _digest(artifact_sha256["worker_claim.json"], "present_claim_hash")
    else:
        claim_time = start_time
        _require(artifact_sha256["worker_claim.json"] is None, "absent_claim_hash")

    common_terminal = {"status", "attempts", "fits", "started_utc", "ended_utc",
                       "manifest_sha256", "started_sha256", "worker_claim_sha256"}
    _require(type(terminal) is dict, "terminal_object")
    status = terminal.get("status")
    _require(status in {"COMPLETE", "STOPPED_NO_RETRY"}, "terminal_status")
    extra = {"result_sha256", "predictions"} if status == "COMPLETE" else {"error_type", "reason"}
    _keys(terminal, common_terminal | extra, "terminal")
    _integer(terminal["attempts"], "terminal_attempts", expected=1)
    _integer(terminal["fits"], "terminal_fits", expected=0)
    _same(terminal["started_utc"], started["started_utc"], "terminal_started_time")
    _same(terminal["manifest_sha256"], artifact_sha256["manifest.json"], "terminal_manifest_binding")
    _same(terminal["started_sha256"], artifact_sha256["started.json"], "terminal_start_binding")
    _same(terminal["worker_claim_sha256"], artifact_sha256["worker_claim.json"],
          "terminal_claim_binding")
    ended = _utc(terminal["ended_utc"], "ended")
    _require(claim_time <= ended and (ended - start_time).total_seconds() <= 120,
             "terminal_wall_order_bound")
    if status == "STOPPED_NO_RETRY":
        for key in ("error_type", "reason"):
            _require(type(terminal[key]) is str and bool(terminal[key]), "stopped_" + key)
        return {"status": "STOPPED_NO_RETRY_NOT_SCALAR_AUDITED", "attempts": 1,
                "fits": 0, "raw_reads": 0, "partial_result_present": result is not None,
                "artifact_sha256": artifact_sha256, "limitations": LIMITATION}

    _require(claim is not None and result is not None, "complete_requires_claim_result")
    _require(ended <= DEADLINE, "complete_deadline")
    _integer(terminal["predictions"], "terminal_predictions", expected=30)
    _same(terminal["result_sha256"], _digest(artifact_sha256["result.json"], "result_hash"),
          "terminal_result_binding")
    _keys(result, {"schema", "subject", "status", "mat_sha256", "role_sha256", "completed_utc",
                   "sampling_rate_hz", "channel_index", "window_samples", "fits",
                   "frequencies_hz", "support", "queries", "summary", "cost", "scope",
                   "access_stages"}, "result")
    _same(result["schema"], "cfeg.mamem-reference-probe-dev-v1.result", "result_schema")
    _same(result["subject"], "S001", "result_subject")
    _same(result["status"], "DEVELOPMENT_DIAGNOSTIC_NOT_EFFICACY", "result_status")
    _same(result["mat_sha256"], MAT_SHA256, "result_mat")
    _same(result["role_sha256"], ROLE_SHA256, "result_role")
    _require(claim_time <= _utc(result["completed_utc"], "completed") <= ended,
             "completion_order")
    for key, expected in {"sampling_rate_hz": 250, "channel_index": 125,
                          "window_samples": 500, "fits": 0}.items():
        _integer(result[key], "result_" + key, expected=expected)
    _same(result["scope"], {
        "a_loaded_variables": ["DIN_1", "samplingRate"],
        "b_loaded_variables": ["eeg", "DIN_1", "samplingRate"],
        "a_eeg_numeric_windows": 0, "b_eeg_numeric_windows": 15,
        "b_eeg_numeric_channels": 1, "query_metadata_extractions": 0, "gate_fits": 0,
        "source_cohort_reads": 0, "held60_openings": 0,
    }, "scope")
    _same(result["access_stages"], ["a_loaded", "support_bank_frozen", "b_loaded",
                                    "all_scores_frozen", "evaluation_done"], "access_stages")
    _keys(result["frequencies_hz"], ARMS, "frequency_arms")
    errors = []
    for arm in ARMS:
        frequencies = _list(result["frequencies_hz"][arm], 5, "frequency_bank")
        frequencies = [_number(value, "frequency") for value in frequencies]
        _require(all(0 < value < 62.5 for value in frequencies)
                 and all(a < b for a, b in pairwise(frequencies)), "frequency_range_order")
    for value, expected in zip(result["frequencies_hz"]["NOMINAL"], NOMINAL):
        _near(value, expected, errors, "nominal_frequency")

    support = _list(result["support"], 5, "support")
    support_groups, support_windows, trial_ends = [], [], []
    for label, record in enumerate(support):
        _keys(record, {"label", "group_index", "start0", "end0", "trial_end0", "event_samples"},
              "support_record")
        _integer(record["label"], "support_label", expected=label)
        group = _integer(record["group_index"], "support_group", lower=8, upper=22)
        start, end = _window(record, 117917, "support")
        trial_end = _integer(record["trial_end0"], "trial_end", upper=117917)
        _require(trial_end == start + 1000, "trial_end_cost_window")
        samples = record["event_samples"]
        _require(type(samples) is list and 4 <= len(samples) <= 500, "support_event_count")
        for value in samples:
            _integer(value, "event_sample", lower=start + 1, upper=end)
        _require(all(a < b for a, b in pairwise(samples)), "support_event_order")
        span = samples[-1] - samples[0]
        _require(0 < span < 500, "support_event_span")
        frequency = 250 * (len(samples) - 1) / (2 * span)
        _near(result["frequencies_hz"]["SAMPLE_SUPPORT"][label], frequency, errors,
              "sample_frequency")
        support_groups.append(group)
        support_windows.append((group, start, end))
        trial_ends.append(trial_end)
    _require(set(support_groups) == {8, 11, 14, 17, 20}, "first_main_support_groups")
    ordered_support = sorted(support_windows)
    _require(all(left[2] <= right[1] for left, right in pairwise(ordered_support)),
             "support_chronology")

    queries = _list(result["queries"], 15, "queries")
    class_counts = [0] * 5
    per_arm = {arm: [0] * 5 for arm in ARMS}
    paired = {"both_correct": 0, "nominal_only": 0, "sample_only": 0, "both_wrong": 0}
    labels, query_windows = [], []
    reconstruction_near_ties = 0
    for index, query in enumerate(queries):
        _keys(query, {"group_index", "start0", "end0", "label", "arms"}, "query_record")
        _integer(query["group_index"], "query_group", expected=index + 8)
        start, end = _window(query, 200000, "query")
        query_windows.append((start, end))
        label = _integer(query["label"], "query_label", upper=4)
        labels.append(label)
        class_counts[label] += 1
        _keys(query["arms"], ARMS, "query_arms")
        correct = {}
        for arm in ARMS:
            row = query["arms"][arm]
            _keys(row, {"projection", "scores", "prediction", "correct"}, "query_arm")
            projection = _list(row["projection"], 5, "projection")
            saved_scores = _list(row["scores"], 5, "scores")
            reconstructed = []
            for vector, score in zip(projection, saved_scores):
                vector = [_number(value, "projection_value")
                          for value in _list(vector, 4, "projection_vector")]
                _require(all(abs(value) <= 1 + BOUND_TOL for value in vector), "projection_bounds")
                projected_energy = math.fsum(value * value for value in vector)
                _require(projected_energy <= 1 + BOUND_TOL, "projection_energy_bounds")
                score = _number(score, "score")
                _require(-BOUND_TOL <= score <= 1 + BOUND_TOL, "score_bounds")
                expected = min(1.0, max(0.0, projected_energy))
                _near(score, expected, errors, "projection_score")
                reconstructed.append(expected)
            # Projection recomputation is tolerance-based, whereas the producer
            # decision is exactly the saved-score argmax. Do not turn an fsum vs
            # numpy.sum sub-ULP near-tie into a false audit failure.
            predicted = max(range(5), key=lambda j: saved_scores[j])
            _integer(row["prediction"], "prediction", expected=predicted, upper=4)
            _require(max(reconstructed) - reconstructed[predicted] <= 2 * FLOAT_TOL,
                     "reconstructed_prediction_consistency")
            ordered_scores = sorted(reconstructed)
            reconstruction_near_ties += int(ordered_scores[-1] - ordered_scores[-2] <= 2 * FLOAT_TOL)
            correct[arm] = predicted == label
            _same(row["correct"], correct[arm], "correct_bool")
            per_arm[arm][label] += int(correct[arm])
        pair = (("both_correct" if correct[ARMS[1]] else "nominal_only")
                if correct[ARMS[0]] else ("sample_only" if correct[ARMS[1]] else "both_wrong"))
        paired[pair] += 1
    _same(class_counts, [3] * 5, "three_queries_per_class")
    _require(all(labels[index] == labels[index + 1] == labels[index + 2]
                 for index in range(0, 15, 3)), "contiguous_class_blocks")
    _require(all(left[1] <= right[0] for left, right in pairwise(query_windows)),
             "query_chronology")
    summary = {}
    for arm in ARMS:
        counts = per_arm[arm]
        total_correct = sum(counts)
        summary[arm] = {"correct": total_correct, "total": 15, "per_class_correct": counts,
                        "signal_present_dev": total_correct >= 12 and min(counts) >= 1}
    summary["paired"] = paired
    summary["decision"] = (
        "REFERENCE_DIAGNOSTIC_COMPLETE" if any(summary[arm]["signal_present_dev"] for arm in ARMS)
        else "NO_ARM_PASSES_STOP"
    )
    _same(result["summary"], summary, "summary")
    cost = result["cost"]
    _keys(cost, {"selected_trials", "selected_stimulus_seconds", "analyzed_support_seconds",
                 "selected_support_elapsed_prefix_seconds", "full_support_record_seconds",
                 "query_ready_elapsed", "setup_seconds"}, "cost")
    _integer(cost["selected_trials"], "selected_trials", expected=5)
    for key, expected in {"selected_stimulus_seconds": 25, "analyzed_support_seconds": 10,
                          "selected_support_elapsed_prefix_seconds": max(trial_ends) / 250,
                          "full_support_record_seconds": 117917 / 250}.items():
        _near(cost[key], expected, errors, "cost_" + key)
    _same(cost["query_ready_elapsed"], "UNKNOWN", "query_ready_unknown")
    _same(cost["setup_seconds"], "UNKNOWN", "setup_unknown")
    return {"status": "PASS_SAVED_PROJECTION_AND_RECORDED_SCOPE", "attempts": 1,
            "fits": 0, "raw_reads": 0, "producer_imports": 0, "queries": 15,
            "predictions": 30, "class_scores": 150, "projection_scalars": 600,
            "reconstruction_near_ties": reconstruction_near_ties,
            "max_scalar_error": max(errors), "summary": summary, "cost": cost,
            "frequencies_hz": result["frequencies_hz"], "artifact_sha256": artifact_sha256,
            "code_sha256": actual_code_sha256, "limitations": LIMITATION}


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite_json_constant:" + value)


def _safe_path(root, relative):
    """Resolve only fixed root-relative names without following any symlink."""
    relative = Path(relative)
    _require(not relative.is_absolute() and ".." not in relative.parts, "root_relative_path")
    path = root
    for part in relative.parts:
        path = path / part
        _require(not path.is_symlink(), "symlink_path")
    return path


def _read_regular(root, relative, cap):
    """Bounded regular-file read; reject all symlink path components."""
    path = _safe_path(root, relative)
    metadata = path.stat()
    _require(stat.S_ISREG(metadata.st_mode) and metadata.st_size <= cap, "file_bound")
    with path.open("rb") as handle:
        body = handle.read(cap + 1)
    _require(len(body) <= cap and len(body) == metadata.st_size, "file_read_bound")
    return body


def load_bundle(root):
    """Read only fixed repository paths, once per artifact; never raw input pins."""
    root = Path(root)
    run = _safe_path(root, RUN_RELATIVE)
    _require(run.is_dir() and not run.is_symlink(), "run_directory")
    entries = list(run.iterdir())
    _require(all(not path.is_symlink() and path.is_file() and path.stat().st_size <= FILE_CAP
                 for path in entries), "run_regular_files")
    _require(sum(path.stat().st_size for path in entries) <= RUN_CAP, "total_run_byte_cap")
    objects, digests, sizes = {}, {}, {}
    for name in ("manifest.json", "started.json", "terminal.json", "worker_claim.json", "result.json"):
        path = run / name
        if name in {"worker_claim.json", "result.json"} and not path.exists():
            objects[name], digests[name] = None, None
            continue
        body = _read_regular(root, RUN_RELATIVE + "/" + name, FILE_CAP)
        objects[name] = json.loads(body, object_pairs_hook=_no_duplicates,
                                   parse_constant=_reject_constant)
        digests[name] = hashlib.sha256(body).hexdigest()
        sizes[name] = len(body)
    code_hashes = {name: hashlib.sha256(_read_regular(root, name, RUN_CAP)).hexdigest()
                   for name in sorted(CODE_PATHS)}
    return objects, digests, code_hashes, sizes


def _timeout(_signum, _frame):
    raise TimeoutError("audit_wall_60_seconds")


def main():
    began = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(60)
    try:
        _require(len(sys.argv) == 1, "no_path_or_retry_arguments")
        objects, hashes, code_hashes, sizes = load_bundle(ROOT)
        report = audit_objects(objects["manifest.json"], objects["started.json"],
                               objects["worker_claim.json"], objects["terminal.json"],
                               objects["result.json"], artifact_sha256=hashes,
                               actual_code_sha256=code_hashes)
        report["artifact_bytes"] = sizes
        report["elapsed_seconds"] = time.monotonic() - began
        print(json.dumps(report, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        print(json.dumps({"status": "AUDIT_FAILED_NO_RETRY", "error_type": type(error).__name__,
                          "reason": str(error), "raw_reads": 0, "fits": 0,
                          "elapsed_seconds": time.monotonic() - began}, allow_nan=False))
        return 1
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
