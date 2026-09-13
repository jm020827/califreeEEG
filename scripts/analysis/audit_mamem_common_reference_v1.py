"""Independent saved-projection/LOPO audit, with no producer imports or raw IO.

Only the frozen 19 repository inputs and 26 new JSON artifacts are read. Previous
A frequency values and previous B event sequences are not numerical inputs. This
does not independently reconstruct DIN segmentation or EEG -> projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import signal
import stat
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_RELATIVE = "docs/reports/mamem_common_reference_v1_run"
RUN = ROOT / RUN_RELATIVE
SUBJECTS = tuple(f"S{number:03}" for number in range(2, 12))
ARMS = ("NOMINAL", "COMMON")
NOMINAL = (6.66, 7.5, 8.57, 10., 12.)
SPEC = "configs/governance/mamem_reference_residual_v1_inputs.json"
PREVIOUS = "docs/reports/mamem_reference_residual_v1_run"
OLD_RESULT = PREVIOUS + "/result.json"
OLD_TERMINAL = PREVIOUS + "/terminal.json"
OLD_AUDIT = "docs/reports/mamem_reference_residual_v1_audit.json"
PROBE = "src/cfeg/mamem_reference_probe_v1.py"
KNOWN_SHA = {
    SPEC: "99a3adfae3765457bb23bf0305519775268007546742b6bff512d33fb3d7135b",
    OLD_RESULT: "50cef4d3482975efe69d4fc5c145a95d2e4ab3822db30203576ce63dad906389",
    OLD_TERMINAL: "2b2eeb8e1e8b6c61447d6fcc932515962da081cc7b54ded917be4c5df89a0ff4",
    OLD_AUDIT: "f54841c75840ae0f2b0b0d911b34c0489501eb846f315a7cff4344a90ff2c1c4",
    PROBE: "a147f8ac18016d521832f33dc9f45f956ac0f8163aa56df5a5da9f38f32eb7e3",
}
RECORD_PATHS = {subject: PREVIOUS + "/record_" + subject + "b.json" for subject in SUBJECTS}
PINS = frozenset({
    SPEC, OLD_RESULT, OLD_TERMINAL, OLD_AUDIT, PROBE,
    "docs/mamem_common_reference_v1_contract.md", "src/cfeg/mamem_common_reference_v1.py",
    "scripts/analysis/run_mamem_common_reference_v1.py",
    "scripts/analysis/audit_mamem_common_reference_v1.py", *RECORD_PATHS.values(),
})
SCOPE = {
    "mat_byte_hashes": 10, "mat_headers": 10, "mat_eeg_loads": 10,
    "din_decodes": 0, "rate_decodes": 0, "eeg_numeric_windows": 150,
    "eeg_numeric_channels": 1, "predictions": 300, "class_scores": 1500,
    "projection_scalars": 6000, "fits": 0, "target_support_trials": 0, "held60_openings": 0,
}
COST = {
    "target_labeled_support_trials": 0, "target_support_stimulus_seconds": 0,
    "source_subjects_per_fold": 9, "source_event_windows_per_fold": 135,
    "source_collection": "PREVIOUSLY_ACQUIRED_NOT_TARGET_CALIBRATION",
    "query_ready_elapsed": "UNKNOWN", "setup_seconds": "UNKNOWN",
}
JSON_NAMES = frozenset({
    "manifest.json", "started.json", "worker_claim.json", "terminal.json", "result.json",
    "scores_frozen.json", *(prefix + subject + ".json" for subject in SUBJECTS
                             for prefix in ("attempt_", "scores_")),
})
FILE_CAP = 1024**2
RUN_CAP = 8 * FILE_CAP
OUTPUT_CAP = 64 * 1024
TOL, BOUND_TOL = 1e-10, 1e-12
DEADLINE = datetime(2026, 9, 13, 16, tzinfo=timezone.utc)
LIMITATIONS = (
    "Saved B frequencies, projection arithmetic and recorded scope only; no raw MAT/EEG/DIN "
    "reads, original DIN segmentation or EEG-to-projection reconstruction, independent label "
    "truth, physical clock validation or OS-I/O tracing. Reused development subjects and shared "
    "source folds; not independent-cohort efficacy, learned-M benefit or calibration savings."
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def keys(value, expected, name):
    require(type(value) is dict and set(value) == set(expected), name + "_keys")


def sequence(value, length, name):
    require(type(value) is list and len(value) == length, name + "_shape")
    return value


def integer(value, name, lower=0, upper=None):
    require(type(value) is int and value >= lower, name + "_integer")
    require(upper is None or value <= upper, name + "_bound")
    return value


def number(value, name):
    require(type(value) in (int, float), name + "_number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(name + "_finite") from error
    require(math.isfinite(result), name + "_finite")
    return result


def compare(actual, expected, name, tolerance=TOL):
    """Strict keys/bools/integers, absolute tolerance only for float quantities."""
    if type(expected) is dict:
        keys(actual, expected, name)
        return max((compare(actual[key], value, name + "." + key, tolerance)
                    for key, value in expected.items()), default=0.)
    if type(expected) is list:
        sequence(actual, len(expected), name)
        return max((compare(a, b, name + f"[{index}]", tolerance)
                    for index, (a, b) in enumerate(zip(actual, expected))), default=0.)
    if type(expected) is float:
        error = abs(number(actual, name) - expected)
        require(error <= tolerance, name + "_arithmetic")
        return error
    require(type(actual) is type(expected) and actual == expected, name + "_value")
    return 0.


def digest(value, name):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            name + "_sha256")
    return value


def stamp(value, name="timestamp"):
    require(type(value) is str, name + "_utc")
    matched = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?(?:Z|\+00:00)", value,
    )
    require(matched is not None, name + "_utc")
    normalized = matched.group(1)
    if matched.group(2) is not None:
        normalized += "." + matched.group(2).ljust(6, "0")
    return datetime.fromisoformat(normalized + "+00:00")


def common_frequencies(previous_frequencies):
    """Select B only; each bank excludes the participant's entire B array.

    A values are deliberately neither converted nor validated numerically.
    Scalar summation provides independence from the producer's array reductions.
    """
    selected = []
    for participant in sequence(previous_frequencies, 10, "previous_participants"):
        b_values = sequence(participant, 2, "previous_runs")[1]
        classes = []
        for row in sequence(b_values, 5, "previous_B_classes"):
            classes.append([number(value, "previous_B_frequency")
                            for value in sequence(row, 3, "previous_B_repeats")])
        require(all(0 < value < 62.5 for row in classes for value in row)
                and all(classes[c][j] < classes[c+1][j] for c in range(4) for j in range(3)),
                "previous_B_frequency_range_order")
        selected.append(classes)
    result = {}
    for p, subject in enumerate(SUBJECTS):
        result[subject] = [math.exp(math.fsum(math.log(selected[o][c][j])
                                            for o in range(10) if o != p
                                            for j in range(3)) / 27)
                           for c in range(5)]
    return result


def input_rows(specification):
    keys(specification, {"schema", "inputs"}, "input_spec")
    compare(specification["schema"], "cfeg.mamem-reference-residual-v1.inputs", "input_schema")
    rows = sequence(specification["inputs"], 20, "input_rows")
    require([(row["subject"], row["run"]) for row in rows]
            == [(subject, run) for subject in SUBJECTS for run in ("a", "b")], "input_roles")
    selected = {}
    for row in rows[1::2]:
        keys(row, {"subject", "run", "archive", "member", "bytes", "crc32", "path", "sha256",
                   "receipt_sha256"}, "input_B_record")
        name = row["subject"] + "b.mat"
        compare(row["path"], "/home/whwovy/data/mamem_recorded_event_source_v2/" + name,
                "input_B_path")
        part = 1 if int(row["subject"][1:]) <= 6 else 2
        compare(row["archive"], f"/home/whwovy/data/mamem_i_v1_20260913/EEG-SSVEP-Part{part}.rar",
                "input_archive")
        compare(row["member"], "EEG-SSVEP-Part1/" + name if part == 1 else name, "input_member")
        integer(row["bytes"], "input_bytes", 1, 512 * FILE_CAP)
        require(type(row["crc32"]) is str and re.fullmatch(r"[0-9a-f]{8}", row["crc32"]), "input_crc")
        digest(row["sha256"], "input_mat")
        digest(row["receipt_sha256"], "input_receipt")
        selected[row["subject"]] = row
    return selected


def validate_record(record, row):
    keys(record, {"subject", "run", "mat_sha256", "receipt_sha256", "total_samples", "records",
                  "completed_utc", "loaded_variables"}, "previous_record")
    for name, expected in {"subject": row["subject"], "run": "b", "mat_sha256": row["sha256"],
                            "receipt_sha256": row["receipt_sha256"],
                            "loaded_variables": ["DIN_1", "samplingRate"]}.items():
        compare(record[name], expected, "previous_record_" + name)
    total = integer(record["total_samples"], "total_samples", 1, 200000)
    windows, labels, counts = [], [], [0] * 5
    previous_end = 0
    for index, trial in enumerate(sequence(record["records"], 15, "previous_main_records")):
        keys(trial, {"group_index", "label", "start0", "end0", "trial_end0", "repeat",
                     "event_samples", "frequency_hz"}, "previous_trial")
        compare(trial["group_index"], index + 8, "main_group")
        label = integer(trial["label"], "previous_label", 0, 4)
        compare(trial["repeat"], counts[label], "previous_repeat")
        counts[label] += 1
        start = integer(trial["start0"], "window_start", 250, total)
        end = integer(trial["end0"], "window_end", 1, total)
        trial_end = integer(trial["trial_end0"], "trial_end", 1, total)
        require(start >= previous_end and end == start + 500 and trial_end == start + 1000,
                "previous_window_bounds")
        previous_end = end
        windows.append({"group_index": index + 8, "start0": start, "end0": end})
        labels.append(label)
        # Do not inspect event_samples or frequency_hz: only the old byte hash
        # and prior audit bind their provenance in this EEG-projection audit.
    compare(counts, [3] * 5, "class_coverage")
    require(all(labels[i] == labels[i+1] == labels[i+2] for i in range(0, 15, 3)), "class_blocks")
    return windows, labels


def score_row(row):
    keys(row, {"projection", "scores", "prediction"}, "score_arm")
    projections = sequence(row["projection"], 5, "projection_classes")
    scores = [number(value, "score") for value in sequence(row["scores"], 5, "class_scores")]
    reconstructed, errors = [], []
    for vector, saved in zip(projections, scores):
        vector = [number(value, "projection") for value in sequence(vector, 4, "projection_rank")]
        require(all(abs(value) <= 1 + BOUND_TOL for value in vector), "projection_bounds")
        energy = math.fsum(value * value for value in vector)
        require(energy <= 1 + BOUND_TOL and -BOUND_TOL <= saved <= 1 + BOUND_TOL, "score_bounds")
        expected = min(1., max(0., energy))
        errors.append(compare(saved, expected, "projection_score"))
        reconstructed.append(expected)
    predicted = max(range(5), key=lambda index: scores[index])
    compare(row["prediction"], predicted, "saved_argmax")
    require(max(reconstructed) - reconstructed[predicted] <= 2 * TOL,
            "reconstructed_argmax_consistency")
    ordered = sorted(reconstructed)
    near_tie = int(ordered[-1] - ordered[-2] <= 2 * TOL)
    return predicted, max(errors), near_tie


def participant_summary(correct, labels):
    result = {}
    for arm in ARMS:
        per_class = [sum(int(hit) for hit, label in zip(correct[arm], labels) if label == c)
                     for c in range(5)]
        total = sum(per_class)
        result[arm] = {"correct": total, "total": 15, "per_class_correct": per_class,
                       "ready": total >= 12 and min(per_class) >= 1}
    return result


def overall_summary(people, paired):
    result, changes = {}, {"common_better": 0, "nominal_better": 0, "tied": 0}
    for arm in ARMS:
        correct = sum(person[arm]["correct"] for person in people)
        ready_count = sum(int(person[arm]["ready"]) for person in people)
        result[arm] = {"correct": correct, "total": 150, "accuracy": correct / 150,
                       "per_class_correct": [sum(person[arm]["per_class_correct"][c]
                                                 for person in people) for c in range(5)],
                       "subjects_ready": ready_count, "ready": correct >= 120 and ready_count >= 7}
    for person in people:
        difference = person["COMMON"]["correct"] - person["NOMINAL"]["correct"]
        changes["common_better" if difference > 0 else "nominal_better" if difference < 0 else "tied"] += 1
    n_ready, c_ready = result["NOMINAL"]["ready"], result["COMMON"]["ready"]
    decision = ("BOTH_ZERO_SUPPORT_READY_DEV" if n_ready and c_ready
                else "COMMON_ONLY_READY_DEV" if c_ready else "NOMINAL_ONLY_READY_DEV" if n_ready
                else "NEITHER_BASELINE_READY_STOP")
    result.update(paired=paired, subject_changes=changes,
                  common_advantage_pp=100 * (result["COMMON"]["correct"] - result["NOMINAL"]["correct"]) / 150,
                  decision=decision)
    return result


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite_json_constant:" + value)


def safe_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "root_relative_path")
    path = root
    for part in relative.parts:
        path /= part
        require(not path.is_symlink(), "symlink_path")
    return path


class Reader:
    """Single bounded read per whitelisted path, including source-code pin bytes."""

    def __init__(self, root, run):
        self.root = Path(root).absolute()
        self.run = Path(run).absolute()
        require(self.root.resolve() == self.root, "canonical_root")
        require(self.run == self.root / RUN_RELATIVE, "fixed_run_path")
        safe_path(self.root, RUN_RELATIVE)
        self.allowed = set(PINS) | {RUN_RELATIVE + "/" + name for name in JSON_NAMES}
        self.bodies, self.hashes = {}, {}

    def read(self, relative):
        require(relative in self.allowed, "read_allowlist")
        if relative not in self.bodies:
            path = safe_path(self.root, relative)
            metadata = path.stat()
            require(stat.S_ISREG(metadata.st_mode) and metadata.st_size <= FILE_CAP, "bounded_file")
            with path.open("rb") as stream:
                body = stream.read(FILE_CAP + 1)
            require(len(body) == metadata.st_size and len(body) <= FILE_CAP, "bounded_read")
            self.bodies[relative] = body
            self.hashes[relative] = hashlib.sha256(body).hexdigest()
        return self.bodies[relative]

    def json(self, relative):
        return json.loads(self.read(relative), object_pairs_hook=unique, parse_constant=reject_constant)

    def new(self, name):
        return self.json(RUN_RELATIVE + "/" + name)

    def sha(self, relative):
        self.read(relative)
        return self.hashes[relative]

    def new_sha(self, name):
        return self.sha(RUN_RELATIVE + "/" + name)


def audit(root=ROOT, run=RUN):
    began = time.monotonic()
    reader = Reader(root, run)
    files = list(reader.run.iterdir())
    require({path.name for path in files} == set(JSON_NAMES) | {"stdout.txt", "stderr.txt"},
            "exact_run_files")
    require(all(path.is_file() and not path.is_symlink() and path.stat().st_size <= FILE_CAP
                for path in files) and sum(path.stat().st_size for path in files) <= RUN_CAP,
            "artifact_budget")
    manifest = reader.new("manifest.json")
    keys(manifest, {"schema", "created_utc", "deadline_utc", "attempts", "scope", "pins",
                    "frequencies_hz"}, "manifest")
    compare(manifest["schema"], "cfeg.mamem-common-reference-v1.manifest", "manifest_schema")
    compare(manifest["attempts"], 1, "manifest_attempts")
    compare(manifest["scope"], SCOPE, "manifest_scope")
    require(stamp(manifest["deadline_utc"]) == DEADLINE, "frozen_deadline")
    keys(manifest["pins"], PINS, "nineteen_pins")
    keys(KNOWN_SHA, {SPEC, OLD_RESULT, OLD_TERMINAL, OLD_AUDIT, PROBE}, "five_known_pins")
    for path in sorted(PINS):
        require(reader.sha(path) == digest(manifest["pins"][path], "pin"), "pin_hash_binding")
    for path, expected in KNOWN_SHA.items():
        compare(manifest["pins"][path], expected, "unchanged_known_pin")

    inputs = input_rows(reader.json(SPEC))
    previous = reader.json(OLD_RESULT)
    compare(previous["schema"], "cfeg.mamem-reference-residual-v1.result", "previous_schema")
    keys(previous["record_sha256"], {s+r for s in SUBJECTS for r in ("a", "b")}, "previous_records")
    old_terminal, old_audit = reader.json(OLD_TERMINAL), reader.json(OLD_AUDIT)
    for field, expected in {"status": "COMPLETE", "attempts": 1, "fits": 0,
                            "result_sha256": reader.sha(OLD_RESULT)}.items():
        compare(old_terminal[field], expected, "old_terminal_" + field)
    for field, expected in {"status": "PASS_SAVED_EVENTS_AND_RECORDED_SCOPE", "raw_reads": 0,
                            "fits": 0, "result_sha256": reader.sha(OLD_RESULT)}.items():
        compare(old_audit[field], expected, "old_audit_" + field)
    require(stamp(previous["completed_utc"]) <= stamp(old_terminal["ended_utc"])
            <= stamp(old_audit["completed_utc"]) <= stamp(manifest["created_utc"]), "prior_before_bank")
    common = common_frequencies(previous["frequencies_hz"])
    max_error = compare(manifest["frequencies_hz"], {"NOMINAL": list(NOMINAL), "COMMON": common},
                        "source_only_banks")

    started, claim, terminal, seal, result = [reader.new(name + ".json") for name in
                                            ("started", "worker_claim", "terminal", "scores_frozen", "result")]
    keys(started, {"status", "attempt", "parent_pid", "started_utc", "manifest_sha256"}, "started")
    keys(claim, {"attempt", "pid", "parent_pid", "started_utc", "manifest_sha256"}, "claim")
    keys(terminal, {"status", "attempts", "fits", "started_utc", "ended_utc", "manifest_sha256",
                    "started_sha256", "worker_claim_sha256", "result_sha256"}, "terminal")
    compare(started["status"], "STARTED", "started_status")
    compare(terminal["status"], "COMPLETE", "complete_terminal_required")
    compare(terminal["fits"], 0, "terminal_fits")
    for data, field in ((started, "attempt"), (claim, "attempt"), (terminal, "attempts")):
        compare(data[field], 1, "single_attempt")
        compare(data["manifest_sha256"], reader.new_sha("manifest.json"), "manifest_binding")
    parent = integer(started["parent_pid"], "parent_pid", 1)
    compare(claim["parent_pid"], parent, "worker_parent")
    require(integer(claim["pid"], "worker_pid", 1) != parent, "distinct_worker_pid")
    compare(terminal["started_utc"], started["started_utc"], "terminal_start_time")
    for field, name in (("started_sha256", "started.json"), ("worker_claim_sha256", "worker_claim.json"),
                         ("result_sha256", "result.json")):
        compare(terminal[field], reader.new_sha(name), "terminal_" + field)
    keys(seal, {"created_utc", "predictions", "scores_sha256"}, "scores_seal")
    compare(seal["predictions"], 300, "seal_predictions")
    keys(seal["scores_sha256"], SUBJECTS, "sealed_subjects")
    keys(result, {"schema", "completed_utc", "scope", "scores_frozen_sha256", "participants",
                  "summary", "cost"}, "result")
    compare(result["schema"], "cfeg.mamem-common-reference-v1.result", "result_schema")
    compare(result["scope"], SCOPE, "result_scope")
    compare(result["scores_frozen_sha256"], reader.new_sha("scores_frozen.json"), "result_seal_binding")
    times = [stamp(manifest["created_utc"]), stamp(started["started_utc"]), stamp(claim["started_utc"]),
             stamp(seal["created_utc"]), stamp(result["completed_utc"]), stamp(terminal["ended_utc"])]
    require(all(a <= b for a, b in pairwise(times)) and times[-1] <= DEADLINE
            and (times[-1] - times[1]).total_seconds() <= 180, "lifetime_times")

    people = sequence(result["participants"], 10, "result_participants")
    paired = {"both_correct": 0, "nominal_only": 0, "common_only": 0, "both_wrong": 0}
    people_summaries, differences, near_ties = [], {}, 0
    last_completed = times[2]
    for subject, person in zip(SUBJECTS, people):
        record_path = RECORD_PATHS[subject]
        compare(previous["record_sha256"][subject + "b"], reader.sha(record_path), "old_record_binding")
        record = reader.json(record_path)
        require(stamp(record["completed_utc"]) <= stamp(old_terminal["ended_utc"]), "previous_record_time")
        windows, labels = validate_record(record, inputs[subject])
        attempt = reader.new("attempt_" + subject + ".json")
        scores = reader.new("scores_" + subject + ".json")
        keys(attempt, {"subject", "started_utc", "mat_sha256", "loaded_variables"}, "file_attempt")
        keys(scores, {"subject", "mat_sha256", "source_record_sha256", "completed_utc", "queries"},
             "saved_scores")
        for data in (attempt, scores):
            compare(data["subject"], subject, "subject_order")
            compare(data["mat_sha256"], inputs[subject]["sha256"], "mat_pin_binding")
        compare(attempt["loaded_variables"], ["eeg"], "eeg_only_decode")
        compare(scores["source_record_sha256"], reader.sha(record_path), "score_window_source")
        compare(seal["scores_sha256"][subject], reader.new_sha("scores_" + subject + ".json"),
                "score_seal_binding")
        require(last_completed <= stamp(attempt["started_utc"]) <= stamp(scores["completed_utc"])
                <= times[3], "score_attempt_seal_order")
        last_completed = stamp(scores["completed_utc"])
        keys(person, {"subject", "queries", "summary"}, "result_person")
        compare(person["subject"], subject, "evaluated_subject_order")
        saved_queries = sequence(scores["queries"], 15, "saved_queries")
        evaluated_queries = sequence(person["queries"], 15, "evaluated_queries")
        correct = {arm: [] for arm in ARMS}
        for window, label, query, evaluated in zip(windows, labels, saved_queries, evaluated_queries):
            keys(query, {*window, "arms"}, "label_free_query")
            compare({key: query[key] for key in window}, window, "saved_window")
            keys(query["arms"], ARMS, "two_score_arms")
            keys(evaluated, {*window, "arms", "label"}, "evaluated_query")
            compare({key: evaluated[key] for key in window}, window, "evaluated_window")
            compare(evaluated["label"], label, "frozen_evaluation_label")
            keys(evaluated["arms"], ARMS, "two_evaluated_arms")
            for arm in ARMS:
                prediction, error, near_tie = score_row(query["arms"][arm])
                max_error = max(max_error, error)
                near_ties += near_tie
                hit = prediction == label
                compare(evaluated["arms"][arm], dict(query["arms"][arm], correct=hit),
                        "evaluation_preserves_scores", tolerance=0.)
                correct[arm].append(hit)
            n_hit, c_hit = correct["NOMINAL"][-1], correct["COMMON"][-1]
            pair = ("both_correct" if n_hit and c_hit else "nominal_only" if n_hit
                    else "common_only" if c_hit else "both_wrong")
            paired[pair] += 1
        summary = participant_summary(correct, labels)
        compare(person["summary"], summary, "person_summary")
        people_summaries.append(summary)
        differences[subject] = summary["COMMON"]["correct"] - summary["NOMINAL"]["correct"]
    expected_summary = overall_summary(people_summaries, paired)
    max_error = max(max_error, compare(result["summary"], expected_summary, "overall_summary"))
    compare(result["cost"], COST, "zero_target_support_cost")
    require(len(reader.bodies) == 45, "exact_45_distinct_reads")
    return {
        "status": "PASS_SAVED_PROJECTION_AND_SOURCE_ONLY_COMMON", "actual_audit_attempts": 1,
        "participants": 10, "eeg_windows": 150, "predictions": 300, "class_scores": 1500,
        "projection_scalars": 6000, "raw_reads": 0, "producer_imports": 0, "fits": 0,
        "max_scalar_error": max_error, "reconstruction_near_ties": near_ties,
        "source_other_subjects_per_bank": 9, "source_repeats_per_class": 27,
        "participant_correct_differences": differences, "summary": expected_summary, "cost": COST,
        "source_frequencies_hz": common, "result_sha256": reader.new_sha("result.json"),
        "verified_input_sha256": dict(sorted(reader.hashes.items())),
        "distinct_files_read": len(reader.bodies), "bytes_read": sum(map(len, reader.bodies.values())),
        "limitations": LIMITATIONS, "elapsed_seconds": time.monotonic() - began,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def timeout_handler(_signum, _frame):
    raise TimeoutError("audit_wall_60_seconds")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    require(output.parent.resolve() == output.parent and output.parent.is_dir(), "output_parent")
    require(output != RUN and RUN not in output.parents, "output_outside_run")
    # Exclusive output reservation precedes all audit input reads. Never overwrite
    # an earlier success, failure, or incomplete audit attempt.
    with output.open("xb") as stream:
        began = time.monotonic()
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
            report = audit()
        except Exception as error:  # noqa: BLE001 -- retain failure without retry
            report = {"status": "AUDIT_FAIL_NO_RETRY", "actual_audit_attempts": 1,
                      "error_type": type(error).__name__, "reason": str(error)[:4096],
                      "raw_reads": 0, "fits": 0, "elapsed_seconds": time.monotonic() - began,
                      "limitations": LIMITATIONS}
        finally:
            signal.alarm(0)
        raw = json.dumps(report, allow_nan=False, sort_keys=True).encode() + b"\n"
        if len(raw) > OUTPUT_CAP:
            report = {"status": "AUDIT_FAIL_NO_RETRY", "reason": "audit_output_cap",
                      "raw_reads": 0, "fits": 0}
            raw = json.dumps(report).encode() + b"\n"
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
        print(raw.decode().strip())
    return 0 if report["status"] == "PASS_SAVED_PROJECTION_AND_SOURCE_ONLY_COMMON" else 1


if __name__ == "__main__":
    raise SystemExit(main())
