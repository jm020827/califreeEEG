"""Generated JSON-only audit tests; no producer import, raw data, or fitted model.

Integration API: ``fixture = build_fixture(tmp_root)`` exposes ``root``, ``run``,
``documents`` (root-relative JSON objects), ``known_sha``, ``timestamp(seconds)``,
``refresh()`` (rewrite synthetic bytes/hash bindings), and ``rebuild_result()``.
Patch the auditor's KNOWN_SHA to fixture.known_sha, then call audit(root, run).
The producer can reuse these generated cached inputs and a mock EEG-only reader.
"""

import copy
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

AUDITOR_PATH = Path(__file__).resolve().parents[1] / "scripts/analysis/audit_mamem_common_reference_v1.py"
SPECIFICATION = importlib.util.spec_from_file_location("independent_common_reference_audit", AUDITOR_PATH)
audit = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(audit)


def encoded(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def row_for(predicted):
    projection = [[.9 if c == predicted else .1, 0., 0., 0.] for c in range(5)]
    return {"projection": projection,
            "scores": [sum(value * value for value in row) for row in projection],
            "prediction": predicted}


def expected_evaluation(scored, records):
    """Test-fixture arithmetic independent of all auditor/producer helpers."""
    participants, correct_pairs = [], {(a, b): 0 for a in (False, True) for b in (False, True)}
    for subject in audit.SUBJECTS:
        queries = []
        totals = {arm: [0] * 5 for arm in audit.ARMS}
        for query, trial in zip(scored[subject]["queries"], records[subject]["records"]):
            label = trial["label"]
            result = copy.deepcopy(query)
            result["label"] = label
            for arm in audit.ARMS:
                hit = result["arms"][arm]["prediction"] == label
                result["arms"][arm]["correct"] = hit
                totals[arm][label] += int(hit)
            correct_pairs[tuple(result["arms"][arm]["correct"] for arm in audit.ARMS)] += 1
            queries.append(result)
        summary = {arm: {"correct": sum(counts), "total": 15, "per_class_correct": counts,
                         "ready": sum(counts) >= 12 and min(counts) >= 1}
                   for arm, counts in totals.items()}
        participants.append({"subject": subject, "queries": queries, "summary": summary})
    summary = {}
    for arm in audit.ARMS:
        per_class = [sum(person["summary"][arm]["per_class_correct"][c] for person in participants)
                     for c in range(5)]
        ready = sum(person["summary"][arm]["ready"] for person in participants)
        summary[arm] = {"correct": sum(per_class), "total": 150, "accuracy": sum(per_class) / 150,
                        "per_class_correct": per_class, "subjects_ready": ready,
                        "ready": sum(per_class) >= 120 and ready >= 7}
    differences = [person["summary"]["COMMON"]["correct"] - person["summary"]["NOMINAL"]["correct"]
                   for person in participants]
    n_ready, c_ready = summary["NOMINAL"]["ready"], summary["COMMON"]["ready"]
    decisions = {(True, True): "BOTH_ZERO_SUPPORT_READY_DEV", (False, True): "COMMON_ONLY_READY_DEV",
                 (True, False): "NOMINAL_ONLY_READY_DEV", (False, False): "NEITHER_BASELINE_READY_STOP"}
    summary.update(
        paired={"both_correct": correct_pairs[True, True], "nominal_only": correct_pairs[True, False],
                "common_only": correct_pairs[False, True], "both_wrong": correct_pairs[False, False]},
        subject_changes={"common_better": sum(d > 0 for d in differences),
                         "nominal_better": sum(d < 0 for d in differences),
                         "tied": sum(d == 0 for d in differences)},
        common_advantage_pp=100 * sum(differences) / 150, decision=decisions[n_ready, c_ready],
    )
    return participants, summary


class SyntheticFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.run = self.root / audit.RUN_RELATIVE
        self.documents = {}
        self.code_bodies = {name: ("synthetic source pin " + name).encode() for name in audit.PINS
                            if not name.endswith(".json")}
        self.known_sha = {}

    @staticmethod
    def timestamp(seconds=0):
        return (datetime(2026, 9, 13, 15, 11, tzinfo=timezone.utc)
                + timedelta(seconds=seconds)).isoformat()

    def new(self, name):
        return self.documents[audit.RUN_RELATIVE + "/" + name]

    def write_document(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(self.documents[relative]))

    def rebuild_result(self):
        records = {s: self.documents[audit.RECORD_PATHS[s]] for s in audit.SUBJECTS}
        scored = {s: self.new("scores_" + s + ".json") for s in audit.SUBJECTS}
        participants, summary = expected_evaluation(scored, records)
        self.new("result.json").update(participants=participants, summary=summary)

    def refresh(self):
        """Rebind generated byte hashes, without repairing semantic poisons."""
        previous = self.documents[audit.OLD_RESULT]
        for subject, path in audit.RECORD_PATHS.items():
            previous["record_sha256"][subject + "b"] = digest(self.documents[path])
            self.new("scores_" + subject + ".json")["source_record_sha256"] = digest(self.documents[path])
        self.documents[audit.OLD_TERMINAL]["result_sha256"] = digest(previous)
        self.documents[audit.OLD_AUDIT]["result_sha256"] = digest(previous)
        pins = {name: digest(self.documents[name]) if name in self.documents
                else hashlib.sha256(self.code_bodies[name]).hexdigest() for name in audit.PINS}
        self.new("manifest.json")["pins"] = pins
        manifest_hash = digest(self.new("manifest.json"))
        for name in ("started.json", "worker_claim.json", "terminal.json"):
            self.new(name)["manifest_sha256"] = manifest_hash
        self.new("scores_frozen.json")["scores_sha256"] = {
            subject: digest(self.new("scores_" + subject + ".json")) for subject in audit.SUBJECTS}
        self.new("result.json")["scores_frozen_sha256"] = digest(self.new("scores_frozen.json"))
        self.new("terminal.json").update(started_sha256=digest(self.new("started.json")),
                                         worker_claim_sha256=digest(self.new("worker_claim.json")),
                                         result_sha256=digest(self.new("result.json")))
        known = {name: pins[name] for name in
                 (audit.SPEC, audit.OLD_RESULT, audit.OLD_TERMINAL, audit.OLD_AUDIT, audit.PROBE)}
        self.known_sha.clear()
        self.known_sha.update(known)
        for name in self.documents:
            self.write_document(name)
        for name, body in self.code_bodies.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        self.run.mkdir(parents=True, exist_ok=True)
        for name in ("stdout.txt", "stderr.txt"):
            (self.run / name).write_bytes(b"")
        return self


def build_fixture(root):
    fixture = SyntheticFixture(root)
    docs = fixture.documents
    inputs = []
    for index, subject in enumerate(audit.SUBJECTS):
        for run in ("a", "b"):
            part = 1 if int(subject[1:]) <= 6 else 2
            name = subject + run + ".mat"
            inputs.append({"subject": subject, "run": run,
                           "archive": f"/home/whwovy/data/mamem_i_v1_20260913/EEG-SSVEP-Part{part}.rar",
                           "member": "EEG-SSVEP-Part1/" + name if part == 1 else name,
                           "bytes": 100000 + index, "crc32": "12345678",
                           "path": "/home/whwovy/data/mamem_recorded_event_source_v2/" + name,
                           "sha256": hashlib.sha256(name.encode()).hexdigest(),
                           "receipt_sha256": hashlib.sha256((name + "receipt").encode()).hexdigest()})
    docs[audit.SPEC] = {"schema": "cfeg.mamem-reference-residual-v1.inputs", "inputs": inputs}
    frequencies = []
    for p in range(10):
        b = [[f * math.exp(.004 * (p - 4.5) + .0002 * (j - 1)) for j in range(3)]
             for f in audit.NOMINAL]
        # A is deliberately nonnumerical yet valid JSON. Neither auditor nor
        # the producer's B-only selection is entitled to convert it to a tensor.
        frequencies.append([{"A_IS_NOT_A_NUMERICAL_INPUT": "poison"}, b])
    docs[audit.OLD_RESULT] = {
        "schema": "cfeg.mamem-reference-residual-v1.result", "completed_utc": fixture.timestamp(-1000),
        "record_sha256": {s + r: "1" * 64 for s in audit.SUBJECTS for r in ("a", "b")},
        "frequencies_hz": frequencies,
    }
    docs[audit.OLD_TERMINAL] = {"status": "COMPLETE", "attempts": 1, "fits": 0,
                               "ended_utc": fixture.timestamp(-999), "result_sha256": "1" * 64}
    docs[audit.OLD_AUDIT] = {"status": "PASS_SAVED_EVENTS_AND_RECORDED_SCOPE", "raw_reads": 0,
                            "fits": 0, "completed_utc": fixture.timestamp(-998), "result_sha256": "1" * 64}
    for p, subject in enumerate(audit.SUBJECTS):
        row = inputs[2 * p + 1]
        trials = []
        # A nonascending class schedule exercises class-vs-group indexing.
        class_order = (3, 0, 4, 1, 2)
        for index in range(15):
            start, label = 1000 + index * 1500, class_order[index // 3]
            trials.append({"group_index": index + 8, "start0": start, "end0": start + 500,
                           "trial_end0": start + 1000, "label": label, "repeat": index % 3,
                           "event_samples": "NOT_RESEGMENTED_IN_THIS_AUDIT",
                           "frequency_hz": "NOT_A_TARGET_BANK_INPUT"})
        docs[audit.RECORD_PATHS[subject]] = {
            "subject": subject, "run": "b", "mat_sha256": row["sha256"],
            "receipt_sha256": row["receipt_sha256"], "total_samples": 24000,
            "records": trials, "completed_utc": fixture.timestamp(-1100 + p),
            "loaded_variables": ["DIN_1", "samplingRate"],
        }
        docs[audit.RUN_RELATIVE + "/attempt_" + subject + ".json"] = {
            "subject": subject, "started_utc": fixture.timestamp(1 + 2 * p),
            "mat_sha256": row["sha256"], "loaded_variables": ["eeg"],
        }
        docs[audit.RUN_RELATIVE + "/scores_" + subject + ".json"] = {
            "subject": subject, "mat_sha256": row["sha256"], "source_record_sha256": "1" * 64,
            "completed_utc": fixture.timestamp(2 + 2 * p),
            "queries": [{**{k: trial[k] for k in ("group_index", "start0", "end0")},
                         "arms": {arm: row_for(trial["label"]) for arm in audit.ARMS}} for trial in trials],
        }
    common = {subject: [math.exp(sum(math.log(frequencies[o][1][c][j]) for o in range(10) if o != p
                                    for j in range(3)) / 27) for c in range(5)]
              for p, subject in enumerate(audit.SUBJECTS)}
    for name, value in {
        "manifest.json": {"schema": "cfeg.mamem-common-reference-v1.manifest",
                          "created_utc": fixture.timestamp(-60), "deadline_utc": "2026-09-13T16:00:00Z",
                          "attempts": 1, "scope": dict(audit.SCOPE), "pins": {},
                          "frequencies_hz": {"NOMINAL": list(audit.NOMINAL), "COMMON": common}},
        "started.json": {"status": "STARTED", "attempt": 1, "parent_pid": 101,
                         "started_utc": fixture.timestamp(), "manifest_sha256": "1" * 64},
        "worker_claim.json": {"attempt": 1, "pid": 102, "parent_pid": 101,
                              "started_utc": fixture.timestamp(.1), "manifest_sha256": "1" * 64},
        "scores_frozen.json": {"created_utc": fixture.timestamp(21), "predictions": 300,
                               "scores_sha256": {}},
        "result.json": {"schema": "cfeg.mamem-common-reference-v1.result",
                        "completed_utc": fixture.timestamp(22), "scope": dict(audit.SCOPE),
                        "scores_frozen_sha256": "1" * 64, "participants": [], "summary": {},
                        "cost": dict(audit.COST)},
        "terminal.json": {"status": "COMPLETE", "attempts": 1, "fits": 0,
                          "started_utc": fixture.timestamp(), "ended_utc": fixture.timestamp(23),
                          "manifest_sha256": "1" * 64, "started_sha256": "1" * 64,
                          "worker_claim_sha256": "1" * 64, "result_sha256": "1" * 64},
    }.items():
        docs[audit.RUN_RELATIVE + "/" + name] = value
    fixture.rebuild_result()
    return fixture.refresh()


@pytest.fixture
def generated(tmp_path, monkeypatch):
    fixture = build_fixture(tmp_path)
    monkeypatch.setattr(audit, "KNOWN_SHA", fixture.known_sha)
    return fixture


def test_complete_audit_and_exact_read_boundary(generated, monkeypatch):
    opened = []
    original = Path.open

    def guarded(path, mode="r", *args, **kwargs):
        if mode == "rb":
            opened.append(str(path.relative_to(generated.root)))
            assert not str(path).endswith((".mat", ".npz"))
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    report = audit.audit(generated.root, generated.run)
    assert report["status"] == "PASS_SAVED_PROJECTION_AND_SOURCE_ONLY_COMMON"
    assert report["max_scalar_error"] < 1e-12
    assert report["reconstruction_near_ties"] == 0
    assert report["distinct_files_read"] == len(opened) == len(set(opened)) == 45
    assert report["projection_scalars"] == 6000 and report["class_scores"] == 1500
    assert report["predictions"] == 300
    assert report["summary"]["decision"] == "BOTH_ZERO_SUPPORT_READY_DEV"
    assert report["raw_reads"] == report["fits"] == report["producer_imports"] == 0
    assert "not independent-cohort efficacy" in report["limitations"]
    assert len(encoded(report)) < audit.OUTPUT_CAP


def test_A_is_not_a_numeric_input_and_own_B_excluded(generated):
    frequencies = copy.deepcopy(generated.documents[audit.OLD_RESULT]["frequencies_hz"])
    before = audit.common_frequencies(frequencies)
    for person in frequencies:
        person[0] = "A poison changes do not matter"
    assert audit.common_frequencies(frequencies) == before
    frequencies[0][1] = [[value * 1.01 for value in row] for row in frequencies[0][1]]
    after = audit.common_frequencies(frequencies)
    assert after["S002"] == before["S002"]
    assert all(after[s] != before[s] for s in audit.SUBJECTS[1:])


@pytest.mark.parametrize("n_ready,c_ready,decision", [
    (False, True, "COMMON_ONLY_READY_DEV"), (True, False, "NOMINAL_ONLY_READY_DEV"),
    (False, False, "NEITHER_BASELINE_READY_STOP"),
])
def test_all_decision_branches(generated, n_ready, c_ready, decision):
    for subject in audit.SUBJECTS:
        for index, query in enumerate(generated.new("scores_" + subject + ".json")["queries"]):
            label = generated.documents[audit.RECORD_PATHS[subject]]["records"][index]["label"]
            for arm, ready in zip(audit.ARMS, (n_ready, c_ready)):
                query["arms"][arm] = row_for(label if ready else (label + 1) % 5)
    generated.rebuild_result()
    generated.refresh()
    assert audit.audit(generated.root, generated.run)["summary"]["decision"] == decision


def test_high_accuracy_with_only_six_subjects_ready_does_not_pass(generated):
    for subject in audit.SUBJECTS[6:]:
        for query in generated.new("scores_" + subject + ".json")["queries"][:4]:
            prediction = query["arms"]["NOMINAL"]["prediction"]
            query["arms"]["NOMINAL"] = row_for((prediction + 1) % 5)
    generated.rebuild_result()
    generated.refresh()
    summary = audit.audit(generated.root, generated.run)["summary"]
    assert summary["NOMINAL"]["correct"] == 134
    assert summary["NOMINAL"]["subjects_ready"] == 6
    assert summary["NOMINAL"]["ready"] is False


def test_per_subject_missing_class_cannot_be_ready(generated):
    subject = "S002"
    for query in generated.new("scores_" + subject + ".json")["queries"][:3]:
        query["arms"]["COMMON"] = row_for(0)
    generated.rebuild_result()
    generated.refresh()
    report = audit.audit(generated.root, generated.run)
    assert report["summary"]["COMMON"]["subjects_ready"] == 9
    assert report["participant_correct_differences"][subject] == -3
    assert report["summary"]["paired"]["nominal_only"] == 3


@pytest.mark.parametrize("document,path,value,reason", [
    ("manifest.json", ["scope", "fits"], False, "manifest_scope"),
    ("manifest.json", ["frequencies_hz", "COMMON", "S002", 0], 6.66, "source_only_banks"),
    ("manifest.json", ["attempts"], True, "manifest_attempts"),
    ("manifest.json", ["deadline_utc"], "2026-09-14T16:00:00Z", "deadline"),
    ("started.json", ["parent_pid"], True, "parent_pid"),
    ("worker_claim.json", ["pid"], 101, "distinct_worker"),
    ("worker_claim.json", ["attempt"], 2, "single_attempt"),
    ("terminal.json", ["ended_utc"], "2026-09-13T15:14:01Z", "lifetime_times"),
    ("terminal.json", ["status"], "STOPPED_NO_RETRY", "complete_terminal"),
    ("scores_frozen.json", ["created_utc"], "2026-09-13T15:11:19Z", "score_attempt_seal_order"),
    ("scores_frozen.json", ["predictions"], 299, "seal_predictions"),
    ("attempt_S002.json", ["loaded_variables"], ["eeg", "DIN_1"], "eeg_only_decode"),
    ("attempt_S002.json", ["subject"], "S003", "subject_order"),
    ("attempt_S003.json", ["started_utc"], "2026-09-13T15:11:01Z", "score_attempt_seal_order"),
    ("scores_S002.json", ["mat_sha256"], "f" * 64, "mat_pin_binding"),
    ("scores_S002.json", ["queries", 0, "start0"], 1001, "saved_window"),
    ("scores_S002.json", ["queries", 0, "arms", "NOMINAL", "prediction"], True, "saved_argmax"),
    ("scores_S002.json", ["queries", 0, "arms", "NOMINAL", "prediction"], 0, "saved_argmax"),
    ("scores_S002.json", ["queries", 0, "arms", "NOMINAL", "scores", 0], .04, "projection_score"),
    ("scores_S002.json", ["queries", 0, "arms", "NOMINAL", "scores", 0], 1.00000000001, "score_bounds"),
    ("scores_S002.json", ["queries", 0, "arms", "NOMINAL", "projection", 0], [1., 1., 0., 0.], "score_bounds"),
    ("scores_S002.json", ["queries", 0, "arms", "NOMINAL", "projection", 0, 0], True, "projection_number"),
    ("scores_S002.json", ["queries", 0, "arms", "NOMINAL", "projection", 0], [.1], "projection_rank"),
    ("result.json", ["participants", 0, "queries", 0, "label"], 0, "frozen_evaluation_label"),
    ("result.json", ["participants", 0, "queries", 0, "arms", "NOMINAL", "correct"], 1,
     "evaluation_preserves_scores"),
    ("result.json", ["participants", 0, "queries", 0, "arms", "NOMINAL", "scores", 0], .011,
     "evaluation_preserves_scores"),
    ("result.json", ["participants", 0, "summary", "NOMINAL", "correct"], 14, "person_summary"),
    ("result.json", ["summary", "COMMON", "subjects_ready"], 9, "overall_summary"),
    ("result.json", ["summary", "paired", "both_correct"], 149, "overall_summary"),
    ("result.json", ["summary", "common_advantage_pp"], 1., "overall_summary"),
    ("result.json", ["summary", "subject_changes", "tied"], 9, "overall_summary"),
    ("result.json", ["summary", "decision"], "COMMON_ONLY_READY_DEV", "overall_summary"),
    ("result.json", ["cost", "target_labeled_support_trials"], 1, "zero_target_support_cost"),
    ("result.json", ["scope", "din_decodes"], 1, "result_scope"),
])
def test_semantic_poison_rejected(generated, document, path, value, reason):
    target = generated.new(document)
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    generated.refresh()
    with pytest.raises(ValueError, match=reason):
        audit.audit(generated.root, generated.run)


@pytest.mark.parametrize("name,field", [
    ("started.json", "manifest_sha256"), ("worker_claim.json", "manifest_sha256"),
    ("terminal.json", "started_sha256"), ("terminal.json", "worker_claim_sha256"),
    ("terminal.json", "result_sha256"), ("result.json", "scores_frozen_sha256"),
])
def test_hash_binding_poison(generated, name, field):
    generated.new(name)[field] = "f" * 64
    generated.write_document(audit.RUN_RELATIVE + "/" + name)
    with pytest.raises(ValueError, match="binding|sha256"):
        audit.audit(generated.root, generated.run)


@pytest.mark.parametrize("path", ["../outside.py", "/absolute.py", "src/extra.py"])
def test_exact_pin_set_precedes_arbitrary_path_access(generated, path):
    generated.new("manifest.json")["pins"][path] = "f" * 64
    generated.write_document(audit.RUN_RELATIVE + "/manifest.json")
    with pytest.raises(ValueError, match="nineteen_pins"):
        audit.audit(generated.root, generated.run)


def test_known_source_pin_is_not_just_self_consistent_manifest(generated, monkeypatch):
    known = dict(generated.known_sha)
    known[audit.OLD_RESULT] = "f" * 64
    monkeypatch.setattr(audit, "KNOWN_SHA", known)
    with pytest.raises(ValueError, match="unchanged_known_pin"):
        audit.audit(generated.root, generated.run)


def test_label_in_score_artifact_is_forbidden(generated):
    generated.new("scores_S002.json")["queries"][0]["label"] = 3
    generated.refresh()
    with pytest.raises(ValueError, match="label_free_query"):
        audit.audit(generated.root, generated.run)


@pytest.mark.parametrize("field,value,reason", [
    ("run", "a", "previous_record_run"), ("total_samples", True, "total_samples"),
    ("receipt_sha256", "f" * 64, "previous_record_receipt"),
])
def test_old_record_role_poison(generated, field, value, reason):
    generated.documents[audit.RECORD_PATHS["S002"]][field] = value
    generated.refresh()
    with pytest.raises(ValueError, match=reason):
        audit.audit(generated.root, generated.run)


@pytest.mark.parametrize("field,value", [("group_index", 9), ("repeat", True), ("end0", 1501),
                                         ("label", 3.), ("trial_end0", 2001)])
def test_old_window_structure_poison(generated, field, value):
    generated.documents[audit.RECORD_PATHS["S002"]]["records"][0][field] = value
    generated.refresh()
    with pytest.raises(ValueError):
        audit.audit(generated.root, generated.run)


def test_input_path_poison(generated):
    generated.documents[audit.SPEC]["inputs"][1]["path"] = "/tmp/outside.mat"
    generated.refresh()
    with pytest.raises(ValueError, match="input_B_path"):
        audit.audit(generated.root, generated.run)


def test_exact_tie_and_sub_tolerance_saved_argmax():
    row = {"projection": [[.5, 0., 0., 0.]] * 5, "scores": [.25] * 5, "prediction": 0}
    assert audit.score_row(row) == (0, 0., 1)
    row["prediction"] = 1
    with pytest.raises(ValueError, match="saved_argmax"):
        audit.score_row(row)
    row["scores"][1] += 1e-13
    predicted, error, near = audit.score_row(row)
    assert predicted == near == 1 and error < audit.TOL


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1"])
def test_nonfinite_and_wrong_numeric_projection_types(value):
    row = row_for(0)
    row["projection"][0][0] = value
    with pytest.raises(ValueError):
        audit.score_row(row)


@pytest.mark.parametrize("body", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_strict_json(generated, body):
    (generated.run / "manifest.json").write_bytes(body)
    with pytest.raises(ValueError):
        audit.audit(generated.root, generated.run)


def test_extra_run_file_and_per_file_cap(generated):
    extra = generated.run / "unexpected.json"
    extra.write_bytes(b"{}")
    with pytest.raises(ValueError, match="exact_run_files"):
        audit.audit(generated.root, generated.run)


def test_per_file_cap(generated):
    (generated.run / "stdout.txt").write_bytes(b"x" * (audit.FILE_CAP + 1))
    with pytest.raises(ValueError, match="artifact_budget"):
        audit.audit(generated.root, generated.run)


def test_reader_refuses_raw_paths_without_opening(generated):
    reader = audit.Reader(generated.root, generated.run)
    with pytest.raises(ValueError, match="read_allowlist"):
        reader.read("/home/whwovy/data/raw.mat")
    assert reader.bodies == {}


@pytest.mark.parametrize("value", ["2026-09-13T15:00:00Z", "2026-09-13T15:00:00.2Z",
                                   "2026-09-13T15:00:00.200000+00:00"])
def test_utc_fraction_compatibility(value):
    assert audit.stamp(value).utcoffset().total_seconds() == 0


def test_cli_reserves_exclusive_output_and_preserves_failure(tmp_path, monkeypatch, capsys):
    calls = []

    def fail():
        calls.append(1)
        raise ValueError("synthetic audit failure")

    monkeypatch.setattr(audit, "audit", fail)
    monkeypatch.setattr(audit.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(audit.signal, "signal", lambda *_: None)
    monkeypatch.setattr(audit.signal, "alarm", lambda *_: None)
    output = tmp_path / "audit.json"
    assert audit.main(["--output", str(output)]) == 1
    before = output.read_bytes()
    assert len(before) < audit.OUTPUT_CAP
    assert json.loads(before)["status"] == "AUDIT_FAIL_NO_RETRY"
    assert json.loads(capsys.readouterr().out)["reason"] == "synthetic audit failure"
    with pytest.raises(FileExistsError):
        audit.main(["--output", str(output)])
    assert calls == [1] and output.read_bytes() == before
