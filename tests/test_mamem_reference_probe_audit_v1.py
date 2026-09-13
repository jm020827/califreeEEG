"""Synthetic JSON-only audit canaries; no EEG, DIN, producer imports or fits."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/analysis/audit_mamem_reference_probe_v1.py"
SPEC = importlib.util.spec_from_file_location("reference_probe_independent_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bind(bundle):
    objects = bundle["objects"]
    hashes = {name: digest(value) if value is not None else None for name, value in objects.items()}
    for name in ("started.json", "worker_claim.json"):
        if objects[name] is not None:
            objects[name]["manifest_sha256"] = hashes["manifest.json"]
            hashes[name] = digest(objects[name])
    terminal = objects["terminal.json"]
    terminal["manifest_sha256"] = hashes["manifest.json"]
    terminal["started_sha256"] = hashes["started.json"]
    terminal["worker_claim_sha256"] = hashes["worker_claim.json"]
    if terminal["status"] == "COMPLETE":
        terminal["result_sha256"] = hashes["result.json"]
    hashes["terminal.json"] = digest(terminal)
    bundle["hashes"] = hashes
    return bundle


def summarize(result):
    summary = {}
    for arm in audit.ARMS:
        counts = [0] * 5
        for query in result["queries"]:
            row = query["arms"][arm]
            row["correct"] = row["prediction"] == query["label"]
            counts[query["label"]] += int(row["correct"])
        summary[arm] = {"correct": sum(counts), "total": 15, "per_class_correct": counts,
                        "signal_present_dev": sum(counts) >= 12 and all(counts)}
        summary[arm]["signal_present_dev"] = bool(summary[arm]["signal_present_dev"])
    pairs = {"both_correct": 0, "nominal_only": 0, "sample_only": 0, "both_wrong": 0}
    lookup = {(True, True): "both_correct", (True, False): "nominal_only",
              (False, True): "sample_only", (False, False): "both_wrong"}
    for query in result["queries"]:
        pairs[lookup[tuple(query["arms"][arm]["correct"] for arm in audit.ARMS)]] += 1
    summary["paired"] = pairs
    summary["decision"] = ("REFERENCE_DIAGNOSTIC_COMPLETE" if any(
        summary[arm]["signal_present_dev"] for arm in audit.ARMS) else "NO_ARM_PASSES_STOP")
    result["summary"] = summary


def arm_row(prediction, label):
    projection = [[.9 if candidate == prediction else .1, 0., 0., 0.] for candidate in range(5)]
    return {"projection": projection, "scores": [sum(x * x for x in vector) for vector in projection],
            "prediction": prediction, "correct": prediction == label}


def fixture():
    code = {name: "1" * 64 for name in audit.CODE_PATHS}
    code["src/cfeg/mamem_reference_probe_v1.py"] = audit.OPERATOR_SHA256
    manifest = {"schema": "cfeg.mamem-reference-probe-dev-v1.manifest", "subject": "S001",
                "created_utc": "2026-09-13T14:00:00Z", "deadline_utc": "2026-09-13T15:00:00Z",
                "attempt_budget": 1, "fits": 0, "queries": 15, "predictions": 30,
                "mat_sha256": dict(audit.MAT_SHA256), "role_sha256": dict(audit.ROLE_SHA256),
                "code_sha256": code}
    started = {"status": "STARTED", "attempt": 1, "parent_pid": 121,
               "started_utc": "2026-09-13T14:10:00Z", "manifest_sha256": "0" * 64}
    claim = {"attempt": 1, "parent_pid": 121, "pid": 122,
             "started_utc": "2026-09-13T14:10:00.2Z", "manifest_sha256": "0" * 64}
    terminal = {"status": "COMPLETE", "attempts": 1, "fits": 0, "predictions": 30,
                "started_utc": started["started_utc"], "ended_utc": "2026-09-13T14:10:02Z",
                "manifest_sha256": "0" * 64, "started_sha256": "0" * 64,
                "worker_claim_sha256": "0" * 64, "result_sha256": "0" * 64}
    support = []
    for label, step in enumerate((19, 17, 15, 13, 11)):
        group = 8 + 3 * label
        start = 500 + 5000 * group
        support.append({"label": label, "group_index": group, "start0": start,
                        "end0": start + 500, "trial_end0": start + 1000,
                        "event_samples": [start + 1 + index * step for index in range(20)]})
    queries = []
    for index in range(15):
        label = index // 3
        start = 500 + 3000 * (8 + index)
        queries.append({"group_index": 8 + index, "start0": start, "end0": start + 500,
                        "label": label, "arms": {arm: arm_row(label, label) for arm in audit.ARMS}})
    result = {
        "schema": "cfeg.mamem-reference-probe-dev-v1.result", "subject": "S001",
        "status": "DEVELOPMENT_DIAGNOSTIC_NOT_EFFICACY", "completed_utc": "2026-09-13T14:10:01Z",
        "mat_sha256": dict(audit.MAT_SHA256), "role_sha256": dict(audit.ROLE_SHA256),
        "sampling_rate_hz": 250, "channel_index": 125, "window_samples": 500, "fits": 0,
        "frequencies_hz": {"NOMINAL": list(audit.NOMINAL),
                           "SAMPLE_SUPPORT": [250 / (2 * step) for step in (19, 17, 15, 13, 11)]},
        "support": support, "queries": queries,
        "cost": {"selected_trials": 5, "selected_stimulus_seconds": 25,
                 "analyzed_support_seconds": 10, "selected_support_elapsed_prefix_seconds":
                 max(row["trial_end0"] for row in support) / 250, "full_support_record_seconds": 471.668,
                 "query_ready_elapsed": "UNKNOWN", "setup_seconds": "UNKNOWN"},
        "scope": {"a_loaded_variables": ["DIN_1", "samplingRate"],
                  "b_loaded_variables": ["eeg", "DIN_1", "samplingRate"],
                  "a_eeg_numeric_windows": 0, "b_eeg_numeric_windows": 15,
                  "b_eeg_numeric_channels": 1, "query_metadata_extractions": 0, "gate_fits": 0,
                  "source_cohort_reads": 0, "held60_openings": 0},
        "access_stages": ["a_loaded", "support_bank_frozen", "b_loaded", "all_scores_frozen",
                          "evaluation_done"],
    }
    summarize(result)
    return bind({"objects": {"manifest.json": manifest, "started.json": started,
                             "worker_claim.json": claim, "terminal.json": terminal,
                             "result.json": result}, "code": copy.deepcopy(code)})


def validate(bundle, rebind=True):
    if rebind:
        bind(bundle)
    objects = bundle["objects"]
    return audit.audit_objects(objects["manifest.json"], objects["started.json"],
                               objects["worker_claim.json"], objects["terminal.json"],
                               objects["result.json"], artifact_sha256=bundle["hashes"],
                               actual_code_sha256=bundle["code"])


def replace_path(bundle, path, value):
    current = bundle["objects"]
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def test_complete_independent_saved_scalar_audit_and_no_mutation():
    bundle = fixture()
    before = copy.deepcopy(bundle)
    report = validate(bundle, rebind=False)
    assert bundle == before
    assert report["status"] == "PASS_SAVED_PROJECTION_AND_RECORDED_SCOPE"
    assert report["summary"]["paired"] == {
        "both_correct": 15, "nominal_only": 0, "sample_only": 0, "both_wrong": 0}
    assert report["max_scalar_error"] < 1e-12
    assert (report["queries"], report["predictions"], report["class_scores"],
            report["projection_scalars"]) == (15, 30, 150, 600)
    assert report["raw_reads"] == report["fits"] == report["producer_imports"] == 0
    assert "no independent EEG-to-projection" in report["limitations"]


@pytest.mark.parametrize("nominal_good,sample_good,decision", [
    (True, False, "REFERENCE_DIAGNOSTIC_COMPLETE"),
    (False, True, "REFERENCE_DIAGNOSTIC_COMPLETE"),
    (False, False, "NO_ARM_PASSES_STOP"),
])
def test_all_prespecified_decisions(nominal_good, sample_good, decision):
    bundle = fixture()
    result = bundle["objects"]["result.json"]
    for query in result["queries"]:
        for arm, good in zip(audit.ARMS, (nominal_good, sample_good)):
            predicted = query["label"] if good else (query["label"] + 1) % 5
            query["arms"][arm] = arm_row(predicted, query["label"])
    summarize(result)
    assert validate(bundle)["summary"]["decision"] == decision


def test_twelve_correct_but_missing_class_does_not_pass():
    bundle = fixture()
    result = bundle["objects"]["result.json"]
    for query in result["queries"][:3]:
        for arm in audit.ARMS:
            query["arms"][arm] = arm_row(1, 0)
    summarize(result)
    report = validate(bundle)
    assert report["summary"]["NOMINAL"]["correct"] == 12
    assert report["summary"]["decision"] == "NO_ARM_PASSES_STOP"


def test_paired_transitions_and_class_minimum():
    bundle = fixture()
    result = bundle["objects"]["result.json"]
    for index, wrong_arms in ((0, ("NOMINAL",)), (3, ("SAMPLE_SUPPORT",)),
                              (6, audit.ARMS)):
        for arm in wrong_arms:
            label = result["queries"][index]["label"]
            result["queries"][index]["arms"][arm] = arm_row((label + 1) % 5, label)
    summarize(result)
    assert validate(bundle)["summary"]["paired"] == {
        "both_correct": 12, "nominal_only": 1, "sample_only": 1, "both_wrong": 1}


@pytest.mark.parametrize("path,value,reason", [
    (["manifest.json", "fits"], False, "integer"),
    (["manifest.json", "attempt_budget"], 2, "value"),
    (["manifest.json", "mat_sha256", "a"], "2" * 64, "manifest_mat"),
    (["manifest.json", "role_sha256", "b"], "2" * 64, "manifest_role"),
    (["manifest.json", "deadline_utc"], "2026-09-14T15:00:00Z", "deadline"),
    (["started.json", "parent_pid"], True, "integer"),
    (["started.json", "started_utc"], "2026-09-13T13:00:00Z", "start_time_order"),
    (["worker_claim.json", "pid"], 121, "distinct_worker"),
    (["worker_claim.json", "parent_pid"], 999, "value"),
    (["worker_claim.json", "attempt"], 2, "value"),
    (["worker_claim.json", "started_utc"], "2026-09-13T14:09:59Z", "claim_time_order"),
    (["terminal.json", "ended_utc"], "2026-09-13T14:12:01Z", "wall_order_bound"),
    (["terminal.json", "fits"], 1, "value"),
    (["terminal.json", "attempts"], 2, "value"),
    (["result.json", "completed_utc"], "2026-09-13T14:10:03Z", "completion_order"),
    (["result.json", "channel_index"], 256, "value"),
    (["result.json", "sampling_rate_hz"], 250., "integer"),
    (["result.json", "scope", "a_eeg_numeric_windows"], 1, "scope"),
    (["result.json", "scope", "source_cohort_reads"], False, "scope"),
    (["result.json", "scope", "query_metadata_extractions"], 1, "scope"),
    (["result.json", "scope", "held60_openings"], 1, "scope"),
    (["result.json", "access_stages"], ["b_loaded", "a_loaded"], "access_stages"),
    (["result.json", "frequencies_hz", "NOMINAL", 0], 6.7, "nominal_frequency"),
    (["result.json", "frequencies_hz", "SAMPLE_SUPPORT", 0], 6.7, "sample_frequency"),
    (["result.json", "support", 0, "group_index"], 9, "first_main_support"),
    (["result.json", "support", 0, "label"], True, "integer"),
    (["result.json", "support", 0, "event_samples", 0], 40500, "integer"),
    (["result.json", "support", 0, "event_samples", 0], 40501., "integer"),
    (["result.json", "support", 0, "event_samples", 0], True, "integer"),
    (["result.json", "support", 0, "event_samples", 0], 40520, "event_order"),
    (["result.json", "support", 0, "trial_end0"], 41501, "cost_window"),
    (["result.json", "queries", 0, "group_index"], 9, "value"),
    (["result.json", "queries", 0, "label"], 0., "integer"),
    (["result.json", "queries", 0, "end0"], 25001, "window_length"),
    (["result.json", "queries", 0, "arms", "NOMINAL", "projection", 0], [1., 1., 0., 0.],
     "energy_bounds"),
    (["result.json", "queries", 0, "arms", "NOMINAL", "projection", 0, 0], float("nan"), "finite"),
    (["result.json", "queries", 0, "arms", "NOMINAL", "projection", 0, 0], True, "number"),
    (["result.json", "queries", 0, "arms", "NOMINAL", "scores", 0], 1.00000000001,
     "score_bounds"),
    (["result.json", "queries", 0, "arms", "NOMINAL", "scores", 0], .8, "arithmetic"),
    (["result.json", "queries", 0, "arms", "NOMINAL", "prediction"], 1, "prediction_value"),
    (["result.json", "queries", 0, "arms", "NOMINAL", "correct"], 1, "correct_bool"),
    (["result.json", "summary", "NOMINAL", "correct"], 14, "summary"),
    (["result.json", "summary", "paired", "both_correct"], 14, "summary"),
    (["result.json", "summary", "decision"], "NO_ARM_PASSES_STOP", "summary"),
    (["result.json", "cost", "selected_support_elapsed_prefix_seconds"], 25, "arithmetic"),
    (["result.json", "cost", "analyzed_support_seconds"], 25, "arithmetic"),
    (["result.json", "cost", "full_support_record_seconds"], 470, "arithmetic"),
    (["result.json", "cost", "query_ready_elapsed"], 406, "unknown"),
])
def test_synthetic_poison_rejected(path, value, reason):
    bundle = fixture()
    replace_path(bundle, path, value)
    with pytest.raises(ValueError, match=reason):
        validate(bundle)


@pytest.mark.parametrize("name,field", [
    ("started.json", "manifest_sha256"), ("worker_claim.json", "manifest_sha256"),
    ("terminal.json", "manifest_sha256"), ("terminal.json", "started_sha256"),
    ("terminal.json", "worker_claim_sha256"), ("terminal.json", "result_sha256"),
])
def test_each_byte_hash_binding(name, field):
    bundle = fixture()
    bundle["objects"][name][field] = "f" * 64
    with pytest.raises(ValueError, match="binding"):
        validate(bundle, rebind=False)


@pytest.mark.parametrize("path", ["../outside.py", "/absolute/outside.py", "src/extra.py"])
def test_exact_six_code_pins_reject_path_injection(path):
    bundle = fixture()
    bundle["objects"]["manifest.json"]["code_sha256"][path] = "f" * 64
    with pytest.raises(ValueError, match="six_code_pins"):
        validate(bundle)


def test_actual_source_bytes_must_match_pinned_code():
    bundle = fixture()
    bundle["code"]["src/cfeg/mamem_events_v1.py"] = "f" * 64
    with pytest.raises(ValueError, match="code_hash_binding"):
        validate(bundle)


def test_operator_cannot_be_replaced_even_with_consistent_manifest():
    bundle = fixture()
    key = "src/cfeg/mamem_reference_probe_v1.py"
    bundle["code"][key] = bundle["objects"]["manifest.json"]["code_sha256"][key] = "f" * 64
    with pytest.raises(ValueError, match="unchanged_operator"):
        validate(bundle)


@pytest.mark.parametrize("target", ["query", "support"])
def test_extra_metadata_or_label_channels_are_not_accepted(target):
    bundle = fixture()
    result = bundle["objects"]["result.json"]
    record = result["queries" if target == "query" else "support"][0]
    record["query_frequency"] = 10
    with pytest.raises(ValueError, match="record_keys"):
        validate(bundle)


def test_exact_score_ties_take_smallest_index():
    bundle = fixture()
    result = bundle["objects"]["result.json"]
    for query in result["queries"]:
        for arm in audit.ARMS:
            query["arms"][arm] = {"projection": [[.5, 0., 0., 0.]] * 5,
                                   "scores": [.25] * 5, "prediction": 0,
                                   "correct": query["label"] == 0}
    summarize(result)
    assert validate(bundle)["reconstruction_near_ties"] == 30
    result["queries"][0]["arms"]["NOMINAL"]["prediction"] = 1
    with pytest.raises(ValueError, match="prediction_value"):
        validate(bundle)


def test_sub_tolerance_recomputed_near_tie_respects_saved_argmax():
    bundle = fixture()
    result = bundle["objects"]["result.json"]
    row = result["queries"][0]["arms"]["NOMINAL"]
    row.update(projection=[[.5, 0., 0., 0.]] * 5,
               scores=[.25, .25 + 1e-13, .25, .25, .25], prediction=1, correct=False)
    summarize(result)
    assert validate(bundle)["reconstruction_near_ties"] == 1


@pytest.mark.parametrize("claim_present,partial_result", [(False, False), (True, False), (True, True)])
def test_stopped_never_promoted_to_scalar_pass(claim_present, partial_result):
    bundle = fixture()
    objects = bundle["objects"]
    terminal = objects["terminal.json"]
    terminal.update(status="STOPPED_NO_RETRY", error_type="SyntheticFailure", reason="canary")
    del terminal["predictions"], terminal["result_sha256"]
    if not claim_present:
        objects["worker_claim.json"] = None
    if not partial_result:
        objects["result.json"] = None
    report = validate(bundle)
    assert report["status"] == "STOPPED_NO_RETRY_NOT_SCALAR_AUDITED"
    assert report["partial_result_present"] is partial_result
    assert "summary" not in report


def test_complete_without_claim_cannot_pass():
    bundle = fixture()
    bundle["objects"]["worker_claim.json"] = None
    with pytest.raises(ValueError, match="complete_requires"):
        validate(bundle)


@pytest.mark.parametrize("text", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_strict_json_poison(text):
    with pytest.raises(ValueError):
        json.loads(text, object_pairs_hook=audit._no_duplicates, parse_constant=audit._reject_constant)


@pytest.mark.parametrize("path", ["../raw.mat", "/home/whwovy/data/raw.mat"])
def test_root_relative_read_guard_before_any_io(path):
    with pytest.raises(ValueError, match="root_relative_path"):
        audit._read_regular(Path("/does-not-exist"), path, 1024)


def test_load_bundle_only_reads_five_json_and_six_fixed_code_paths(monkeypatch):
    bundle = fixture()
    root = Path("/synthetic-root")
    touched = []
    bodies = {audit.RUN_RELATIVE + "/" + name: json.dumps(value).encode()
              for name, value in bundle["objects"].items()}
    bodies.update({name: b"synthetic-source-only" for name in audit.CODE_PATHS})

    def read_regular(received_root, relative, cap):
        assert received_root == root
        touched.append(relative)
        assert relative in bodies and len(bodies[relative]) < cap
        return bodies[relative]

    monkeypatch.setattr(audit, "_read_regular", read_regular)
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    monkeypatch.setattr(Path, "iterdir", lambda _path: iter(()))
    monkeypatch.setattr(Path, "exists", lambda _path: True)
    objects, hashes, code_hashes, sizes = audit.load_bundle(root)
    assert objects == bundle["objects"]
    assert set(hashes) == audit.ARTIFACT_NAMES
    assert set(code_hashes) == audit.CODE_PATHS
    assert set(sizes) == audit.ARTIFACT_NAMES
    assert len(touched) == len(set(touched)) == 11
    assert all(not name.endswith((".mat", ".npz")) for name in touched)
