"""Synthetic operator mocks only: no MAT/cache, subprocess, pipelines or fits."""

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "mamem_source_runner_v2",
    Path(__file__).resolve().parents[1] / "scripts/analysis/run_mamem_source_v2.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    report, cache = tmp_path / "report", tmp_path / "cache"
    report.mkdir()
    cache.mkdir()
    monkeypatch.setattr(runner, "RUN", report)
    monkeypatch.setattr(runner, "CACHE", cache)
    monkeypatch.setattr(runner, "remaining", lambda cap: cap)
    return report, cache


def cfg_fixture():
    selected = [
        {
            "subject": f"S{n:03}",
            "run": run,
            "archive": str(runner.SOURCE / "EEG-SSVEP-Part1.rar"),
            "member": f"EEG-SSVEP-Part1/S{n:03}{run}.mat",
            "bytes": 100,
            "crc32": "12345678",
        }
        for n in range(1, 12)
        for run in "ab"
    ]
    pins = dict.fromkeys(runner.CODE, "a" * 64)
    pins.update(runner.UNCHANGED)
    return {
        "schema": "cfeg.mamem-source-v2",
        "created_utc": "synthetic",
        "deadline_utc": runner.DEADLINE,
        "selected": selected,
        "archives": runner.ARCHIVE_SHA.copy(),
        "inventory_sha256": runner.INVENTORY_SHA,
        "code_sha256": pins,
        "development_subject": "S001",
        "max_real_fits": 80,
        "max_generated_fits": 0,
        "query_ready_elapsed": "UNKNOWN",
    }


def preflight_fixture(phase="generated"):
    subjects = runner.phase_subjects(phase)
    folds = []
    for target in subjects:
        source = sorted(set(subjects) - {target})
        for k in (1, 2):
            folds.append(
                {
                    "target": target,
                    "k": k,
                    "source": source,
                    "sham": {
                        "donor_indices": list(range(5 * len(source))),
                        "changed_fraction": 1.0,
                        "exact_changed_fraction": 1.0,
                        "singleton_rows": 0,
                        "strata": 5,
                    },
                    "prior_audit": [
                        {
                            "target": target,
                            "pseudo_target": s,
                            "contributors": sorted(set(source) - {s}),
                        }
                        for s in source
                    ],
                    "degenerate_oracles": 0,
                }
            )
    return {
        "folds": folds,
        "total_folds": len(folds),
        "total_scalar_oracles": 240 if phase == "generated" else 1800,
        "metadata_extracted_from_query": False,
    }


def test_json_exclusive_finite_bounded(isolated):
    report, _ = isolated
    path = report / "receipt.json"
    runner.save(path, {"attempt": 1})
    assert runner.read_json(path) == {"attempt": 1}
    with pytest.raises(FileExistsError):
        runner.save(path, {"attempt": 2})
    with pytest.raises(ValueError, match="json_cap"):
        runner.save(report / "large.json", {"value": "x" * 100}, cap=10)
    with pytest.raises(ValueError, match="json_read_cap"):
        runner.read_json(path, cap=1)
    with pytest.raises(ValueError):
        runner.save(report / "nan.json", {"value": float("nan")})


def test_exact_manifest_and_transitive_pins():
    runner.validate_manifest(cfg_fixture())
    assert {
        "src/cfeg/mamem_events_v1.py",
        "src/cfeg/mamem_events_v2.py",
        "src/cfeg/mamem_signal_v1.py",
        "src/cfeg/mamem_shrinkage_v1.py",
    } <= set(runner.CODE)
    assert runner.GENERATED_PERIODS_MS == (75.5, 66.5, 58.5, 52.5, 43.5)
    assert [1000 // (2 * int(p)) for p in runner.GENERATED_PERIODS_MS] == [6, 7, 8, 9, 11]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda c: c["code_sha256"].pop("src/cfeg/mamem_events_v1.py"), "exact_code_pin_keyset"),
        (lambda c: c["code_sha256"].update({"../escape.py": "a" * 64}), "exact_code_pin_keyset"),
        (
            lambda c: c["code_sha256"].update({"src/cfeg/mamem_signal_v1.py": "b" * 64}),
            "unchanged_v1_pins",
        ),
        (lambda c: c.update(max_generated_fits=32), "frozen_budget_source"),
        (lambda c: c.update(deadline_utc="2099-01-01"), "frozen_schema_deadline"),
        (lambda c: c["selected"].pop(), "exact_22_members"),
        (lambda c: c["selected"][0].update(member="../S001a.mat"), "member_scope"),
        (lambda c: c["selected"][0].update(archive="/tmp/EEG-SSVEP-Part1.rar"), "archive_scope"),
    ],
)
def test_manifest_rejects_scope_changes(mutation, reason):
    cfg = cfg_fixture()
    mutation(cfg)
    with pytest.raises(ValueError, match=reason):
        runner.validate_manifest(cfg)


@pytest.mark.parametrize("phase", ["development", "anything", "", None])
def test_fit_phase_allowlist(phase):
    with pytest.raises(ValueError, match="fit_phase"):
        runner.claim_child("fit-child", phase)


@pytest.mark.parametrize("name", ["S001c", "S000a", "S012a", "../S001a", "", None])
def test_io_name_allowlist(name):
    with pytest.raises(ValueError, match="io_name"):
        runner.claim_child("io-child", name)


@pytest.mark.parametrize("phase", ["generated", "real"])
def test_preflight_exact_roles_counts(phase):
    runner.validate_preflight(preflight_fixture(phase), phase)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda p: p.update(total_scalar_oracles=239), "preflight_counts"),
        (lambda p: p["folds"].pop(), "preflight_fold_ids"),
        (lambda p: p["folds"][0]["source"].append("FAKE0"), "preflight_source_exclusion"),
        (
            lambda p: p["folds"][0]["prior_audit"][0]["contributors"].append("FAKE0"),
            "preflight_pseudo_exclusion",
        ),
        (lambda p: p["folds"][0]["sham"]["donor_indices"].pop(), "sham_donor_permutation"),
        (
            lambda p: [f["sham"].update(changed_fraction=0.0) for f in p["folds"]],
            "preflight_sham_eligibility",
        ),
    ],
)
def test_preflight_rejects_role_count_errors(mutation, reason):
    report = preflight_fixture()
    mutation(report)
    with pytest.raises(ValueError, match=reason):
        runner.validate_preflight(report, "generated")


def test_generated_phase_has_no_fit_or_accuracy_call(isolated, monkeypatch, capsys):
    report, cache = isolated
    monkeypatch.setattr(runner, "manifest", lambda: cfg_fixture())
    data = {s: object() for s in runner.phase_subjects("generated")}
    monkeypatch.setattr(runner, "generated_data", lambda: data)
    folds = copy.deepcopy(preflight_fixture()["folds"])
    for fold in folds:
        fold["y"] = np.zeros((15, 2))
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("No fit, optimizer or accuracy summary is allowed")

    engine = types.ModuleType("cfeg.mamem_shrinkage_v1")
    engine.prepare = lambda actual: calls.append(("prepare", actual is data)) or actual
    engine.preflight = lambda actual: calls.append(("preflight", actual is data)) or folds
    engine.run_folds = forbidden
    engine.summarize = forbidden
    monkeypatch.setitem(sys.modules, "cfeg.mamem_shrinkage_v1", engine)
    runner.fit_child("generated")
    assert calls == [("prepare", True), ("preflight", True)]
    assert json.loads(capsys.readouterr().out) == {
        "phase": "generated",
        "status": "COMPLETE",
        "fits": 0,
    }
    runner.validate_fit_result("generated")
    assert not list(cache.iterdir())
    assert not (report / "generated_ledger.jsonl").exists()


def test_fit_ledger_unique_ordered_80_and_zero_generated(isolated):
    runner.validate_fit_ledger("generated")
    for i, (subject, k, arm) in enumerate(
        (s, k, arm) for s in runner.phase_subjects("real") for k in (1, 2) for arm in runner.ARMS
    ):
        for kind in ("FIT_STARTED", "FIT_COMPLETE"):
            runner.append("real", kind, {"target": subject, "k": k, "arm": arm, "fit_index": i + 1})
    assert runner.validate_fit_ledger("real") == 80
    runner.append("real", "FIT_COMPLETE", {"target": "S002", "k": 1, "arm": "Q", "fit_index": 1})
    with pytest.raises(ValueError, match="unique_fit_ledger_count"):
        runner.validate_fit_ledger("real")
    runner.append(
        "generated", "FIT_STARTED", {"target": "FAKE0", "k": 1, "arm": "Q", "fit_index": 1}
    )
    with pytest.raises(ValueError, match="unique_fit_ledger_count"):
        runner.validate_fit_ledger("generated")


def test_unvalidated_exit_zero_cannot_complete(isolated, monkeypatch):
    report, _ = isolated
    runner.save(report / "manifest.json", {"synthetic": True})
    monkeypatch.setattr(runner, "manifest", cfg_fixture)
    monkeypatch.setattr(runner, "budget", lambda: None)
    monkeypatch.setattr(runner, "subprocess_child", lambda *args: None)
    runner.execute("generated")
    terminal = runner.read_json(report / "generated_terminal.json")
    assert terminal["status"] == "STOPPED_NO_RETRY"
    assert terminal["observed_counts"]["FIT_STARTED"] == 0
    with pytest.raises(FileExistsError):
        runner.execute("generated")


def test_child_claim_wrong_parent_rejected_before_data(isolated, monkeypatch):
    report, _ = isolated
    monkeypatch.setattr(runner, "child_limits", lambda kind: None)
    monkeypatch.setattr(runner.os, "getppid", lambda: 300)
    runner.save(
        report / "fit-child_generated_authorization.json",
        {
            "phase": "generated",
            "kind": "fit",
            "args": ["fit-child", "generated"],
            "parent_pid": 301,
        },
    )
    runner.save(report / "generated_attempt.json", {"parent_pid": 301})
    with pytest.raises(ValueError, match="active_parent_required"):
        runner.claim_child("fit-child", "generated")
    assert not (report / "fit-child_generated_claim.json").exists()


def test_terminal_status_alone_not_accepted(isolated):
    report, _ = isolated
    runner.save(report / "generated_terminal.json", {"phase": "generated", "status": "COMPLETE"})
    with pytest.raises(ValueError, match="validated_phase_terminal"):
        runner.phase_complete("generated")


def test_child_stdout_requires_exact_identity_and_claim(isolated, monkeypatch):
    report, _ = isolated
    runner.save(
        report / "io-child_S002a_claim.json",
        {
            "pid": 4,
            "parent_pid": runner.os.getpid(),
            "phase": "real",
            "args": ["io-child", "S002a"],
            "one_attempt": True,
        },
    )
    calls = []
    monkeypatch.setattr(runner, "validate_trial_receipt", lambda name: calls.append(name))
    status = {
        "status": "FEATURES_COMPLETE",
        "phase": "real",
        "file": "S002a",
        "windows": 15,
        "fits": 0,
    }
    runner.validate_child_output("io", ["io-child", "S002a"], status, 4)
    assert calls == ["S002a"]
    with pytest.raises(ValueError, match="io_child_status"):
        runner.validate_child_output("io", ["io-child", "S002a"], dict(status, windows=14), 4)
    with pytest.raises(ValueError, match="child_claim_identity"):
        runner.validate_child_output("io", ["io-child", "S002a"], status, 5)


def trial_receipt_fixture(cache, name):
    cfg = cfg_fixture()
    row = next(r for r in cfg["selected"] if r["subject"] + r["run"] == name)
    # This is a tiny JSON sentinel, NOT a numeric cache or an NPZ pipeline run.
    feature = cache / (name + "_features.npz")
    runner.save(feature, {"synthetic_non_numeric_sentinel": True})
    runner.save(
        cache / (name + "_input.json"),
        dict(row, path=str(cache / (name + ".mat")), sha256="a" * 64),
    )
    records = [
        {
            "group_index": 8 + i,
            "label": i // 3,
            "start0": 100 + 2500 * i,
            "end0": 600 + 2500 * i,
            "trial_end0": 1100 + 2500 * i,
            "event_count": 25,
            "metadata": [0.001, 0.2] if name.endswith("a") else None,
        }
        for i in range(15)
    ]
    receipt = {
        "subject": name[:4],
        "run": name[-1],
        "records": records,
        "feature_path": str(feature),
        "feature_sha256": runner.sha(feature),
        "mat_sha256": "a" * 64,
        "fully_decoded_variables": ["eeg", "DIN_1", "samplingRate"],
        "numerical_eeg_scope": "synthetic",
        "physical_units": "UNVERIFIED",
        "labels": "synthetic",
        "metadata_extracted": name.endswith("a"),
    }
    return cfg, receipt


@pytest.mark.parametrize("name", ["S002a", "S002b"])
def test_trial_receipt_exact_support_query_binding(isolated, name):
    _, cache = isolated
    cfg, receipt = trial_receipt_fixture(cache, name)
    runner.save(cache / (name + "_trials.json"), receipt)
    assert runner.validate_trial_receipt(name, cfg) == receipt


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda r: r.update(subject="S003"), "trial_role_scope"),
        (lambda r: r.update(feature_path="/tmp/wrong.npz"), "exact_feature_path_hash"),
        (lambda r: r["records"].pop(), "trial_group_coverage"),
        (lambda r: r["records"][0].update(metadata=None), "trial_metadata_role"),
        (lambda r: r["records"][0].update(start0=True), "trial_integer_fields"),
        (lambda r: r["records"][0].update(end0=599), "trial_window_order"),
        (lambda r: r["records"][0].update(label=4), "trial_label_coverage"),
    ],
)
def test_trial_receipt_rejects_wrong_scope(isolated, mutation, reason):
    _, cache = isolated
    cfg, receipt = trial_receipt_fixture(cache, "S002a")
    mutation(receipt)
    runner.save(cache / "S002a_trials.json", receipt)
    with pytest.raises(ValueError, match=reason):
        runner.validate_trial_receipt("S002a", cfg)


def test_active_child_claim_is_exclusive(isolated, monkeypatch):
    report, _ = isolated
    monkeypatch.setattr(runner, "child_limits", lambda kind: None)
    monkeypatch.setattr(runner.os, "getppid", lambda: 300)
    monkeypatch.setattr(runner, "sha", lambda path: "a" * 64)
    runner.save(
        report / "fit-child_generated_authorization.json",
        {
            "phase": "generated",
            "kind": "fit",
            "args": ["fit-child", "generated"],
            "parent_pid": 300,
            "script_sha256": "a" * 64,
            "manifest_sha256": "a" * 64,
        },
    )
    runner.save(
        report / "generated_attempt.json",
        {"phase": "generated", "attempt": 1, "parent_pid": 300, "manifest_sha256": "a" * 64},
    )
    runner.claim_child("fit-child", "generated")
    with pytest.raises(FileExistsError):
        runner.claim_child("fit-child", "generated")


def saved_fold_fixture():
    fold = {
        "oracle": [[0.5, 0.5] for _ in range(45)],
        "models": {},
        "lambdas": {},
        "train_x": {},
        "eval_x": {},
        "scores": {},
        "predictions": {},
        "accuracy": {},
        "truth": [j for j in range(5) for _ in range(3)],
        "support_prefix_samples": 1000,
        "support_prefix_seconds": 4.0,
        "query_ready_elapsed_seconds": None,
    }
    for arm in runner.ARMS:
        dim = 16 if arm == "Q" else 18
        fold["models"][arm] = {
            "mean": [0.0] * dim,
            "scale": [1.0] * dim,
            "coef": [[0.0, 0.0] for _ in range(dim)],
            "intercept": [0.5, 0.5],
        }
        fold["lambdas"][arm] = [[0.5, 0.5] for _ in range(5)]
        fold["train_x"][arm] = [[0.0] * dim for _ in range(45)]
        fold["eval_x"][arm] = [[0.0] * dim for _ in range(5)]
    for arm in (*runner.ARMS, "TARGET_ONLY", "SOURCE_ONLY", "ZERO_SHOT"):
        fold["scores"][arm] = [[0.0] * 5 for _ in range(15)]
        fold["predictions"][arm] = [0] * 15
        fold["accuracy"][arm] = 0.2
    return fold


def test_saved_output_shape_validation_requires_no_model_forward():
    runner.validate_saved_fold_arrays(saved_fold_fixture())


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda f: f["oracle"].pop(), "oracle_shape_range"),
        (lambda f: f["models"]["QM"]["scale"].__setitem__(0, 0.0), "saved_model_input_dimensions"),
        (lambda f: f["lambdas"]["QM"][0].__setitem__(0, 1.1), "saved_model_input_dimensions"),
        (lambda f: f["predictions"]["QM"].__setitem__(0, 1), "saved_score_prediction_shape"),
        (lambda f: f["accuracy"].update(QM=0.3), "saved_accuracy_arithmetic"),
        (lambda f: f.update(query_ready_elapsed_seconds=4.0), "saved_cost_scope"),
    ],
)
def test_saved_output_rejects_bad_shapes_and_claims(mutation, reason):
    fold = saved_fold_fixture()
    mutation(fold)
    with pytest.raises(ValueError, match=reason):
        runner.validate_saved_fold_arrays(fold)
