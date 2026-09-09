"""Contract tests only: artificial score tensors, no registered generation/fit."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import n1_metadata_generated_efficacy_run as run
from cfeg.analysis.task_trca_n1_evaluation import ARMS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/n1_metadata_generated_efficacy_v1.json").read_text())
SPEC = importlib.util.spec_from_file_location(
    "generated_launcher", ROOT / "scripts/launch_n1_metadata_generated_efficacy.py"
)
LAUNCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCH)


def mock_scores():
    """Hand-authored score oracle, not the prospective DGP or real predictions."""
    config = {**CONFIG, "evaluation_groups": 4}
    scores = np.zeros((2, 8, 10, 48, 12))
    oracle = np.zeros((2, 8, 48, 12))
    for regime in range(2):
        for record in range(8):
            for arm in range(10):
                ncorrect = 36 if regime == 0 and ARMS[arm] == "QM" else 24
                prediction = np.where(
                    np.arange(48) < ncorrect, np.arange(48) % 12, (np.arange(48) + 1) % 12
                )
                scores[regime, record, arm, np.arange(48), prediction] = 1
            prediction = np.where(np.arange(48) < 42, np.arange(48) % 12, (np.arange(48) + 1) % 12)
            oracle[regime, record, np.arange(48), prediction] = 1
    valid = {
        "finite_arrays": True,
        "common_q_hashes": True,
        "constant_support_geometry": True,
        "donors_cross_group": True,
    }
    actuation = {
        regime: {
            "coefficient_max_abs": 1.0,
            "gradient_max": 1.0,
            "max_abs_R": 1.0,
            "max_abs_F": 1.0,
            "max_abs_J": 1.0,
            "max_abs_score": 1.0,
            "argmax_changed": 96,
        }
        for regime in run.REGIMES
    }
    return config, scores, oracle, valid, actuation


def summary(config, scores, oracle, valid, actuation):
    return run.summarize(
        scores, oracle, np.tile(np.arange(4), 2), ARMS, config, validity=valid, actuation=actuation
    )


def test_mock_score_pass_and_group_unit():
    result = summary(*mock_scores())
    assert result["terminal"] == "GENERATED_M_CAPACITY_PASS"
    assert np.asarray(result["group_accuracy_percent"]).shape == (2, 4, 11)
    assert result["contrasts"]["coupled_QM_minus_Q"] == {
        "mean_pp": 25.0,
        "mcse_pp": 0.0,
        "low_pp": 25.0,
        "high_pp": 25.0,
        "n_groups": 4,
    }
    assert result["contrasts"]["null_QM_minus_Q"]["high_pp"] == 0


@pytest.mark.parametrize(
    "failure,terminal",
    [
        ("validity", "EFFICACY_NOT_EVALUATED"),
        ("oracle", "POSITIVE_CONTROL_NOT_QUALIFIED"),
        ("null", "NEGATIVE_CONTROL_NOT_QUALIFIED"),
        ("learner", "LEARNER_CAPACITY_NOT_ESTABLISHED"),
        ("actuation", "LEARNER_CAPACITY_NOT_ESTABLISHED"),
    ],
)
def test_gate_precedence(failure, terminal):
    config, scores, oracle, valid, actuation = mock_scores()
    if failure == "validity":
        valid["common_q_hashes"] = False
        oracle[:] = scores[:, :, ARMS.index("Q")]
    elif failure == "oracle":
        oracle[:] = scores[:, :, ARMS.index("Q")]
    elif failure == "null":
        scores[1, :, ARMS.index("QM")] = scores[0, :, ARMS.index("QM")]
    elif failure == "learner":
        scores[0, :, ARMS.index("QM")] = scores[0, :, ARMS.index("Q")]
    else:
        actuation["coupled"]["gradient_max"] = 0
    assert summary(config, scores, oracle, valid, actuation)["terminal"] == terminal


def test_grid_and_finite_rejection():
    config, scores, oracle, valid, actuation = mock_scores()
    with pytest.raises(ValueError, match="group grid"):
        run.summarize(
            scores,
            oracle,
            np.repeat(np.arange(4), 2),
            ARMS,
            config,
            validity=valid,
            actuation=actuation,
        )
    scores[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        summary(config, scores, oracle, valid, actuation)


def test_interval_is_sample_mcse_and_two_sided():
    values = np.array([0, 2, 4, 6])
    row = run.interval(values)
    assert row["mean_pp"] == 3
    assert row["mcse_pp"] == pytest.approx(values.std(ddof=1) / 2)
    assert row["mean_pp"] - row["low_pp"] == pytest.approx(row["high_pp"] - row["mean_pp"])
    with pytest.raises(ValueError):
        run.interval([1])


def test_exclusive_publication_and_no_pickle(tmp_path):
    path = tmp_path / "record.json"
    run.publish(path, {"a": 1})
    before = run.descriptor(path)
    with pytest.raises(FileExistsError):
        run.publish(path, {"a": 2})
    assert run.descriptor(path) == before
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        run.descriptor(link)
    with pytest.raises(ValueError):
        run.save_arrays(tmp_path / "unsafe.npz", x=np.array([object()]))


def test_launcher_fast_child_and_file_backed_logs(tmp_path):
    (tmp_path / "logs").mkdir()
    result = LAUNCH.run_child(
        [sys.executable, "-c", "print('component-only')"], tmp_path, "component", 5, CONFIG
    )
    assert result["exit_code"] == 0
    assert result["watchdog_reason"] is None
    assert result["process_group_remaining"] is False
    assert (tmp_path / "logs/component.stdout.log").read_text() == "component-only\n"
    assert result["peak_rss_kib"] > 0


def test_launcher_timeout_no_retry(tmp_path):
    (tmp_path / "logs").mkdir()
    result = LAUNCH.run_child(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        "timed",
        0.1,
        CONFIG,
        grace=0.1,
    )
    assert result["exit_code"] != 0
    assert result["watchdog_reason"] == "TIME_LIMIT"
    assert result["process_group_remaining"] is False
    with pytest.raises(FileExistsError):
        LAUNCH.run_child([sys.executable, "-c", "pass"], tmp_path, "timed", 5, CONFIG)


def test_registration_requires_exact_scope_and_config(tmp_path):
    with pytest.raises(ValueError, match="dedicated generated-study"):
        LAUNCH.prepare(tmp_path, tmp_path / "none.json")
    assert LAUNCH.digest(LAUNCH.CONFIG)["sha256"] == LAUNCH.CONFIG_SHA
    assert LAUNCH.ENV["CUDA_VISIBLE_DEVICES"] == ""


def test_symlink_and_hardlink_pins_rejected(tmp_path):
    p = tmp_path / "file"
    p.write_bytes(b"x")
    alias = tmp_path / "alias"
    alias.symlink_to(p)
    with pytest.raises(ValueError):
        LAUNCH.digest(alias)
    hardlink = tmp_path / "hard"
    os.link(p, hardlink)
    with pytest.raises(ValueError, match="single-link"):
        LAUNCH.digest(p)


def test_child_start_receipt_failure_reaps_owned_process(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    processes = []
    original_detach = LAUNCH.detach

    def capture(*args):
        process = original_detach(*args)
        processes.append(process)
        return process

    def fail_publish(*_args):
        raise OSError("mock start-receipt failure")

    monkeypatch.setattr(LAUNCH, "detach", capture)
    monkeypatch.setattr(LAUNCH, "publish", fail_publish)
    with pytest.raises(OSError, match="mock start-receipt"):
        LAUNCH.run_child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            "receipt_failure",
            5,
            CONFIG,
        )
    assert len(processes) == 1
    assert processes[0].returncode is not None
    assert not LAUNCH.group_exists(processes[0].pid)


def test_launch_handshake_waits_for_seal_then_checks_exact_identity(tmp_path):
    path = tmp_path / "supervisor.start.json"
    path.write_text("{unfinished")
    with pytest.raises(TimeoutError, match="primary not started"):
        LAUNCH.await_launch_receipt(tmp_path, "test-sha", seconds=0.04)
    # Test-only completion of an unsealed publication. No actual supervisor is spawned.
    path.write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpid()}))
    path.chmod(0o400)
    LAUNCH.publish(tmp_path / "launch.claim.json", {"registration_sha256": "test-sha"})
    LAUNCH.await_launch_receipt(tmp_path, "test-sha", seconds=0.1)
    with pytest.raises(ValueError, match="identity mismatch"):
        LAUNCH.await_launch_receipt(tmp_path, "different-sha", seconds=0.1)


def test_launch_handshake_rejects_alias(tmp_path):
    target = tmp_path / "receipt"
    target.write_text("{}")
    target.chmod(0o400)
    (tmp_path / "supervisor.start.json").symlink_to(target)
    with pytest.raises(ValueError, match="regular single-link"):
        LAUNCH.await_launch_receipt(tmp_path, "test-sha", seconds=0.1)


def test_real_producer_to_independent_audit_with_explicit_mock_optimizer(tmp_path, monkeypatch):
    """Engineering linkage only: no optimizer runs and no registered seed is used.

    Trace records below are hand-authored mocks, not learned-model evidence. The
    generated waveform/score bridge is exercised without observing a trained
    learner's efficacy or using this toy result to change the prospective DGP.
    """
    import torch

    from cfeg.analysis import n1_metadata_generated_efficacy_audit as cold
    from cfeg.analysis import task_trca_n1_learning as learn

    calls = []

    def mock_head(dimensions, _loss, _regularization, _device):
        calls.append(dimensions)
        trace = tuple(
            {"step": step, "loss_before_step": 1.0, "ce_before_step": 1.0, "gradient_norm": 0.0}
            for step in range(1, 201)
        )
        return learn.HeadFit(learn._readonly(np.zeros(dimensions + 1)), 1.0, 1.0, trace)

    monkeypatch.setattr(learn, "_fit_head", mock_head)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda _: None)
    config = {**CONFIG, "seed": 917, "fit_groups": 2, "evaluation_groups": 2}
    output = tmp_path / "mock-engineering-only"
    output.mkdir()
    result = run.primary(output, config)
    assert calls == [15, 2, 2, 2, 15, 2, 2, 2]
    assert result["terminal"] != "GENERATED_M_CAPACITY_PASS"
    audit = cold.audit(output, config)
    assert audit["status"] == "PASS", audit["errors"]
    assert audit["recomputed_summary"]["terminal"] == result["terminal"]
    rows = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    names = [r["event"] for r in rows]
    assert names.index("both_models_frozen_before_query_decode") < names.index(
        "query_decode_started"
    )
    original = run.descriptor(output / "result.json")
    with pytest.raises(ValueError, match="new empty"):
        run.primary(output, config)
    assert run.descriptor(output / "result.json") == original
