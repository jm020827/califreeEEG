from __future__ import annotations

import pytest
import torch
import yaml

from cfeg.runtime import RuntimeMeasurement, cuda_execution_probe, resolve_device
from cfeg.train_loop import _environment_contract, _record_runtime_attempt


def test_explicit_cpu_runtime_is_allowed() -> None:
    assert resolve_device({"runtime": {"device": "cpu"}}).type == "cpu"


def test_invalid_runtime_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="runtime.device"):
        resolve_device({"runtime": {"device": "tpu"}})


def test_required_cuda_fails_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device({"runtime": {"device": "cuda"}})


def test_environment_contract_is_yaml_serializable() -> None:
    dumped = yaml.safe_dump({"runtime_contract": _environment_contract()})
    assert str(torch.__version__) in dumped


def test_cpu_runtime_measurement_uses_null_cuda_memory_fields() -> None:
    measurement = RuntimeMeasurement.start(
        {"runtime": {"device": "cpu"}}, torch.device("cpu"), run_mode="unit_test"
    )
    result = measurement.finish(status="completed")

    assert result["schema"] == "cfeg.runtime-metrics.v1"
    assert result["resolved_device"] == "cpu"
    assert result["execution_device_type"] == "cpu"
    assert result["peak_memory_allocated_bytes"] is None
    assert result["peak_memory_reserved_bytes"] is None
    assert result["cuda_oom"] is False


def test_cpu_runtime_measurement_records_named_phases() -> None:
    measurement = RuntimeMeasurement.start(
        {"runtime": {"device": "cpu"}}, torch.device("cpu"), run_mode="unit_test"
    )
    measurement.start_phase("train", epoch=2)
    phase = measurement.finish_phase()
    result = measurement.finish(status="completed")

    assert phase["name"] == "train"
    assert phase["epoch"] == 2
    assert phase["status"] == "completed"
    assert result["phases"] == [phase]


def test_runtime_measurement_closes_active_failed_phase() -> None:
    measurement = RuntimeMeasurement.start(
        {"runtime": {"device": "cpu"}}, torch.device("cpu"), run_mode="unit_test"
    )
    measurement.start_phase("validation", epoch=1)
    result = measurement.finish(status="failed", error=RuntimeError("broken"))

    assert result["phases"][0]["status"] == "failed"
    assert result["phases"][0]["error_type"] == "RuntimeError"


def test_runtime_attempt_journal_preserves_prior_attempts_and_uses_max_peak(tmp_path) -> None:
    first, _ = _record_runtime_attempt(
        tmp_path,
        {
            "schema": "cfeg.runtime-metrics.v1",
            "status": "failed",
            "elapsed_time_sec": 2.0,
            "peak_memory_allocated_bytes": 10,
            "peak_memory_reserved_bytes": 20,
            "cuda_oom": True,
        },
    )
    second, path = _record_runtime_attempt(
        tmp_path,
        {
            "schema": "cfeg.runtime-metrics.v1",
            "status": "completed",
            "elapsed_time_sec": 3.0,
            "peak_memory_allocated_bytes": 8,
            "peak_memory_reserved_bytes": 12,
            "cuda_oom": False,
        },
    )

    assert first["attempt_count"] == 1
    assert second["attempt_count"] == 2
    assert second["elapsed_time_sec"] == 5.0
    assert second["peak_memory_allocated_bytes"] == 10
    assert second["peak_memory_reserved_bytes"] == 20
    assert second["cuda_oom"] is True
    assert path.is_file()
    assert len(list((tmp_path / "runtime_attempts").glob("attempt_*.json"))) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration probe")
def test_real_cuda_probe_runs_forward_and_backward() -> None:
    result = cuda_execution_probe()
    assert result["device_name"]
    assert result["compute_capability"] == list(torch.cuda.get_device_capability())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration measurement")
def test_real_cuda_runtime_measurement_records_peak() -> None:
    device = torch.device("cuda")
    measurement = RuntimeMeasurement.start(
        {"runtime": {"device": "cuda"}}, device, run_mode="unit_test"
    )
    value = torch.ones((1024, 1024), device=device)
    result = measurement.finish(status="completed")

    assert value.is_cuda
    assert result["peak_memory_allocated_bytes"] >= value.numel() * value.element_size()
    assert result["peak_memory_reserved_bytes"] >= result["peak_memory_allocated_bytes"]
