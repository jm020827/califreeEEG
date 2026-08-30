from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field

import torch


def resolve_device(cfg: dict) -> torch.device:
    runtime = cfg.get("runtime", {})
    requested = str(runtime.get("device", "auto")).lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, cuda, cpu.")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "runtime.device=cuda but CUDA is unavailable. Use the project CUDA environment; "
            "set runtime.device=cpu only for an explicitly CPU-scoped diagnostic."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and bool(runtime.get("probe_cuda", False)):
        cuda_execution_probe(device)
    return device


def cuda_execution_probe(device: torch.device | str = "cuda") -> dict[str, object]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA execution probe requires an available CUDA device.")
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    resolved = torch.device("cuda", index)
    torch.cuda.set_device(index)
    torch.cuda.init()
    left = torch.arange(16, dtype=torch.float32, device=resolved).reshape(4, 4)
    left.requires_grad_(True)
    right = torch.eye(4, dtype=torch.float32, device=resolved)
    loss = (left @ right).square().mean()
    loss.backward()
    torch.cuda.synchronize(resolved)
    if left.grad is None or not torch.isfinite(left.grad).all():
        raise RuntimeError("CUDA probe produced a missing or non-finite gradient.")
    return {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "device": str(resolved),
        "device_name": torch.cuda.get_device_name(resolved),
        "compute_capability": list(torch.cuda.get_device_capability(resolved)),
        "loss": float(loss.detach().cpu()),
    }


@dataclass
class RuntimeMeasurement:
    requested_device: str
    device: torch.device
    run_mode: str
    started_at: float
    baseline_allocated_bytes: int | None
    baseline_reserved_bytes: int | None
    start_free_bytes: int | None
    start_total_bytes: int | None
    phases: list[dict[str, object]] = field(default_factory=list)
    active_phase: dict[str, object] | None = None

    def start_phase(self, name: str, *, epoch: int | None = None) -> None:
        if self.active_phase is not None:
            raise RuntimeError(f"Runtime phase {self.active_phase['name']!r} is still active.")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.active_phase = {
            "name": str(name),
            "epoch": epoch,
            "started_at": time.perf_counter(),
        }

    def finish_phase(
        self, *, status: str = "completed", error: BaseException | None = None
    ) -> dict[str, object]:
        if self.active_phase is None:
            raise RuntimeError("No runtime phase is active.")
        peak_allocated = peak_reserved = None
        telemetry_errors: list[str] = []
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
                peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
                peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
            except RuntimeError as exc:
                telemetry_errors.append(f"phase_telemetry:{type(exc).__name__}:{exc}")
        record = {
            "name": self.active_phase["name"],
            "epoch": self.active_phase["epoch"],
            "status": status,
            "elapsed_time_sec": float(
                time.perf_counter() - float(self.active_phase["started_at"])
            ),
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "telemetry_errors": telemetry_errors,
        }
        self.phases.append(record)
        self.active_phase = None
        return record

    @classmethod
    def start(cls, cfg: dict, device: torch.device, *, run_mode: str) -> RuntimeMeasurement:
        requested = str(cfg.get("runtime", {}).get("device", "auto"))
        resolved = _normalized_device(device)
        baseline_allocated = baseline_reserved = free_bytes = total_bytes = None
        if resolved.type == "cuda":
            gc.collect()
            torch.cuda.synchronize(resolved)
            torch.cuda.empty_cache()
            baseline_allocated = int(torch.cuda.memory_allocated(resolved))
            baseline_reserved = int(torch.cuda.memory_reserved(resolved))
            try:
                free_bytes, total_bytes = [
                    int(value) for value in torch.cuda.mem_get_info(resolved)
                ]
            except (AttributeError, RuntimeError):
                free_bytes = total_bytes = None
            torch.cuda.reset_peak_memory_stats(resolved)
        return cls(
            requested_device=requested,
            device=resolved,
            run_mode=run_mode,
            started_at=time.perf_counter(),
            baseline_allocated_bytes=baseline_allocated,
            baseline_reserved_bytes=baseline_reserved,
            start_free_bytes=free_bytes,
            start_total_bytes=total_bytes,
        )

    def finish(
        self,
        *,
        status: str,
        error: BaseException | None = None,
    ) -> dict[str, object]:
        if self.active_phase is not None:
            self.finish_phase(status="failed" if error is not None else status, error=error)
        telemetry_errors: list[str] = []
        peak_allocated = peak_reserved = final_allocated = final_reserved = None
        end_free = end_total = None
        device_name = compute_capability = device_total_memory = None
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError as exc:
                telemetry_errors.append(f"synchronize:{type(exc).__name__}:{exc}")
            for field_name, function in (
                ("peak_allocated", torch.cuda.max_memory_allocated),
                ("peak_reserved", torch.cuda.max_memory_reserved),
                ("final_allocated", torch.cuda.memory_allocated),
                ("final_reserved", torch.cuda.memory_reserved),
            ):
                try:
                    value = int(function(self.device))
                except RuntimeError as exc:
                    telemetry_errors.append(f"{field_name}:{type(exc).__name__}:{exc}")
                    value = None
                if field_name == "peak_allocated":
                    peak_allocated = value
                elif field_name == "peak_reserved":
                    peak_reserved = value
                elif field_name == "final_allocated":
                    final_allocated = value
                else:
                    final_reserved = value
            try:
                end_free, end_total = [
                    int(value) for value in torch.cuda.mem_get_info(self.device)
                ]
            except (AttributeError, RuntimeError) as exc:
                telemetry_errors.append(f"mem_get_info:{type(exc).__name__}:{exc}")
            try:
                properties = torch.cuda.get_device_properties(self.device)
                device_name = str(properties.name)
                compute_capability = list(torch.cuda.get_device_capability(self.device))
                device_total_memory = int(properties.total_memory)
            except RuntimeError as exc:
                telemetry_errors.append(f"device_properties:{type(exc).__name__}:{exc}")
        phase_allocated = [
            int(value)
            for phase in self.phases
            if (value := phase.get("peak_memory_allocated_bytes")) is not None
        ]
        phase_reserved = [
            int(value)
            for phase in self.phases
            if (value := phase.get("peak_memory_reserved_bytes")) is not None
        ]
        if phase_allocated:
            peak_allocated = max([peak_allocated or 0, *phase_allocated])
        if phase_reserved:
            peak_reserved = max([peak_reserved or 0, *phase_reserved])
        return {
            "schema": "cfeg.runtime-metrics.v1",
            "status": status,
            "run_mode": self.run_mode,
            "measurement_scope": (
                "after_device_probe_and_cache_cleanup_through_final_synchronized_operation"
            ),
            "requested_device": self.requested_device,
            "resolved_device": str(self.device),
            "execution_device_type": self.device.type,
            "logical_device_index": self.device.index,
            "device_name": device_name,
            "compute_capability": compute_capability,
            "device_total_memory_bytes": device_total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "baseline_memory_allocated_bytes": self.baseline_allocated_bytes,
            "baseline_memory_reserved_bytes": self.baseline_reserved_bytes,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "final_memory_allocated_bytes": final_allocated,
            "final_memory_reserved_bytes": final_reserved,
            "start_free_memory_bytes": self.start_free_bytes,
            "end_free_memory_bytes": end_free,
            "reported_total_memory_bytes": end_total or self.start_total_bytes,
            "elapsed_time_sec": float(time.perf_counter() - self.started_at),
            "cuda_oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "telemetry_errors": telemetry_errors,
            "phases": self.phases,
        }


def _normalized_device(device: torch.device) -> torch.device:
    if device.type != "cuda":
        return torch.device("cpu")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.device("cuda", index)
