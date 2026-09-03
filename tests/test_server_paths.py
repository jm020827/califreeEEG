from __future__ import annotations

from pathlib import Path


def _env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_server_env_uses_current_ssd3_mount():
    env = _env_example()
    assert env["PROJECT_ROOT"] == "/home/jm020827/califreeEEG"
    assert env["HF_HOME"] == "/mnt/ssd3/jm020827/cache/huggingface"
    assert env["HF_HUB_CACHE"] == "/mnt/ssd3/jm020827/cache/huggingface/hub"
    assert env["EEG_DATA_ROOT"] == ("/mnt/ssd3/jm020827/califreeEEG/eeg_data")
    assert env["WANDB_DIR"] == ("/mnt/ssd3/jm020827/califreeEEG/wandb")
    assert env["CFEG_EXPERIMENT_ROOT"] == "/mnt/ssd3/jm020827/califreeEEG/experiments"
    assert env["WANDB_ENTITY"] == "jm020827"
    assert env["WANDB_PROJECT"] == "calibration-free-eeg"
