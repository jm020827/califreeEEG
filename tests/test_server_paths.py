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


def test_interns_env_uses_real_nvme_and_ddn_mounts():
    env = _env_example()
    assert env["PROJECT_ROOT"] == "/root/work/jm020827/califreeEEG"
    assert env["HF_HOME"] == "/mnt/nvme/cache/interns/hf"
    assert env["HF_HUB_CACHE"] == "/mnt/nvme/cache/interns/hf/hub"
    assert env["EEG_DATA_ROOT"] == (
        "/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage/eeg_data"
    )
    assert env["WANDB_DIR"] == (
        "/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage/wandb"
    )
    assert env["WANDB_ENTITY"] == "jmsmlove02"
    assert env["WANDB_PROJECT"] == "calibration-free-eeg"
