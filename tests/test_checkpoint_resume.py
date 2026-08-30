from __future__ import annotations

import json
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from cfeg import train_loop
from cfeg.data.io_hdf5 import write_processed_hdf5
from cfeg.train_loop import run_training
from cfeg.utils.checkpoint import (
    capture_rng_state,
    load_resume_state,
    restore_rng_state,
    save_resume_state,
)


def test_two_slot_resume_journal_commits_latest_generation(tmp_path) -> None:
    first = save_resume_state(tmp_path, {"value": torch.tensor([1])})
    second = save_resume_state(tmp_path, {"value": torch.tensor([2])})
    state, commit = load_resume_state(tmp_path)

    assert first["slot"] != second["slot"]
    assert commit["generation"] == 1
    assert state["generation"] == 1
    assert state["value"].item() == 2


def test_resume_journal_fails_closed_on_committed_slot_corruption(tmp_path) -> None:
    commit = save_resume_state(tmp_path, {"value": torch.tensor([1])})
    state_path = tmp_path / ".resume" / commit["state_path"]
    state_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="missing or corrupted"):
        load_resume_state(tmp_path)


def test_uncommitted_other_slot_does_not_replace_committed_state(tmp_path) -> None:
    commit = save_resume_state(tmp_path, {"value": torch.tensor([1])})
    other = tmp_path / ".resume" / f"state.{1 - int(commit['slot'])}.pt"
    torch.save({"schema": "cfeg.exact-resume.v1", "value": torch.tensor([99])}, other)

    state, observed = load_resume_state(tmp_path)

    assert observed == commit
    assert state["value"].item() == 1


def test_rng_state_round_trip_includes_loader_generator() -> None:
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    generator = torch.Generator().manual_seed(3)
    state = capture_rng_state(generator)
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )
    random.random()
    np.random.random()
    torch.rand(())
    torch.rand((), generator=generator)

    restore_rng_state(state, generator)
    observed = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )

    assert observed == expected


def test_resume_loader_rejects_commit_tampering(tmp_path) -> None:
    save_resume_state(tmp_path, {"value": 1})
    commit_path = tmp_path / ".resume" / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["state_sha256"] = "0" * 64
    commit_path.write_text(json.dumps(commit), encoding="utf-8")

    with pytest.raises(ValueError, match="missing or corrupted"):
        load_resume_state(tmp_path)


def test_interrupted_exact_resume_matches_uninterrupted_training(tmp_path) -> None:
    processed = _write_tiny_processed_dataset(tmp_path / "processed")
    reference_cfg = _tiny_training_config(processed, tmp_path / "reference")
    interrupted_cfg = _tiny_training_config(processed, tmp_path / "interrupted")

    run_training(deepcopy(reference_cfg), resume_exact=True)
    with pytest.raises(RuntimeError, match="Injected exact-resume crash"):
        run_training(
            interrupted_cfg,
            resume_exact=True,
            _test_crash_after_epoch_commit=2,
        )
    resumed = run_training(interrupted_cfg, resume_exact=True)

    reference_state, _ = load_resume_state(reference_cfg["output_dir"])
    resumed_state, _ = load_resume_state(interrupted_cfg["output_dir"])
    assert reference_state["progress"] == resumed_state["progress"]
    assert resumed_state["progress"]["stop_reason"] == "max_epochs"
    for name, expected in reference_state["training_state"]["model_state"].items():
        torch.testing.assert_close(
            resumed_state["training_state"]["model_state"][name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    assert resumed["resumed_from_epoch"] == 2
    assert (tmp_path / "interrupted" / "training_completion.json").is_file()


def test_terminal_early_stop_resume_does_not_run_an_extra_epoch(
    tmp_path, monkeypatch
) -> None:
    processed = _write_tiny_processed_dataset(tmp_path / "processed")
    cfg = _tiny_training_config(processed, tmp_path / "early-stop")
    cfg["train"].update(
        {
            "epochs": 6,
            "checkpoint_selection": "validation_best",
            "validation_epochs": [],
            "early_stop_patience": 1,
        }
    )

    def constant_validation(*args, **kwargs):
        return {"accuracy": 0.5, "balanced_accuracy": 0.5, "nll": 1.0, "ece": 0.0}

    monkeypatch.setattr(train_loop, "evaluate_loader", constant_validation)
    with pytest.raises(RuntimeError, match="Injected exact-resume crash"):
        run_training(cfg, resume_exact=True, _test_crash_after_epoch_commit=2)

    before, _ = load_resume_state(cfg["output_dir"])
    assert before["progress"]["completed_epoch"] == 2
    assert before["progress"]["stop_reason"] == "early_stop"
    resumed = run_training(cfg, resume_exact=True)
    after, _ = load_resume_state(cfg["output_dir"])

    assert resumed["completed_epochs"] == 2
    assert resumed["resumed_from_epoch"] == 2
    assert after["progress"] == before["progress"]


def _write_tiny_processed_dataset(root):
    rng = np.random.default_rng(5)
    n_subjects = 4
    samples_per_subject = 4
    labels = np.tile(np.asarray([0, 1, 0, 1]), n_subjects)
    signals = rng.normal(size=(len(labels), 2, 40)).astype(np.float32)
    signals += labels[:, None, None].astype(np.float32) * 0.2
    write_processed_hdf5(
        root,
        signals,
        np.ones((len(labels), 2), dtype=bool),
        labels,
    )
    rows = []
    for index, label in enumerate(labels):
        subject = index // samples_per_subject
        rows.append(
            {
                "sample_id": f"s{subject}-trial{index}",
                "h5_index": index,
                "dataset_id": "resume_synthetic",
                "subject_id": f"s{subject}",
                "session_id": "session",
                "run_id": "run",
                "trial_id": f"trial-{index}",
                "label": int(label),
                "stimulus_frequency_hz": 2.0 + 2.0 * int(label),
                "stimulus_phase_rad": 0.0,
                "sfreq_original": 20.0,
                "sfreq_processed": 20.0,
                "window_start_sec": 0.0,
                "window_duration_sec": 2.0,
                "reference": "average",
                "hardware_id": "synthetic",
                "cap_type": "synthetic",
                "electrode_type": "synthetic",
                "n_channels_original": 2,
                "n_channels_used": 2,
                "channel_names_original": ["Fp1", "Fp2"],
                "channel_names_used": ["Fp1", "Fp2"],
                "canonical_channel_ids": [1, 2],
                "impedance_mean_kohm": None,
                "impedance_max_kohm": None,
                "reattach_flag": None,
                "time_since_last_session_hours": None,
                "environment_note_code": "synthetic",
                "source_file": "generated",
            }
        )
    pd.DataFrame(rows).to_json(root / "manifest.jsonl", orient="records", lines=True)
    (root / "class_map.json").write_text(
        json.dumps(
            {
                "0": {"label": 0, "stimulus_frequency_hz": 2.0},
                "1": {"label": 1, "stimulus_frequency_hz": 4.0},
            }
        ),
        encoding="utf-8",
    )
    return root


def _tiny_training_config(processed, output_dir) -> dict:
    return {
        "seed": 11,
        "output_dir": str(output_dir),
        "protocol": {"metadata_contract_version": "legacy"},
        "data": {
            "processed_dirs": [str(processed)],
            "split": "cross_subject_train_val",
            "split_seed": 42,
            "val_ratio": 0.25,
            "batch_size": 4,
            "eval_batch_size": 4,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "persistent_hdf5_handles": False,
            "preload_hdf5_to_memory": True,
        },
        "model": {
            "n_classes": 2,
            "c_max": 2,
            "t_len": 40,
            "target_sfreq": 20,
            "d_model": 16,
            "patch_size": 10,
            "depth": 1,
            "n_heads": 2,
            "backbone": {"name": "tiny_transformer"},
            "condition_encoder": {"enabled": False, "external_metadata_mode": "observed"},
            "adapter": {"enabled": False},
            "latent": {"enabled": False},
        },
        "train": {
            "epochs": 4,
            "checkpoint_selection": "fixed_final",
            "validation_epochs": [4],
            "lr": 0.001,
            "weight_decay": 0.0,
            "amp": False,
            "eval_amp": False,
            "grad_clip_norm": 1.0,
            "early_stop_patience": 4,
            "log_interval": 0,
        },
        "checkpoint": {"save_trainable_only": False},
        "loss": {"lambda_cons": 0.0, "lambda_logit_cons": 0.0, "beta_kl": 0.0},
        "tracking": {"wandb": {"enabled": False, "mode": "disabled"}},
        "evaluation": {"open_test": False, "trial_time_sec": 2.0},
        "runtime": {"device": "cpu", "probe_cuda": False},
        "augment": {
            "categorical_metadata_dropout_prob": 0.0,
            "make_two_views": False,
            "channel_dropout_prob": 0.0,
            "channel_subset_prob": 0.0,
            "noise_std_range": [0.0, 0.0],
            "time_shift_samples": 0,
        },
    }
