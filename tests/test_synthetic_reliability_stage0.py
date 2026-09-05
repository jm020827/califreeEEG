from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from cfeg.analysis.synthetic_reliability_stage0 import (
    M_FEATURES,
    PREDICTOR_FORBIDDEN,
    Q_FEATURES,
    _release_stage0_execution,
    _reserve_stage0_execution,
    run_stage0,
    validate_stage0_output_tree,
    write_stage0_outputs,
)
from cfeg.data.ssvep_synthetic import generate_quality_synthetic_processed
from cfeg.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _asset(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic_quality_v1"
    generate_quality_synthetic_processed(
        root,
        n_subjects=3,
        n_blocks_per_interface=2,
        n_repetitions_per_class_per_block=1,
        n_classes=2,
        duration_sec=0.5,
        seed=31,
    )
    return root


def _plan() -> dict:
    plan = load_config(
        REPO / "configs/analysis/synthetic_reliability_stage0.yaml",
        strict_env=False,
    )
    plan["generator"].update(
        {
            "seed": 31,
            "n_subjects": 3,
            "n_blocks_per_interface": 2,
            "n_repetitions_per_class_per_block": 1,
            "n_classes": 2,
            "duration_sec": 0.5,
            "expected_n_samples": 24,
            "expected_n_quality_rows": 192,
        }
    )
    for field in (
        "metadata_partial_r2_positive_subjects_min",
        "channel_rank_rho_gain_positive_subjects_min",
        "block_shuffle_positive_subjects_min",
        "channel_permutation_positive_subjects_min",
    ):
        plan["gate"][field] = 2
    plan["leakage_audit"] = {
        "label_only_subject_macro_r2_max": 1e6,
        "class_target_mean_range_max": 1e6,
    }
    return plan


def test_stage0_is_subject_disjoint_allowlisted_and_deterministic(tmp_path) -> None:
    root = _asset(tmp_path)

    first = run_stage0(root, _plan())
    second = run_stage0(root, _plan())

    predictions = first["predictions"]
    assert len(predictions) == 3 * 2 * 2 * 2 * 8
    assert predictions.groupby("subject_id").size().eq(2 * 2 * 2 * 8).all()
    assert not PREDICTOR_FORBIDDEN & set(Q_FEATURES + M_FEATURES)
    assert "label" not in predictions
    assert first["receipt"]["integrity"]["all_checks_passed"] is True
    assert first["receipt"]["population_inference_allowed"] is False
    assert first["receipt"]["decoder_checkpoint_reuse_allowed"] is False
    assert first["receipt"]["n_subjects"] == 3
    assert len(first["subject_metrics"]) == 3
    assert np.isfinite(first["subject_metrics"].select_dtypes(include="number")).all().all()
    np.testing.assert_allclose(
        first["predictions"].filter(like="prediction_").to_numpy(),
        second["predictions"].filter(like="prediction_").to_numpy(),
        rtol=0.0,
        atol=0.0,
    )
    assert first["receipt"]["split_assignment_sha256"] == second["receipt"][
        "split_assignment_sha256"
    ]
    for fold in first["fold_models"].values():
        assert fold["control_changed_fraction"]["block_bundle_shuffle"] > 0.0
        assert fold["control_changed_fraction"]["channel_impedance_permutation"] > 0.0


def test_stage0_plan_rejects_scope_or_target_drift(tmp_path) -> None:
    root = _asset(tmp_path)
    promoted = copy.deepcopy(_plan())
    promoted["population_inference_allowed"] = True
    with pytest.raises(ValueError, match="must prohibit"):
        run_stage0(root, promoted)

    wrong_target = copy.deepcopy(_plan())
    wrong_target["primary_target"] = "oracle_signal_fraction"
    with pytest.raises(ValueError, match="realized dropout"):
        run_stage0(root, wrong_target)

    wrong_seed = copy.deepcopy(_plan())
    wrong_seed["generator"]["seed"] += 1
    with pytest.raises(ValueError, match="generator recipe mismatch"):
        run_stage0(root, wrong_seed)


def test_stage0_rejects_jsonl_parquet_semantic_drift(tmp_path) -> None:
    root = _asset(tmp_path)
    parquet = root / "manifest.parquet"
    frame = pd.read_parquet(parquet)
    frame.loc[0, "label"] = 1 - int(frame.loc[0, "label"])
    frame.to_parquet(parquet, index=False)

    with pytest.raises(ValueError, match="manifests differ semantically"):
        run_stage0(root, _plan())


def test_stage0_rejects_observed_tensor_grid_or_preprocess_drift(tmp_path) -> None:
    tensor_root = _asset(tmp_path / "tensor")
    signal_path = tensor_root / "signals.h5"
    with h5py.File(signal_path, "r") as handle:
        x = handle["x"][:, :, :-1]
        mask = handle["channel_mask"][:]
        y = handle["y"][:]
    signal_path.unlink()
    with h5py.File(signal_path, "w") as handle:
        handle.create_dataset("x", data=x)
        handle.create_dataset("channel_mask", data=mask)
        handle.create_dataset("y", data=y)
    with pytest.raises(ValueError, match="signal tensor shape"):
        run_stage0(tensor_root, _plan())

    grid_root = _asset(tmp_path / "grid")
    for name in ("manifest.jsonl", "manifest.parquet"):
        path = grid_root / name
        frame = (
            pd.read_json(path, lines=True)
            if path.suffix == ".jsonl"
            else pd.read_parquet(path)
        )
        frame.loc[0, "run_id"] = "block99"
        if path.suffix == ".jsonl":
            frame.to_json(path, orient="records", lines=True)
        else:
            frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="acquisition block|factorial class grid"):
        run_stage0(grid_root, _plan())

    preprocess_root = _asset(tmp_path / "preprocess")
    preprocess_path = preprocess_root / "preprocess_config.yaml"
    preprocess = yaml.safe_load(preprocess_path.read_text(encoding="utf-8"))
    preprocess["target_sfreq"] = 199.0
    preprocess_path.write_text(yaml.safe_dump(preprocess), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocess_config"):
        run_stage0(preprocess_root, _plan())


def test_stage0_outputs_are_hashed_and_one_shot_on_any_existing_root(tmp_path) -> None:
    result = run_stage0(_asset(tmp_path), _plan())
    output = tmp_path / "stage0"

    receipt_path = write_stage0_outputs(result, output)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for name, contract in receipt["artifacts"].items():
        assert contract["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="one-shot"):
        write_stage0_outputs(result, output)

    empty_output = tmp_path / "empty-stage0"
    empty_output.mkdir()
    with pytest.raises(FileExistsError, match="one-shot"):
        write_stage0_outputs(result, empty_output)


def test_stage0_execution_reservation_is_exclusive(tmp_path) -> None:
    output = tmp_path / "stage0"
    reservation = _reserve_stage0_execution(output)

    with pytest.raises(FileExistsError, match="already reserved"):
        _reserve_stage0_execution(output)

    _release_stage0_execution(reservation)
    assert not reservation.exists()


def test_stage0_output_recomputation_rejects_forged_receipt(tmp_path) -> None:
    result = run_stage0(_asset(tmp_path), _plan())
    output = tmp_path / "stage0"
    receipt_path = write_stage0_outputs(result, output)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "passed" if receipt["status"] != "passed" else "failed"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt field 'status'"):
        validate_stage0_output_tree(result, output)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = result["receipt"]["status"]
    receipt["artifacts"] = {}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact inventory"):
        validate_stage0_output_tree(result, output)


def test_stage0_output_recomputation_rejects_provenance_or_tree_drift(tmp_path) -> None:
    result = run_stage0(_asset(tmp_path), _plan())
    output = tmp_path / "stage0"
    receipt_path = write_stage0_outputs(result, output)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_tree_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="source_tree_sha256"):
        validate_stage0_output_tree(result, output)

    receipt_path.write_text(
        json.dumps({**result["receipt"], "artifacts": receipt["artifacts"]}),
        encoding="utf-8",
    )
    (output / "unexpected").mkdir()
    with pytest.raises(ValueError, match="missing or unexpected"):
        validate_stage0_output_tree(result, output)


def test_stage0_rehashes_input_asset_before_publication(tmp_path) -> None:
    asset = _asset(tmp_path)
    result = run_stage0(asset, _plan())
    asset_info = asset / "asset_info.json"
    asset_info.write_bytes(asset_info.read_bytes() + b"\n")
    output = tmp_path / "stage0"

    with pytest.raises(ValueError, match="input asset changed"):
        write_stage0_outputs(result, output)

    assert not output.exists()
