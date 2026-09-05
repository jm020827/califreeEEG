from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from cfeg.metadata_calibration_checkpoint import (
    _publish_file_noreplace,
    capture_stage1_freeze,
    verify_full_composite_checkpoint,
    write_full_composite_checkpoint,
)
from cfeg.models.backbones.spectral_transformer import SpectralEEGTransformerBackbone
from cfeg.models.metadata_calibration_composite import MetadataCalibrationComposite
from cfeg.models.metadata_calibration_prior import BoundedMetadataCalibrationPrior


def _model() -> MetadataCalibrationComposite:
    backbone = SpectralEEGTransformerBackbone(
        c_max=64,
        t_len=400,
        target_sfreq=200.0,
        d_model=16,
        depth=1,
        n_heads=4,
        dropout=0.0,
    )
    prior = BoundedMetadataCalibrationPrior(
        embedding_dim=16,
        n_classes=12,
        q_feature_dim=56,
        metadata_feature_dim=18,
        hidden_dim=8,
    )
    return MetadataCalibrationComposite(backbone=backbone, prior=prior)


def _write(model: MetadataCalibrationComposite, tmp_path: Path, freeze):
    return write_full_composite_checkpoint(
        model,
        checkpoint_path=tmp_path / "composite.safetensors",
        receipt_path=tmp_path / "composite.receipt.json",
        phase="source_development",
        seed=42,
        outer_fold=0,
        stage1_freeze=freeze,
        stage1_state_sha256="1" * 64,
        stage1_recipe_sha256="2" * 64,
        stage2_recipe_sha256="3" * 64,
        plan_sha256="4" * 64,
        decision_receipt_sha256="5" * 64,
        asset_receipt_sha256="6" * 64,
        asset_fingerprint_bundle_sha256="7" * 64,
        class_map_sha256="8" * 64,
        source_commit="9" * 40,
        source_tree_sha256="a" * 64,
        source_tag="metadata-calibration-freeze-test",
    )


def test_full_checkpoint_binds_stage1_common_state_and_only_m_trainables(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = _model()
    freeze = capture_stage1_freeze(model)
    model.freeze_common_path_for_stage2()
    with torch.no_grad():
        model.prior.metadata_precision_encoder[-1].bias.add_(0.1)

    receipt = _write(model, tmp_path, freeze)
    verified = verify_full_composite_checkpoint(
        tmp_path / "composite.safetensors", tmp_path / "composite.receipt.json"
    )
    assert verified == receipt
    assert receipt["full_model_state"] is True
    assert receipt["trainable_only_checkpoint"] is False
    assert receipt["stage1_common_state_sha256"] == receipt["final_common_state_sha256"]
    assert receipt["initial_metadata_state_sha256"] != receipt["final_metadata_state_sha256"]
    assert all(
        name.startswith("prior.metadata_precision_encoder.")
        for name in receipt["stage2_trainable_names"]
    )
    checkpoint = load_file(tmp_path / "composite.safetensors", device="cpu")
    assert any(name.startswith("backbone.") for name in checkpoint)
    assert any(name.startswith("prior.") for name in checkpoint)


def test_writer_rejects_common_path_drift_and_existing_targets(tmp_path: Path) -> None:
    model = _model()
    freeze = capture_stage1_freeze(model)
    model.freeze_common_path_for_stage2()
    with torch.no_grad():
        model.prior.source_anchor.add_(0.01)
    with pytest.raises(ValueError, match="Common backbone/Q/anchor state changed"):
        _write(model, tmp_path, freeze)

    clean = _model()
    clean_freeze = capture_stage1_freeze(clean)
    clean.freeze_common_path_for_stage2()
    _write(clean, tmp_path, clean_freeze)
    with pytest.raises(FileExistsError, match="must be new"):
        _write(clean, tmp_path, clean_freeze)


def test_verifier_rejects_receipt_tampering(tmp_path: Path) -> None:
    model = _model()
    freeze = capture_stage1_freeze(model)
    model.freeze_common_path_for_stage2()
    _write(model, tmp_path, freeze)
    receipt_path = tmp_path / "composite.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["seed"] = 99
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt payload hash"):
        verify_full_composite_checkpoint(tmp_path / "composite.safetensors", receipt_path)


def test_checkpoint_file_publication_never_replaces_existing_target(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.write_bytes(b"new")
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="destination already exists"):
        _publish_file_noreplace(staging, destination)

    assert staging.read_bytes() == b"new"
    assert destination.read_bytes() == b"existing"
