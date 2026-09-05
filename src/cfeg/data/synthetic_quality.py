from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfeg.data.schema import load_manifest

QUALITY_TRUTH_FILENAME = "quality_truth.jsonl"
QUALITY_TRUTH_SCHEMA = "cfeg.synthetic-quality-truth.v1"
SYNTHETIC_QUALITY_ASSET_FILES = (
    "asset_info.json",
    "class_map.json",
    "manifest.jsonl",
    "manifest.parquet",
    "preprocess_config.yaml",
    "quality_target_spec.json",
    "quality_truth.jsonl",
    "signals.h5",
)
QUALITY_TRUTH_COLUMNS = [
    "sample_id",
    "canonical_channel_id",
    "acquisition_block_id",
    "latent_badness",
    "clean_gain",
    "injected_noise_std",
    "injected_mains_amplitude",
    "dropout_fraction",
    "artifact_loading",
    "oracle_signal_fraction",
    "clean_signal_power",
    "realized_signal_fraction",
    "realized_clean_signal_mse",
]
QUALITY_TARGET_COLUMNS = QUALITY_TRUTH_COLUMNS[3:]
FORBIDDEN_QUALITY_TRUTH_COLUMNS = {
    "label",
    "class_id",
    "stimulus_frequency_hz",
    "stimulus_phase_rad",
}


def load_synthetic_quality_truth(processed_dir: str | Path) -> pd.DataFrame:
    """Load and integrity-check simulation truth outside the model data path."""

    root = Path(processed_dir)
    info_path = root / "asset_info.json"
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    if info.get("dataset_revision") != "synthetic_quality_v1":
        raise ValueError("Quality truth is available only for synthetic_quality_v1.")
    artifacts = info.get("analysis_only_artifacts") or {}
    if not isinstance(artifacts, dict) or set(artifacts) != {
        QUALITY_TRUTH_FILENAME,
        "quality_target_spec.json",
    }:
        raise TypeError("asset_info.json must bind the exact two analysis-only artifacts.")
    for name, contract in artifacts.items():
        if Path(name).name != name or not isinstance(contract, dict):
            raise TypeError("Synthetic analysis-only artifact contracts must be simple files.")
        artifact_path = root / name
        observed_artifact_sha256 = (
            _sha256_file(artifact_path) if artifact_path.is_file() else None
        )
        if (
            name == QUALITY_TRUTH_FILENAME
            and contract.get("sha256") != observed_artifact_sha256
        ):
            raise ValueError("quality_truth.jsonl digest differs from asset_info.json.")
        if (
            contract.get("schema") != QUALITY_TRUTH_SCHEMA
            or contract.get("model_input") is not False
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
            or contract.get("sha256") != observed_artifact_sha256
        ):
            raise ValueError("Synthetic quality-truth access contract is invalid.")
    contract = artifacts[QUALITY_TRUTH_FILENAME]
    truth_path = root / QUALITY_TRUTH_FILENAME
    observed_sha256 = _sha256_file(truth_path)
    if contract.get("sha256") != observed_sha256:
        raise ValueError("quality_truth.jsonl digest differs from asset_info.json.")

    if info.get("quality_target_spec") != "quality_target_spec.json":
        raise ValueError("Synthetic quality-target specification path must be canonical.")
    spec_path = root / "quality_target_spec.json"
    with spec_path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if (
        spec.get("schema") != QUALITY_TRUTH_SCHEMA
        or spec.get("quality_truth_sha256") != observed_sha256
        or spec.get("model_input") is not False
        or spec.get("scope") != "simulation_engineering_only"
        or spec.get("row_unit") != "sample_x_canonical_channel"
        or spec.get("quality_mechanism_label_effect")
        != "none_balanced_by_complete_block"
        or spec.get("trial_rng_key_includes_label") is not True
        or spec.get("targets") != QUALITY_TARGET_COLUMNS
    ):
        raise ValueError("Synthetic quality-target specification is stale or invalid.")

    truth = pd.read_json(truth_path, lines=True)
    missing = sorted(set(QUALITY_TRUTH_COLUMNS) - set(truth.columns))
    extra = sorted(set(truth.columns) - set(QUALITY_TRUTH_COLUMNS))
    forbidden = sorted(FORBIDDEN_QUALITY_TRUTH_COLUMNS & set(truth.columns))
    if missing or extra or forbidden or list(truth.columns) != QUALITY_TRUTH_COLUMNS:
        raise ValueError(
            "Synthetic quality truth schema mismatch: "
            f"missing={missing}, extra={extra}, forbidden={forbidden}."
        )
    if truth.duplicated(["sample_id", "canonical_channel_id"]).any():
        raise ValueError("Synthetic quality truth sample/channel keys must be unique.")
    numeric = truth[["canonical_channel_id", *QUALITY_TARGET_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Synthetic quality truth contains non-finite values.")
    manifest = load_manifest(root)
    manifest_ids = set(manifest["sample_id"].astype(str))
    if set(truth["sample_id"].astype(str)) != manifest_ids:
        raise ValueError("Synthetic quality truth and manifest sample sets differ.")
    manifest_blocks = manifest.set_index("sample_id")["acquisition_block_id"].astype(str)
    expected_blocks = truth["sample_id"].astype(str).map(manifest_blocks)
    if not np.array_equal(
        expected_blocks.to_numpy(str),
        truth["acquisition_block_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Synthetic quality truth block IDs differ from the manifest.")
    manifest_ids_by_sample = manifest.set_index("sample_id")["canonical_channel_ids"]
    expected_keys = {
        (str(sample_id), int(channel_id))
        for sample_id, values in manifest_ids_by_sample.items()
        for channel_id in np.asarray(values, dtype=int)
        if int(channel_id) > 0
    }
    observed_keys = set(
        zip(
            truth["sample_id"].astype(str),
            truth["canonical_channel_id"].astype(int),
        )
    )
    if observed_keys != expected_keys:
        raise ValueError("Synthetic quality truth channel keys differ from active manifest slots.")
    bounded = (
        "latent_badness",
        "clean_gain",
        "dropout_fraction",
        "oracle_signal_fraction",
        "realized_signal_fraction",
    )
    if any(not truth[name].between(0.0, 1.0).all() for name in bounded):
        raise ValueError("Synthetic bounded quality targets must lie in [0,1].")
    nonnegative = (
        "injected_noise_std",
        "injected_mains_amplitude",
        "artifact_loading",
        "clean_signal_power",
        "realized_clean_signal_mse",
    )
    if any((truth[name] < 0.0).any() for name in nonnegative):
        raise ValueError("Synthetic power/noise quality targets must be non-negative.")
    reconstructed = truth["clean_signal_power"] / (
        truth["clean_signal_power"] + truth["realized_clean_signal_mse"] + 1e-8
    )
    if not np.allclose(
        reconstructed,
        truth["realized_signal_fraction"],
        rtol=2e-7,
        atol=1e-9,
    ):
        raise ValueError("Synthetic realized-signal target violates its frozen formula.")
    return truth


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
