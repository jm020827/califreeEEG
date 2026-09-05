from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from cfeg.data import ssvep_synthetic
from cfeg.data.collate import collate_eeg
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.schema import load_manifest
from cfeg.data.ssvep_synthetic import generate_quality_synthetic_processed
from cfeg.data.synthetic_quality import load_synthetic_quality_truth
from cfeg.train_loop import _processed_asset_provenance


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_quality_synthetic_publication_is_fresh_and_atomic(tmp_path, monkeypatch) -> None:
    failed_output = tmp_path / "failed"

    def fail_after_partial_write(out_dir, **_kwargs):
        (out_dir / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected generation failure")

    monkeypatch.setattr(
        ssvep_synthetic,
        "_generate_quality_synthetic_processed_into",
        fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="injected generation failure"):
        ssvep_synthetic.generate_quality_synthetic_processed(failed_output)
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))
    assert not (tmp_path / ".failed.generation-reservation").exists()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        ssvep_synthetic.generate_quality_synthetic_processed(occupied)


def test_quality_synthetic_is_block_balanced_and_targets_stay_out_of_model(tmp_path) -> None:
    out_dir = tmp_path / "synthetic_quality_v1"
    result = generate_quality_synthetic_processed(
        out_dir,
        n_subjects=3,
        n_blocks_per_interface=1,
        n_repetitions_per_class_per_block=1,
        n_classes=3,
        duration_sec=0.5,
        seed=5,
    )
    assert result["n_samples"] == 18
    assert result["n_quality_rows"] == 18 * 8

    manifest = load_manifest(out_dir)
    assert not {
        "injected_noise_std",
        "injected_mains_amplitude",
        "dropout_fraction",
        "realized_clean_signal_mse",
    } & set(manifest.columns)
    grouped_labels = manifest.groupby("acquisition_block_id")["label"].apply(
        lambda values: sorted(values.tolist())
    )
    assert all(labels == [0, 1, 2] for labels in grouped_labels)
    balance = manifest.groupby(["electrode_type", "label"]).size().unstack()
    assert balance.nunique(axis=1).eq(1).all()
    assert manifest.groupby("acquisition_block_id")["metadata_measurement_id"].nunique().eq(1).all()
    assert manifest.groupby("acquisition_block_id")["impedance_mean_kohm"].nunique().eq(1).all()

    quality = load_synthetic_quality_truth(out_dir)
    assert set(quality["sample_id"]) == set(manifest["sample_id"])
    assert quality.groupby("sample_id").size().eq(8).all()
    assert not {"label", "stimulus_frequency_hz", "stimulus_phase_rad"} & set(
        quality.columns
    )
    spec = json.loads((out_dir / "quality_target_spec.json").read_text(encoding="utf-8"))
    assert spec["quality_mechanism_label_effect"] == "none_balanced_by_complete_block"
    assert spec["trial_rng_key_includes_label"] is True
    assert spec["model_input"] is False
    assert spec["quality_truth_sha256"] == _sha256(out_dir / "quality_truth.jsonl")
    assert quality["realized_signal_fraction"].between(0.0, 1.0).all()
    np.testing.assert_allclose(
        quality["realized_signal_fraction"],
        quality["clean_signal_power"]
        / (
            quality["clean_signal_power"]
            + quality["realized_clean_signal_mse"]
            + 1e-8
        ),
        rtol=2e-7,
    )
    provenance = _processed_asset_provenance([out_dir])
    fingerprints = provenance["datasets"][0]["analysis_only_artifact_fingerprints"]
    assert set(fingerprints) == {"quality_target_spec.json", "quality_truth.jsonl"}

    dataset = EEGProcessedDataset(
        [out_dir],
        expected_revisions={"synthetic_quality": "synthetic_quality_v1"},
        expected_protocol={
            "metadata_contract_version": "0.4-dev",
            "query_qc_extractor_version": (
                "filtered_cropped_pre_zscore_channel_std_median_v1"
            ),
            "external_continuous_schema": "impedance_mean_max_v1",
        },
    )
    batch = collate_eeg(
        [dataset[0], dataset[1]],
        metadata_contract_version="0.4-dev",
        external_metadata_mode="observed",
    )
    assert set(batch) == {"x", "y", "sample_id", "split_meta", "cond"}
    assert "quality_targets" not in batch["cond"]
    assert (~batch["cond"]["channel_impedance_missing"]).sum(dim=-1).eq(8).all()


def test_injected_noise_has_a_positive_impedance_mechanism(tmp_path) -> None:
    out_dir = tmp_path / "synthetic_quality_v1"
    generate_quality_synthetic_processed(
        out_dir,
        n_subjects=4,
        n_blocks_per_interface=2,
        n_repetitions_per_class_per_block=1,
        n_classes=2,
        duration_sec=0.5,
        seed=9,
    )
    manifest = load_manifest(out_dir).set_index("sample_id")
    quality = load_synthetic_quality_truth(out_dir)
    impedances = []
    injected_noise = []
    for row in quality.itertuples(index=False):
        vector = manifest.loc[row.sample_id, "impedance_kohm_by_channel"]
        active_ids = np.asarray(manifest.loc[row.sample_id, "canonical_channel_ids"])
        slot = int(np.flatnonzero(active_ids == row.canonical_channel_id)[0])
        impedances.append(float(vector[slot]))
        injected_noise.append(float(row.injected_noise_std))
    correlation = np.corrcoef(np.log1p(impedances), injected_noise)[0, 1]
    assert correlation > 0.5


def test_quality_metadata_and_truth_rng_are_independent_of_class_count(tmp_path) -> None:
    three = tmp_path / "three"
    four = tmp_path / "four"
    common = {
        "n_subjects": 3,
        "n_blocks_per_interface": 1,
        "n_repetitions_per_class_per_block": 1,
        "duration_sec": 0.5,
        "seed": 17,
    }
    generate_quality_synthetic_processed(three, n_classes=3, **common)
    generate_quality_synthetic_processed(four, n_classes=4, **common)
    manifest_three = load_manifest(three)
    manifest_four = load_manifest(four)
    bundle_columns = [
        "acquisition_block_id",
        "electrode_type",
        "impedance_mean_kohm",
        "impedance_max_kohm",
        "metadata_measurement_time_sec",
    ]
    block_three = manifest_three[bundle_columns].drop_duplicates().sort_values(
        "acquisition_block_id"
    ).reset_index(drop=True)
    block_four = manifest_four[bundle_columns].drop_duplicates().sort_values(
        "acquisition_block_id"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(block_three, block_four)

    truth_three = load_synthetic_quality_truth(three).sort_values(
        ["sample_id", "canonical_channel_id"]
    )
    truth_four = load_synthetic_quality_truth(four)
    truth_four = truth_four.loc[truth_four["sample_id"].isin(truth_three["sample_id"])]
    truth_four = truth_four.sort_values(["sample_id", "canonical_channel_id"])
    pd.testing.assert_frame_equal(
        truth_three.reset_index(drop=True),
        truth_four.reset_index(drop=True),
    )


def test_quality_truth_loader_rejects_tampering(tmp_path) -> None:
    out_dir = tmp_path / "synthetic_quality_v1"
    generate_quality_synthetic_processed(
        out_dir,
        n_subjects=3,
        n_blocks_per_interface=1,
        n_repetitions_per_class_per_block=1,
        n_classes=2,
        duration_sec=0.5,
        seed=21,
    )
    truth_path = out_dir / "quality_truth.jsonl"
    truth_path.write_bytes(truth_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="digest differs"):
        load_synthetic_quality_truth(out_dir)
    with pytest.raises(ValueError, match="digest mismatch"):
        _processed_asset_provenance([out_dir])


def test_quality_truth_loader_rejects_noncanonical_spec_path(tmp_path) -> None:
    out_dir = tmp_path / "synthetic_quality_v1"
    generate_quality_synthetic_processed(
        out_dir,
        n_subjects=3,
        n_blocks_per_interface=1,
        n_repetitions_per_class_per_block=1,
        n_classes=2,
        duration_sec=0.5,
        seed=21,
    )
    info_path = out_dir / "asset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["quality_target_spec"] = "../quality_target_spec.json"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="path must be canonical"):
        load_synthetic_quality_truth(out_dir)
