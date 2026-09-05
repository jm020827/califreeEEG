from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

import cfeg.data.choi2019_partition as partition_module
from cfeg.data.choi2019_partition import (
    construct_partition,
    verify_governance_contract,
    write_partition_bundle_atomically,
)
from cfeg.data.schema import REQUIRED_MANIFEST_COLUMNS

_ROOT = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict]:
    data = yaml.safe_load((_ROOT / "configs/data/choi2019.yaml").read_text(encoding="utf-8"))
    anchor = yaml.safe_load(
        (_ROOT / "configs/baselines/fbcca_choi2019_bandwise_v1.yaml").read_text(encoding="utf-8")
    )
    data = copy.deepcopy(data)
    data["expected"]["n_subjects"] = 1
    data["expected"]["n_trials_total"] = 480
    return data, anchor


def _prepared_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    data, anchor = _configs()
    rows = []
    h5_index = 0
    channel_ids = [0] * 64
    for channel_id in anchor["evaluation_view"]["evaluation_canonical_channel_ids"]:
        channel_ids[channel_id - 1] = channel_id
    for day in (1, 2):
        for band_index, band in enumerate(data["processing_band_order"]):
            frequencies = data["stimulus_by_band"][band]["frequencies_hz"]
            for session in (1, 2):
                for run_trial in range(40):
                    within_label = run_trial % 4
                    row = {column: None for column in REQUIRED_MANIFEST_COLUMNS}
                    row.update(
                        {
                            "sample_id": (
                                f"choi2019_sub001_day{day:02d}_{band.lower()}_"
                                f"session{session:02d}_trial{run_trial + 1:03d}"
                            ),
                            "h5_index": h5_index,
                            "dataset_id": "choi2019",
                            "subject_id": "sub001",
                            "session_id": f"day{day:02d}",
                            "run_id": f"{band.lower()}_session{session:02d}",
                            "trial_id": f"trial{run_trial + 1:03d}",
                            "label": band_index * 4 + within_label,
                            "stimulus_frequency_hz": frequencies[within_label],
                            "stimulus_phase_rad": None,
                            "sfreq_original": 200.0,
                            "sfreq_processed": 200.0,
                            "window_start_sec": 0.14,
                            "window_duration_sec": 2.0,
                            "reference": "fcz",
                            "hardware_id": "brain_products_brainamp",
                            "cap_type": "unknown",
                            "electrode_type": "active_contact_medium_unreported",
                            "n_channels_original": 33,
                            "n_channels_used": 33,
                            "channel_names_original": data["channel_names"],
                            "channel_names_used": data["channel_names"],
                            "canonical_channel_ids": channel_ids,
                            "environment_note_code": "unknown",
                            "source_file": "fixture.mat",
                            "day_index": day,
                            "frequency_band": band.lower(),
                            "within_band_label": within_label,
                            "source_marker_sample_index_1based": 100 + run_trial * 200,
                            "source_marker_event_index": run_trial * 2,
                        }
                    )
                    rows.append(row)
                    h5_index += 1
    manifest = pd.DataFrame(rows)
    manifest.to_json(tmp_path / "manifest.jsonl", orient="records", lines=True)
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    masks = np.ones((len(rows), 64), dtype=bool)
    with h5py.File(tmp_path / "signals.h5", "w") as handle:
        handle.create_dataset("channel_mask", data=masks)
    return tmp_path, data, anchor


def test_partition_is_deterministic_nested_disjoint_and_query_immutable(tmp_path: Path) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path)
    first = construct_partition(processed, data, anchor)
    second = construct_partition(processed, data, anchor)

    assert first.assignment_bytes == second.assignment_bytes
    assert first.summary["participant_band_groups"] == 3
    assert first.summary["assignment_records"] == 1068
    assert first.summary["sample_identity_records"] == 480
    assert first.summary["contains_scores_predictions_or_decoding_outcomes"] is False
    assert first.summary["evaluation_channels"]["all_signal_masks_true"] is True

    assignments = pd.DataFrame(first.assignments)
    group = assignments.loc[
        (assignments["subject_id"] == "sub001") & (assignments["frequency_band"] == "LOW")
    ]
    queries = []
    prior_support: set[str] = set()
    for budget in (0, 1, 3, 5):
        selected = group.loc[group["budget_trials_per_class"] == budget]
        support = selected.loc[selected["role"] == "support"]
        query_ids = set(selected.loc[selected["role"] == "query", "sample_id"])
        assert len(support) == 4 * budget
        assert support.groupby("within_band_label").size().to_dict() == (
            {} if budget == 0 else {label: budget for label in range(4)}
        )
        assert prior_support <= set(support["sample_id"])
        assert not set(support["sample_id"]) & query_ids
        prior_support = set(support["sample_id"])
        queries.append(query_ids)
    assert all(query_ids == queries[0] for query_ids in queries)


def test_partition_rejects_a_masked_frozen_evaluation_channel(tmp_path: Path) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path)
    channel_id = anchor["evaluation_view"]["evaluation_canonical_channel_ids"][0]
    with h5py.File(processed / "signals.h5", "r+") as handle:
        handle["channel_mask"][0, channel_id - 1] = False

    with pytest.raises(ValueError, match="masks a frozen evaluation channel"):
        construct_partition(processed, data, anchor)


def test_partition_rejects_missing_or_reordered_day1_marker_grid(tmp_path: Path) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path)
    manifest = pd.read_json(processed / "manifest.jsonl", lines=True)
    manifest.loc[0, "source_marker_event_index"] = 3
    manifest.to_json(processed / "manifest.jsonl", orient="records", lines=True)
    manifest.to_parquet(processed / "manifest.parquet", index=False)

    with pytest.raises(ValueError, match="run marker grid drift"):
        construct_partition(processed, data, anchor)


def test_partition_rejects_day_index_session_mismatch(tmp_path: Path) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path)
    manifest = pd.read_json(processed / "manifest.jsonl", lines=True)
    manifest.loc[0, "day_index"] = 2
    manifest.to_json(processed / "manifest.jsonl", orient="records", lines=True)
    manifest.to_parquet(processed / "manifest.parquet", index=False)

    with pytest.raises(ValueError, match="day_index/session_id correspondence drift"):
        construct_partition(processed, data, anchor)


def test_partition_rejects_implicit_or_bit_drifted_subband_weights(tmp_path: Path) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path)
    anchor["common_parameters"]["subband_weights"][1] = np.nextafter(
        anchor["common_parameters"]["subband_weights"][1], np.inf
    ).item()
    with pytest.raises(ValueError, match="differ bitwise"):
        construct_partition(processed, data, anchor)

    _, _, anchor = _prepared_fixture(tmp_path / "implicit")
    del anchor["common_parameters"]["subband_weights"]
    with pytest.raises(ValueError, match="one-dimensional vector"):
        construct_partition(tmp_path / "implicit", data, anchor)

    _, _, anchor = _prepared_fixture(tmp_path / "band_override")
    anchor["band_parameters"]["MID"]["subband_weights"] = list(
        anchor["common_parameters"]["subband_weights"]
    )
    anchor["band_parameters"]["MID"]["subband_weights"][2] = np.nextafter(
        anchor["band_parameters"]["MID"]["subband_weights"][2], -np.inf
    ).item()
    with pytest.raises(ValueError, match="MID resolved weights differ bitwise"):
        construct_partition(tmp_path / "band_override", data, anchor)


def _contract(path: Path) -> dict[str, int | str]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_governance_binds_complete_processed_and_repository_inventories(tmp_path: Path) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path / "processed")
    plan = construct_partition(processed, data, anchor)
    for name in (
        "asset_info.json",
        "class_map.json",
        "manifest.parquet",
        "preprocess_config.yaml",
        "questionnaire_normalized.csv",
    ):
        (processed / name).write_bytes(f"frozen-{name}".encode())

    repository = tmp_path / "repository"
    governed_contracts = (
        "configs/baselines/fbcca_choi2019_bandwise_v1.yaml",
        "configs/canonical_channels.yaml",
        "configs/data/choi2019.yaml",
    )
    for relative in governed_contracts:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen-{relative}".encode())
    partition_fields = (
        "assignment_records",
        "assignment_bytes",
        "partition_assignment_sha256",
        "sample_identity_records",
        "sample_identity_sha256",
        "group_query_identity_sha256",
    )
    governance = {
        "schema": "cfeg.choi2019-processed-partition-governance.v1",
        "freeze_status": "frozen_pre_outcome",
        "dataset_revision": data["dataset_revision"],
        "role": "immutable_processed_asset_and_partition_identity_contract",
        "contains_scores_predictions_or_decoding_outcomes": False,
        "processed_assets": {
            name: _contract(processed / name)
            for name in (
                "asset_info.json",
                "class_map.json",
                "manifest.jsonl",
                "manifest.parquet",
                "preprocess_config.yaml",
                "questionnaire_normalized.csv",
                "signals.h5",
            )
        },
        "repository_contracts": {
            relative: _contract(repository / relative) for relative in governed_contracts
        },
        "partition": {name: plan.summary[name] for name in partition_fields},
        "evaluation_channels": plan.summary["evaluation_channels"],
        "filterbank_weight_contract": plan.summary["filterbank_weight_contract"],
    }
    receipt = verify_governance_contract(processed, repository, governance, plan)
    assert receipt["status"] == "verified_without_decoding_outcomes"

    (repository / governed_contracts[0]).write_bytes(b"post-freeze drift")
    with pytest.raises(ValueError, match="Governed Choi file drift"):
        verify_governance_contract(processed, repository, governance, plan)

    incomplete = copy.deepcopy(governance)
    del incomplete["processed_assets"]["signals.h5"]
    with pytest.raises(ValueError, match="processed-asset inventory drift"):
        verify_governance_contract(processed, repository, incomplete, plan)


def test_partition_bundle_is_atomic_and_uses_stable_config_label(tmp_path: Path) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path / "processed")
    plan = construct_partition(processed, data, anchor)
    config = tmp_path / "governance.yaml"
    config.write_text("frozen: true\n", encoding="utf-8")
    destination = tmp_path / "partition"
    receipt = write_partition_bundle_atomically(
        destination,
        plan,
        {"status": "verified_without_decoding_outcomes"},
        config,
        governance_config_label="configs/data/frozen.yaml",
    )

    persisted = json.loads((destination / "governance_receipt.json").read_text())
    assert persisted == receipt
    assert persisted["governance_config"]["path"] == "configs/data/frozen.yaml"
    assert (
        persisted["partition_artifact"]["sha256"] == (plan.summary["partition_assignment_sha256"])
    )
    detached = (destination / "governance_receipt.sha256").read_text().split()[0]
    assert (
        detached
        == hashlib.sha256((destination / "governance_receipt.json").read_bytes()).hexdigest()
    )
    for name in (
        "partition_assignments.jsonl",
        "governance_receipt.json",
        "governance_receipt.sha256",
    ):
        assert stat.S_IMODE((destination / name).stat().st_mode) == 0o400
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_partition_bundle_atomically(
            destination,
            plan,
            {},
            config,
            governance_config_label="configs/data/frozen.yaml",
        )


def test_partition_paths_reject_symlink_traversal(tmp_path: Path) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path / "processed")
    processed_alias = tmp_path / "processed_alias"
    processed_alias.symlink_to(processed, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink path component"):
        construct_partition(processed_alias, data, anchor)

    plan = construct_partition(processed, data, anchor)
    config = tmp_path / "governance.yaml"
    config.write_text("frozen: true\n", encoding="utf-8")
    config_alias = tmp_path / "governance_alias.yaml"
    config_alias.symlink_to(config)
    with pytest.raises(ValueError, match="Symlink path component"):
        write_partition_bundle_atomically(
            tmp_path / "config_symlink_partition", plan, {}, config_alias
        )

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    parent_alias = tmp_path / "parent_alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink path component"):
        write_partition_bundle_atomically(parent_alias / "partition", plan, {}, config)


def test_partition_fsync_failure_preserves_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed, data, anchor = _prepared_fixture(tmp_path / "processed")
    plan = construct_partition(processed, data, anchor)
    config = tmp_path / "governance.yaml"
    config.write_text("frozen: true\n", encoding="utf-8")
    destination = tmp_path / "partition"

    def fail_fsync(_: Path) -> None:
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(partition_module, "_fsync_directory", fail_fsync)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        write_partition_bundle_atomically(destination, plan, {}, config)
    assert not destination.exists()
    staging = list(tmp_path.glob(".partition.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / "partition_assignments.jsonl").is_file()
