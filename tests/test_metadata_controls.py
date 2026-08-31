from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.data.collate import build_vocabularies, collate_eeg
from cfeg.data.metadata_controls import build_development_control_plan
from cfeg.data.schema import EEGSample
from cfeg.models.full_model import ConditionedEEGDecoder


def _sample(
    sample_id: str,
    *,
    electrode: str = "dry",
    impedance: float | None = 10.0,
    reference: str | None = "forehead",
    query: float = 20.0,
    value: float = 1.0,
) -> EEGSample:
    return EEGSample(
        x=np.full((4, 16), value, dtype=np.float32),
        y=0,
        sample_id=sample_id,
        dataset_id="wearable",
        subject_id="sub001",
        session_id=electrode,
        channel_mask=np.ones(4, dtype=bool),
        canonical_channel_ids=np.arange(1, 5, dtype=np.int64),
        sfreq=200.0,
        reference=reference,
        hardware_id="public_unknown",
        electrode_type=electrode,
        cap_type="wearable",
        n_channels_used=4,
        impedance_mean_kohm=impedance,
        impedance_max_kohm=None if impedance is None else impedance * 2,
        reattach_flag=None,
        time_since_last_session_hours=None,
        query_signal_std=query,
        query_signal_std_by_channel=np.full(4, query, dtype=np.float32),
        impedance_kohm_by_channel=(
            None
            if impedance is None
            else np.full(4, impedance, dtype=np.float32)
        ),
    )


def _manifest_for_shuffle() -> pd.DataFrame:
    rows = []
    for split_name, subject in (("train", "sub001"), ("val", "sub002")):
        for label in (0, 1):
            for replicate in (0, 1):
                impedance = 100.0 if replicate == 0 else 10.0
                rows.append(
                    {
                        "sample_id": f"{split_name}-{label}-{replicate}",
                        "dataset_id": "wearable",
                        "subject_id": subject,
                        "label": label,
                        "run_id": f"block{replicate + 1:02d}",
                        "electrode_type": "dry" if replicate == 0 else "wet",
                        "reference": "forehead",
                        "cap_type": "wearable",
                        "impedance_mean_kohm": impedance,
                        "impedance_max_kohm": impedance * 2,
                        "impedance_kohm_by_channel": [impedance],
                    }
                )
    return pd.DataFrame(rows)


def _counterfactual_manifest() -> pd.DataFrame:
    rows = []
    for label in (0, 1):
        for block in (1, 2):
            for electrode in ("dry", "wet"):
                rows.append(
                    {
                        "sample_id": f"sub001-{block}-{label}-{electrode}",
                        "dataset_id": "wearable",
                        "subject_id": "sub001",
                        "label": label,
                        "run_id": f"block{block:02d}",
                        "window_start_sec": 0.0,
                        "window_duration_sec": 2.0,
                        "electrode_type": electrode,
                        "reference": "forehead",
                        "cap_type": "wearable",
                        "impedance_mean_kohm": 100.0 if electrode == "dry" else 10.0,
                        "impedance_max_kohm": 200.0 if electrode == "dry" else 20.0,
                        "impedance_kohm_by_channel": (
                            [100.0, 120.0] if electrode == "dry" else [10.0, 12.0]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_within_class_shuffle_is_deterministic_split_local_and_label_preserving() -> None:
    manifest = _manifest_for_shuffle()
    split = {"train": np.arange(4), "val": np.arange(4, 8), "test": np.asarray([], dtype=int)}

    first = build_development_control_plan(
        manifest, split, control="within_class_shuffle", seed=42
    )
    second = build_development_control_plan(
        manifest, split, control="within_class_shuffle", seed=42
    )

    assert first.contract["donor_mapping_sha256"] == second.contract["donor_mapping_sha256"]
    mapping = first.donor_mapping
    assert not (mapping["sample_id"] == mapping["donor_sample_id"]).any()
    assert (mapping["target_label"] == mapping["donor_label"]).all()
    assert first.contract["pair_coverage"] == 1.0
    assert first.contract["effective_external_metadata_changed_fraction"] == 1.0
    assert first.contract["donor_scope"] == (
        "within_dataset_block_derangement_label_aligned"
    )
    for split_name in ("train", "val"):
        ids = set(manifest.iloc[split[split_name]]["sample_id"])
        selected = mapping[mapping["split"] == split_name]
        assert set(selected["sample_id"]) == ids
        assert set(selected["donor_sample_id"]) == ids
        # Both labels in one target acquisition block must use the same donor
        # acquisition block; row-wise metadata mosaics are invalid.
        target_to_donor = selected.groupby(
            ["target_subject_id", "target_electrode_type", "target_run_id"]
        )[["donor_subject_id", "donor_electrode_type", "donor_run_id"]]
        assert all(len(group.drop_duplicates()) == 1 for _, group in target_to_donor)


def test_counterfactual_pair_is_exact_opposite_electrode_involution() -> None:
    manifest = _counterfactual_manifest()
    plan = build_development_control_plan(
        manifest,
        {"val": np.arange(len(manifest))},
        control="counterfactual_wet_dry",
        seed=42,
    )
    mapping = plan.donor_mapping.set_index("sample_id")

    assert plan.contract["condition_flip_fraction"] == 1.0
    assert plan.contract["pair_coverage"] == 1.0
    assert plan.contract["donor_scope"] == (
        "same_dataset_subject_label_block_window_opposite_electrode"
    )
    for sample_id, row in mapping.iterrows():
        donor_id = row["donor_sample_id"]
        assert mapping.loc[donor_id, "donor_sample_id"] == sample_id
        assert row["target_subject_id"] == row["donor_subject_id"]
        assert row["target_label"] == row["donor_label"]
        assert row["target_electrode_type"] != row["donor_electrode_type"]


def test_counterfactual_rejects_incomplete_wet_dry_pair() -> None:
    manifest = _counterfactual_manifest().iloc[:-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="exactly one dry and one wet"):
        build_development_control_plan(
            manifest,
            {"val": np.arange(len(manifest))},
            control="counterfactual_wet_dry",
            seed=42,
        )


def test_metadata_only_removes_eeg_structure_and_query_but_keeps_external() -> None:
    sample = _sample("target", electrode="dry", impedance=100.0, query=99.0)
    vocab = build_vocabularies([sample])
    batch = collate_eeg(
        [sample],
        vocabularies=vocab,
        metadata_contract_version="0.4-dev",
        external_metadata_mode="observed",
        development_control="metadata_only",
    )

    assert batch["x"].eq(0).all()
    assert batch["cond"]["channel_ids"].eq(0).all()
    assert not batch["cond"]["channel_mask"].any()
    assert batch["cond"]["query_qc"].eq(0).all()
    assert batch["cond"]["query_qc_missing"].all()
    assert batch["cond"]["external_continuous"].ne(0).all()
    assert batch["cond"]["electrode_type"].item() != 0


def test_missingness_only_preserves_availability_but_removes_values() -> None:
    observed = _sample("observed", impedance=10.0, reference="forehead")
    missing = _sample("missing", impedance=None, reference=None)
    vocab = build_vocabularies([observed, missing])
    batch = collate_eeg(
        [observed, missing],
        vocabularies=vocab,
        metadata_contract_version="0.4-dev",
        external_metadata_mode="observed",
        development_control="missingness_only",
    )
    cond = batch["cond"]

    assert batch["x"].eq(0).all()
    assert cond["external_continuous"].eq(0).all()
    assert not cond["external_continuous_missing"][0].any()
    assert cond["external_continuous_missing"][1].all()
    assert cond["reference"][0].item() != 0
    assert cond["reference"][1].item() == 0


def test_donor_override_changes_only_external_bundle() -> None:
    target = _sample("target", electrode="dry", impedance=100.0, query=33.0, value=3.0)
    donor = _sample("donor", electrode="wet", impedance=10.0, query=99.0, value=9.0)
    vocab = build_vocabularies([target, donor])
    clean = collate_eeg(
        [target], vocabularies=vocab, metadata_contract_version="0.4-dev"
    )
    swapped = collate_eeg(
        [target],
        vocabularies=vocab,
        metadata_contract_version="0.4-dev",
        development_control="within_class_shuffle",
        external_metadata_overrides={"target": vars(donor)},
    )

    torch.testing.assert_close(swapped["x"], clean["x"])
    torch.testing.assert_close(swapped["y"], clean["y"])
    torch.testing.assert_close(swapped["cond"]["channel_ids"], clean["cond"]["channel_ids"])
    torch.testing.assert_close(swapped["cond"]["query_qc"], clean["cond"]["query_qc"])
    assert swapped["cond"]["electrode_type"].item() != clean["cond"]["electrode_type"].item()
    assert not torch.equal(
        swapped["cond"]["external_continuous"], clean["cond"]["external_continuous"]
    )


def test_shuffle_train_can_use_donor_metadata_then_evaluate_clean() -> None:
    target = _sample("target", electrode="dry", impedance=100.0, query=33.0, value=3.0)
    donor = _sample("donor", electrode="wet", impedance=10.0, query=99.0, value=9.0)
    vocab = build_vocabularies([target, donor])
    train_batch = collate_eeg(
        [target],
        vocabularies=vocab,
        metadata_contract_version="0.4-dev",
        development_control="none",
        external_metadata_overrides={"target": vars(donor)},
    )
    validation_batch = collate_eeg(
        [target],
        vocabularies=vocab,
        metadata_contract_version="0.4-dev",
        development_control="none",
    )
    cfg = {
        "protocol": {
            "metadata_contract_version": "0.4-dev",
            "development_control": "none",
        },
        "model": {
            "n_classes": 2,
            "c_max": 4,
            "t_len": 16,
            "d_model": 16,
            "patch_size": 4,
            "depth": 1,
            "n_heads": 4,
            "backbone": {"name": "tiny_transformer"},
            "conditioning": {
                "architecture": "physical_hybrid_v1",
                "hidden_dim": 8,
                "common_query_film": {"enabled": True},
                "external_global_film": {"enabled": True},
                "channel_quality": {
                    "enabled": True,
                    "source": "channel_impedance",
                    "placement": "pre_channel_embedding",
                    "max_gain_delta": 0.25,
                },
            },
            "condition_encoder": {
                "enabled": True,
                "n_prompt_tokens": 0,
                "fields": ["electrode_type"],
                "include_continuous": True,
                "include_channels": False,
                "external_metadata_mode": "observed",
            },
            "adapter": {"enabled": False},
            "latent": {"enabled": False},
        },
    }
    model = ConditionedEEGDecoder(
        cfg, vocab_sizes={field: len(values) for field, values in vocab.items()}
    ).eval()

    assert train_batch["cond"]["development_control"] == "none"
    assert validation_batch["cond"]["development_control"] == "none"
    assert not torch.equal(
        train_batch["cond"]["external_continuous"],
        validation_batch["cond"]["external_continuous"],
    )
    with torch.no_grad():
        model(train_batch["x"], train_batch["cond"])
        model(validation_batch["x"], validation_batch["cond"])


def test_declared_external_override_never_silently_falls_back() -> None:
    sample = _sample("target", electrode="dry")

    with pytest.raises(KeyError, match="no row"):
        collate_eeg(
            [sample],
            metadata_contract_version="0.4-dev",
            external_metadata_mode="observed",
            external_metadata_overrides={},
        )


def test_metadata_only_model_is_invariant_to_x_structure_and_query() -> None:
    sample = _sample("target", electrode="dry", impedance=100.0)
    vocab = build_vocabularies([sample])
    cfg = {
        "protocol": {
            "metadata_contract_version": "0.4-dev",
            "development_control": "metadata_only",
        },
        "model": {
            "n_classes": 2,
            "c_max": 4,
            "t_len": 16,
            "d_model": 16,
            "patch_size": 4,
            "depth": 1,
            "n_heads": 4,
            "backbone": {"name": "tiny_transformer"},
            "condition_encoder": {
                "enabled": True,
                "fields": ["reference", "electrode_type", "cap_type"],
                "include_channels": False,
                "external_metadata_mode": "observed",
            },
            "adapter": {"enabled": False},
            "latent": {"enabled": False},
        },
    }
    model = ConditionedEEGDecoder(
        cfg, vocab_sizes={field: len(values) for field, values in vocab.items()}
    ).eval()
    batch = collate_eeg(
        [sample],
        vocabularies=vocab,
        metadata_contract_version="0.4-dev",
        development_control="metadata_only",
    )
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch["cond"].items()}
    changed["query_qc"].fill_(2.0)
    changed["query_qc_missing"].fill_(False)
    changed["channel_ids"].fill_(3)
    changed["channel_mask"].fill_(True)

    with torch.no_grad():
        first = model(torch.randn_like(batch["x"]), batch["cond"]).logits
        second = model(torch.randn_like(batch["x"]), changed).logits
    torch.testing.assert_close(first, second)
