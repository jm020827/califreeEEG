from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.constants import METADATA_CONTRACT_V04_DEV
from cfeg.data.metadata_calibration_features import (
    WEARABLE_CALIBRATION_CHANNEL_IDS,
    build_wearable_metadata_features,
)
from cfeg.data.metadata_calibration_interventions import (
    apply_resolved_metadata_to_condition,
    build_metadata_intervention_mapping,
    metadata_calibration_base_input_sha256,
    metadata_intervention_mapping_sha256,
    resolve_metadata_intervention_batch,
    resolved_context_usage_sha256,
)
from cfeg.metadata_calibration_execution import resolve_cell_execution_spec


def _manifest() -> pd.DataFrame:
    rows = []
    for subject in ("sub004", "sub006"):
        for interface_index, interface in enumerate(("dry", "wet")):
            for block in range(1, 11):
                impedance = np.arange(8, dtype=np.float32) + block + 100 * interface_index
                rows.append(
                    {
                        "dataset_id": "wearable",
                        "subject_id": subject,
                        "electrode_type": interface,
                        "run_id": f"block{block:02d}",
                        "impedance_channel_ids": list(WEARABLE_CALIBRATION_CHANNEL_IDS),
                        "impedance_kohm_by_channel": impedance.copy(),
                    }
                )
    return pd.DataFrame(rows)


def _row_bindings(prefix: str, *identities: str) -> dict[str, tuple[str, ...]]:
    return {
        "row_tokens": tuple(
            f"{prefix}_{hashlib.sha256(identity.encode()).hexdigest()}"
            for identity in identities
        ),
        "base_input_sha256s": tuple(
            hashlib.sha256(f"input:{identity}".encode()).hexdigest()
            for identity in identities
        ),
    }


def _model_row_bindings(
    prefix: str,
    identity: str,
    x: torch.Tensor,
    channel_mask: torch.Tensor,
) -> dict[str, tuple[str, ...]]:
    values = _row_bindings(prefix, identity)
    values["base_input_sha256s"] = (
        metadata_calibration_base_input_sha256(x[0], channel_mask[0]),
    )
    return values


def test_correct_and_opposite_interface_maps_are_block_level_and_label_blind() -> None:
    manifest = _manifest()
    correct = build_metadata_intervention_mapping(manifest, context="correct")
    opposite = build_metadata_intervention_mapping(manifest, context="opposite_interface")

    assert len(correct.frame) == len(manifest)
    assert correct.frame["metadata_mode"].eq("observed").all()
    assert correct.frame["donor_run_id"].eq(correct.frame["run_id"]).all()
    assert correct.frame["donor_electrode_type"].eq(correct.frame["electrode_type"]).all()
    assert opposite.frame["donor_run_id"].eq(opposite.frame["run_id"]).all()
    assert opposite.frame["donor_electrode_type"].ne(opposite.frame["electrode_type"]).all()

    for forbidden in ("label", "sample_id", "trial_id"):
        with pytest.raises(ValueError, match="exact label-free allowlist"):
            build_metadata_intervention_mapping(
                manifest.assign(**{forbidden: "forbidden"}), context="correct"
            )


def test_block_shuffle_is_a_within_interface_derangement_and_never_crosses_partition() -> None:
    mapping = build_metadata_intervention_mapping(
        _manifest(), context="block_shuffle", shuffle_seed=17
    ).frame
    assert mapping["donor_subject_id"].eq(mapping["subject_id"]).all()
    assert mapping["donor_electrode_type"].eq(mapping["electrode_type"]).all()
    assert mapping["donor_run_id"].ne(mapping["run_id"]).all()
    support = mapping["run_id"].isin([f"block{index:02d}" for index in range(1, 6)])
    assert mapping.loc[support, "donor_run_id"].isin(
        [f"block{index:02d}" for index in range(1, 6)]
    ).all()
    assert mapping.loc[~support, "donor_run_id"].isin(
        [f"block{index:02d}" for index in range(6, 11)]
    ).all()
    per_block = mapping.groupby(["subject_id", "electrode_type", "run_id"])[
        "donor_run_id"
    ].nunique()
    assert per_block.eq(1).all()


def test_stale_and_null_contexts_have_explicit_missing_semantics() -> None:
    manifest = _manifest()
    stale = build_metadata_intervention_mapping(manifest, context="stale").frame
    first = stale["run_id"].eq("block01")
    assert stale.loc[first, "metadata_mode"].eq("all_missing").all()
    assert stale.loc[first, "donor_run_id"].isna().all()
    sixth = stale["run_id"].eq("block06")
    assert stale.loc[sixth, "donor_run_id"].eq("block05").all()

    missing = build_metadata_intervention_mapping(manifest, context="all_missing").frame
    off = build_metadata_intervention_mapping(manifest, context="exact_off").frame
    assert missing["metadata_mode"].eq("all_missing").all()
    assert off["metadata_mode"].eq("off").all()
    assert missing["donor_run_id"].isna().all()
    assert off["donor_run_id"].isna().all()


def test_mapping_digest_is_order_invariant_and_rejects_bad_block_context() -> None:
    manifest = _manifest()
    result = build_metadata_intervention_mapping(manifest, context="correct")
    shuffled = result.frame.sample(frac=1.0, random_state=8).reset_index(drop=True)
    assert metadata_intervention_mapping_sha256(shuffled) == result.sha256

    broken = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one row per block"):
        build_metadata_intervention_mapping(broken, context="correct")

    wrong_width = manifest.copy()
    wrong_width.at[0, "impedance_kohm_by_channel"] = np.full(7, 1.0)
    with pytest.raises(ValueError, match="eight channel"):
        build_metadata_intervention_mapping(wrong_width, context="correct")


def test_resolver_changes_both_interface_and_impedance_for_opposite_context() -> None:
    manifest = _manifest()
    channel_ids = torch.zeros((2, 64), dtype=torch.long)
    channel_ids[:, :8] = torch.tensor(WEARABLE_CALIBRATION_CHANNEL_IDS)
    channel_mask = channel_ids.gt(0)
    x = torch.zeros((2, 64, 400), dtype=torch.float32)
    x[1] = 1.0
    bindings = _row_bindings("q", "opposite-a", "opposite-b")
    bindings["base_input_sha256s"] = tuple(
        metadata_calibration_base_input_sha256(x[index], channel_mask[index])
        for index in range(2)
    )
    mapping = build_metadata_intervention_mapping(manifest, context="opposite_interface")
    spec = resolve_cell_execution_spec(
        role="A_QM",
        context="opposite_interface",
        phase="held_participant_evaluation",
    )
    resolved = resolve_metadata_intervention_batch(
        subject_ids=("sub004", "sub006"),
        electrode_types=("dry", "wet"),
        run_ids=("block06", "block07"),
        **bindings,
        mapping=mapping,
        sealed_block_context=manifest,
        cell_spec=spec,
        phase="held_participant_evaluation",
        batch_kind="query",
    )

    assert resolved.electrode_types == ("wet", "dry")
    expected_wet = np.arange(8, dtype=np.float32) + 106
    expected_dry = np.arange(8, dtype=np.float32) + 7
    expected = np.stack(
        [
            np.log1p(expected_wet) / np.log1p(100.0),
            np.log1p(expected_dry) / np.log1p(100.0),
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(resolved.impedance_by_channel, expected, rtol=2e-7)
    assert not resolved.impedance_missing_by_channel.any()
    assert len(resolved.resolved_context_values_sha256) == 64
    applied = apply_resolved_metadata_to_condition(
        x,
        {
            "channel_ids": channel_ids,
            "channel_mask": channel_mask,
            "metadata_contract_version": METADATA_CONTRACT_V04_DEV,
        },
        resolved,
    )
    values, missing = build_wearable_metadata_features(
        applied.condition,
        electrode_types=applied.electrode_types,
    )
    assert torch.equal(values[:, :2], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert torch.allclose(values[:, 2:10], torch.tensor(expected))
    assert not missing.any()


def test_resolved_context_digest_binds_values_and_all_missing_override() -> None:
    manifest = _manifest()
    mapping = build_metadata_intervention_mapping(manifest, context="correct")
    spec = resolve_cell_execution_spec(
        role="A_QM", context="correct", phase="held_participant_evaluation"
    )
    kwargs = {
        "subject_ids": ("sub004",),
        "electrode_types": ("dry",),
        "run_ids": ("block06",),
        **_row_bindings("q", "digest-row"),
        "mapping": mapping,
        "cell_spec": spec,
        "phase": "held_participant_evaluation",
        "batch_kind": "query",
    }
    original = resolve_metadata_intervention_batch(
        sealed_block_context=manifest,
        **kwargs,
    )
    changed_manifest = manifest.copy(deep=True)
    changed_manifest.at[5, "impedance_kohm_by_channel"] = np.full(8, 42.0)
    changed = resolve_metadata_intervention_batch(
        sealed_block_context=changed_manifest,
        **kwargs,
    )
    assert original.resolved_context_values_sha256 != changed.resolved_context_values_sha256

    support_only_spec = resolve_cell_execution_spec(
        role="A_QM", context="support_metadata_only", phase="source_development"
    )
    missing_query = resolve_metadata_intervention_batch(
        subject_ids=("sub004",),
        electrode_types=("dry",),
        run_ids=("block06",),
        **_row_bindings("q", "missing-query"),
        mapping=mapping,
        sealed_block_context=manifest,
        cell_spec=support_only_spec,
        phase="source_development",
        batch_kind="query",
    )
    assert missing_query.electrode_types == (None,)
    assert missing_query.metadata_modes == ("all_missing",)
    assert np.array_equal(missing_query.impedance_by_channel, np.zeros((1, 8)))
    assert missing_query.impedance_missing_by_channel.all()


def test_resolved_values_are_installed_at_official_channel_positions() -> None:
    manifest = _manifest()
    channel_ids = torch.zeros((1, 64), dtype=torch.long)
    channel_ids[0, :8] = torch.tensor(WEARABLE_CALIBRATION_CHANNEL_IDS)
    channel_mask = channel_ids.gt(0)
    x = torch.zeros((1, 64, 400), dtype=torch.float32)
    mapping = build_metadata_intervention_mapping(manifest, context="correct")
    spec = resolve_cell_execution_spec(
        role="A_QM", context="correct", phase="held_participant_evaluation"
    )
    resolved = resolve_metadata_intervention_batch(
        subject_ids=("sub004",),
        electrode_types=("dry",),
        run_ids=("block02",),
        **_model_row_bindings("s", "support-row", x, channel_mask),
        mapping=mapping,
        sealed_block_context=manifest,
        cell_spec=spec,
        phase="held_participant_evaluation",
        batch_kind="support",
    )
    cond = {
        "channel_ids": channel_ids,
        "channel_mask": channel_mask,
        "channel_impedance": torch.full_like(channel_ids, 9.0, dtype=torch.float32),
        "channel_impedance_missing": torch.zeros_like(channel_ids, dtype=torch.bool),
    }
    applied = apply_resolved_metadata_to_condition(x, cond, resolved)
    updated = applied.condition

    assert updated is not cond
    assert applied.electrode_types == ("dry",)
    assert applied.metadata_modes == ("observed",)
    assert applied.resolved_context_values_sha256 == resolved.resolved_context_values_sha256
    assert torch.equal(cond["channel_impedance"], torch.full((1, 64), 9.0))
    assert torch.allclose(
        updated["channel_impedance"][0, :8],
        torch.tensor(resolved.impedance_by_channel[0]),
    )
    assert updated["channel_impedance_missing"][0, 8:].all()


def test_resolver_rejects_mapping_or_context_drift() -> None:
    manifest = _manifest()
    mapping = build_metadata_intervention_mapping(manifest, context="correct")
    spec = resolve_cell_execution_spec(
        role="A_QM", context="correct", phase="held_participant_evaluation"
    )
    mapping.frame.at[0, "donor_run_id"] = "block02"
    with pytest.raises(ValueError, match="content drifted"):
        resolve_metadata_intervention_batch(
            subject_ids=("sub004",),
            electrode_types=("dry",),
            run_ids=("block01",),
            **_row_bindings("s", "drift-row"),
            mapping=mapping,
            sealed_block_context=manifest,
            cell_spec=spec,
            phase="held_participant_evaluation",
            batch_kind="support",
        )

    mapping = replace(
        mapping,
        sha256=metadata_intervention_mapping_sha256(mapping.frame),
    )
    with pytest.raises(ValueError, match="donor semantics"):
        resolve_metadata_intervention_batch(
            subject_ids=("sub004",),
            electrode_types=("dry",),
            run_ids=("block01",),
            **_row_bindings("s", "semantic-row"),
            mapping=mapping,
            sealed_block_context=manifest,
            cell_spec=spec,
            phase="held_participant_evaluation",
            batch_kind="support",
        )

    leaked_context = manifest.assign(label=0)
    fresh_mapping = build_metadata_intervention_mapping(manifest, context="correct")
    with pytest.raises(ValueError, match="exact label-free allowlist"):
        resolve_metadata_intervention_batch(
            subject_ids=("sub004",),
            electrode_types=("dry",),
            run_ids=("block06",),
            **_row_bindings("q", "leaked-row"),
            mapping=fresh_mapping,
            sealed_block_context=leaked_context,
            cell_spec=spec,
            phase="held_participant_evaluation",
            batch_kind="query",
        )


def test_resolver_rejects_noncanonical_cell_and_ledger_is_batch_order_invariant() -> None:
    manifest = _manifest()
    mapping = build_metadata_intervention_mapping(manifest, context="correct")
    spec = resolve_cell_execution_spec(
        role="A_QM", context="correct", phase="held_participant_evaluation"
    )
    with pytest.raises(ValueError, match="frozen execution dispatch"):
        resolve_metadata_intervention_batch(
            subject_ids=("sub004",),
            electrode_types=("dry",),
            run_ids=("block06",),
            **_row_bindings("q", "noncanonical-row"),
            mapping=mapping,
            sealed_block_context=manifest,
            cell_spec=replace(spec, query_metadata_mode="all_missing"),
            phase="held_participant_evaluation",
            batch_kind="query",
        )

    common = {
        "mapping": mapping,
        "sealed_block_context": manifest,
        "cell_spec": spec,
        "phase": "held_participant_evaluation",
        "batch_kind": "query",
    }
    first = resolve_metadata_intervention_batch(
        subject_ids=("sub004",),
        electrode_types=("dry",),
        run_ids=("block06",),
        **_row_bindings("q", "ledger-row-6"),
        **common,
    )
    second = resolve_metadata_intervention_batch(
        subject_ids=("sub004",),
        electrode_types=("dry",),
        run_ids=("block07",),
        **_row_bindings("q", "ledger-row-7"),
        **common,
    )
    together = resolve_metadata_intervention_batch(
        subject_ids=("sub004", "sub004"),
        electrode_types=("dry", "dry"),
        run_ids=("block06", "block07"),
        **_row_bindings("q", "ledger-row-6", "ledger-row-7"),
        **common,
    )
    expected_tokens = (*first.row_tokens, *second.row_tokens)
    assert resolved_context_usage_sha256(
        [first, second], expected_row_tokens=expected_tokens
    ) == resolved_context_usage_sha256(
        [second, first], expected_row_tokens=reversed(expected_tokens)
    )
    assert resolved_context_usage_sha256(
        [first, second], expected_row_tokens=expected_tokens
    ) == resolved_context_usage_sha256(
        [together], expected_row_tokens=together.row_tokens
    )
    with pytest.raises(ValueError, match="exact expected row tokens"):
        resolved_context_usage_sha256(
            [first],
            expected_row_tokens=expected_tokens,
        )
    with pytest.raises(ValueError, match="read-only"):
        first.impedance_by_channel[0, 0] = 1.0

    changed_values = first.impedance_by_channel.copy()
    changed_values[0, 0] += 0.1
    tampered = replace(first, impedance_by_channel=changed_values)
    x = torch.zeros((1, 64, 400), dtype=torch.float32)
    channel_ids = torch.zeros((1, 64), dtype=torch.long)
    channel_ids[:, :8] = torch.tensor(WEARABLE_CALIBRATION_CHANNEL_IDS)
    with pytest.raises(ValueError, match="drifted"):
        apply_resolved_metadata_to_condition(
            x,
            {"channel_ids": channel_ids, "channel_mask": channel_ids.gt(0)},
            tampered,
        )


def test_resolver_rejects_wrong_pool_shuffle_seed_and_channel_order() -> None:
    manifest = _manifest()
    spec = resolve_cell_execution_spec(
        role="A_QM", context="correct", phase="held_participant_evaluation"
    )
    correct = build_metadata_intervention_mapping(manifest, context="correct")
    with pytest.raises(ValueError, match="outside its declared support/query pool"):
        resolve_metadata_intervention_batch(
            subject_ids=("sub004",),
            electrode_types=("dry",),
            run_ids=("block01",),
            **_row_bindings("q", "wrong-pool"),
            mapping=correct,
            sealed_block_context=manifest,
            cell_spec=spec,
            phase="held_participant_evaluation",
            batch_kind="query",
        )

    shuffle = build_metadata_intervention_mapping(
        manifest,
        context="block_shuffle",
        shuffle_seed=17,
    )
    shuffle_spec = resolve_cell_execution_spec(
        role="A_QM", context="block_shuffle", phase="held_participant_evaluation"
    )
    with pytest.raises(ValueError, match="frozen seed"):
        resolve_metadata_intervention_batch(
            subject_ids=("sub004",),
            electrode_types=("dry",),
            run_ids=("block06",),
            **_row_bindings("q", "wrong-seed"),
            mapping=shuffle,
            sealed_block_context=manifest,
            cell_spec=shuffle_spec,
            phase="held_participant_evaluation",
            batch_kind="query",
        )

    reordered = manifest.copy(deep=True)
    reordered.at[0, "impedance_channel_ids"] = list(
        reversed(WEARABLE_CALIBRATION_CHANNEL_IDS)
    )
    with pytest.raises(ValueError, match="official eight-channel"):
        build_metadata_intervention_mapping(reordered, context="correct")


def test_stale_first_support_block_is_all_missing_in_final_18d_features() -> None:
    manifest = _manifest()
    channel_ids = torch.zeros((1, 64), dtype=torch.long)
    channel_ids[0, :8] = torch.tensor(WEARABLE_CALIBRATION_CHANNEL_IDS)
    channel_mask = channel_ids.gt(0)
    x = torch.zeros((1, 64, 400), dtype=torch.float32)
    mapping = build_metadata_intervention_mapping(manifest, context="stale")
    spec = resolve_cell_execution_spec(
        role="A_QM", context="stale", phase="held_participant_evaluation"
    )
    resolved = resolve_metadata_intervention_batch(
        subject_ids=("sub004",),
        electrode_types=("dry",),
        run_ids=("block01",),
        **_model_row_bindings("s", "stale-first-row", x, channel_mask),
        mapping=mapping,
        sealed_block_context=manifest,
        cell_spec=spec,
        phase="held_participant_evaluation",
        batch_kind="support",
    )
    applied = apply_resolved_metadata_to_condition(
        x,
        {
            "channel_ids": channel_ids,
            "channel_mask": channel_mask,
            "metadata_contract_version": METADATA_CONTRACT_V04_DEV,
        },
        resolved,
    )
    values, missing = build_wearable_metadata_features(
        applied.condition,
        electrode_types=applied.electrode_types,
    )
    assert torch.equal(values, torch.zeros_like(values))
    assert missing.all()
