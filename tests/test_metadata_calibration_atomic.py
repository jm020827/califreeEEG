from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from cfeg.analysis.metadata_calibration_efficiency import (
    MetadataCalibrationBundleBindings,
    MetadataCalibrationBundleSpec,
)
from cfeg.identity import canonical_identity_sha256
from cfeg.metadata_calibration_atomic import (
    seal_metadata_calibration_private_staging,
    validate_and_summarize_support_rows,
)
from cfeg.metadata_calibration_baseline_contract import (
    all_baseline_execution_bindings,
)
from cfeg.metadata_calibration_execution import cell_execution_contract_sha256


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _token(prefix: str, value: str) -> str:
    return f"{prefix}_{_hash(value)}"


def _spec() -> MetadataCalibrationBundleSpec:
    return MetadataCalibrationBundleSpec(n_classes=3, seeds=(1,), sensitivity_resamples=50)


def _bindings() -> MetadataCalibrationBundleBindings:
    return MetadataCalibrationBundleBindings(
        plan_sha256="a" * 64,
        execution_manifest_sha256="b" * 64,
        decision_receipt_sha256="c" * 64,
        asset_receipt_sha256="d" * 64,
        asset_fingerprint_bundle_sha256="e" * 64,
        class_map_sha256="f" * 64,
        source_tree_sha256="1" * 64,
    )


def _query() -> pd.DataFrame:
    rows = []
    for subject in ("s0", "s1"):
        for condition in ("dry", "wet"):
            raw = [f"{subject}:{condition}:query:{index}" for index in range(6)]
            identity = canonical_identity_sha256(raw)
            for item in raw:
                rows.append(
                    {
                        "query_token": _token("q", item),
                        "query_identity_sha256": identity,
                        "subject_id": subject,
                        "electrode_type": condition,
                        "checkpoint_group": "held",
                    }
                )
    return pd.DataFrame(rows)


def _support_rows(query: pd.DataFrame, spec: MetadataCalibrationBundleSpec) -> pd.DataFrame:
    rows = []
    cells = query[["subject_id", "electrode_type", "checkpoint_group"]].drop_duplicates()
    for cell in cells.to_dict(orient="records"):
        for budget in (1, 3, 5):
            for block in range(1, budget + 1):
                for label in range(spec.n_classes):
                    identity = (
                        f"{cell['subject_id']}:{cell['electrode_type']}:block{block:02d}:{label}"
                    )
                    rows.append(
                        {
                            **cell,
                            "budget": budget,
                            "support_token": _token("s", identity),
                            "run_id": f"block{block:02d}",
                            "support_label": label,
                            "base_input_sha256": _hash(f"input:{identity}"),
                        }
                    )
    return pd.DataFrame(rows)


def _candidate(
    query: pd.DataFrame,
    support: pd.DataFrame,
    spec: MetadataCalibrationBundleSpec,
) -> pd.DataFrame:
    support_lookup = {
        (row.subject_id, row.electrode_type, row.checkpoint_group, row.budget): (
            row.support_identity_sha256
        )
        for row in support.itertuples(index=False)
    }
    rows = []
    for role, context in spec.contexts:
        for seed in spec.seeds:
            for budget in spec.budgets:
                for query_row in query.to_dict(orient="records"):
                    rows.append(
                        {
                            **query_row,
                            "candidate_id": spec.candidate_id,
                            "phase": spec.phase,
                            **_bindings().as_dict(),
                            "checkpoint_sha256": _hash(
                                f"checkpoint:{seed}:{query_row['checkpoint_group']}"
                            ),
                            "role": role,
                            "context": context,
                            "seed": seed,
                            "budget": budget,
                            "support_identity_sha256": support_lookup[
                                (
                                    query_row["subject_id"],
                                    query_row["electrode_type"],
                                    query_row["checkpoint_group"],
                                    budget,
                                )
                            ],
                            "intervention_mapping_sha256": _hash(
                                f"{role}:{context}"
                            ),
                            "resolved_context_usage_sha256": _hash(
                                f"{role}:{context}:{budget}:"
                                f"{query_row['checkpoint_group']}"
                            ),
                            "cell_execution_contract_sha256": (
                                cell_execution_contract_sha256(phase=spec.phase)
                            ),
                            **{f"prob_{label:03d}": 1.0 / spec.n_classes for label in range(3)},
                        }
                    )
    return pd.DataFrame(rows)


def _baseline(
    query: pd.DataFrame,
    support: pd.DataFrame,
    spec: MetadataCalibrationBundleSpec,
) -> pd.DataFrame:
    budgets = {
        "strict_FBCCA": (0,),
        "target_template_correlation": (1, 3, 5),
        "target_filterbank_eTRCA": (3, 5),
        "same3_filterbank_eTRCA": (1,),
        "chiang2021_LST_filterbank_eTRCA": (1, 3, 5),
    }
    support_lookup = {
        (row.subject_id, row.electrode_type, row.checkpoint_group, row.budget): (
            row.support_identity_sha256
        )
        for row in support.itertuples(index=False)
    }
    empty = canonical_identity_sha256([])
    execution_bindings = all_baseline_execution_bindings()
    rows = []
    for adapter, applicable in budgets.items():
        for budget in applicable:
            for query_row in query.to_dict(orient="records"):
                rows.append(
                    {
                        **query_row,
                        "candidate_id": spec.candidate_id,
                        "phase": spec.phase,
                        **_bindings().as_dict(),
                        "adapter_id": adapter,
                        "budget": budget,
                        "support_identity_sha256": support_lookup[
                            (
                                query_row["subject_id"],
                                query_row["electrode_type"],
                                query_row["checkpoint_group"],
                                budget,
                            )
                        ],
                        "source_pool_identity_sha256": (
                            _hash(
                                f"source:{query_row['checkpoint_group']}:"
                                f"{query_row['electrode_type']}"
                            )
                            if adapter == "chiang2021_LST_filterbank_eTRCA"
                            else empty
                        ),
                        "implementation_sha256": execution_bindings[
                            adapter
                        ].implementation_sha256,
                        "config_sha256": execution_bindings[adapter].config_sha256,
                        "adapter_contract_sha256": execution_bindings[
                            adapter
                        ].adapter_contract_sha256,
                        "score_schema": execution_bindings[adapter].score_schema,
                        "score_interpretation": execution_bindings[
                            adapter
                        ].score_interpretation,
                        **{f"score_{label:03d}": float(label) for label in range(3)},
                    }
                )
    return pd.DataFrame(rows)


def _source_pools() -> dict[tuple[str, str], str]:
    return {
        ("held", interface): _hash(f"source:held:{interface}")
        for interface in ("dry", "wet")
    }


def test_private_seal_rehashes_nested_support_and_complete_baselines() -> None:
    spec = _spec()
    query = _query()
    support_rows = _support_rows(query, spec)
    support, support_hash = validate_and_summarize_support_rows(
        support_rows, expected_query_unlabeled=query, spec=spec
    )
    seal = seal_metadata_calibration_private_staging(
        _candidate(query, support, spec),
        _baseline(query, support, spec),
        support_rows,
        expected_query_unlabeled=query,
        expected_source_pools=_source_pools(),
        bindings=_bindings(),
        spec=spec,
    ).as_dict()
    assert seal["query_outcomes_loaded"] is False
    assert seal["complete_candidate_and_mandatory_baseline_grid"] is True
    assert seal["support_rows_sha256"] == support_hash


def test_private_seal_fails_on_support_or_baseline_partial_grid() -> None:
    spec = _spec()
    query = _query()
    support_rows = _support_rows(query, spec)
    support, _ = validate_and_summarize_support_rows(
        support_rows, expected_query_unlabeled=query, spec=spec
    )
    baseline = _baseline(query, support, spec)
    with pytest.raises(ValueError, match="complete query grid"):
        seal_metadata_calibration_private_staging(
            _candidate(query, support, spec),
            baseline.iloc[:-1],
            support_rows,
            expected_query_unlabeled=query,
            expected_source_pools=_source_pools(),
            bindings=_bindings(),
            spec=spec,
        )

    broken_support = support_rows.copy()
    target = broken_support.index[broken_support["budget"].eq(3)][0]
    broken_support.loc[target, "run_id"] = "block05"
    with pytest.raises(ValueError, match="complete nested blocks"):
        validate_and_summarize_support_rows(
            broken_support, expected_query_unlabeled=query, spec=spec
        )


def test_support_rows_reject_label_proxy_columns_and_token_drift() -> None:
    spec = _spec()
    query = _query()
    rows = _support_rows(query, spec)
    with pytest.raises(ValueError, match="exact atomic schema"):
        validate_and_summarize_support_rows(
            rows.assign(sample_id="raw_target02"),
            expected_query_unlabeled=query,
            spec=spec,
        )
    drift = rows.copy()
    repeated = drift["support_token"].eq(drift.iloc[0]["support_token"])
    changed = drift.index[repeated][-1]
    drift.loc[changed, "base_input_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="changed run, label, or input"):
        validate_and_summarize_support_rows(
            drift, expected_query_unlabeled=query, spec=spec
        )

    cross_cell = rows.copy()
    source_token = cross_cell.loc[
        cross_cell["subject_id"].eq("s0")
        & cross_cell["electrode_type"].eq("dry")
        & cross_cell["run_id"].eq("block01")
        & cross_cell["support_label"].eq(0),
        "support_token",
    ].iloc[0]
    target = (
        cross_cell["subject_id"].eq("s1")
        & cross_cell["electrode_type"].eq("wet")
        & cross_cell["run_id"].eq("block01")
        & cross_cell["support_label"].eq(0)
    )
    cross_cell.loc[target, "support_token"] = source_token
    with pytest.raises(ValueError, match="reused across different"):
        validate_and_summarize_support_rows(
            cross_cell, expected_query_unlabeled=query, spec=spec
        )
