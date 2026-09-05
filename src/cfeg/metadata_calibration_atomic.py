from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfeg.analysis.metadata_calibration_efficiency import (
    MetadataCalibrationBundleBindings,
    MetadataCalibrationBundleSpec,
    validate_metadata_calibration_private_staging,
)
from cfeg.identity import canonical_identity_sha256
from cfeg.metadata_calibration_baseline_contract import (
    all_baseline_execution_bindings,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUPPORT_TOKEN = re.compile(r"s_[0-9a-f]{64}")
_SUPPORT_ROW_COLUMNS = {
    "checkpoint_group",
    "subject_id",
    "electrode_type",
    "budget",
    "support_token",
    "run_id",
    "support_label",
    "base_input_sha256",
}
_BASELINE_BINDING_COLUMNS = {
    "plan_sha256",
    "execution_manifest_sha256",
    "decision_receipt_sha256",
    "asset_receipt_sha256",
    "asset_fingerprint_bundle_sha256",
    "class_map_sha256",
    "source_tree_sha256",
}
_BASELINE_FIXED_COLUMNS = {
    "candidate_id",
    "phase",
    "checkpoint_group",
    "adapter_id",
    "budget",
    "query_token",
    "subject_id",
    "electrode_type",
    "support_identity_sha256",
    "query_identity_sha256",
    "source_pool_identity_sha256",
    "implementation_sha256",
    "config_sha256",
    "adapter_contract_sha256",
    "score_schema",
    "score_interpretation",
    *_BASELINE_BINDING_COLUMNS,
}
_BASELINE_BUDGETS = {
    "strict_FBCCA": (0,),
    "target_template_correlation": (1, 3, 5),
    "target_filterbank_eTRCA": (3, 5),
    "same3_filterbank_eTRCA": (1,),
    "chiang2021_LST_filterbank_eTRCA": (1, 3, 5),
}


@dataclass(frozen=True)
class PrivateStagingSeal:
    candidate_staging_sha256: str
    baseline_staging_sha256: str
    support_rows_sha256: str
    combined_private_tree_sha256: str
    query_outcomes_loaded: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "cfeg.metadata-calibration-private-seal.v1",
            "candidate_staging_sha256": self.candidate_staging_sha256,
            "baseline_staging_sha256": self.baseline_staging_sha256,
            "support_rows_sha256": self.support_rows_sha256,
            "combined_private_tree_sha256": self.combined_private_tree_sha256,
            "query_outcomes_loaded": self.query_outcomes_loaded,
            "complete_candidate_and_mandatory_baseline_grid": True,
        }


def validate_and_summarize_support_rows(
    support_rows: pd.DataFrame,
    *,
    expected_query_unlabeled: pd.DataFrame,
    spec: MetadataCalibrationBundleSpec,
) -> tuple[pd.DataFrame, str]:
    """Rebuild support cell identities from exact sample-level nested blocks."""

    missing = _SUPPORT_ROW_COLUMNS - set(support_rows.columns)
    extra = set(support_rows.columns) - _SUPPORT_ROW_COLUMNS
    if missing or extra:
        raise ValueError(
            "Support rows must use the exact atomic schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    frame = support_rows.loc[:, sorted(_SUPPORT_ROW_COLUMNS)].copy()
    for column in ("checkpoint_group", "subject_id", "electrode_type", "run_id"):
        frame[column] = frame[column].astype(str)
    frame["support_token"] = frame["support_token"].astype(str)
    if not frame["support_token"].map(lambda value: bool(_SUPPORT_TOKEN.fullmatch(value))).all():
        raise ValueError("Support identities must be opaque s_<HMAC-SHA256> tokens.")
    frame["budget"] = _strict_integer(frame["budget"], name="support budget")
    frame["support_label"] = _strict_integer(
        frame["support_label"], name="support label"
    )
    if not frame["base_input_sha256"].astype(str).map(_is_sha256).all():
        raise ValueError("Every support row must bind its actual base input by SHA-256.")
    if frame.duplicated(
        ["checkpoint_group", "subject_id", "electrode_type", "budget", "support_token"]
    ).any():
        raise ValueError("Support rows contain a duplicate sample within one budget.")
    if set(frame["budget"]) != {1, 3, 5}:
        raise ValueError("Sample-level support rows must contain exact nonzero budgets 1,3,5.")
    if frame["support_label"].min() < 0 or frame["support_label"].max() >= spec.n_classes:
        raise ValueError("Support label falls outside the frozen class vocabulary.")

    query_required = {
        "query_token",
        "subject_id",
        "electrode_type",
        "checkpoint_group",
    }
    if not query_required.issubset(expected_query_unlabeled.columns):
        raise ValueError("Expected query view cannot define complete support cells.")
    query_cells = expected_query_unlabeled.loc[
        :, ["subject_id", "electrode_type", "checkpoint_group"]
    ].astype(str).drop_duplicates()
    observed_cells = frame.loc[
        :, ["subject_id", "electrode_type", "checkpoint_group"]
    ].drop_duplicates()
    merged_cells = query_cells.merge(observed_cells, how="outer", indicator=True)
    if not merged_cells["_merge"].eq("both").all():
        raise ValueError("Support participant-interface cells differ from the query view.")
    if set(frame["support_token"]).intersection(
        expected_query_unlabeled["query_token"].astype(str)
    ):
        raise ValueError("Support and query opaque tokens must be disjoint.")

    summary_rows: list[dict[str, object]] = []
    all_row_records: list[dict[str, object]] = []
    for cell in query_cells.sort_values(
        ["checkpoint_group", "subject_id", "electrode_type"], kind="mergesort"
    ).itertuples(index=False):
        cell_mask = (
            frame["checkpoint_group"].eq(cell.checkpoint_group)
            & frame["subject_id"].eq(cell.subject_id)
            & frame["electrode_type"].eq(cell.electrode_type)
        )
        token_sets: dict[int, set[str]] = {0: set()}
        for budget in spec.budgets:
            if budget == 0:
                rows = frame.iloc[0:0].copy()
            else:
                rows = frame.loc[cell_mask & frame["budget"].eq(budget)].copy()
                _validate_support_budget_rows(rows, budget=budget, n_classes=spec.n_classes)
            token_sets[budget] = set(rows["support_token"].astype(str))
            records = _canonical_support_records(rows)
            summary_rows.append(
                {
                    "checkpoint_group": cell.checkpoint_group,
                    "subject_id": cell.subject_id,
                    "electrode_type": cell.electrode_type,
                    "budget": budget,
                    "support_identity_sha256": _json_sha256(records),
                }
            )
            all_row_records.extend(records)
        if not token_sets[1] < token_sets[3] or not token_sets[3] < token_sets[5]:
            raise ValueError("Support tokens must be strictly nested from k1 to k3 to k5.")
        repeated = frame.loc[cell_mask].sort_values(
            ["support_token", "budget"], kind="mergesort"
        )
        consistency = repeated.groupby("support_token", sort=False).agg(
            run_count=("run_id", "nunique"),
            label_count=("support_label", "nunique"),
            input_count=("base_input_sha256", "nunique"),
        )
        if not consistency.eq(1).all().all():
            raise ValueError("A nested support token changed run, label, or input hash.")
    token_binding = frame.groupby("support_token", sort=False).agg(
        checkpoint_groups=("checkpoint_group", "nunique"),
        subjects=("subject_id", "nunique"),
        interfaces=("electrode_type", "nunique"),
        runs=("run_id", "nunique"),
        labels=("support_label", "nunique"),
        inputs=("base_input_sha256", "nunique"),
    )
    if not token_binding.eq(1).all().all():
        raise ValueError("A support token is reused across different samples or participant cells.")
    summary = pd.DataFrame(summary_rows)
    expected_rows = len(query_cells) * len(spec.budgets)
    if len(summary) != expected_rows:
        raise RuntimeError("Support summary did not cover every cell and budget.")
    return summary, _json_sha256(all_row_records)


def validate_baseline_private_staging(
    scores: pd.DataFrame,
    *,
    expected_query_unlabeled: pd.DataFrame,
    expected_support: pd.DataFrame,
    expected_source_pools: dict[tuple[str, str], str],
    bindings: MetadataCalibrationBundleBindings,
    spec: MetadataCalibrationBundleSpec,
) -> dict[str, object]:
    """Validate every mandatory deterministic baseline score before label access."""

    score_columns = tuple(f"score_{label:03d}" for label in range(spec.n_classes))
    allowed = {*_BASELINE_FIXED_COLUMNS, *score_columns}
    missing = allowed - set(scores.columns)
    extra = set(scores.columns) - allowed
    if missing or extra:
        raise ValueError(
            "Baseline staging must use its exact allowlist; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    frame = scores.loc[:, sorted(allowed)].copy()
    if set(frame["candidate_id"].astype(str)) != {spec.candidate_id}:
        raise ValueError("Baseline candidate binding is wrong.")
    if set(frame["phase"].astype(str)) != {spec.phase}:
        raise ValueError("Baseline phase binding is wrong.")
    for column, value in bindings.as_dict().items():
        if set(frame[column].astype(str)) != {value}:
            raise ValueError(f"Baseline staging differs from frozen {column}.")
    frame["budget"] = _strict_integer(frame["budget"], name="baseline budget")
    observed_adapters = set(frame["adapter_id"].astype(str))
    if observed_adapters != set(_BASELINE_BUDGETS):
        raise ValueError("Baseline staging omits or adds a mandatory adapter.")
    expected_groups = {
        (adapter, budget)
        for adapter, budgets in _BASELINE_BUDGETS.items()
        for budget in budgets
    }
    if set(frame[["adapter_id", "budget"]].itertuples(index=False, name=None)) != expected_groups:
        raise ValueError("Baseline adapter-budget grid differs from the frozen ledger.")
    if len(frame) != len(expected_query_unlabeled) * len(expected_groups):
        raise ValueError("Baseline staging is not a complete query grid.")
    key = ["adapter_id", "budget", "query_token"]
    if frame.duplicated(key).any():
        raise ValueError("Baseline staging contains duplicate query scores.")

    query_columns = [
        "query_token",
        "subject_id",
        "electrode_type",
        "checkpoint_group",
        "query_identity_sha256",
    ]
    query = expected_query_unlabeled.loc[:, query_columns].copy()
    merged = frame.merge(
        query,
        on="query_token",
        how="left",
        suffixes=("", "_expected"),
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Baseline staging contains an unsealed query token.")
    for column in (
        "subject_id",
        "electrode_type",
        "checkpoint_group",
        "query_identity_sha256",
    ):
        if not merged[column].astype(str).eq(merged[f"{column}_expected"].astype(str)).all():
            raise ValueError(f"Baseline {column} differs from the sealed query view.")
    merged = merged.merge(
        expected_support,
        on=["subject_id", "electrode_type", "checkpoint_group", "budget"],
        how="left",
        suffixes=("", "_expected"),
        validate="many_to_one",
    )
    if not merged["support_identity_sha256"].astype(str).eq(
        merged["support_identity_sha256_expected"].astype(str)
    ).all():
        raise ValueError("Baseline support binding differs from the rebuilt support rows.")
    for column in (
        "query_identity_sha256",
        "support_identity_sha256",
        "source_pool_identity_sha256",
        "implementation_sha256",
        "config_sha256",
        "adapter_contract_sha256",
    ):
        if not merged[column].astype(str).map(_is_sha256).all():
            raise ValueError(f"Baseline {column} contains a non-SHA-256 value.")
    numeric_scores = merged.loc[:, score_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_scores).all():
        raise ValueError("Baseline staging contains a non-finite raw score.")
    per_adapter = merged.groupby("adapter_id", sort=False).agg(
        implementation=("implementation_sha256", "nunique"),
        config=("config_sha256", "nunique"),
        contract=("adapter_contract_sha256", "nunique"),
        schema=("score_schema", "nunique"),
        interpretation=("score_interpretation", "nunique"),
    )
    if not per_adapter.eq(1).all().all():
        raise ValueError("A baseline adapter changed implementation/config/schema within the grid.")
    canonical_bindings = all_baseline_execution_bindings()
    for adapter_id, execution in canonical_bindings.items():
        selected = merged.loc[merged["adapter_id"].astype(str).eq(adapter_id)]
        exact = {
            "implementation_sha256": execution.implementation_sha256,
            "config_sha256": execution.config_sha256,
            "adapter_contract_sha256": execution.adapter_contract_sha256,
            "score_schema": execution.score_schema,
            "score_interpretation": execution.score_interpretation,
        }
        if any(set(selected[field].astype(str)) != {value} for field, value in exact.items()):
            raise ValueError(f"Baseline {adapter_id} differs from its canonical execution binding.")
    empty_identity = canonical_identity_sha256([])
    non_lst = merged["adapter_id"].ne("chiang2021_LST_filterbank_eTRCA")
    if set(merged.loc[non_lst, "source_pool_identity_sha256"].astype(str)) != {
        empty_identity
    }:
        raise ValueError("Only LST may bind a nonempty labeled source pool.")
    lst_source_counts = merged.loc[~non_lst].groupby(
        ["checkpoint_group", "electrode_type"], sort=False
    )["source_pool_identity_sha256"].nunique()
    if not lst_source_counts.eq(1).all() or (
        merged.loc[~non_lst, "source_pool_identity_sha256"].astype(str) == empty_identity
    ).any():
        raise ValueError(
            "LST must bind one nonempty matching-interface source pool per checkpoint group."
        )
    per_group = merged.loc[~non_lst, [
        "checkpoint_group",
        "electrode_type",
        "source_pool_identity_sha256",
    ]].drop_duplicates()
    if per_group.duplicated(["checkpoint_group", "source_pool_identity_sha256"]).any():
        raise ValueError("Dry and wet LST source-pool identities must remain distinct.")
    observed_source = {
        (str(row.checkpoint_group), str(row.electrode_type)): str(
            row.source_pool_identity_sha256
        )
        for row in per_group.itertuples(index=False)
    }
    if observed_source != dict(expected_source_pools):
        raise ValueError("LST source pools differ from exact sealed sample-level identities.")
    content_hash = _baseline_content_sha256(merged, score_columns=score_columns)
    return {
        "schema": "cfeg.metadata-calibration-baseline-private-validation.v1",
        "rows": len(merged),
        "query_outcomes_loaded": False,
        "complete_mandatory_grid": True,
        "raw_scores_only": True,
        "deterministic_seed_column_present": False,
        "staging_content_sha256": content_hash,
    }


def seal_metadata_calibration_private_staging(
    candidate_predictions: pd.DataFrame,
    baseline_scores: pd.DataFrame,
    support_rows: pd.DataFrame,
    *,
    expected_query_unlabeled: pd.DataFrame,
    expected_source_pools: dict[tuple[str, str], str],
    bindings: MetadataCalibrationBundleBindings,
    spec: MetadataCalibrationBundleSpec,
) -> PrivateStagingSeal:
    """Seal complete candidate and mandatory-baseline grids before any label join."""

    expected_support, support_hash = validate_and_summarize_support_rows(
        support_rows,
        expected_query_unlabeled=expected_query_unlabeled,
        spec=spec,
    )
    candidate = validate_metadata_calibration_private_staging(
        candidate_predictions,
        expected_query_unlabeled=expected_query_unlabeled,
        expected_support=expected_support,
        bindings=bindings,
        spec=spec,
    )
    baseline = validate_baseline_private_staging(
        baseline_scores,
        expected_query_unlabeled=expected_query_unlabeled,
        expected_support=expected_support,
        expected_source_pools=expected_source_pools,
        bindings=bindings,
        spec=spec,
    )
    components = {
        "candidate": candidate["staging_content_sha256"],
        "baseline": baseline["staging_content_sha256"],
        "support": support_hash,
    }
    return PrivateStagingSeal(
        candidate_staging_sha256=str(components["candidate"]),
        baseline_staging_sha256=str(components["baseline"]),
        support_rows_sha256=support_hash,
        combined_private_tree_sha256=_json_sha256(components),
    )


def _validate_support_budget_rows(
    rows: pd.DataFrame, *, budget: int, n_classes: int
) -> None:
    expected_runs = {f"block{index:02d}" for index in range(1, budget + 1)}
    if len(rows) != budget * n_classes or set(rows["run_id"]) != expected_runs:
        raise ValueError(f"Support budget k={budget} is not exact complete nested blocks.")
    for _run_id, block in rows.groupby("run_id", sort=False):
        labels = sorted(block["support_label"].astype(int).tolist())
        if labels != list(range(n_classes)):
            raise ValueError("Every support block must contain each class exactly once.")


def _canonical_support_records(rows: pd.DataFrame) -> list[dict[str, object]]:
    columns = [
        "checkpoint_group",
        "subject_id",
        "electrode_type",
        "budget",
        "support_token",
        "run_id",
        "support_label",
        "base_input_sha256",
    ]
    ordered = rows.loc[:, columns].sort_values(
        ["support_token", "run_id"], kind="mergesort"
    )
    return ordered.to_dict(orient="records")


def _baseline_content_sha256(
    frame: pd.DataFrame, *, score_columns: tuple[str, ...]
) -> str:
    key = ["adapter_id", "budget", "query_token"]
    ordered = frame.sort_values(key, kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    for column in sorted(set(ordered.columns) - set(score_columns)):
        values = "\n".join(ordered[column].astype(str).tolist()).encode("utf-8")
        digest.update(column.encode("utf-8"))
        digest.update(len(values).to_bytes(8, "big"))
        digest.update(values)
    scores = ordered.loc[:, score_columns].to_numpy(dtype="<f8", copy=True)
    digest.update(scores.tobytes(order="C"))
    return digest.hexdigest()


def _strict_integer(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integers.")
    return numeric.astype(int)


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(_SHA256.fullmatch(str(value)))


def nominal_mc_standard_error(probability: float, draws: int) -> float:
    """Small public helper used by atomic-receipt tests and diagnostics."""

    if not 0.0 <= probability <= 1.0 or draws <= 0:
        raise ValueError("Probability and draw count are invalid.")
    return math.sqrt(probability * (1.0 - probability) / draws)
