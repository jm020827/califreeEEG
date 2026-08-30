from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cfeg.constants import (
    PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
    PROTOCOL_V04_EXTERNAL_CONTINUOUS_FIELDS,
)

DEVELOPMENT_CONTROL_NONE = "none"
DEVELOPMENT_CONTROLS = {
    DEVELOPMENT_CONTROL_NONE,
    "metadata_only",
    "missingness_only",
    "within_class_shuffle",
    "counterfactual_wet_dry",
}
DONOR_CONTROLS = {"within_class_shuffle", "counterfactual_wet_dry"}

_EXTERNAL_AUDIT_FIELDS = [
    *PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
    *PROTOCOL_V04_EXTERNAL_CONTINUOUS_FIELDS,
    "impedance_kohm_by_channel",
]


@dataclass
class DevelopmentControlPlan:
    name: str
    seed: int
    contract: dict[str, Any]
    donor_mapping: pd.DataFrame = field(default_factory=pd.DataFrame)
    external_sources: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def sources_for(self, split_name: str) -> dict[str, dict[str, Any]] | None:
        sources = self.external_sources.get(split_name)
        return sources or None


def normalize_development_control(value: Any) -> str:
    name = str(value or DEVELOPMENT_CONTROL_NONE).strip().lower()
    if name not in DEVELOPMENT_CONTROLS:
        raise ValueError(
            f"Unknown protocol.development_control={name!r}; expected one of "
            f"{sorted(DEVELOPMENT_CONTROLS)}."
        )
    return name


def build_development_control_plan(
    manifest: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    *,
    control: str,
    seed: int,
) -> DevelopmentControlPlan:
    """Build a deterministic, auditable S1--S3 negative-control treatment.

    Donor controls exchange only the external metadata bundle. They never load
    donor EEG and never expose donor identifiers to the model.
    """
    name = normalize_development_control(control)
    missingness = _missingness_contract(manifest)
    base_contract: dict[str, Any] = {
        "schema": "cfeg.development-control.v2",
        "name": name,
        "seed": int(seed),
        "development_only": name != DEVELOPMENT_CONTROL_NONE,
        "external_fields": list(_EXTERNAL_AUDIT_FIELDS),
        "donor_scope": None,
        "donor_unit": None,
        "label_alignment": None,
        "donor_mapping_sha256": None,
        "mapping_row_count": 0,
        "fixed_point_count": 0,
        "pair_coverage": None,
        "effective_external_metadata_changed_fraction": None,
        "field_changed_fraction": {},
        "condition_flip_fraction": None,
        **missingness,
    }
    if name == DEVELOPMENT_CONTROL_NONE:
        base_contract["input_access"] = "configured_primary_input_contract"
        return DevelopmentControlPlan(name=name, seed=int(seed), contract=base_contract)
    if name == "metadata_only":
        base_contract.update(
            {
                "input_access": "external_metadata_only",
                "signal_access": "zeroed",
                "structure_access": "removed",
                "query_qc_access": "removed",
                "external_value_access": "observed",
                "external_missingness_access": "observed",
            }
        )
        return DevelopmentControlPlan(name=name, seed=int(seed), contract=base_contract)
    if name == "missingness_only":
        base_contract.update(
            {
                "input_access": "external_metadata_missingness_only",
                "signal_access": "zeroed",
                "structure_access": "removed",
                "query_qc_access": "removed",
                "external_value_access": "zeroed_or_present_token",
                "external_missingness_access": "observed",
                "assay_potency": (
                    "not_testable_single_natural_pattern"
                    if missingness["natural_missingness_pattern_count"] < 2
                    else "natural_patterns_present"
                ),
            }
        )
        return DevelopmentControlPlan(name=name, seed=int(seed), contract=base_contract)

    mapping_tables: list[pd.DataFrame] = []
    sources_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    split_seed_offsets = {"train": 0, "val": 1, "test": 2}
    for split_name in ("train", "val", "test"):
        selected = np.asarray(split_indices.get(split_name, []), dtype=int)
        if not len(selected):
            continue
        if name == "within_class_shuffle":
            donor_indices = metadata_shuffle_indices(
                manifest,
                selected,
                seed=int(seed) + split_seed_offsets[split_name],
            )
            scope = "within_dataset_block_derangement_label_aligned"
        else:
            donor_indices = counterfactual_wet_dry_indices(manifest, selected)
            scope = "same_dataset_subject_label_block_window_opposite_electrode"
        table = _mapping_audit_table(
            manifest,
            selected,
            donor_indices,
            split_name=split_name,
        )
        mapping_tables.append(table)
        targets = manifest.iloc[selected]
        donors = manifest.iloc[donor_indices]
        sources_by_split[split_name] = {
            str(target["sample_id"]): donor.to_dict()
            for (_, target), (_, donor) in zip(targets.iterrows(), donors.iterrows())
        }

    if not mapping_tables:
        raise ValueError(f"Development control {name!r} has no eligible split rows.")
    mapping = pd.concat(mapping_tables, ignore_index=True)
    fixed_points = int((mapping["sample_id"] == mapping["donor_sample_id"]).sum())
    if fixed_points:
        raise ValueError(f"Development control {name!r} produced {fixed_points} fixed points.")
    changed_columns = [column for column in mapping if column.startswith("changed_")]
    field_changed = {
        column.removeprefix("changed_"): float(mapping[column].astype(bool).mean())
        for column in changed_columns
    }
    mapping_payload = mapping.to_csv(index=False).encode("utf-8")
    base_contract.update(
        {
            "input_access": "eeg_structure_query_qc_plus_donor_external_metadata",
            "signal_access": "observed_target",
            "structure_access": "observed_target",
            "query_qc_access": "observed_target",
            "external_value_access": "donor",
            "external_missingness_access": "donor",
            "donor_scope": scope,
            "donor_unit": (
                "subject_session_electrode_run_block"
                if name == "within_class_shuffle"
                else "opposite_electrode_row_pair"
            ),
            "label_alignment": (
                "exact_label_and_window_within_donor_block"
                if name == "within_class_shuffle"
                else "exact_label_and_window"
            ),
            "donor_mapping_sha256": hashlib.sha256(mapping_payload).hexdigest(),
            "mapping_row_count": len(mapping),
            "fixed_point_count": fixed_points,
            "pair_coverage": float(len(mapping) / sum(len(v) for v in split_indices.values())),
            "effective_external_metadata_changed_fraction": float(
                mapping["external_bundle_changed"].astype(bool).mean()
            ),
            "field_changed_fraction": field_changed,
            "condition_flip_fraction": float(mapping["changed_electrode_type"].mean()),
        }
    )
    return DevelopmentControlPlan(
        name=name,
        seed=int(seed),
        contract=base_contract,
        donor_mapping=mapping,
        external_sources=sources_by_split,
    )


def metadata_shuffle_indices(
    manifest: pd.DataFrame,
    selected_indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Derange acquisition blocks while keeping every donor label/window aligned.

    A wearable block contains all stimulus labels under one shared metadata
    measurement. Row-wise shuffling would synthesize an impossible block whose
    labels receive metadata from different acquisitions. This routine assigns
    one donor block to the whole target block, then matches rows by label/window.
    """
    selected = np.asarray(selected_indices, dtype=int)
    if len(selected) < 2:
        raise ValueError("within_class_shuffle requires at least two selected samples.")
    view = manifest.iloc[selected].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    required = {"dataset_id", "subject_id", "label", "run_id"}
    missing = sorted(required - set(view.columns))
    if missing:
        raise ValueError(f"within_class_shuffle is missing block columns: {missing}.")
    block_columns = ["dataset_id", "subject_id"]
    block_columns.extend(
        column for column in ("session_id", "electrode_type", "run_id") if column in view
    )
    match_columns = ["label"]
    match_columns.extend(
        column
        for column in ("window_start_sec", "window_duration_sec")
        if column in view
    )

    blocks: list[dict[str, Any]] = []
    for block_key, positions in view.groupby(block_columns, dropna=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        block = view.iloc[positions]
        _validate_external_bundle_is_block_constant(block, block_key)
        row_keys = [
            tuple(_canonical_value(row[column]) for column in match_columns)
            for _, row in block.iterrows()
        ]
        if len(row_keys) != len(set(row_keys)):
            raise ValueError(
                "within_class_shuffle requires unique label/window rows inside each block; "
                f"block={block_key!r}."
            )
        blocks.append(
            {
                "dataset_id": str(block.iloc[0]["dataset_id"]),
                "positions": positions,
                "row_map": dict(zip(row_keys, positions)),
                "signature": tuple(sorted(row_keys)),
            }
        )

    donor_positions = np.full(len(selected), -1, dtype=int)
    strata: dict[tuple[str, tuple], list[int]] = {}
    for block_index, block in enumerate(blocks):
        key = (block["dataset_id"], block["signature"])
        strata.setdefault(key, []).append(block_index)
    for stratum_key, block_indices in strata.items():
        if len(block_indices) < 2:
            raise ValueError(
                "within_class_shuffle cannot form a block derangement for singleton "
                f"stratum={stratum_key!r}."
            )
        permutation = _random_derangement(len(block_indices), rng)
        for local_index, donor_local_index in enumerate(permutation):
            target = blocks[block_indices[local_index]]
            donor = blocks[block_indices[int(donor_local_index)]]
            for row_key, target_position in target["row_map"].items():
                donor_positions[target_position] = donor["row_map"][row_key]
    if np.any(donor_positions < 0):
        raise RuntimeError("within_class_shuffle left an acquisition-block row unmapped.")
    return selected[donor_positions]


def _validate_external_bundle_is_block_constant(block: pd.DataFrame, block_key: Any) -> None:
    for field_name in _EXTERNAL_AUDIT_FIELDS:
        if field_name not in block:
            continue
        values = {_canonical_value(value) for value in block[field_name]}
        if len(values) != 1:
            raise ValueError(
                "External metadata must be constant within an acquisition block; "
                f"field={field_name!r}, block={block_key!r}."
            )


def counterfactual_wet_dry_indices(
    manifest: pd.DataFrame,
    selected_indices: np.ndarray,
) -> np.ndarray:
    """Pair each wearable row with its opposite wet/dry acquisition metadata."""
    selected = np.asarray(selected_indices, dtype=int)
    view = manifest.iloc[selected].reset_index(drop=True)
    required = {"dataset_id", "subject_id", "label", "run_id", "electrode_type"}
    missing = sorted(required - set(view.columns))
    if missing:
        raise ValueError(f"counterfactual_wet_dry is missing pairing columns: {missing}.")
    group_columns = ["dataset_id", "subject_id", "label", "run_id"]
    for optional in ("window_start_sec", "window_duration_sec"):
        if optional in view:
            group_columns.append(optional)
    donor_positions = np.full(len(view), -1, dtype=int)
    for key, positions in view.groupby(group_columns, dropna=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        electrode = view.iloc[positions]["electrode_type"].astype(str).str.lower().to_numpy()
        if len(positions) != 2 or set(electrode) != {"dry", "wet"}:
            raise ValueError(
                "counterfactual_wet_dry requires exactly one dry and one wet row per "
                f"dataset×subject×label×block×window group; group={key!r}, "
                f"electrodes={electrode.tolist()}."
            )
        donor_positions[positions[0]] = positions[1]
        donor_positions[positions[1]] = positions[0]
    if np.any(donor_positions < 0):  # pragma: no cover - group loop covers every row
        raise RuntimeError("counterfactual_wet_dry left an unpaired row.")
    return selected[donor_positions]


def _mapping_audit_table(
    manifest: pd.DataFrame,
    selected: np.ndarray,
    donors: np.ndarray,
    *,
    split_name: str,
) -> pd.DataFrame:
    target_view = manifest.iloc[selected].reset_index(drop=True)
    donor_view = manifest.iloc[donors].reset_index(drop=True)
    table = pd.DataFrame(
        {
            "split": split_name,
            "sample_id": target_view["sample_id"].astype(str),
            "donor_sample_id": donor_view["sample_id"].astype(str),
            "target_subject_id": target_view["subject_id"].astype(str),
            "donor_subject_id": donor_view["subject_id"].astype(str),
            "target_label": target_view["label"].astype(int),
            "donor_label": donor_view["label"].astype(int),
            "target_electrode_type": target_view["electrode_type"].astype(str),
            "donor_electrode_type": donor_view["electrode_type"].astype(str),
        }
    )
    for field_name in ("session_id", "run_id"):
        if field_name in target_view and field_name in donor_view:
            table[f"target_{field_name}"] = target_view[field_name].astype(str)
            table[f"donor_{field_name}"] = donor_view[field_name].astype(str)
    bundle_changed = np.zeros(len(table), dtype=bool)
    for field_name in _EXTERNAL_AUDIT_FIELDS:
        if field_name not in target_view or field_name not in donor_view:
            changed = np.zeros(len(table), dtype=bool)
        else:
            changed = np.asarray(
                [
                    _canonical_value(left) != _canonical_value(right)
                    for left, right in zip(target_view[field_name], donor_view[field_name])
                ],
                dtype=bool,
            )
        table[f"changed_{field_name}"] = changed
        bundle_changed |= changed
    table["external_bundle_changed"] = bundle_changed
    return table


def _missingness_contract(manifest: pd.DataFrame) -> dict[str, Any]:
    fields = [
        *PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
        *PROTOCOL_V04_EXTERNAL_CONTINUOUS_FIELDS,
    ]
    available_fields = [field_name for field_name in fields if field_name in manifest]
    patterns: list[str] = []
    for _, row in manifest.iterrows():
        pattern = {
            field_name: not _is_missing(row[field_name]) for field_name in available_fields
        }
        patterns.append(json.dumps(pattern, sort_keys=True, separators=(",", ":")))
    counts = pd.Series(patterns, dtype="object").value_counts().sort_index()
    return {
        "natural_missingness_fields": available_fields,
        "natural_missingness_pattern_count": len(counts),
        "natural_missingness_pattern_frequencies": {
            hashlib.sha256(pattern.encode("utf-8")).hexdigest()[:12]: int(count)
            for pattern, count in counts.items()
        },
    }


def _random_derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    positions = np.arange(size)
    for _ in range(256):
        candidate = rng.permutation(size)
        if np.all(candidate != positions):
            return candidate
    return np.roll(positions, int(rng.integers(1, size)))


def _canonical_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = [None if _is_missing(item) else item for item in value]
    elif _is_missing(value):
        value = None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "<na>", "nat"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
