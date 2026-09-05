from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from cfeg.data.metadata_calibration_features import MetadataFeatureAblation
from cfeg.data.metadata_calibration_interventions import MetadataIntervention

ExperimentPhase = Literal["source_development", "held_participant_evaluation"]
QualityMode = Literal["off", "observed"]
MetadataMode = Literal["off", "observed", "all_missing"]


@dataclass(frozen=True)
class CellExecutionSpec:
    """Exact runtime dispatch for one declared candidate cell."""

    role: str
    context: str
    query_q_mode: QualityMode
    support_q_mode: QualityMode
    query_metadata_mode: MetadataMode
    support_metadata_mode: MetadataMode
    query_metadata_ablation: MetadataFeatureAblation
    support_metadata_ablation: MetadataFeatureAblation
    intervention: MetadataIntervention

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _cell(
    role: str,
    context: str,
    *,
    q_mode: QualityMode,
    query_metadata_mode: MetadataMode,
    support_metadata_mode: MetadataMode,
    intervention: MetadataIntervention,
    ablation: MetadataFeatureAblation = "full",
) -> CellExecutionSpec:
    return CellExecutionSpec(
        role=role,
        context=context,
        query_q_mode=q_mode,
        support_q_mode=q_mode,
        query_metadata_mode=query_metadata_mode,
        support_metadata_mode=support_metadata_mode,
        query_metadata_ablation=ablation,
        support_metadata_ablation=ablation,
        intervention=intervention,
    )


_HELD_CELLS = {
    ("A_0", "exact_off"): _cell(
        "A_0",
        "exact_off",
        q_mode="off",
        query_metadata_mode="off",
        support_metadata_mode="off",
        intervention="exact_off",
    ),
    ("A_M", "correct"): _cell(
        "A_M",
        "correct",
        q_mode="off",
        query_metadata_mode="observed",
        support_metadata_mode="observed",
        intervention="correct",
    ),
    ("A_Q", "exact_off"): _cell(
        "A_Q",
        "exact_off",
        q_mode="observed",
        query_metadata_mode="off",
        support_metadata_mode="off",
        intervention="exact_off",
    ),
    **{
        ("A_QM", context): _cell(
            "A_QM",
            context,
            q_mode="observed",
            query_metadata_mode=("all_missing" if context == "all_missing" else "observed"),
            support_metadata_mode=("all_missing" if context == "all_missing" else "observed"),
            intervention=context,
        )
        for context in (
            "correct",
            "all_missing",
            "block_shuffle",
            "stale",
            "opposite_interface",
        )
    },
}

_DEVELOPMENT_DIAGNOSTICS = {
    ("A_QM", "support_metadata_only"): CellExecutionSpec(
        role="A_QM",
        context="support_metadata_only",
        query_q_mode="observed",
        support_q_mode="observed",
        query_metadata_mode="all_missing",
        support_metadata_mode="observed",
        query_metadata_ablation="full",
        support_metadata_ablation="full",
        intervention="correct",
    ),
    ("A_QM", "query_metadata_only"): CellExecutionSpec(
        role="A_QM",
        context="query_metadata_only",
        query_q_mode="observed",
        support_q_mode="observed",
        query_metadata_mode="observed",
        support_metadata_mode="all_missing",
        query_metadata_ablation="full",
        support_metadata_ablation="full",
        intervention="correct",
    ),
    ("A_QM", "interface_only"): _cell(
        "A_QM",
        "interface_only",
        q_mode="observed",
        query_metadata_mode="observed",
        support_metadata_mode="observed",
        intervention="correct",
        ablation="interface_only",
    ),
    ("A_QM", "impedance_only"): _cell(
        "A_QM",
        "impedance_only",
        q_mode="observed",
        query_metadata_mode="observed",
        support_metadata_mode="observed",
        intervention="correct",
        ablation="impedance_only",
    ),
}


def resolve_cell_execution_spec(
    *, role: str, context: str, phase: ExperimentPhase
) -> CellExecutionSpec:
    """Resolve role/context names to model modes without caller discretion."""

    if phase not in {"source_development", "held_participant_evaluation"}:
        raise ValueError(f"Unknown metadata-calibration phase {phase!r}.")
    key = (str(role), str(context))
    allowed = dict(_HELD_CELLS)
    if phase == "source_development":
        allowed.update(_DEVELOPMENT_DIAGNOSTICS)
    if key not in allowed:
        raise ValueError(f"Cell {key!r} is not declared for phase {phase!r}.")
    return allowed[key]


def execution_cells_for_phase(
    *, phase: ExperimentPhase
) -> tuple[tuple[str, str], ...]:
    """Return the exact ordered role/context grid for one experiment phase."""

    keys = set(_HELD_CELLS)
    if phase == "source_development":
        keys.update(_DEVELOPMENT_DIAGNOSTICS)
    elif phase != "held_participant_evaluation":
        raise ValueError(f"Unknown metadata-calibration phase {phase!r}.")
    return tuple(sorted(keys))


def cell_execution_contract_sha256(*, phase: ExperimentPhase) -> str:
    """Hash the complete canonical dispatch table for producer receipts."""

    keys = execution_cells_for_phase(phase=phase)
    rows = [
        resolve_cell_execution_spec(role=role, context=context, phase=phase).as_dict()
        for role, context in keys
    ]
    contract = {
        "schema": "cfeg.metadata-calibration-cell-dispatch.v1",
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "cells": rows,
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
