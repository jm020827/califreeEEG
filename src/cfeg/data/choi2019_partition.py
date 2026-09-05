from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from cfeg.baselines.fbcca import resolve_filterbank_parameters
from cfeg.data.preprocess import CanonicalChannelMap
from cfeg.data.schema import load_manifest, validate_manifest

_FROZEN_SUBBAND_WEIGHTS = np.asarray(
    [
        1.25,
        0.6704482076268572,
        0.5032785618838642,
        0.42677669529663687,
        0.38374806099528436,
        0.35649051737437876,
        0.33782687899303776,
    ],
    dtype=np.float64,
)
_GOVERNED_PROCESSED_ASSETS = {
    "asset_info.json",
    "class_map.json",
    "manifest.jsonl",
    "manifest.parquet",
    "preprocess_config.yaml",
    "questionnaire_normalized.csv",
    "signals.h5",
}
_GOVERNED_REPOSITORY_CONTRACTS = {
    "configs/baselines/fbcca_choi2019_bandwise_v1.yaml",
    "configs/canonical_channels.yaml",
    "configs/data/choi2019.yaml",
}


@dataclass(frozen=True)
class ChoiPartitionPlan:
    assignments: tuple[dict[str, Any], ...]
    assignment_bytes: bytes
    summary: dict[str, Any]


def construct_partition(
    processed_dir: Path,
    data_cfg: dict[str, Any],
    anchor_cfg: dict[str, Any],
) -> ChoiPartitionPlan:
    """Construct and validate the frozen Choi cross-day partition without scores."""

    processed_dir = require_safe_directory(processed_dir, "Choi processed root")
    require_safe_regular_file(processed_dir / "manifest.parquet", "Choi Parquet manifest")
    require_safe_regular_file(processed_dir / "manifest.jsonl", "Choi JSONL manifest")
    manifest = load_manifest(processed_dir)
    validate_manifest(manifest)
    _validate_manifest_grid(manifest, data_cfg)
    weight_receipt = _validate_anchor_weights(anchor_cfg)
    channel_receipt = _validate_evaluation_channels(
        manifest,
        processed_dir / "signals.h5",
        anchor_cfg,
    )

    budgets = [int(value) for value in anchor_cfg["support_query"]["budgets_trials_per_class"]]
    if budgets != [0, 1, 3, 5]:
        raise ValueError(f"Choi calibration budgets drifted: {budgets}")
    processing_bands = [str(value) for value in data_cfg["processing_band_order"]]
    if processing_bands != ["LOW", "MID", "HIGH"]:
        raise ValueError(f"Choi processing-band order drifted: {processing_bands}")
    if data_cfg.get("acquisition_band_order") != "unavailable_not_reported":
        raise ValueError("Choi acquisition-band order must remain explicitly unavailable")

    assignments: list[dict[str, Any]] = []
    group_receipts: list[dict[str, Any]] = []
    subjects = sorted(manifest["subject_id"].astype(str).unique())
    for subject_id in subjects:
        for band in processing_bands:
            group = manifest.loc[
                (manifest["subject_id"].astype(str) == subject_id)
                & (manifest["frequency_band"].astype(str).str.upper() == band)
            ].copy()
            day1 = group.loc[group["session_id"].astype(str) == "day01"].copy()
            day2 = group.loc[group["session_id"].astype(str) == "day02"].copy()
            support_blocks = _pseudo_blocks(day1, band)
            queries = _chronological(day2, band)
            if len(queries) != 80:
                raise ValueError(f"Expected 80 Day-2 queries for {subject_id}/{band}")

            query_ids = tuple(queries["sample_id"].astype(str))
            support_sets: dict[int, set[str]] = {}
            for budget in budgets:
                support_rows = [row for block in support_blocks[:budget] for row in block]
                support_ids = {str(row["sample_id"]) for row in support_rows}
                if len(support_ids) != 4 * budget:
                    raise ValueError(
                        f"Support count/identity drift for {subject_id}/{band}/k={budget}"
                    )
                if support_ids & set(query_ids):
                    raise ValueError(f"Support/query overlap for {subject_id}/{band}/k={budget}")
                if budget and not support_sets[budgets[budgets.index(budget) - 1]] <= support_ids:
                    raise ValueError(f"Support sets are not nested for {subject_id}/{band}")
                support_sets[budget] = support_ids
                class_counts = pd.Series(
                    [int(row["within_band_label"]) for row in support_rows], dtype="int64"
                ).value_counts()
                if budget and class_counts.to_dict() != {label: budget for label in range(4)}:
                    raise ValueError(
                        f"Support class counts drift for {subject_id}/{band}/k={budget}: "
                        f"{class_counts.to_dict()}"
                    )

                for row in support_rows:
                    assignments.append(_assignment_record(row, band, budget, role="support"))
                for _, row in queries.iterrows():
                    assignments.append(_assignment_record(row, band, budget, role="query"))

            group_receipts.append(
                {
                    "subject_id": subject_id,
                    "frequency_band": band,
                    "day1_trials": len(day1),
                    "complete_pseudo_blocks": len(support_blocks),
                    "day2_query_trials": len(queries),
                    "query_sample_ids_sha256": _string_list_sha256(query_ids),
                }
            )

    assignment_bytes = _canonical_jsonl(assignments)
    sample_identity_bytes = _canonical_jsonl(_sample_identity_records(manifest))
    expected_groups = int(data_cfg["expected"]["n_subjects"]) * len(processing_bands)
    if len(group_receipts) != expected_groups:
        raise ValueError(f"Expected {expected_groups} participant-band groups")
    expected_assignments = expected_groups * (4 * sum(budgets) + 80 * len(budgets))
    if len(assignments) != expected_assignments:
        raise ValueError(
            f"Expected {expected_assignments} partition assignments, observed {len(assignments)}"
        )
    summary = {
        "schema": "cfeg.choi2019-partition-plan.v1",
        "dataset_revision": data_cfg["dataset_revision"],
        "contains_scores_predictions_or_decoding_outcomes": False,
        "acquisition_band_order": "unavailable_not_reported",
        "processing_band_order": processing_bands,
        "processing_order_is_acquisition_metadata": False,
        "budgets_trials_per_class": budgets,
        "support_day": "day01",
        "query_day": "day02",
        "participant_band_groups": expected_groups,
        "complete_pseudo_blocks_per_group": 20,
        "query_trials_per_group_per_budget": 80,
        "assignment_records": len(assignments),
        "assignment_bytes": len(assignment_bytes),
        "partition_assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "sample_identity_records": len(manifest),
        "sample_identity_sha256": hashlib.sha256(sample_identity_bytes).hexdigest(),
        "group_query_identity_sha256": hashlib.sha256(_canonical_jsonl(group_receipts)).hexdigest(),
        "evaluation_channels": channel_receipt,
        "filterbank_weight_contract": weight_receipt,
    }
    _validate_assignment_plan(assignments, summary)
    return ChoiPartitionPlan(tuple(assignments), assignment_bytes, summary)


def _float64_bytes(values: Any) -> bytes:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Choi subband weights must be a finite one-dimensional vector")
    return array.astype("<f8", copy=False).tobytes(order="C")


def _validate_anchor_weights(anchor_cfg: dict[str, Any]) -> dict[str, Any]:
    """Require explicit, bitwise-stable V2 c4 weights for every Choi band."""

    common = anchor_cfg["common_parameters"]
    explicit = common.get("subband_weights")
    expected_bytes = _float64_bytes(_FROZEN_SUBBAND_WEIGHTS)
    if _float64_bytes(explicit) != expected_bytes:
        raise ValueError("Choi common subband weights differ bitwise from frozen V2 c4 weights")
    if common.get("subband_weight_resolution") != (
        "explicit_vector_only_no_runtime_exponentiation"
    ):
        raise ValueError("Choi execution must forbid runtime subband-weight exponentiation")
    if common.get("weight_formula_role") != "historical_provenance_only":
        raise ValueError("Choi exponent/offset must remain historical provenance only")

    resolved_receipts: dict[str, Any] = {}
    for band in ("LOW", "MID", "HIGH"):
        selected = anchor_cfg["band_parameters"][band]
        supplied_weights = selected.get("subband_weights", explicit)
        runtime = {
            "bands": selected["bands"],
            "weights": supplied_weights,
            "order": common["order"],
            "n_harmonics": selected["n_harmonics"],
            "regularization": common["regularization"],
            "filter_family": common["filter_family"],
            "passband_ripple_db": common["passband_ripple_db"],
            "reproduction_contract": selected["reproduction_contract"],
        }
        resolved = resolve_filterbank_parameters(
            runtime, float(anchor_cfg["sampling_frequency_hz"])
        )
        if _float64_bytes(resolved["weights"]) != expected_bytes:
            raise ValueError(
                f"Choi {band} resolved weights differ bitwise from frozen V2 c4 weights"
            )
        resolved_receipts[band] = {
            "n_subbands": len(resolved["weights"]),
            "resolved_float64_le_sha256": hashlib.sha256(
                _float64_bytes(resolved["weights"])
            ).hexdigest(),
            "bitwise_equal_to_explicit_common_vector": True,
        }
    return {
        "resolution": "explicit_vector_only_no_runtime_exponentiation",
        "float64_little_endian_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "weights": explicit,
        "bands": resolved_receipts,
    }


def _validate_manifest_grid(manifest: pd.DataFrame, cfg: dict[str, Any]) -> None:
    required = {
        "day_index",
        "frequency_band",
        "within_band_label",
        "source_marker_event_index",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Choi partition manifest is missing columns: {missing}")
    expected = cfg["expected"]
    if len(manifest) != int(expected["n_trials_total"]):
        raise ValueError(f"Choi partition manifest row-count drift: {len(manifest)}")
    subjects = set(manifest["subject_id"].astype(str))
    n_subjects = int(expected["n_subjects"])
    expected_subjects = {f"sub{value:03d}" for value in range(1, n_subjects + 1)}
    if subjects != expected_subjects:
        raise ValueError("Choi partition subject inventory drift")
    if set(manifest["session_id"].astype(str)) != {"day01", "day02"}:
        raise ValueError("Choi partition day/session inventory drift")
    expected_day_index = manifest["session_id"].astype(str).map({"day01": 1, "day02": 2})
    if not np.array_equal(
        manifest["day_index"].astype(int).to_numpy(), expected_day_index.to_numpy(dtype=int)
    ):
        raise ValueError("Choi day_index/session_id correspondence drift")
    if set(manifest["frequency_band"].astype(str)) != {"low", "mid", "high"}:
        raise ValueError("Choi partition frequency-band inventory drift")
    if set(manifest["within_band_label"].astype(int)) != {0, 1, 2, 3}:
        raise ValueError("Choi within-band label inventory drift")

    group_counts = manifest.groupby(["subject_id", "session_id", "frequency_band"]).size()
    if len(group_counts) != n_subjects * 2 * 3 or not group_counts.eq(80).all():
        raise ValueError("Choi participant/day/band grid must contain exactly 80 trials")
    class_counts = manifest.groupby(
        ["subject_id", "session_id", "frequency_band", "within_band_label"]
    ).size()
    if len(class_counts) != n_subjects * 2 * 3 * 4 or not class_counts.eq(20).all():
        raise ValueError("Choi participant/day/band/class grid must contain exactly 20 trials")

    for (subject_id, session_id, run_id), run in manifest.groupby(
        ["subject_id", "session_id", "run_id"], sort=False
    ):
        match = re.fullmatch(r"(low|mid|high)_session(01|02)", str(run_id))
        if not match or set(run["frequency_band"].astype(str)) != {match.group(1)}:
            raise ValueError(f"Malformed Choi run identity: {subject_id}/{session_id}/{run_id}")
        markers = sorted(run["source_marker_event_index"].astype(int))
        if len(run) != 40 or markers != list(range(0, 80, 2)):
            raise ValueError(f"Choi run marker grid drift: {subject_id}/{session_id}/{run_id}")
        if not run.groupby("within_band_label").size().eq(10).all():
            raise ValueError(f"Choi run class counts drift: {subject_id}/{session_id}/{run_id}")

    for band in cfg["processing_band_order"]:
        expected_frequencies = [
            float(value) for value in cfg["stimulus_by_band"][band]["frequencies_hz"]
        ]
        selected = manifest.loc[manifest["frequency_band"].astype(str) == band.lower()]
        observed = (
            selected["within_band_label"].astype(int).map(dict(enumerate(expected_frequencies)))
        )
        if not np.allclose(observed.to_numpy(float), selected["stimulus_frequency_hz"]):
            raise ValueError(f"Choi label/frequency mapping drift for {band}")


def _validate_evaluation_channels(
    manifest: pd.DataFrame,
    signals_path: Path,
    anchor_cfg: dict[str, Any],
) -> dict[str, Any]:
    view = anchor_cfg["evaluation_view"]
    names = [str(value) for value in view["evaluation_channel_names"]]
    ids = [int(value) for value in view["evaluation_canonical_channel_ids"]]
    if names != ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]:
        raise ValueError(f"Frozen Choi evaluation-channel names drifted: {names}")
    canonical = CanonicalChannelMap.from_yaml()
    if canonical.get_ids(names) != ids:
        raise ValueError("Frozen Choi evaluation channel names/IDs disagree with canonical map")
    required_names = {name.upper() for name in names}
    for value in manifest["channel_names_used"]:
        observed_names = {str(name).upper() for name in value}
        if not required_names <= observed_names:
            raise ValueError("A Choi manifest row is missing a frozen evaluation channel")
    for value in manifest["canonical_channel_ids"]:
        row_ids = np.asarray(value, dtype=int)
        if len(row_ids) != 64 or any(row_ids[channel_id - 1] != channel_id for channel_id in ids):
            raise ValueError("A Choi manifest row has an unbound evaluation-channel slot")
    if signals_path.is_symlink() or not signals_path.is_file():
        raise ValueError(f"Missing or unsafe Choi signals file: {signals_path}")
    slots = np.asarray(ids, dtype=int) - 1
    with h5py.File(signals_path, "r") as handle:
        if "channel_mask" not in handle or len(handle["channel_mask"]) != len(manifest):
            raise ValueError("Choi channel-mask/manifest length mismatch")
        for start in range(0, len(manifest), 1024):
            selected = handle["channel_mask"][start : start + 1024, slots]
            if selected.shape[1] != len(ids) or not np.asarray(selected, dtype=bool).all():
                raise ValueError("A Choi signal row masks a frozen evaluation channel")
    return {
        "names": names,
        "canonical_channel_ids": ids,
        "all_manifest_rows_present": True,
        "all_signal_masks_true": True,
    }


def _session_number(run_id: object, expected_band: str) -> int:
    match = re.fullmatch(r"(low|mid|high)_session(01|02)", str(run_id))
    if not match or match.group(1) != expected_band.lower():
        raise ValueError(f"Unexpected Choi run_id for {expected_band}: {run_id!r}")
    return int(match.group(2))


def _chronological(frame: pd.DataFrame, band: str) -> pd.DataFrame:
    result = frame.copy()
    result["_source_session_number"] = [_session_number(value, band) for value in result["run_id"]]
    return result.sort_values(
        ["_source_session_number", "source_marker_event_index", "sample_id"],
        kind="stable",
    )


def _pseudo_blocks(day1: pd.DataFrame, band: str) -> list[list[pd.Series]]:
    by_class = {
        label: _chronological(day1.loc[day1["within_band_label"].astype(int) == label], band)
        for label in range(4)
    }
    if any(len(frame) != 20 for frame in by_class.values()):
        raise ValueError(f"Choi Day-1 class count drift for pseudo-block construction: {band}")
    blocks = []
    for block_index in range(1, 21):
        block = []
        for label in range(4):
            row = by_class[label].iloc[block_index - 1].copy()
            row["_pseudo_block_index"] = block_index
            block.append(row)
        if {int(row["within_band_label"]) for row in block} != {0, 1, 2, 3}:
            raise ValueError(f"Incomplete Choi pseudo-block {block_index} for {band}")
        blocks.append(block)
    return blocks


def _assignment_record(
    row: pd.Series,
    band: str,
    budget: int,
    *,
    role: str,
) -> dict[str, Any]:
    return {
        "budget_trials_per_class": budget,
        "frequency_band": band,
        "h5_index": int(row["h5_index"]),
        "pseudo_block_index": (int(row["_pseudo_block_index"]) if role == "support" else None),
        "role": role,
        "sample_id": str(row["sample_id"]),
        "source_marker_event_index": int(row["source_marker_event_index"]),
        "source_session_number": int(row["_source_session_number"]),
        "subject_id": str(row["subject_id"]),
        "within_band_label": int(row["within_band_label"]),
    }


def _sample_identity_records(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for _, row in manifest.sort_values("h5_index").iterrows():
        records.append(
            {
                "day_index": int(row["day_index"]),
                "frequency_band": str(row["frequency_band"]).upper(),
                "h5_index": int(row["h5_index"]),
                "run_id": str(row["run_id"]),
                "sample_id": str(row["sample_id"]),
                "session_id": str(row["session_id"]),
                "source_marker_event_index": int(row["source_marker_event_index"]),
                "source_marker_sample_index_1based": int(row["source_marker_sample_index_1based"]),
                "stimulus_frequency_hz": float(row["stimulus_frequency_hz"]),
                "subject_id": str(row["subject_id"]),
                "trial_id": str(row["trial_id"]),
                "within_band_label": int(row["within_band_label"]),
            }
        )
    return records


def _canonical_jsonl(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
        for record in records
    )


def _string_list_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical_jsonl([{"sample_id": value} for value in values])).hexdigest()


def _validate_assignment_plan(assignments: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    frame = pd.DataFrame(assignments)
    budgets = summary["budgets_trials_per_class"]
    for (subject_id, band), group in frame.groupby(["subject_id", "frequency_band"]):
        query_reference: set[str] | None = None
        prior_support: set[str] = set()
        for budget in budgets:
            selected = group.loc[group["budget_trials_per_class"] == budget]
            support = selected.loc[selected["role"] == "support"]
            query = selected.loc[selected["role"] == "query"]
            support_ids = set(support["sample_id"])
            query_ids = set(query["sample_id"])
            if len(support_ids) != 4 * budget or len(query_ids) != 80:
                raise ValueError(f"Partition count drift for {subject_id}/{band}/k={budget}")
            if support_ids & query_ids or not prior_support <= support_ids:
                raise ValueError(f"Partition nesting/disjointness drift for {subject_id}/{band}")
            if query_reference is not None and query_ids != query_reference:
                raise ValueError(f"Query set changed across budgets for {subject_id}/{band}")
            query_reference = query_ids
            prior_support = support_ids


def verify_governance_contract(
    processed_dir: Path,
    repository_root: Path,
    governance_cfg: dict[str, Any],
    plan: ChoiPartitionPlan,
) -> dict[str, Any]:
    """Bind the deterministic partition to frozen processed/config bytes."""

    processed_dir = require_safe_directory(processed_dir, "Choi processed root")
    repository_root = require_safe_directory(repository_root, "Choi repository root")
    if governance_cfg.get("schema") != "cfeg.choi2019-processed-partition-governance.v1":
        raise ValueError("Unexpected Choi governance schema")
    if (
        governance_cfg.get("freeze_status") != "frozen_pre_outcome"
        or governance_cfg.get("contains_scores_predictions_or_decoding_outcomes") is not False
    ):
        raise ValueError("Choi governance must remain frozen and outcome-free")
    if governance_cfg.get("role") != "immutable_processed_asset_and_partition_identity_contract":
        raise ValueError("Choi governance role drift")
    if governance_cfg.get("dataset_revision") != plan.summary["dataset_revision"]:
        raise ValueError("Choi governance dataset revision drift")
    if set(governance_cfg["processed_assets"]) != _GOVERNED_PROCESSED_ASSETS:
        raise ValueError("Choi governed processed-asset inventory drift")
    if set(governance_cfg["repository_contracts"]) != _GOVERNED_REPOSITORY_CONTRACTS:
        raise ValueError("Choi governed repository-contract inventory drift")
    verified_assets = _verify_file_contracts(processed_dir, governance_cfg["processed_assets"])
    verified_contracts = _verify_file_contracts(
        repository_root, governance_cfg["repository_contracts"]
    )
    expected = governance_cfg["partition"]
    actual = {
        "assignment_records": plan.summary["assignment_records"],
        "assignment_bytes": plan.summary["assignment_bytes"],
        "partition_assignment_sha256": plan.summary["partition_assignment_sha256"],
        "sample_identity_records": plan.summary["sample_identity_records"],
        "sample_identity_sha256": plan.summary["sample_identity_sha256"],
        "group_query_identity_sha256": plan.summary["group_query_identity_sha256"],
    }
    if actual != expected:
        raise ValueError(f"Choi partition contract drift: expected={expected}, observed={actual}")
    if plan.summary["evaluation_channels"] != governance_cfg["evaluation_channels"]:
        raise ValueError("Choi evaluation-channel governance drift")
    if plan.summary["filterbank_weight_contract"] != governance_cfg["filterbank_weight_contract"]:
        raise ValueError("Choi explicit filterbank-weight governance drift")
    return {
        "schema": "cfeg.choi2019-processed-partition-governance-receipt.v1",
        "status": "verified_without_decoding_outcomes",
        "dataset_revision": governance_cfg["dataset_revision"],
        "contains_scores_predictions_or_decoding_outcomes": False,
        "processed_assets": verified_assets,
        "repository_contracts": verified_contracts,
        "partition": actual,
        "evaluation_channels": plan.summary["evaluation_channels"],
        "filterbank_weight_contract": plan.summary["filterbank_weight_contract"],
    }


def _verify_file_contracts(root: Path, contracts: dict[str, Any]) -> dict[str, Any]:
    verified = {}
    for relative, expected in contracts.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe governed Choi relative path: {relative!r}")
        path = require_safe_regular_file(root / relative_path, "governed Choi file")
        digest = hashlib.sha256()
        size = 0
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError(f"Governed Choi path is not a regular file: {path}")
            while chunk := handle.read(8 * 1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
        actual = {"bytes": size, "sha256": digest.hexdigest()}
        if actual != expected:
            raise ValueError(
                f"Governed Choi file drift for {relative}: expected={expected}, observed={actual}"
            )
        verified[relative] = actual
    return verified


def write_partition_bundle_atomically(
    out_dir: Path,
    plan: ChoiPartitionPlan,
    governance_receipt: dict[str, Any],
    governance_config_path: Path,
    *,
    governance_config_label: str | None = None,
) -> dict[str, Any]:
    """Write canonical assignments and a governance receipt by directory rename."""

    out_dir = _absolute_without_symlink_traversal(out_dir, must_exist=False)
    governance_config_path = require_safe_regular_file(
        governance_config_path, "Choi governance config"
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    require_safe_directory(out_dir.parent, "Choi partition output parent")
    if os.path.lexists(out_dir):
        raise FileExistsError(f"Refusing to overwrite Choi partition bundle: {out_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=str(out_dir.parent)))
    assignment_path = staging / "partition_assignments.jsonl"
    _write_durable_read_only(assignment_path, plan.assignment_bytes)
    config_digest = _sha256_path(governance_config_path)
    receipt = {
        **governance_receipt,
        "governance_config": {
            "path": governance_config_label or str(governance_config_path),
            "bytes": governance_config_path.stat().st_size,
            "sha256": config_digest,
        },
        "partition_artifact": {
            "name": assignment_path.name,
            "bytes": assignment_path.stat().st_size,
            "sha256": _sha256_path(assignment_path),
        },
        "receipt_integrity": {
            "algorithm": "sha256",
            "detached_digest_file": "governance_receipt.sha256",
            "covered_file": "governance_receipt.json",
        },
    }
    receipt_path = staging / "governance_receipt.json"
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_durable_read_only(receipt_path, receipt_bytes)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    digest_path = staging / "governance_receipt.sha256"
    digest_bytes = f"{receipt_sha256}  governance_receipt.json\n".encode("ascii")
    _write_durable_read_only(digest_path, digest_bytes)
    if receipt["partition_artifact"]["sha256"] != plan.summary["partition_assignment_sha256"]:
        raise ValueError("Written Choi partition digest differs from constructed plan")
    if _sha256_path(receipt_path) != receipt_sha256:
        raise ValueError("Written Choi governance receipt differs from its detached digest")
    expected_modes = {
        assignment_path.name: stat.S_IMODE(assignment_path.stat(follow_symlinks=False).st_mode),
        receipt_path.name: stat.S_IMODE(receipt_path.stat(follow_symlinks=False).st_mode),
        digest_path.name: stat.S_IMODE(digest_path.stat(follow_symlinks=False).st_mode),
    }
    if set(expected_modes.values()) != {0o400}:
        raise ValueError(f"Choi partition files are not read-only: {expected_modes}")
    _fsync_directory(staging)
    if os.path.lexists(out_dir):
        raise FileExistsError(f"Choi partition target appeared; staging preserved at {staging}")
    os.rename(staging, out_dir)
    _fsync_directory(out_dir.parent)
    return receipt


def _absolute_without_symlink_traversal(path: Path, *, must_exist: bool) -> Path:
    """Return a lexical absolute path after rejecting symlinks in every existing component."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.anchor)
    missing = False
    for component in absolute.parts[1:]:
        current /= component
        if missing:
            continue
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            missing = True
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Symlink path component is forbidden for Choi assets: {current}")
    if must_exist and missing:
        raise FileNotFoundError(f"Required Choi path does not exist: {absolute}")
    return absolute


def require_safe_directory(path: Path, description: str) -> Path:
    absolute = _absolute_without_symlink_traversal(path, must_exist=True)
    if not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise ValueError(f"{description} is not a directory: {absolute}")
    return absolute


def require_safe_regular_file(path: Path, description: str) -> Path:
    absolute = _absolute_without_symlink_traversal(path, must_exist=True)
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"{description} is not a regular file: {absolute}")
    return absolute


def _write_durable_read_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o400)
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"Choi governed path is not a regular file: {path}")
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
