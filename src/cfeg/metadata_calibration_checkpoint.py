from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from cfeg.models.metadata_calibration_composite import MetadataCalibrationComposite

CHECKPOINT_FORMAT = "safetensors_tensor_map_v1"
CHECKPOINT_RECEIPT_SCHEMA = "cfeg.metadata-calibration-composite-receipt.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METADATA_PREFIX = "prior.metadata_precision_encoder."
_CHECKPOINT_RECEIPT_FIELDS = {
    "schema",
    "candidate_id",
    "phase",
    "seed",
    "outer_fold",
    "checkpoint_sha256",
    "checkpoint_format",
    "full_model_state",
    "trainable_only_checkpoint",
    "parameter_schema_sha256",
    "stage1_common_state_sha256",
    "final_common_state_sha256",
    "initial_metadata_state_sha256",
    "final_metadata_state_sha256",
    "stage2_trainable_names",
    "stage1_state_sha256",
    "stage1_recipe_sha256",
    "stage2_recipe_sha256",
    "plan_sha256",
    "decision_receipt_sha256",
    "asset_receipt_sha256",
    "asset_fingerprint_bundle_sha256",
    "class_map_sha256",
    "source_tree_sha256",
    "source_commit",
    "source_tag",
    "receipt_payload_sha256",
}


@dataclass(frozen=True)
class Stage1FreezeBinding:
    common_state_sha256: str
    initial_metadata_state_sha256: str
    parameter_schema_sha256: str


def capture_stage1_freeze(model: MetadataCalibrationComposite) -> Stage1FreezeBinding:
    """Bind the exact common path immediately before stage-2 M fitting."""

    state = model.state_dict()
    return Stage1FreezeBinding(
        common_state_sha256=tensor_state_sha256(_common_state(state)),
        initial_metadata_state_sha256=tensor_state_sha256(_metadata_state(state)),
        parameter_schema_sha256=parameter_schema_sha256(state),
    )


def write_full_composite_checkpoint(
    model: MetadataCalibrationComposite,
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    phase: Literal["source_development", "held_participant_evaluation"],
    seed: int,
    outer_fold: int | None,
    stage1_freeze: Stage1FreezeBinding,
    stage1_state_sha256: str,
    stage1_recipe_sha256: str,
    stage2_recipe_sha256: str,
    plan_sha256: str,
    decision_receipt_sha256: str,
    asset_receipt_sha256: str,
    asset_fingerprint_bundle_sha256: str,
    class_map_sha256: str,
    source_commit: str,
    source_tree_sha256: str,
    source_tag: str,
) -> dict[str, Any]:
    """Atomically save one full state and a receipt proving stage-2 isolation."""

    checkpoint_path = Path(checkpoint_path)
    receipt_path = Path(receipt_path)
    if checkpoint_path.exists() or receipt_path.exists():
        raise FileExistsError("Composite checkpoint and receipt paths must be new.")
    if phase == "source_development" and outer_fold not in {0, 1, 2}:
        raise ValueError("Development checkpoints require outer_fold 0, 1, or 2.")
    if phase == "held_participant_evaluation" and outer_fold is not None:
        raise ValueError("Held-participant checkpoints must use outer_fold=None.")
    hash_fields = {
        "stage1_state_sha256": stage1_state_sha256,
        "stage1_recipe_sha256": stage1_recipe_sha256,
        "stage2_recipe_sha256": stage2_recipe_sha256,
        "plan_sha256": plan_sha256,
        "decision_receipt_sha256": decision_receipt_sha256,
        "asset_receipt_sha256": asset_receipt_sha256,
        "asset_fingerprint_bundle_sha256": asset_fingerprint_bundle_sha256,
        "class_map_sha256": class_map_sha256,
        "source_tree_sha256": source_tree_sha256,
    }
    if not all(_is_sha256(value) for value in hash_fields.values()):
        raise ValueError("Every composite checkpoint binding except source_commit must be SHA-256.")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_commit)):
        raise ValueError("source_commit must be a full lowercase Git commit hash.")
    if not source_tag.strip():
        raise ValueError("source_tag must be nonempty.")

    state = model.state_dict()
    final_common = tensor_state_sha256(_common_state(state))
    if final_common != stage1_freeze.common_state_sha256:
        raise ValueError("Common backbone/Q/anchor state changed after the stage-1 freeze.")
    parameter_schema = parameter_schema_sha256(state)
    if parameter_schema != stage1_freeze.parameter_schema_sha256:
        raise ValueError("Composite parameter schema changed after the stage-1 freeze.")
    trainable_names = model.stage2_trainable_parameter_names()
    if not trainable_names or not all(name.startswith(_METADATA_PREFIX) for name in trainable_names):
        raise ValueError("Only prior.metadata_precision_encoder parameters may remain trainable.")

    serialized_state = {
        str(name): value.detach().contiguous().cpu() for name, value in state.items()
    }
    _atomic_safetensors_save(serialized_state, checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    receipt: dict[str, Any] = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "candidate_id": "metadata-calibration-efficiency-v1",
        "phase": phase,
        "seed": int(seed),
        "outer_fold": outer_fold,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "full_model_state": True,
        "trainable_only_checkpoint": False,
        "parameter_schema_sha256": parameter_schema,
        "stage1_common_state_sha256": stage1_freeze.common_state_sha256,
        "final_common_state_sha256": final_common,
        "initial_metadata_state_sha256": stage1_freeze.initial_metadata_state_sha256,
        "final_metadata_state_sha256": tensor_state_sha256(_metadata_state(state)),
        "stage2_trainable_names": sorted(trainable_names),
        **hash_fields,
        "source_commit": source_commit,
        "source_tag": source_tag,
    }
    receipt["receipt_payload_sha256"] = _json_payload_sha256(receipt)
    _atomic_json_save(receipt, receipt_path)
    return receipt


def verify_full_composite_checkpoint(
    checkpoint_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or set(receipt) != _CHECKPOINT_RECEIPT_FIELDS:
        raise ValueError("Composite checkpoint receipt has an invalid exact schema.")
    payload_digest = receipt.pop("receipt_payload_sha256", None)
    if payload_digest != _json_payload_sha256(receipt):
        raise ValueError("Composite receipt payload hash is invalid.")
    receipt["receipt_payload_sha256"] = payload_digest
    if (
        receipt.get("schema") != CHECKPOINT_RECEIPT_SCHEMA
        or receipt.get("candidate_id") != "metadata-calibration-efficiency-v1"
        or receipt.get("phase") not in {
            "source_development",
            "held_participant_evaluation",
        }
        or type(receipt.get("seed")) is not int
        or receipt.get("checkpoint_format") != CHECKPOINT_FORMAT
        or receipt.get("full_model_state") is not True
        or receipt.get("trainable_only_checkpoint") is not False
    ):
        raise ValueError("Unknown composite checkpoint receipt schema.")
    if (
        receipt["phase"] == "source_development"
        and type(receipt.get("outer_fold")) is not int
    ) or (
        receipt["phase"] == "held_participant_evaluation"
        and receipt.get("outer_fold") is not None
    ):
        raise ValueError("Composite receipt phase/fold binding is invalid.")
    hash_fields = {
        key: value
        for key, value in receipt.items()
        if key.endswith("_sha256") and key != "receipt_payload_sha256"
    }
    if not hash_fields or not all(_is_sha256(value) for value in hash_fields.values()):
        raise ValueError("Composite receipt contains an invalid SHA-256 binding.")
    if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source_commit"))) or not str(
        receipt.get("source_tag", "")
    ).strip():
        raise ValueError("Composite receipt source revision binding is invalid.")
    if receipt.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise ValueError("Composite checkpoint file hash differs from its receipt.")
    state = load_file(checkpoint_path, device="cpu")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Composite checkpoint has no full state dictionary.")
    if not any(str(key).startswith("backbone.") for key in state) or not any(
        str(key).startswith("prior.") for key in state
    ):
        raise ValueError("Composite checkpoint is missing backbone or prior state.")
    if parameter_schema_sha256(state) != receipt["parameter_schema_sha256"]:
        raise ValueError("Composite checkpoint parameter schema hash is invalid.")
    if tensor_state_sha256(_common_state(state)) != receipt["final_common_state_sha256"]:
        raise ValueError("Composite checkpoint common-state hash is invalid.")
    if tensor_state_sha256(_metadata_state(state)) != receipt["final_metadata_state_sha256"]:
        raise ValueError("Composite checkpoint metadata-state hash is invalid.")
    if receipt["stage1_common_state_sha256"] != receipt["final_common_state_sha256"]:
        raise ValueError("Receipt does not prove exact stage-1 common-state preservation.")
    trainable_names = receipt.get("stage2_trainable_names")
    expected_trainables = sorted(_metadata_state(state))
    if (
        not isinstance(trainable_names, list)
        or not trainable_names
        or any(type(name) is not str for name in trainable_names)
        or len(trainable_names) != len(set(trainable_names))
        or trainable_names != expected_trainables
        or any(not name.startswith(_METADATA_PREFIX) for name in trainable_names)
    ):
        raise ValueError("Receipt does not bind the exact metadata-only trainable set.")
    return receipt


def parameter_schema_sha256(state: Mapping[str, torch.Tensor]) -> str:
    schema = [
        {"name": str(name), "dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in sorted(state.items())
    ]
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().contiguous().cpu()
        header = json.dumps(
            {"name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = (
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            if tensor.numel()
            else b""
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _common_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if not name.startswith(_METADATA_PREFIX)}


def _metadata_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    selected = {name: value for name, value in state.items() if name.startswith(_METADATA_PREFIX)}
    if not selected:
        raise ValueError("Composite state has no metadata residual parameters.")
    return selected


def _atomic_safetensors_save(
    state: Mapping[str, torch.Tensor], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        save_file(dict(state), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_file_noreplace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(encoded)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        _publish_file_noreplace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(_SHA256.fullmatch(str(value)))


def _publish_file_noreplace(staging: Path, destination: Path) -> None:
    """Atomically link a new file without ever replacing a raced destination."""

    try:
        os.link(staging, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError("Checkpoint artifact destination already exists.") from error
    staging.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
