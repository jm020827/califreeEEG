from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cfeg.baselines.metadata_calibration_adapter import (
    DESCRIPTORS,
    GovernedCalibrationBaselineAdapter,
)
from cfeg.utils.config import load_config

_REPOSITORY = Path(__file__).resolve().parents[2]
_SCORE_INTERPRETATION = "uncalibrated_raw_class_score_argmax_only"
_SCORE_SCHEMAS = {
    "strict_FBCCA": "strict_fbcca_bound_filterbank_raw_scores_v1",
    "target_template_correlation": "multichannel_mean_template_pearson_v1",
    "target_filterbank_eTRCA": "filterbank_ensemble_trca_weighted_signed_square_v1",
    "same3_filterbank_eTRCA": "same3_filterbank_ensemble_trca_common_protocol_v1",
    "chiang2021_LST_filterbank_eTRCA": (
        "chiang2021_lst_filterbank_ensemble_trca_cleanroom_v1"
    ),
}


@dataclass(frozen=True)
class BaselineExecutionBinding:
    adapter_id: str
    implementation_sha256: str
    config_sha256: str
    adapter_contract_sha256: str
    score_schema: str
    score_interpretation: str
    filterbank_path: Path | None
    filterbank_file_sha256: str | None
    filterbank: dict[str, Any] | None


def baseline_execution_binding(adapter_id: str) -> BaselineExecutionBinding:
    if adapter_id not in DESCRIPTORS:
        raise ValueError(f"Unknown baseline adapter {adapter_id!r}.")
    descriptor = DESCRIPTORS[adapter_id]  # type: ignore[index]
    if adapter_id == "strict_FBCCA":
        filterbank_path = _REPOSITORY / "configs/baselines/fbcca_chen2015_m3.yaml"
    elif adapter_id == "target_template_correlation":
        filterbank_path = None
    else:
        filterbank_path = (
            _REPOSITORY / "configs/baselines/fb_etrac_chiang2021_common_protocol.yaml"
        )
    filterbank = (
        None
        if filterbank_path is None
        else load_config(filterbank_path, strict_env=False)
    )
    filterbank_hash = (
        None if filterbank_path is None else _sha256_file(filterbank_path)
    )
    wearable_path = _REPOSITORY / "configs/data/wearable.yaml"
    wearable = (
        load_config(wearable_path, strict_env=False)
        if descriptor.uses_frequency_codebook
        else None
    )
    config_payload = {
        "schema": "cfeg.metadata-calibration-baseline-config-binding.v1",
        "adapter_id": adapter_id,
        "n_classes": 12,
        "sfreq": 200.0,
        "official_canonical_channel_ids": [56, 55, 57, 54, 58, 62, 61, 63],
        "filterbank_file_sha256": filterbank_hash,
        "filterbank_payload": filterbank,
        "wearable_config_file_sha256": (
            _sha256_file(wearable_path)
            if descriptor.uses_frequency_codebook
            else None
        ),
        "frequency_codebook": (
            list(wearable["class_frequencies"])
            if wearable is not None
            else None
        ),
        "phase_codebook": (
            list(wearable["class_phases"])
            if wearable is not None
            else None
        ),
    }
    implementation_files = (
        "src/cfeg/baselines/metadata_calibration_adapter.py",
        "src/cfeg/baselines/calibration.py",
        "src/cfeg/baselines/fbcca.py",
        "src/cfeg/data/metadata_calibration_baseline_sealed.py",
    )
    implementation_payload = {
        "schema": "cfeg.metadata-calibration-baseline-implementation-binding.v1",
        "adapter_id": adapter_id,
        "files": [
            {
                "path": relative,
                "file_sha256": _sha256_file(_REPOSITORY / relative),
            }
            for relative in implementation_files
        ],
    }
    defaults = GovernedCalibrationBaselineAdapter(
        adapter_id,  # type: ignore[arg-type]
        n_classes=12,
        filterbank=filterbank,
        filterbank_config_path=filterbank_path,
        filterbank_config_sha256=filterbank_hash,
    )
    contract_payload = {
        "schema": "cfeg.metadata-calibration-baseline-adapter-contract.v1",
        "descriptor": asdict(descriptor),
        "lst_ridge_ratio": defaults.lst_ridge_ratio,
        "trca_ridge_ratio": defaults.trca_ridge_ratio,
        "same_seed": defaults.same_seed,
        "score_schema": _SCORE_SCHEMAS[adapter_id],
        "score_interpretation": _SCORE_INTERPRETATION,
        "argmax_tie_break": "lowest_zero_based_class_id",
        "seed_reduction": "none_deterministic_once",
    }
    return BaselineExecutionBinding(
        adapter_id=adapter_id,
        implementation_sha256=_json_sha256(implementation_payload),
        config_sha256=_json_sha256(config_payload),
        adapter_contract_sha256=_json_sha256(contract_payload),
        score_schema=_SCORE_SCHEMAS[adapter_id],
        score_interpretation=_SCORE_INTERPRETATION,
        filterbank_path=filterbank_path,
        filterbank_file_sha256=filterbank_hash,
        filterbank=filterbank,
    )


def all_baseline_execution_bindings() -> dict[str, BaselineExecutionBinding]:
    return {
        adapter_id: baseline_execution_binding(adapter_id)
        for adapter_id in DESCRIPTORS
    }


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
