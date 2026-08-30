from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from cfeg.constants import (
    CONDITION_CATEGORICAL_FIELDS,
    METADATA_CONTRACT_V04_DEV,
    PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS,
)
from cfeg.data.collate import (
    build_vocabularies,
    collate_eeg,
    metadata_collate_kwargs,
    overwrite_external_metadata,
)
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.loader import resolve_loader_settings
from cfeg.data.metadata_controls import (
    build_development_control_plan,
    metadata_shuffle_indices,
)
from cfeg.data.preprocess import CanonicalChannelMap
from cfeg.data.schema import load_manifest
from cfeg.data.transforms import _update_n_channels
from cfeg.governance import (
    GovernanceError,
    bind_cohort,
    current_source_revision_contract,
    validate_checkpoint_evaluation_access,
)
from cfeg.metrics import classification_metrics, confusion_matrix, itr_bits_per_min
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.runtime import resolve_device
from cfeg.seed import seed_everything
from cfeg.train_loop import _processed_asset_provenance, _sha256_json, _to_device
from cfeg.utils.checkpoint import load_checkpoint, load_checkpoint_model_state, save_json


def run_evaluation(
    eval_cfg: dict,
    ckpt_path: str | Path,
    *,
    access_route: str = "generic_evaluation",
) -> dict:
    seed_everything(int(eval_cfg.get("seed", 42)))
    context = load_evaluation_context(eval_cfg, ckpt_path, access_route)
    mode = eval_cfg.get("mode", eval_cfg.get("run_name", "standard"))
    _validate_fresh_governed_evaluation_output(eval_cfg, context, str(mode))
    if mode == "calibration":
        return _run_calibration(eval_cfg, context)
    scenarios = _scenarios(eval_cfg, context, mode)
    rows: list[dict] = []
    confusion_rows: list[dict] = []
    donor_artifacts: list[tuple[str, pd.DataFrame]] = []
    scenario_provenance: list[dict] = []
    trial_time_sec = float(eval_cfg.get("trial_time_sec", 2.0))
    baseline_accuracy, baseline_itr = _reference_metrics(
        mode, context, trial_time_sec=trial_time_sec
    )
    for name, indices, perturb in scenarios:
        loader = _loader(context, indices)
        y_true, logits, sample_ids = collect_predictions(
            context["model"], loader, context["device"], perturb=perturb
        )
        metrics = classification_metrics(y_true, logits, trial_time_sec=trial_time_sec)
        perturbation_audit = getattr(perturb, "audit", {}) if perturb is not None else {}
        metrics.update(
            {key: value for key, value in perturbation_audit.items() if not key.startswith("_")}
        )
        donor_mapping = perturbation_audit.get("_donor_mapping")
        if donor_mapping is not None:
            donor_artifacts.append((name, donor_mapping))
        scenario_provenance.append(
            {
                "scenario": name,
                "n_samples": len(sample_ids),
                "sample_id_identity_sha256": _sample_id_identity_sha256(sample_ids),
                "metadata_control": {
                    key: value
                    for key, value in perturbation_audit.items()
                    if not key.startswith("_")
                },
            }
        )
        if baseline_accuracy is None:
            baseline_accuracy = metrics["accuracy"]
        if baseline_itr is None:
            baseline_itr = metrics.get("itr_bits_per_min")
        accuracy_drop = float(baseline_accuracy - metrics["accuracy"])
        accuracy_drop_rate = _relative_drop(baseline_accuracy, metrics["accuracy"])
        metrics.update(
            {
                "reference_accuracy": baseline_accuracy,
                "accuracy_drop": accuracy_drop,
                "accuracy_drop_rate": accuracy_drop_rate,
                "generalization_drop": accuracy_drop,
                "generalization_drop_rate": accuracy_drop_rate,
                "scenario": name,
                "mode": mode,
            }
        )
        if baseline_itr is not None and "itr_bits_per_min" in metrics:
            metrics["reference_itr_bits_per_min"] = baseline_itr
            metrics["itr_drop"] = float(baseline_itr - metrics["itr_bits_per_min"])
            metrics["itr_drop_rate"] = _relative_drop(baseline_itr, metrics["itr_bits_per_min"])
        rows.append(metrics)
        matrix = confusion_matrix(y_true, logits.argmax(axis=1), logits.shape[1])
        confusion_rows.extend(_confusion_rows(name, matrix))
        _save_predictions(eval_cfg, context, name, sample_ids, y_true, logits)

    output_csv = _output_path(eval_cfg, context, mode)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    pd.DataFrame(confusion_rows).to_csv(
        output_csv.with_name(f"{output_csv.stem}_confusion.csv"), index=False
    )
    for scenario_name, mapping in donor_artifacts:
        mapping.to_csv(
            output_csv.with_name(f"{output_csv.stem}_{scenario_name}_donors.csv"), index=False
        )
    provenance_path = _save_evaluation_provenance(
        eval_cfg,
        context,
        output_csv,
        scenario_provenance,
    )
    return {
        "output_csv": str(output_csv),
        "results": rows,
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256_file(provenance_path),
    }


def _reference_metrics(
    mode: str, context: dict, *, trial_time_sec: float
) -> tuple[float | None, float | None]:
    """Use source validation metrics as the zero-shot transfer baseline."""
    if mode not in {"cross_dataset", "cross_condition"}:
        return None, None
    value = context["checkpoint"].get("best_metric")
    if value is None:
        return None, None
    accuracy = float(value)
    n_classes = int(context["train_config"]["model"]["n_classes"])
    return accuracy, itr_bits_per_min(n_classes, accuracy, trial_time_sec)


def _validate_fresh_governed_evaluation_output(
    eval_cfg: dict, context: dict, mode: str
) -> None:
    if not context["research_access"].governed:
        return
    output = _output_path(eval_cfg, context, mode)
    existing = sorted(path for path in output.parent.glob(f"{output.stem}*") if path.is_file())
    if existing:
        raise GovernanceError(
            "Governed evaluation output prefix is not fresh: "
            f"{output}. Existing artifacts are never overwritten; choose a new output_csv."
        )


def _relative_drop(reference: float, observed: float) -> float:
    if abs(reference) < 1e-12:
        return 0.0
    return float((reference - observed) / reference)


def run_channel_stress_eval(eval_cfg: dict, ckpt_path: str | Path) -> dict:
    cfg = copy.deepcopy(eval_cfg)
    cfg["mode"] = "channel_stress"
    return run_evaluation(cfg, ckpt_path)


def load_evaluation_context(
    eval_cfg: dict,
    ckpt_path: str | Path,
    access_route: str = "generic_evaluation",
) -> dict:
    checkpoint_path = Path(ckpt_path).expanduser().resolve()
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    research_access = validate_checkpoint_evaluation_access(
        cfg,
        eval_cfg,
        checkpoint_path=checkpoint_path,
        access_route=access_route,
    )
    device = resolve_device(cfg)
    data_dirs = eval_cfg.get("data", {}).get("processed_dirs") or cfg["data"]["processed_dirs"]
    asset_manifest = pd.concat([load_manifest(root) for root in data_dirs], ignore_index=True)
    cohort = bind_cohort(asset_manifest, research_access)
    expected_revisions = eval_cfg.get("data", {}).get("expected_revisions") or cfg.get(
        "data", {}
    ).get("expected_revisions")
    _validate_checkpoint_asset_revisions(ckpt, expected_revisions or {})
    expected_counts = eval_cfg.get("data", {}).get("expected_dataset_counts") or cfg.get(
        "data", {}
    ).get("expected_dataset_counts")
    dataset = EEGProcessedDataset(
        data_dirs,
        expected_revisions=expected_revisions,
        expected_protocol=cfg.get("protocol"),
        expected_dataset_counts=expected_counts,
        require_audit_receipt=bool(
            eval_cfg.get("data", {}).get("require_audit_receipt")
            or cfg.get("data", {}).get("require_audit_receipt", False)
        ),
        allowed_subject_ids=(
            set(cohort.selected_subject_ids) if research_access.governed else None
        ),
        persistent_hdf5_handles=bool(
            (eval_cfg.get("data") or {}).get(
                "persistent_hdf5_handles",
                cfg.get("data", {}).get("persistent_hdf5_handles", False),
            )
        ),
        preload_hdf5_to_memory=bool(
            (eval_cfg.get("data") or {}).get(
                "preload_hdf5_to_memory",
                cfg.get("data", {}).get("preload_hdf5_to_memory", False),
            )
        ),
    )
    _validate_checkpoint_dataset_provenance(ckpt, dataset)
    vocab = ckpt.get("vocabularies") or build_vocabularies()
    model = ConditionedEEGDecoder(
        cfg, vocab_sizes={key: len(value) for key, value in vocab.items()}
    )
    load_checkpoint_model_state(model, ckpt)
    model.to(device)
    manifest = pd.DataFrame([entry[2] for entry in dataset.entries])
    if tuple(sorted(manifest["subject_id"].astype(str).unique())) != cohort.selected_subject_ids:
        raise ValueError("Evaluation dataset cohort view differs from its governance binding.")
    return {
        "checkpoint": ckpt,
        "checkpoint_path": checkpoint_path,
        "train_config": cfg,
        "dataset": dataset,
        "manifest": manifest,
        "vocab": vocab,
        "device": device,
        "model": model,
        "research_access": research_access,
        "cohort": cohort,
        "eval_config": eval_cfg,
    }


def _validate_checkpoint_asset_revisions(
    checkpoint: dict, expected_revisions: dict[str, str]
) -> None:
    if not expected_revisions:
        return
    entries = checkpoint.get("asset_info", {}).get("datasets", [])
    observed = {
        str(entry.get("dataset_id")): str(entry.get("dataset_revision"))
        for entry in entries
        if entry.get("dataset_id") is not None
    }
    for dataset_id, expected_revision in expected_revisions.items():
        if observed.get(str(dataset_id)) != str(expected_revision):
            raise ValueError(
                "Checkpoint/data revision mismatch: "
                f"checkpoint has {dataset_id}={observed.get(str(dataset_id), 'missing')!r}, "
                f"evaluation requires {expected_revision!r}. Retrain from the verified "
                "processed revision instead of reusing a stale checkpoint."
            )


def _validate_checkpoint_dataset_provenance(checkpoint: dict, dataset) -> None:
    checkpoint_entries = checkpoint.get("asset_info", {}).get("datasets", [])
    checkpoint_by_id = {
        str(entry.get("dataset_id")): entry
        for entry in checkpoint_entries
        if entry.get("dataset_id") is not None
    }
    trained_paths = [
        str(path).lower()
        for path in checkpoint.get("config", {}).get("data", {}).get("processed_dirs", [])
    ]
    checkpoint_data_cfg = checkpoint.get("config", {}).get("data", {})
    claimed_ids = set(checkpoint_data_cfg.get("expected_revisions", {}))
    claimed_ids.update(str(value) for value in checkpoint_data_cfg.get("train_datasets", []))
    claimed_ids.update(str(value) for value in checkpoint_data_cfg.get("test_datasets", []))
    protocol = checkpoint.get("config", {}).get("protocol", {})
    if str(protocol.get("metadata_contract_version", "legacy")) == "0.4-dev":
        current_asset_info = _processed_asset_provenance(dataset.roots)
        checkpoint_asset_info = checkpoint.get("asset_info")
        if checkpoint_asset_info is None or not _provenance_values_equal(
            checkpoint_asset_info, current_asset_info
        ):
            raise ValueError(
                "Protocol 0.4 checkpoint asset bytes differ from the current processed "
                "asset provenance. Re-evaluate with the exact audited training asset."
            )
        expected_asset_hash = (
            checkpoint.get("config", {}).get("runtime_contract", {}).get("asset_provenance_sha256")
        )
        if expected_asset_hash != _sha256_json(current_asset_info):
            raise ValueError(
                "Protocol 0.4 checkpoint runtime asset hash does not match its current "
                "processed asset fingerprints."
            )
    provenance_fields = [
        "dataset_revision",
        "query_qc_extractor_version",
        "external_continuous_schema",
    ]
    for current in dataset.asset_infos:
        dataset_id = current.get("dataset_id")
        if dataset_id is None:
            continue
        dataset_id = str(dataset_id)
        previous = checkpoint_by_id.get(dataset_id)
        checkpoint_claims_dataset = dataset_id in claimed_ids or any(
            dataset_id.lower() in path for path in trained_paths
        )
        if previous is None:
            if checkpoint_claims_dataset:
                raise ValueError(
                    f"Checkpoint was trained on {dataset_id!r} but has no revisioned dataset "
                    "provenance. Refuse to combine it with a current processed asset."
                )
            continue
        fields = list(provenance_fields)
        if str(current.get("dataset_revision")) == "wearable_v3":
            fields.extend(
                [
                    "impedance_numeric_signature",
                    "processed_subject_count",
                ]
            )
        for field in fields:
            expected = previous.get(field)
            observed = current.get(field)
            must_compare = (
                field == "dataset_revision"
                or str(protocol.get("metadata_contract_version", "legacy")) == "0.4-dev"
            )
            if field not in provenance_fields:
                must_compare = True
            if must_compare and not _provenance_values_equal(expected, observed):
                raise ValueError(
                    f"Checkpoint/current dataset provenance mismatch for {dataset_id}.{field}: "
                    f"checkpoint={expected!r}, current={observed!r}."
                )


def _provenance_values_equal(left, right) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
    *,
    perturb: Callable[[torch.Tensor, dict], tuple[torch.Tensor, dict]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in loader:
        batch = _to_device(batch, device)
        if perturb is not None:
            if getattr(perturb, "uses_sample_ids", False):
                batch["x"], batch["cond"] = perturb(
                    batch["x"], batch["cond"], batch["sample_id"]
                )
            else:
                batch["x"], batch["cond"] = perturb(batch["x"], batch["cond"])
        out = model(batch["x"], batch["cond"], use_latent=False)
        labels.append(batch["y"].detach().cpu().numpy())
        logits.append(out.logits.detach().float().cpu().numpy())
        sample_ids.extend(batch["sample_id"])
    if not labels:
        raise ValueError("Evaluation selection contains no samples.")
    return np.concatenate(labels), np.concatenate(logits), sample_ids


def _scenarios(eval_cfg: dict, context: dict, mode: str):
    if mode == "channel_stress":
        test_indices = _held_out_evaluation_indices(eval_cfg, context, mode=mode)
        return [
            (name, test_indices, _channel_perturbation(name))
            for name in eval_cfg.get("channel_sets", ["all"])
        ]
    if mode == "cross_dataset":
        test_datasets = eval_cfg.get("test_datasets") or eval_cfg.get("data", {}).get(
            "test_datasets"
        )
        if not test_datasets:
            raise ValueError("cross_dataset evaluation requires test_datasets.")
        indices = _held_out_evaluation_indices(eval_cfg, context, mode=mode)
        if eval_cfg.get("common_label_subset_only", True):
            indices = _common_label_indices(context, indices)
        return [("zero_shot_" + "_".join(test_datasets), indices, None)]
    if mode == "cross_condition":
        indices = _held_out_evaluation_indices(eval_cfg, context, mode=mode)
        return [("zero_shot_condition", indices, None)]
    if mode == "robustness":
        test_indices = _held_out_evaluation_indices(eval_cfg, context, mode=mode)
        scenarios = [("clean", test_indices, None)]
        for spec in eval_cfg.get("perturbations", []):
            name = str(spec["name"])
            kind = spec.get("type", spec.get("name"))
            perturbation = (
                _global_metadata_donor_perturbation(spec, context, test_indices)
                if kind in {"metadata_shuffle", "metadata_counterfactual_wet_dry"}
                else _robustness_perturbation(spec)
            )
            scenarios.append((name, test_indices, perturbation))
        return scenarios
    test_indices = _held_out_evaluation_indices(eval_cfg, context, mode=mode)
    return [("standard", test_indices, None)]


def _held_out_evaluation_indices(eval_cfg: dict, context: dict, *, mode: str) -> np.ndarray:
    """Resolve an auditable held-out selection for robustness-style evaluations."""
    manifest = context["manifest"]
    data_cfg = eval_cfg.get("data", {})
    split_name = data_cfg.get("split")
    test_datasets = eval_cfg.get("test_datasets") or data_cfg.get("test_datasets")
    test_filter = eval_cfg.get("test_filter") or data_cfg.get("test_filter")

    if split_name is not None and str(split_name) not in {"test", "val"}:
        raise ValueError(
            f"{mode} data.split must be 'test' or governed development 'val', got "
            f"{split_name!r}."
        )
    if split_name is None and not test_datasets and not test_filter:
        raise ValueError(
            f"{mode} evaluation has no safe held-out selection. Refusing to evaluate the "
            "entire manifest because it may mix train/val/test rows. Set data.split: test "
            "to use the checkpoint-adjacent split.csv, or declare test_datasets/test_filter."
        )

    mask = np.ones(len(manifest), dtype=bool)
    if split_name in {"test", "val"}:
        split_indices = _checkpoint_split_indices(context, split=str(split_name))
        split_mask = np.zeros(len(manifest), dtype=bool)
        split_mask[split_indices] = True
        mask &= split_mask
    if test_datasets:
        values = test_datasets if isinstance(test_datasets, list) else [test_datasets]
        mask &= manifest["dataset_id"].astype(str).isin([str(value) for value in values]).to_numpy()
    if test_filter:
        mask &= _manifest_filter(manifest, test_filter)

    selected = np.flatnonzero(mask)
    if not len(selected):
        raise ValueError(
            f"{mode} held-out selection contains no samples after applying split and "
            "target filters."
        )
    return selected


def _checkpoint_split_indices(context: dict, *, split: str) -> np.ndarray:
    checkpoint_path = Path(context["checkpoint_path"])
    split_path = checkpoint_path.parent / "split.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Requested data.split={split!r}, but checkpoint split manifest is missing: "
            f"{split_path}. Declare an explicit test_datasets/test_filter only when the "
            "evaluation manifest is independently known to be held out."
        )
    table = pd.read_csv(split_path, dtype=str)
    required = {"sample_id", "split"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Checkpoint split manifest {split_path} is missing columns: {missing}")
    if table["sample_id"].duplicated().any():
        duplicated = sorted(table.loc[table["sample_id"].duplicated(), "sample_id"].unique())
        raise ValueError(
            f"Checkpoint split manifest {split_path} contains duplicate sample_id values: "
            f"{duplicated[:5]}"
        )
    _validate_split_assignment_contract(context, table, split_path)
    selected_ids = set(
        table.loc[table["split"].astype(str).str.lower().eq(split.lower()), "sample_id"].astype(str)
    )
    if not selected_ids:
        raise ValueError(
            f"Checkpoint split manifest {split_path} has no rows assigned to {split!r}."
        )
    manifest_ids = context["manifest"]["sample_id"].astype(str)
    if manifest_ids.duplicated().any():
        duplicated = sorted(manifest_ids[manifest_ids.duplicated()].unique())
        raise ValueError(
            "Evaluation manifest contains duplicate sample_id values, so split provenance "
            f"is ambiguous: {duplicated[:5]}"
        )
    indices = np.flatnonzero(manifest_ids.isin(selected_ids).to_numpy())
    if not len(indices):
        raise ValueError(
            f"No data in the evaluation manifest matches the {split!r} sample_id values from "
            f"{split_path}. Check processed_dirs and checkpoint provenance."
        )
    matched_ids = set(manifest_ids.iloc[indices])
    missing_ids = sorted(selected_ids - matched_ids)
    if missing_ids:
        raise ValueError(
            f"Evaluation manifest is missing {len(missing_ids)} sample_id values assigned to "
            f"{split!r} in {split_path}: {missing_ids[:5]}"
        )
    return indices


def _validate_split_assignment_contract(
    context: dict, table: pd.DataFrame, split_path: Path
) -> None:
    cfg = context.get("train_config") or context.get("checkpoint", {}).get("config") or {}
    expected = cfg.get("runtime_contract", {}).get("split_assignment_sha256")
    protocol_version = cfg.get("protocol", {}).get("metadata_contract_version")
    if expected is None:
        if protocol_version == METADATA_CONTRACT_V04_DEV:
            raise ValueError(
                "Protocol 0.4 checkpoint lacks runtime_contract.split_assignment_sha256."
            )
        return
    assignments = [
        {"sample_id": str(sample_id), "split": str(split).lower()}
        for sample_id, split in table[["sample_id", "split"]].itertuples(index=False, name=None)
        if str(split).lower() in {"train", "val", "test"}
    ]
    assignments.sort(key=lambda row: row["sample_id"])
    payload = json.dumps(assignments, sort_keys=True, separators=(",", ":"), default=str)
    observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if observed != str(expected):
        raise ValueError(
            f"Checkpoint split manifest {split_path} no longer matches the checkpoint "
            "runtime split-assignment contract."
        )


def _loader(
    context: dict, indices: np.ndarray, *, training_role: bool = False
) -> DataLoader:
    cfg = context["train_config"]
    data_cfg = {
        **cfg["data"],
        **((context.get("eval_config") or {}).get("data") or {}),
    }
    settings = resolve_loader_settings(data_cfg, device_type=context["device"].type)
    collate = partial(
        collate_eeg,
        vocabularies=context["vocab"],
        **metadata_collate_kwargs(cfg),
    )
    return DataLoader(
        Subset(context["dataset"], indices.tolist()),
        batch_size=(
            settings.train_batch_size if training_role else settings.eval_batch_size
        ),
        shuffle=False,
        **settings.kwargs(),
        collate_fn=collate,
    )


def _channel_perturbation(channel_set: str):
    if channel_set == "all":
        return None
    keep_ids = _channel_set_ids(channel_set)

    def apply(x, cond):
        cond = _clone_cond(cond)
        keep = torch.zeros_like(cond["channel_mask"])
        for channel_id in keep_ids:
            keep |= cond["channel_ids"].eq(channel_id)
        keep &= cond["channel_mask"]
        x = x * keep.unsqueeze(-1).to(x.dtype)
        cond["channel_mask"] = keep
        _update_n_channels(cond)
        _invalidate_query_qc(cond)
        return x, cond

    return apply


def _robustness_perturbation(spec: dict):
    kind = spec["type"] if "type" in spec else spec["name"]
    if kind == "compose":
        transforms = [_robustness_perturbation(item) for item in spec.get("transforms", [])]

        def compose(x, cond):
            for transform in transforms:
                x, cond = transform(x, cond)
            return x, cond

        return compose
    if kind == "channel_subset":
        transform = _channel_perturbation(str(spec["channel_set"]))
        return transform or (lambda x, cond: (x, cond))

    rng = np.random.default_rng(int(spec.get("seed", 42)))

    def apply(x, cond):
        cond = _clone_cond(cond)
        if kind == "metadata_missing":
            rows = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            _mask_condition_metadata(cond, rows, all_metadata=True)
        elif kind == "metadata_missing_ratio":
            ratio = float(spec.get("ratio", 0.5))
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"metadata missing ratio must be within [0,1], got {ratio}")
            rows = torch.as_tensor(
                rng.random(x.shape[0]) < ratio, dtype=torch.bool, device=x.device
            )
            _mask_condition_metadata(cond, rows, all_metadata=True)
        elif kind == "metadata_group_missing":
            rows = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            _mask_condition_metadata(
                cond,
                rows,
                categorical_fields=list(spec.get("categorical_fields", [])),
                continuous_indices=[int(value) for value in spec.get("continuous_indices", [])],
                channels=bool(spec.get("channels", False)),
                external_continuous_indices=[
                    int(value) for value in spec.get("external_continuous_indices", [])
                ],
                query_qc_indices=[int(value) for value in spec.get("query_qc_indices", [])],
                channel_impedance=bool(spec.get("channel_impedance", False)),
            )
        elif kind == "metadata_shuffle":
            raise ValueError(
                "metadata_shuffle requires a held-out global permutation; construct it "
                "through robustness _scenarios rather than as a batch-local transform."
            )
        elif kind == "downsample":
            factor = float(spec.get("factor", 0.5))
            length = max(2, round(x.shape[-1] * factor))
            x = F.interpolate(
                F.interpolate(x, size=length, mode="linear", align_corners=False),
                size=x.shape[-1],
                mode="linear",
                align_corners=False,
            )
            _invalidate_query_qc(cond)
        elif kind == "rereference":
            mask = cond["channel_mask"].unsqueeze(-1).to(x.dtype)
            mean = (x * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(
                1.0
            )
            x = (x - mean) * mask
            _invalidate_query_qc(cond)
        elif kind == "gaussian_noise":
            active = cond["channel_mask"].unsqueeze(-1).to(x.dtype)
            x = x + active * torch.randn_like(x) * float(spec.get("std", 0.1))
            _invalidate_query_qc(cond)
        elif kind == "band_limited_noise":
            low, high = [float(value) for value in spec.get("band_hz", [8.0, 16.0])]
            sfreq = float(spec.get("sfreq", 200.0))
            noise = torch.randn_like(x)
            spectrum = torch.fft.rfft(noise, dim=-1)
            frequencies = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq).to(x.device)
            band = ((frequencies >= low) & (frequencies <= high)).to(spectrum.dtype)
            noise = torch.fft.irfft(spectrum * band, n=x.shape[-1], dim=-1)
            scale = noise.std(dim=-1, keepdim=True).clamp_min(1e-6)
            active = cond["channel_mask"].unsqueeze(-1).to(x.dtype)
            x = x + active * float(spec.get("std", 0.1)) * noise / scale
            _invalidate_query_qc(cond)
        else:
            raise KeyError(f"Unknown robustness perturbation: {kind}")
        return x, cond

    return apply


def _global_metadata_donor_perturbation(
    spec: dict,
    context: dict,
    selected_indices: np.ndarray,
):
    if (
        metadata_collate_kwargs(context["train_config"])["metadata_contract_version"]
        != METADATA_CONTRACT_V04_DEV
    ):
        raise ValueError("Manifest-only metadata donor controls require Protocol 0.4-dev.")
    kind = str(spec.get("type", spec.get("name")))
    control = (
        "within_class_shuffle"
        if kind == "metadata_shuffle"
        else "counterfactual_wet_dry"
    )
    seed = int(spec.get("seed", 42))
    plan = build_development_control_plan(
        context["manifest"],
        {"val": np.asarray(selected_indices, dtype=int)},
        control=control,
        seed=seed,
    )
    sources = plan.sources_for("val") or {}

    def apply(x, cond, sample_ids):
        replaced = overwrite_external_metadata(cond, sample_ids, sources, context["vocab"])
        return x, replaced

    apply.uses_sample_ids = True
    apply.audit = {
        "metadata_control_schema": plan.contract["schema"],
        "metadata_control_name": control,
        "metadata_control_scope": plan.contract["donor_scope"],
        "metadata_control_seed": seed,
        "metadata_control_effective_changed_fraction": plan.contract[
            "effective_external_metadata_changed_fraction"
        ],
        "metadata_control_condition_flip_fraction": plan.contract[
            "condition_flip_fraction"
        ],
        "metadata_control_donor_mapping_sha256": plan.contract[
            "donor_mapping_sha256"
        ],
        "metadata_control_field_changed_fraction": json.dumps(
            plan.contract["field_changed_fraction"], sort_keys=True
        ),
        "_donor_mapping": plan.donor_mapping,
    }
    return apply


def _metadata_shuffle_indices(
    manifest: pd.DataFrame,
    selected_indices: np.ndarray,
    *,
    seed: int,
    external_only: bool = False,
) -> np.ndarray:
    """Compatibility wrapper for the shared deterministic control implementation."""
    del external_only
    return metadata_shuffle_indices(manifest, selected_indices, seed=seed)


def _mask_condition_metadata(
    cond: dict,
    rows: torch.Tensor,
    *,
    categorical_fields: list[str] | None = None,
    continuous_indices: list[int] | None = None,
    channels: bool = False,
    external_continuous_indices: list[int] | None = None,
    query_qc_indices: list[int] | None = None,
    channel_impedance: bool = False,
    all_metadata: bool = False,
) -> None:
    if _uses_v04_contract(cond):
        _mask_v04_metadata(
            cond,
            rows,
            categorical_fields=categorical_fields,
            external_continuous_indices=external_continuous_indices,
            query_qc_indices=query_qc_indices,
            channels=channels,
            channel_impedance=channel_impedance,
            all_metadata=all_metadata,
        )
        return
    categorical_fields = (
        list(CONDITION_CATEGORICAL_FIELDS) if all_metadata else (categorical_fields or [])
    )
    continuous_indices = (
        list(range(cond["continuous"].shape[1])) if all_metadata else (continuous_indices or [])
    )
    for field in categorical_fields:
        if field not in CONDITION_CATEGORICAL_FIELDS:
            raise KeyError(f"Unknown categorical metadata field: {field}")
        cond[field][rows] = 0
    for index in continuous_indices:
        if index < 0 or index >= cond["continuous"].shape[1]:
            raise IndexError(f"continuous metadata index out of range: {index}")
        cond["continuous"][rows, index] = 0.0
        cond["continuous_missing"][rows, index] = True
    if channels or all_metadata:
        ids = cond.get("condition_channel_ids", cond["channel_ids"]).clone()
        mask = cond.get("condition_channel_mask", cond["channel_mask"]).clone()
        ids[rows] = 0
        mask[rows] = False
        cond["condition_channel_ids"] = ids
        cond["condition_channel_mask"] = mask


def _mask_v04_metadata(
    cond: dict,
    rows: torch.Tensor,
    *,
    categorical_fields: list[str] | None,
    external_continuous_indices: list[int] | None,
    query_qc_indices: list[int] | None,
    channels: bool,
    channel_impedance: bool,
    all_metadata: bool,
) -> None:
    fields = (
        list(PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS)
        if all_metadata
        else (categorical_fields or [])
    )
    for field in fields:
        if field not in PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS:
            raise KeyError(f"Field {field!r} is not external metadata in Protocol 0.4-dev.")
        cond[field][rows] = 0

    indices = (
        list(range(cond["external_continuous"].shape[1]))
        if all_metadata
        else (external_continuous_indices or [])
    )
    for index in indices:
        if index < 0 or index >= cond["external_continuous"].shape[1]:
            raise IndexError(f"external continuous metadata index out of range: {index}")
        cond["external_continuous"][rows, index] = 0.0
        cond["external_continuous_missing"][rows, index] = True

    if all_metadata or channel_impedance:
        cond["channel_impedance"][rows] = 0.0
        cond["channel_impedance_missing"][rows] = True

    for index in query_qc_indices or []:
        if index < 0 or index >= cond["query_qc"].shape[1]:
            raise IndexError(f"query QC index out of range: {index}")
        cond["query_qc"][rows, index] = 0.0
        cond["query_qc_missing"][rows, index] = True
        cond["channel_query_qc"][rows] = 0.0
        cond["channel_query_qc_missing"][rows] = True

    # Geometry/channel structure is never part of external-metadata missingness.
    # It can only be removed by an explicitly named structure ablation.
    if channels:
        ids = cond.get("condition_channel_ids", cond["channel_ids"]).clone()
        mask = cond.get("condition_channel_mask", cond["channel_mask"]).clone()
        ids[rows] = 0
        mask[rows] = False
        cond["condition_channel_ids"] = ids
        cond["condition_channel_mask"] = mask


def _uses_v04_contract(cond: dict) -> bool:
    return str(cond.get("metadata_contract_version", "legacy")) == METADATA_CONTRACT_V04_DEV


def _invalidate_query_qc(cond: dict) -> None:
    if "query_qc" not in cond:
        return
    cond["query_qc"] = torch.zeros_like(cond["query_qc"])
    cond["query_qc_missing"] = torch.ones_like(cond["query_qc_missing"])
    if "channel_query_qc" in cond:
        cond["channel_query_qc"] = torch.zeros_like(cond["channel_query_qc"])
        cond["channel_query_qc_missing"] = torch.ones_like(cond["channel_query_qc_missing"])


def _run_calibration(eval_cfg: dict, context: dict) -> dict:
    manifest = context["manifest"]
    selected_indices = _held_out_evaluation_indices(eval_cfg, context, mode="calibration")
    target_mask = np.zeros(len(manifest), dtype=bool)
    target_mask[selected_indices] = True
    target_groups = (
        manifest.loc[target_mask, ["dataset_id", "subject_id"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["dataset_id", "subject_id"])
    )
    subjects = list(target_groups.itertuples(index=False, name=None))
    max_subjects = eval_cfg.get("max_subjects")
    if max_subjects:
        subjects = subjects[: int(max_subjects)]
    if not subjects:
        raise ValueError("Calibration target filters contain no subjects.")

    budgets = [int(value) for value in eval_cfg.get("calibration_trials_per_class", [0, 1, 3, 5])]
    if not budgets:
        raise ValueError("calibration_trials_per_class must contain at least one budget.")
    if any(budget < 0 for budget in budgets):
        raise ValueError(f"Calibration budgets must be non-negative, got {budgets}.")
    max_budget = max(budgets)
    seed = int(eval_cfg.get("seed", 42))

    partitions: dict[tuple[str, str], tuple[dict[int, np.ndarray], np.ndarray]] = {}
    partition_rows: list[dict] = []
    all_query_indices: list[np.ndarray] = []
    for dataset_id, subject in subjects:
        subject_indices = np.flatnonzero(
            target_mask
            & manifest["dataset_id"].astype(str).eq(dataset_id).to_numpy()
            & manifest["subject_id"].astype(str).eq(subject).to_numpy()
        )
        support_by_label, query = _subject_calibration_partition(
            manifest,
            subject_indices,
            max_budget=max_budget,
            seed=seed,
        )
        partitions[(dataset_id, subject)] = (support_by_label, query)
        all_query_indices.append(query)
        for support_pool in support_by_label.values():
            for rank, index in enumerate(support_pool, start=1):
                partition_rows.append(
                    _calibration_partition_row(
                        manifest,
                        int(index),
                        partition="support_pool",
                        support_rank=rank,
                    )
                )
        for index in query:
            partition_rows.append(
                _calibration_partition_row(
                    manifest, int(index), partition="query", support_rank=None
                )
            )

    fixed_query_indices = np.concatenate(all_query_indices)
    fixed_query_ids = manifest.iloc[fixed_query_indices]["sample_id"].astype(str).tolist()
    query_identity_sha256 = _sample_id_identity_sha256(fixed_query_ids)
    support_pool_count = sum(
        len(indices)
        for support_by_label, _ in partitions.values()
        for indices in support_by_label.values()
    )
    rows = []
    subject_rows: list[dict] = []
    prediction_rows: list[dict] = []
    base_model = context["model"].to("cpu")
    for budget in budgets:
        all_y: list[np.ndarray] = []
        all_logits: list[np.ndarray] = []
        observed_query_ids: list[str] = []
        budget_subject_scores: list[float] = []
        evaluated_subjects = 0
        support_count = 0
        for dataset_id, subject in subjects:
            support_by_label, evaluation = partitions[(dataset_id, subject)]
            calibration = _calibration_support_indices(support_by_label, budget)
            adaptation_seed = _calibration_adaptation_seed(
                seed, dataset_id=dataset_id, subject_id=subject, budget=budget
            )
            support_count += len(calibration)
            if budget == 0:
                model = base_model.to(context["device"])
            else:
                model = copy.deepcopy(base_model).to(context["device"])
                _calibrate_model(
                    model,
                    context,
                    calibration,
                    eval_cfg,
                    seed=adaptation_seed,
                )
            y_true, logits, sample_ids = collect_predictions(
                model, _loader(context, evaluation), context["device"]
            )
            all_y.append(y_true)
            all_logits.append(logits)
            observed_query_ids.extend(sample_ids)
            subject_metrics = classification_metrics(
                y_true,
                logits,
                trial_time_sec=float(eval_cfg.get("trial_time_sec", 2.0)),
            )
            budget_subject_scores.append(float(subject_metrics["balanced_accuracy"]))
            subject_rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": subject,
                    "calibration_trials_per_class": budget,
                    "n_support_trials": len(calibration),
                    "adaptation_seed": adaptation_seed,
                    "query_identity_sha256": query_identity_sha256,
                    **subject_metrics,
                }
            )
            prediction_rows.extend(
                _calibration_prediction_rows(
                    dataset_id=dataset_id,
                    subject_id=subject,
                    budget=budget,
                    query_identity_sha256=query_identity_sha256,
                    sample_ids=sample_ids,
                    y_true=y_true,
                    logits=logits,
                )
            )
            evaluated_subjects += 1
            if budget > 0:
                del model
        observed_query_sha256 = _sample_id_identity_sha256(observed_query_ids)
        if observed_query_sha256 != query_identity_sha256:
            raise RuntimeError(
                "Calibration query identity changed during evaluation; refusing incomparable "
                f"results for k={budget}."
            )
        base_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics = classification_metrics(
            np.concatenate(all_y),
            np.concatenate(all_logits),
            trial_time_sec=float(eval_cfg.get("trial_time_sec", 2.0)),
        )
        metrics.update(
            {
                "mode": "calibration",
                "calibration_trials_per_class": budget,
                "n_subjects": evaluated_subjects,
                "n_support_trials": support_count,
                "n_support_pool_trials": support_pool_count,
                "n_query_trials": len(observed_query_ids),
                "max_calibration_trials_per_class": max_budget,
                "query_identity_sha256": query_identity_sha256,
                "mean_subject_balanced_accuracy": float(np.mean(budget_subject_scores)),
                "median_subject_balanced_accuracy": float(np.median(budget_subject_scores)),
                "worst_subject_balanced_accuracy": float(np.min(budget_subject_scores)),
            }
        )
        rows.append(metrics)
    output_csv = _output_path(eval_cfg, context, "calibration")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    partition_csv = output_csv.with_name(f"{output_csv.stem}_partition.csv")
    pd.DataFrame(partition_rows).to_csv(partition_csv, index=False)
    subject_csv = output_csv.with_name(f"{output_csv.stem}_subject_metrics.csv")
    pd.DataFrame(subject_rows).to_csv(subject_csv, index=False)
    predictions_csv = output_csv.with_name(f"{output_csv.stem}_predictions.csv")
    pd.DataFrame(prediction_rows).to_csv(predictions_csv, index=False)
    return {
        "output_csv": str(output_csv),
        "partition_csv": str(partition_csv),
        "subject_csv": str(subject_csv),
        "predictions_csv": str(predictions_csv),
        "query_identity_sha256": query_identity_sha256,
        "results": rows,
    }


def _calibration_prediction_rows(
    *,
    dataset_id: str,
    subject_id: str,
    budget: int,
    query_identity_sha256: str,
    sample_ids: list[str],
    y_true: np.ndarray,
    logits: np.ndarray,
) -> list[dict]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = logits.argmax(axis=1)
    return [
        {
            "dataset_id": dataset_id,
            "subject_id": subject_id,
            "calibration_trials_per_class": budget,
            "query_identity_sha256": query_identity_sha256,
            "sample_id": sample_id,
            "label": int(label),
            "prediction": int(prediction),
            "confidence": float(confidence),
        }
        for sample_id, label, prediction, confidence in zip(
            sample_ids,
            y_true,
            predictions,
            probabilities.max(axis=1),
        )
    ]


def _subject_calibration_partition(
    manifest: pd.DataFrame,
    indices: np.ndarray,
    *,
    max_budget: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if max_budget < 0:
        raise ValueError(f"max_budget must be non-negative, got {max_budget}.")
    support_by_label: dict[int, np.ndarray] = {}
    query: list[int] = []
    for label in sorted(manifest.iloc[indices]["label"].astype(int).unique()):
        label_mask = manifest.iloc[indices]["label"].astype(int).to_numpy() == label
        candidates = indices[label_mask].copy()
        candidates = np.asarray(
            sorted(
                candidates.tolist(),
                key=lambda index: _calibration_order_key(
                    manifest.iloc[int(index)]["sample_id"], seed=seed, label=label
                ),
            ),
            dtype=int,
        )
        if len(candidates) <= max_budget:
            subject_values = sorted(set(manifest.iloc[indices]["subject_id"].astype(str).tolist()))
            subject = ",".join(subject_values)
            raise ValueError(
                f"Calibration subject {subject!r}, label {label} has {len(candidates)} trials, "
                f"but max k={max_budget} would leave no fixed query trial. At least "
                f"{max_budget + 1} trials per subject and label are required."
            )
        support_by_label[int(label)] = candidates[:max_budget]
        query.extend(candidates[max_budget:].tolist())
    if not query:
        raise ValueError("Calibration partition leaves no fixed query samples.")
    return support_by_label, np.asarray(query, dtype=int)


def _subject_calibration_indices(
    manifest: pd.DataFrame,
    indices: np.ndarray,
    budget: int,
    *,
    seed: int,
    max_budget: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper that supports a fixed max-budget partition."""
    reserved_budget = budget if max_budget is None else max_budget
    if budget > reserved_budget:
        raise ValueError(
            f"Calibration budget k={budget} exceeds reserved max budget k={reserved_budget}."
        )
    support_by_label, query = _subject_calibration_partition(
        manifest, indices, max_budget=reserved_budget, seed=seed
    )
    return _calibration_support_indices(support_by_label, budget), query


def _calibration_support_indices(
    support_by_label: dict[int, np.ndarray], budget: int
) -> np.ndarray:
    if budget < 0:
        raise ValueError(f"Calibration budget must be non-negative, got {budget}.")
    if any(budget > len(indices) for indices in support_by_label.values()):
        maximum = min((len(indices) for indices in support_by_label.values()), default=0)
        raise ValueError(
            f"Calibration budget k={budget} exceeds reserved support pool k={maximum}."
        )
    selected = [indices[:budget] for _, indices in sorted(support_by_label.items())]
    nonempty = [indices for indices in selected if len(indices)]
    return np.concatenate(nonempty) if nonempty else np.array([], dtype=int)


def _calibration_order_key(sample_id: object, *, seed: int, label: int) -> bytes:
    value = f"{seed}\0{label}\0{sample_id}".encode()
    return hashlib.sha256(value).digest()


def _sample_id_identity_sha256(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(str(value) for value in sample_ids):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _calibration_partition_row(
    manifest: pd.DataFrame,
    index: int,
    *,
    partition: str,
    support_rank: int | None,
) -> dict:
    row = manifest.iloc[index]
    return {
        "sample_id": str(row["sample_id"]),
        "dataset_id": str(row["dataset_id"]),
        "subject_id": str(row["subject_id"]),
        "label": int(row["label"]),
        "partition": partition,
        "support_rank": support_rank,
    }


def _calibration_adaptation_seed(
    base_seed: int, *, dataset_id: str, subject_id: str, budget: int
) -> int:
    encoded = f"{base_seed}\x1f{dataset_id}\x1f{subject_id}\x1f{budget}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _calibrate_model(
    model,
    context: dict,
    indices: np.ndarray,
    eval_cfg: dict,
    *,
    seed: int,
) -> None:
    seed_everything(seed)
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [model.head]
    if eval_cfg.get("tune_adapter", True) and model.adapter is not None:
        modules.append(model.adapter)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(eval_cfg.get("lr", 1e-3)),
    )
    loader = _loader(context, indices, training_role=True)
    model.train()
    for _ in range(int(eval_cfg.get("epochs", 10))):
        for batch in loader:
            batch = _to_device(batch, context["device"])
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["x"], batch["cond"], use_latent=False)
            loss = F.cross_entropy(out.logits, batch["y"])
            loss.backward()
            optimizer.step()


def _common_label_indices(context: dict, indices: np.ndarray) -> np.ndarray:
    class_map = context["checkpoint"].get("class_map") or {}
    allowed = {round(float(value["stimulus_frequency_hz"]), 4) for value in class_map.values()}
    frequencies = context["manifest"].iloc[indices]["stimulus_frequency_hz"].astype(float)
    keep = frequencies.round(4).isin(allowed).to_numpy()
    selected = indices[keep]
    if not len(selected):
        raise ValueError(
            "No stimulus-frequency overlap exists between checkpoint classes and target dataset. "
            "Use a dataset-specific head/transfer config instead of pretending labels are shared."
        )
    return selected


def _manifest_filter(manifest: pd.DataFrame, filters: dict) -> np.ndarray:
    mask = np.ones(len(manifest), dtype=bool)
    for column, expected in filters.items():
        values = expected if isinstance(expected, list) else [expected]
        mask &= manifest[column].astype(str).isin([str(value) for value in values]).to_numpy()
    return mask


def _output_path(eval_cfg: dict, context: dict, mode: str) -> Path:
    configured = eval_cfg.get("output_csv")
    if configured:
        return Path(configured)
    output_dir = Path(context["train_config"].get("output_dir", "outputs/debug"))
    return output_dir / "eval" / f"{mode}.csv"


def _save_predictions(
    eval_cfg: dict,
    context: dict,
    scenario: str,
    sample_ids: list[str],
    y_true: np.ndarray,
    logits: np.ndarray,
) -> None:
    if not eval_cfg.get("save_predictions", True):
        return
    output = _output_path(eval_cfg, context, eval_cfg.get("mode", "eval"))
    path = output.with_name(f"{output.stem}_{scenario}_predictions.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": y_true,
            "prediction": logits.argmax(axis=1),
            "confidence": probabilities.max(axis=1),
        }
    ).to_csv(path, index=False)


def _save_evaluation_provenance(
    eval_cfg: dict,
    context: dict,
    output_csv: Path,
    scenarios: list[dict],
) -> Path:
    provenance_path = output_csv.with_name(f"{output_csv.stem}_provenance.json")
    artifact_paths = sorted(
        path
        for path in output_csv.parent.glob(f"{output_csv.stem}*")
        if path.is_file() and path != provenance_path
    )
    train_runtime = context["train_config"].get("runtime_contract", {})
    access = context["research_access"]
    cohort = context["cohort"]
    payload = {
        "schema": "cfeg.evaluation-provenance.v1",
        "mode": str(eval_cfg.get("mode", eval_cfg.get("run_name", "standard"))),
        "confirmatory_claim_allowed": access.execution_phase == "confirmatory_training",
        "checkpoint_path": str(context["checkpoint_path"]),
        "checkpoint_sha256": _sha256_file(context["checkpoint_path"]),
        "checkpoint_runtime_contract": train_runtime,
        "evaluation_config_sha256": _sha256_json(eval_cfg),
        "research_access": access.contract(),
        "cohort": cohort.contract(),
        "current_source_revision": current_source_revision_contract(),
        "scenarios": scenarios,
        "artifacts": {
            path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    save_json(provenance_path, payload)
    return provenance_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confusion_rows(scenario: str, matrix: np.ndarray) -> list[dict]:
    return [
        {"scenario": scenario, "true_label": i, "predicted_label": j, "count": int(matrix[i, j])}
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
        if matrix[i, j]
    ]


def _clone_cond(cond: dict) -> dict:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in cond.items()}


def _channel_set_ids(name: str) -> list[int]:
    with Path("configs/channel_sets.yaml").open("r", encoding="utf-8") as handle:
        sets = yaml.safe_load(handle)
    if name not in sets:
        raise KeyError(f"Unknown channel set {name}. Known: {sorted(sets)}")
    return CanonicalChannelMap.from_yaml().get_ids(list(sets[name]))
