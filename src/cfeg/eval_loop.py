from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from cfeg.constants import CONDITION_CATEGORICAL_FIELDS
from cfeg.data.collate import build_vocabularies, collate_eeg
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.preprocess import CanonicalChannelMap
from cfeg.data.transforms import _update_n_channels
from cfeg.metrics import classification_metrics, confusion_matrix, itr_bits_per_min
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.seed import seed_everything
from cfeg.train_loop import _to_device
from cfeg.utils.checkpoint import load_checkpoint


def run_evaluation(eval_cfg: dict, ckpt_path: str | Path) -> dict:
    seed_everything(int(eval_cfg.get("seed", 42)))
    context = load_evaluation_context(eval_cfg, ckpt_path)
    mode = eval_cfg.get("mode", eval_cfg.get("run_name", "standard"))
    if mode == "calibration":
        return _run_calibration(eval_cfg, context)
    scenarios = _scenarios(eval_cfg, context, mode)
    rows: list[dict] = []
    confusion_rows: list[dict] = []
    trial_time_sec = float(eval_cfg.get("trial_time_sec", 2.0))
    baseline_accuracy, baseline_itr = _reference_metrics(
        mode, context, trial_time_sec=trial_time_sec
    )
    for name, indices, perturb in scenarios:
        loader = _loader(context, indices)
        y_true, logits, sample_ids = collect_predictions(
            context["model"], loader, context["device"], perturb=perturb
        )
        metrics = classification_metrics(
            y_true, logits, trial_time_sec=trial_time_sec
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
            metrics["itr_drop_rate"] = _relative_drop(
                baseline_itr, metrics["itr_bits_per_min"]
            )
        rows.append(metrics)
        matrix = confusion_matrix(y_true, logits.argmax(axis=1), logits.shape[1])
        confusion_rows.extend(_confusion_rows(name, matrix))
        _save_predictions(eval_cfg, context, name, sample_ids, y_true, logits)

    output_csv = _output_path(eval_cfg, context, mode)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    pd.DataFrame(confusion_rows).to_csv(output_csv.with_name(f"{output_csv.stem}_confusion.csv"), index=False)
    return {"output_csv": str(output_csv), "results": rows}


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


def _relative_drop(reference: float, observed: float) -> float:
    if abs(reference) < 1e-12:
        return 0.0
    return float((reference - observed) / reference)


def run_channel_stress_eval(eval_cfg: dict, ckpt_path: str | Path) -> dict:
    cfg = copy.deepcopy(eval_cfg)
    cfg["mode"] = "channel_stress"
    return run_evaluation(cfg, ckpt_path)


def load_evaluation_context(eval_cfg: dict, ckpt_path: str | Path) -> dict:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    data_dirs = eval_cfg.get("data", {}).get("processed_dirs") or cfg["data"]["processed_dirs"]
    dataset = EEGProcessedDataset(data_dirs)
    vocab = ckpt.get("vocabularies") or build_vocabularies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedEEGDecoder(cfg, vocab_sizes={key: len(value) for key, value in vocab.items()})
    strict = not bool(ckpt.get("save_trainable_only", False))
    model.load_state_dict(ckpt["model_state"], strict=strict)
    model.to(device)
    manifest = pd.DataFrame([entry[2] for entry in dataset.entries])
    return {
        "checkpoint": ckpt,
        "train_config": cfg,
        "dataset": dataset,
        "manifest": manifest,
        "vocab": vocab,
        "device": device,
        "model": model,
    }


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
            batch["x"], batch["cond"] = perturb(batch["x"], batch["cond"])
        out = model(batch["x"], batch["cond"], use_latent=False)
        labels.append(batch["y"].detach().cpu().numpy())
        logits.append(out.logits.detach().float().cpu().numpy())
        sample_ids.extend(batch["sample_id"])
    if not labels:
        raise ValueError("Evaluation selection contains no samples.")
    return np.concatenate(labels), np.concatenate(logits), sample_ids


def _scenarios(eval_cfg: dict, context: dict, mode: str):
    manifest = context["manifest"]
    all_indices = np.arange(len(manifest))
    if mode == "channel_stress":
        return [
            (name, all_indices, _channel_perturbation(name))
            for name in eval_cfg.get("channel_sets", ["all"])
        ]
    if mode == "cross_dataset":
        test_datasets = eval_cfg.get("test_datasets") or eval_cfg.get("data", {}).get("test_datasets")
        if not test_datasets:
            raise ValueError("cross_dataset evaluation requires test_datasets.")
        mask = manifest["dataset_id"].astype(str).isin(test_datasets).to_numpy()
        indices = np.flatnonzero(mask)
        if eval_cfg.get("common_label_subset_only", True):
            indices = _common_label_indices(context, indices)
        return [("zero_shot_" + "_".join(test_datasets), indices, None)]
    if mode == "cross_condition":
        filters = eval_cfg.get("test_filter", {})
        indices = np.flatnonzero(_manifest_filter(manifest, filters))
        return [("zero_shot_condition", indices, None)]
    if mode == "robustness":
        scenarios = [("clean", all_indices, None)]
        for spec in eval_cfg.get("perturbations", []):
            name = str(spec["name"])
            scenarios.append((name, all_indices, _robustness_perturbation(spec)))
        return scenarios
    return [("standard", all_indices, None)]


def _loader(context: dict, indices: np.ndarray) -> DataLoader:
    cfg = context["train_config"]
    collate = partial(collate_eeg, vocabularies=context["vocab"])
    return DataLoader(
        Subset(context["dataset"], indices.tolist()),
        batch_size=int(cfg["data"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
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
            )
        elif kind == "metadata_shuffle":
            order = torch.arange(x.shape[0] - 1, -1, -1, device=x.device)
            for field in CONDITION_CATEGORICAL_FIELDS:
                cond[field] = cond[field][order]
            cond["continuous"] = cond["continuous"][order]
            cond["continuous_missing"] = cond["continuous_missing"][order]
            cond["condition_channel_ids"] = cond["channel_ids"][order]
            cond["condition_channel_mask"] = cond["channel_mask"][order]
        elif kind == "downsample":
            factor = float(spec.get("factor", 0.5))
            length = max(2, int(round(x.shape[-1] * factor)))
            x = F.interpolate(
                F.interpolate(x, size=length, mode="linear", align_corners=False),
                size=x.shape[-1],
                mode="linear",
                align_corners=False,
            )
        elif kind == "rereference":
            mask = cond["channel_mask"].unsqueeze(-1).to(x.dtype)
            mean = (x * mask).sum(dim=1, keepdim=True) / mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            x = (x - mean) * mask
        elif kind == "gaussian_noise":
            active = cond["channel_mask"].unsqueeze(-1).to(x.dtype)
            x = x + active * torch.randn_like(x) * float(spec.get("std", 0.1))
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
        else:
            raise KeyError(f"Unknown robustness perturbation: {kind}")
        return x, cond

    return apply


def _mask_condition_metadata(
    cond: dict,
    rows: torch.Tensor,
    *,
    categorical_fields: list[str] | None = None,
    continuous_indices: list[int] | None = None,
    channels: bool = False,
    all_metadata: bool = False,
) -> None:
    categorical_fields = (
        list(CONDITION_CATEGORICAL_FIELDS) if all_metadata else (categorical_fields or [])
    )
    continuous_indices = (
        list(range(cond["continuous"].shape[1]))
        if all_metadata
        else (continuous_indices or [])
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


def _run_calibration(eval_cfg: dict, context: dict) -> dict:
    manifest = context["manifest"]
    target = eval_cfg.get("test_datasets")
    target_mask = (
        manifest["dataset_id"].astype(str).isin(target).to_numpy()
        if target
        else np.ones(len(manifest), dtype=bool)
    )
    if eval_cfg.get("test_filter"):
        target_mask &= _manifest_filter(manifest, eval_cfg["test_filter"])
    subjects = sorted(manifest.loc[target_mask, "subject_id"].astype(str).unique())
    max_subjects = eval_cfg.get("max_subjects")
    if max_subjects:
        subjects = subjects[: int(max_subjects)]
    rows = []
    base_model = context["model"].to("cpu")
    for budget in [int(value) for value in eval_cfg.get("calibration_trials_per_class", [0, 1, 3, 5])]:
        all_y: list[np.ndarray] = []
        all_logits: list[np.ndarray] = []
        evaluated_subjects = 0
        for subject in subjects:
            subject_indices = np.flatnonzero(
                target_mask & manifest["subject_id"].astype(str).eq(subject).to_numpy()
            )
            calibration, evaluation = _subject_calibration_indices(
                manifest, subject_indices, budget, seed=int(eval_cfg.get("seed", 42))
            )
            if not len(evaluation):
                continue
            if budget == 0:
                model = base_model.to(context["device"])
            else:
                model = copy.deepcopy(base_model).to(context["device"])
                _calibrate_model(model, context, calibration, eval_cfg)
            y_true, logits, _ = collect_predictions(
                model, _loader(context, evaluation), context["device"]
            )
            all_y.append(y_true)
            all_logits.append(logits)
            evaluated_subjects += 1
            if budget > 0:
                del model
        if not all_y:
            continue
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
            }
        )
        rows.append(metrics)
    output_csv = _output_path(eval_cfg, context, "calibration")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return {"output_csv": str(output_csv), "results": rows}


def _subject_calibration_indices(
    manifest: pd.DataFrame, indices: np.ndarray, budget: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if budget <= 0:
        return np.array([], dtype=int), indices
    rng = np.random.default_rng(seed)
    calibration: list[int] = []
    evaluation: list[int] = []
    for label in sorted(manifest.iloc[indices]["label"].astype(int).unique()):
        candidates = indices[manifest.iloc[indices]["label"].astype(int).to_numpy() == label].copy()
        rng.shuffle(candidates)
        calibration.extend(candidates[:budget].tolist())
        evaluation.extend(candidates[budget:].tolist())
    return np.asarray(calibration, dtype=int), np.asarray(evaluation, dtype=int)


def _calibrate_model(model, context: dict, indices: np.ndarray, eval_cfg: dict) -> None:
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
    loader = _loader(context, indices)
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
    allowed = {
        round(float(value["stimulus_frequency_hz"]), 4)
        for value in class_map.values()
    }
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
