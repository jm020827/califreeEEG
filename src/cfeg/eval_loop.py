from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cfeg.data.collate import build_vocabularies, collate_eeg
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.preprocess import CanonicalChannelMap
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.train_loop import _to_device, evaluate_loader
from cfeg.utils.checkpoint import load_checkpoint


def run_channel_stress_eval(eval_cfg: dict, ckpt_path: str | Path) -> dict:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    data_dirs = eval_cfg.get("data", {}).get("processed_dirs") or cfg["data"]["processed_dirs"]
    dataset = EEGProcessedDataset(data_dirs)
    vocab = ckpt.get("vocabularies") or build_vocabularies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedEEGDecoder(cfg, vocab_sizes={k: len(v) for k, v in vocab.items()}).to(device)
    strict = not bool(ckpt.get("save_trainable_only", False))
    model.load_state_dict(ckpt["model_state"], strict=strict)
    results = []
    for channel_set in eval_cfg.get("channel_sets", ["all"]):
        collate = partial(collate_eeg, vocabularies=vocab)
        loader = DataLoader(dataset, batch_size=int(cfg["data"].get("batch_size", 16)), shuffle=False, collate_fn=collate)
        metrics = _evaluate_with_channel_set(model, loader, device, channel_set)
        metrics["channel_set"] = channel_set
        results.append(metrics)
    out_csv = Path(eval_cfg.get("output_csv", Path(cfg.get("output_dir", "outputs/debug")) / "eval" / "channel_stress.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    return {"output_csv": str(out_csv), "results": results}


def _evaluate_with_channel_set(model, loader, device, channel_set: str) -> dict[str, float]:
    if channel_set == "all":
        return evaluate_loader(model, loader, device)
    keep_ids = _channel_set_ids(channel_set)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            ids = batch["cond"]["channel_ids"]
            keep = torch.zeros_like(batch["cond"]["channel_mask"])
            for cid in keep_ids:
                keep |= ids.eq(cid)
            keep &= batch["cond"]["channel_mask"]
            batch["x"] = batch["x"] * keep.unsqueeze(-1).to(batch["x"].dtype)
            batch["cond"]["channel_mask"] = keep
            out = model(batch["x"], batch["cond"], use_latent=False)
            pred = out.logits.argmax(dim=-1)
            correct += int((pred == batch["y"]).sum().item())
            total += int(batch["y"].numel())
    return {"accuracy": correct / max(total, 1), "nll": float("nan")}


def _channel_set_ids(name: str) -> list[int]:
    with Path("configs/channel_sets.yaml").open("r", encoding="utf-8") as f:
        sets = yaml.safe_load(f)
    if name not in sets:
        raise KeyError(f"Unknown channel set {name}. Known: {sorted(sets)}")
    cmap = CanonicalChannelMap.from_yaml()
    return cmap.get_ids(list(sets[name]))
