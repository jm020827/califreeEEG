from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cfeg.data.collate import build_vocabularies, collate_eeg
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.data.splits import make_cross_subject_split
from cfeg.data.transforms import make_two_views
from cfeg.losses import kl_normal, representation_consistency_loss, symmetric_kl_logits
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.seed import seed_everything
from cfeg.utils.checkpoint import save_checkpoint, save_json
from cfeg.utils.config import save_config
from cfeg.utils.params import count_parameters


def run_training(cfg: dict, *, dry_run: bool = False) -> dict:
    seed_everything(int(cfg.get("seed", 42)))
    full_ds = EEGProcessedDataset(cfg["data"]["processed_dirs"])
    manifest = pd.DataFrame([entry[2] for entry in full_ds.entries])
    split_name = cfg["data"].get("split", "cross_subject")
    if split_name == "cross_subject":
        split = make_cross_subject_split(
            manifest,
            seed=int(cfg.get("seed", 42)),
            val_ratio=float(cfg["data"].get("val_ratio", 0.2)),
            test_ratio=float(cfg["data"].get("test_ratio", 0.2)),
        )
    else:
        idx = torch.randperm(len(full_ds)).numpy()
        n_val = max(1, int(0.2 * len(idx)))
        split = type("Split", (), {"train": idx[n_val:], "val": idx[:n_val], "test": idx[:n_val]})()

    vocab = build_vocabularies()
    collate = partial(collate_eeg, vocabularies=vocab)
    train_loader = DataLoader(
        Subset(full_ds, split.train.tolist()),
        batch_size=int(cfg["data"].get("batch_size", 16)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        Subset(full_ds, split.val.tolist() if len(split.val) else split.test.tolist()),
        batch_size=int(cfg["data"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedEEGDecoder(cfg).to(device)
    params = count_parameters(model)

    batch = next(iter(train_loader))
    batch = _to_device(batch, device)
    with torch.no_grad():
        out = model(batch["x"], batch["cond"])
    dry_result = {"n_samples": len(full_ds), "logits_shape": tuple(out.logits.shape), "params": params}
    if dry_run:
        return dry_result

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = Path(cfg.get("output_dir", "outputs/debug"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    save_json(output_dir / "vocab.json", vocab)
    save_json(output_dir / "params.json", params)
    save_json(output_dir / "class_map.json", full_ds.class_map)
    wandb_run = _init_wandb(cfg, output_dir, params, len(split.train), len(split.val))
    if wandb_run is not None and cfg.get("tracking", {}).get("wandb", {}).get("watch_model", False):
        _watch_wandb(model)

    best_acc = -1.0
    patience = int(cfg["train"].get("early_stop_patience", 10))
    save_trainable_only = bool(cfg.get("checkpoint", {}).get("save_trainable_only", True))
    stale = 0
    metrics_rows = []
    for epoch in range(1, int(cfg["train"].get("epochs", 20)) + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, scaler, cfg, device, amp_enabled)
        val = evaluate_loader(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics_val.csv", index=False)
        _log_wandb(
            wandb_run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/accuracy": val["accuracy"],
                "val/nll": val["nll"],
                "lr": optimizer.param_groups[0]["lr"],
                "best/accuracy": max(best_acc, val["accuracy"]),
            },
            step=epoch,
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            config=cfg,
            epoch=epoch,
            best_metric=max(best_acc, val["accuracy"]),
            vocabularies=vocab,
            class_map=full_ds.class_map,
            asset_info={"processed_dirs": [str(p) for p in cfg["data"]["processed_dirs"]]},
            save_trainable_only=save_trainable_only,
        )
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            stale = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                config=cfg,
                epoch=epoch,
                best_metric=best_acc,
                vocabularies=vocab,
                class_map=full_ds.class_map,
                asset_info={"processed_dirs": [str(p) for p in cfg["data"]["processed_dirs"]]},
                save_trainable_only=save_trainable_only,
            )
            _log_checkpoint_artifact(wandb_run, output_dir / "best.pt", cfg, epoch)
        else:
            stale += 1
            if stale >= patience:
                break
    _finish_wandb(wandb_run)
    return {"best_accuracy": best_acc, "output_dir": str(output_dir), "params": params}


def _train_epoch(model, loader, optimizer, scaler, cfg, device, amp_enabled: bool) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss = _step_loss(model, batch, cfg)
        scaler.scale(loss).backward()
        grad_clip = cfg["train"].get("grad_clip_norm")
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * batch["x"].shape[0]
        count += batch["x"].shape[0]
    return total / max(count, 1)


def _step_loss(model, batch, cfg) -> torch.Tensor:
    loss_cfg = cfg.get("loss", {})
    aug = cfg.get("augment", {})
    if aug.get("make_two_views", True):
        (x1, cond1), (x2, cond2) = make_two_views(
            batch["x"],
            batch["cond"],
            channel_dropout_prob=float(aug.get("channel_dropout_prob", 0.2)),
            min_channels=int(aug.get("min_channels", 4)),
            noise_std_range=tuple(aug.get("noise_std_range", [0.01, 0.05])),
            time_shift_samples=int(aug.get("time_shift_samples", 8)),
        )
    else:
        x1, cond1 = batch["x"], batch["cond"]
        x2, cond2 = batch["x"], batch["cond"]
    y = batch["y"]
    out1 = model(x1, cond1, use_latent=True)
    out2 = model(x2, cond2, use_latent=True)
    loss = F.cross_entropy(out1.logits, y) + F.cross_entropy(out2.logits, y)
    ce_zero_weight = float(loss_cfg.get("ce_zero_weight", 0.5))
    if out1.logits_zero is not None:
        loss = loss + ce_zero_weight * (
            F.cross_entropy(out1.logits_zero, y) + F.cross_entropy(out2.logits_zero, y)
        )
    loss = loss + float(loss_cfg.get("lambda_cons", 0.1)) * representation_consistency_loss(out1.h, out2.h)
    loss = loss + float(loss_cfg.get("lambda_logit_cons", 0.05)) * symmetric_kl_logits(
        out1.logits, out2.logits
    )
    if out1.mu is not None:
        loss = loss + float(loss_cfg.get("beta_kl", 0.001)) * (
            kl_normal(out1.mu, out1.logvar).to(loss.device)
            + kl_normal(out2.mu, out2.logvar).to(loss.device)
        )
    return loss


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch["x"], batch["cond"], use_latent=False)
        loss = F.cross_entropy(out.logits, batch["y"])
        pred = out.logits.argmax(dim=-1)
        correct += int((pred == batch["y"]).sum().item())
        total += int(batch["y"].numel())
        total_loss += float(loss.cpu()) * batch["y"].numel()
    return {"accuracy": correct / max(total, 1), "nll": total_loss / max(total, 1)}


def _to_device(batch, device):
    out = dict(batch)
    out["x"] = batch["x"].to(device)
    out["y"] = batch["y"].to(device)
    out["cond"] = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch["cond"].items()
    }
    return out


def _init_wandb(cfg: dict, output_dir: Path, params: dict, n_train: int, n_val: int):
    wandb_cfg = cfg.get("tracking", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed.\n"
            "Install it on the GPU pod with:\n"
            "  python -m pip install wandb\n"
            "or reinstall project requirements:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    wandb_dir = Path(wandb_cfg.get("dir") or os.environ.get("WANDB_DIR", str(output_dir)))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=wandb_cfg.get("project", "calibration-free-eeg"),
        entity=wandb_cfg.get("entity"),
        name=cfg.get("run_name"),
        dir=str(wandb_dir),
        mode=wandb_cfg.get("mode", "online"),
        tags=wandb_cfg.get("tags"),
        config=cfg,
    )
    run.summary["params/total"] = params["total"]
    run.summary["params/trainable"] = params["trainable"]
    run.summary["data/n_train"] = int(n_train)
    run.summary["data/n_val"] = int(n_val)
    return run


def _log_wandb(run, metrics: dict, step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def _log_checkpoint_artifact(run, ckpt_path: Path, cfg: dict, epoch: int) -> None:
    if run is None or not cfg.get("tracking", {}).get("wandb", {}).get("log_model", False):
        return
    import wandb

    artifact = wandb.Artifact(f"{cfg.get('run_name', 'cfeg')}-best", type="checkpoint")
    artifact.add_file(str(ckpt_path))
    run.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])


def _watch_wandb(model) -> None:
    import wandb

    wandb.watch(model, log="gradients", log_freq=100)


def _finish_wandb(run) -> None:
    if run is not None:
        run.finish()
