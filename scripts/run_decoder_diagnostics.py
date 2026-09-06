#!/usr/bin/env python
"""One-GPU diagnostic suite: frozen REVE probes, CCA, harmonic, and EEGNet.

All trial features are deterministic and label-free. Fitted normalization and
classifiers use training subjects only; test subjects never select checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.utils.data import DataLoader

from cfeg.data.collate import collate_eeg
from cfeg.data.datasets import EEGProcessedDataset
from cfeg.decoder_diagnostics import (
    OCCIPITAL_IDS,
    EEGNetDecoder,
    ReferenceCCA,
    calibration_partition,
    subject_protocols,
)
from cfeg.metrics import classification_metrics
from cfeg.models.backbones.reve import REVEBackbone
from cfeg.models.heads import HarmonicPowerPrior
from cfeg.seed import seed_everything


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def tracking(args, name, **config):
    return wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        dir=os.environ["WANDB_DIR"],
        mode="online" if not args.offline else "offline",
        name=name,
        group=args.root.name,
        tags=["decoder-diagnostics", "single-gpu", "no-metadata", f"seed-{args.seed}"],
        config={**vars(args), "root": str(args.root), **config},
    )


def extract_cache(args, dataset, manifest, device):
    cache = args.root / "cache"
    cache.mkdir(exist_ok=True)
    identity = {
        "sample_ids": manifest.sample_id.tolist(),
        "model": "brain-bzh/reve-base",
        "revision": (Path(os.environ["HF_HUB_CACHE"]) / "models--brain-bzh--reve-base/refs/main")
        .read_text()
        .strip(),
        "occipital_ids": OCCIPITAL_IDS,
        "assets": {
            str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for root in dataset.roots
            for p in [root / "signals.h5", root / "manifest.parquet", root / "class_map.json"]
        },
    }
    done = cache / "complete.json"
    names = ("eeg", "reve_mean", "reve_occ8", "harmonic", "cca")
    if done.exists():
        if json.loads(done.read_text())["identity"] != identity:
            raise ValueError("Cache inputs changed; choose a new study root")
        return {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}

    run = tracking(args, f"{args.root.name}_feature_extraction", stage="cache")
    try:
        backbone = (
            REVEBackbone(
                {
                    "hf_model": "brain-bzh/reve-base",
                    "hf_positions": "brain-bzh/reve-positions",
                    "cache_dir": os.environ["HF_HUB_CACHE"],
                    "local_files_only": True,
                    "freeze": True,
                }
            )
            .to(device)
            .eval()
        )
        names64 = [backbone.canonical_map.id_to_name[i] for i in range(1, 65)]
        keep, _, _ = backbone._resolve_positions(names64)
        kept_ids = [i + 1 for i in keep]
        occ = [kept_ids.index(i) for i in OCCIPITAL_IDS]
        freqs = [float(dataset.class_map[str(i)]["stimulus_frequency_hz"]) for i in range(40)]
        harmonic = (
            HarmonicPowerPrior(
                freqs,
                200,
                400,
                n_harmonics=5,
                channel_ids=OCCIPITAL_IDS,
                harmonic_weighting="uniform",
                trainable_scale=False,
            )
            .to(device)
            .eval()
        )
        cca = ReferenceCCA(freqs, 200, 400, harmonics=5).to(device).eval()
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_eeg,
            persistent_workers=True,
        )
        arrays = {}
        offset = 0
        for step, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            if not cond["channel_mask"].all():
                raise ValueError("This spatial-feature control requires full 64-channel trials")
            with torch.inference_mode():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = backbone(x, cond, return_tokens=True)
                tokens = out.tokens.float().reshape(len(x), len(keep), -1, backbone.d_model)
                outputs = {
                    "eeg": x,
                    "reve_mean": out.h.float(),
                    "reve_occ8": tokens[:, occ].mean(dim=2).flatten(1),
                    "harmonic": harmonic(x, cond["channel_mask"]),
                    "cca": cca(x[:, [i - 1 for i in OCCIPITAL_IDS]]),
                }
            expected = manifest.iloc[offset : offset + len(x)]
            if batch["sample_id"] != expected.sample_id.tolist():
                raise ValueError("Cache ordering disagrees with manifest")
            if not np.array_equal(batch["y"].numpy(), expected.label.to_numpy()):
                raise ValueError("HDF5 labels disagree with manifest")
            for name, values in outputs.items():
                values = values.float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite features in {name}")
                if name not in arrays:
                    arrays[name] = np.lib.format.open_memmap(
                        cache / f"{name}.npy",
                        mode="w+",
                        dtype="float32",
                        shape=(len(dataset), *values.shape[1:]),
                    )
                arrays[name][offset : offset + len(x)] = values
            offset += len(x)
            if step == 1 or step % 20 == 0 or offset == len(dataset):
                print(f"cache {offset}/{len(dataset)} trials", flush=True)
                run.log({"cache/trials": offset, "cache/fraction": offset / len(dataset)})
        for array in arrays.values():
            array.flush()
        save_json(
            done, {"identity": identity, "completed_at": datetime.now(timezone.utc).isoformat()}
        )
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    return {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}


def batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def input_tensor(array, indices, device, stats=None):
    x = torch.from_numpy(np.array(array[indices], copy=True)).to(device)
    if stats is not None:
        x = (x - stats[0]) / stats[1]
    return x


@torch.no_grad()
def predict(model, array, indices, device, stats=None):
    model.eval()
    return np.concatenate(
        [
            model(input_tensor(array, part, device, stats)).float().cpu().numpy()
            for part in batches(indices, 256)
        ]
    )


def fit_model(args, method, array, labels, split, directory, device, run):
    seed_everything(args.seed)
    is_eegnet = method == "eegnet"
    model = EEGNetDecoder() if is_eegnet else nn.Linear(array.shape[1], 40)
    model.to(device)
    stats = None
    if not is_eegnet:
        train = np.asarray(array[split.train])
        stats = tuple(
            torch.from_numpy(x).to(device)
            for x in (train.mean(axis=0), train.std(axis=0).clip(1e-6))
        )
        del train
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    best, stale, rows, best_state = -1.0, 0, [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for indices in batches(rng.permutation(split.train), 128 if is_eegnet else 512):
            x = input_tensor(array, indices, device, stats)
            y = torch.as_tensor(labels[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss: {method}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if is_eegnet:
                model.constrain_weights()
            total += float(loss.detach()) * len(indices)
        val = predict(model, array, split.val, device, stats)
        accuracy = float(np.mean(val.argmax(axis=1) == labels[split.val]))
        row = {"epoch": epoch, "train_loss": total / len(split.train), "val_accuracy": accuracy}
        rows.append(row)
        pd.DataFrame(rows).to_csv(directory / "metrics_val.csv", index=False)
        run.log(row, step=epoch)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{directory.name} epoch={epoch} loss={row['train_loss']:.4f} val={accuracy:.4f}",
                flush=True,
            )
        if accuracy > best:
            best, stale = accuracy, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state": best_state,
                    "epoch": epoch,
                    "val_accuracy": best,
                    "feature_stats": None if stats is None else [x.cpu() for x in stats],
                    "method": method,
                    "seed": args.seed,
                },
                directory / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    run.summary["best/val_accuracy"] = best
    run.summary["params/trainable"] = sum(p.numel() for p in model.parameters())
    return model, stats


def record_predictions(directory, manifest, indices, logits, name):
    table = manifest.iloc[indices][["sample_id", "dataset_id", "subject_id", "label"]].copy()
    table["prediction"] = logits.argmax(axis=1)
    table.to_csv(directory / f"{name}_predictions.csv", index=False)
    rows = []
    for dataset in sorted(table.dataset_id.unique()):
        selected = table.dataset_id.eq(dataset).to_numpy()
        metric = classification_metrics(table.label.to_numpy()[selected], logits[selected])
        subjects = []
        for subject in sorted(table.loc[selected, "subject_id"].unique()):
            mask = selected & table.subject_id.eq(subject).to_numpy()
            subjects.append(float(np.mean(table.label.to_numpy()[mask] == logits[mask].argmax(1))))
        rows.append(
            {
                "dataset": dataset,
                **metric,
                "n_subjects": len(subjects),
                "subject_mean_accuracy": float(np.mean(subjects)),
            }
        )
    return rows


def calibrate(args, model, stats, array, labels, manifest, split, directory, device, run):
    """Fixed-budget head-only adaptation on unseen BETA subjects, without selection."""
    held_out = split.test[manifest.iloc[split.test].dataset_id.eq("beta").to_numpy()]
    rows, assignments = [], []
    for number, subject in enumerate(sorted(manifest.iloc[held_out].subject_id.unique())):
        indices = held_out[manifest.iloc[held_out].subject_id.eq(subject).to_numpy()]
        support, query = calibration_partition(labels[indices], args.seed + number)
        support, query = indices[support], indices[query]
        for part, role in [(support, "support"), (query, "query")]:
            for index in part:
                assignments.append(
                    {
                        "sample_id": manifest.iloc[index].sample_id,
                        "role": role,
                        "subject_id": subject,
                    }
                )
        before = predict(model, array, query, device, stats)
        adapted = copy.deepcopy(model)
        adapted.train()
        optimizer = torch.optim.AdamW(adapted.parameters(), lr=0.001, weight_decay=0.01)
        anchors = [p.detach().clone() for p in model.parameters()]
        x = input_tensor(array, support, device, stats)
        y = torch.as_tensor(labels[support], device=device)
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(adapted(x), y)
            loss = loss + 0.01 * sum(
                (p - a).square().mean() for p, a in zip(adapted.parameters(), anchors)
            )
            loss.backward()
            optimizer.step()
        after = predict(adapted, array, query, device, stats)
        for k, logits in [(0, before), (1, after)]:
            rows.append(
                {
                    "subject_id": subject,
                    "trials_per_class": k,
                    "support_trials": k * len(support),
                    **classification_metrics(labels[query], logits),
                }
            )
            record_predictions(directory, manifest, query, logits, f"calibration_{subject}_k{k}")
    pd.DataFrame(assignments).to_csv(directory / "calibration_split.csv", index=False)
    pd.DataFrame(rows).to_csv(directory / "calibration_metrics.csv", index=False)
    for k in (0, 1):
        values = [r["accuracy"] for r in rows if r["trials_per_class"] == k]
        run.summary[f"calibration/beta/k{k}/subject_mean_accuracy"] = float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU before running this suite")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    seed_everything(args.seed)
    args.root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "commit": commit,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(args.root / "study.json", metadata)
    dataset = EEGProcessedDataset(
        [
            Path(os.environ["EEG_DATA_ROOT"]) / "processed" / f"{name}_v1"
            for name in ("wang", "beta")
        ]
    )
    manifest = pd.DataFrame([entry[2] for entry in dataset.entries])
    labels = manifest.label.to_numpy(dtype=np.int64)
    protocols = subject_protocols(manifest, args.seed)
    arrays = extract_cache(args, dataset, manifest, device)
    torch.cuda.empty_cache()
    results, failures = [], []
    for protocol in ("wang", "beta", "pooled", "wang_to_beta"):
        split = protocols[protocol]
        for method in ("harmonic", "cca", "reve_mean", "reve_occ8", "eegnet"):
            directory = args.root / f"{protocol}_{method}_s{args.seed}"
            directory.mkdir(exist_ok=True)
            if (directory / "complete.json").exists():
                results.extend(json.loads((directory / "complete.json").read_text())["results"])
                continue
            assignment = manifest[["sample_id", "dataset_id", "subject_id", "label"]].copy()
            assignment["split"] = "unused"
            for name in ("train", "val", "test"):
                assignment.loc[getattr(split, name), "split"] = name
            assignment.to_csv(directory / "split.csv", index=False)
            run = tracking(
                args,
                f"20260906_{protocol}_{method}_s{args.seed}",
                method=method,
                protocol=protocol,
                source_commit=commit,
                train_samples=len(split.train),
                val_samples=len(split.val),
                test_samples=len(split.test),
            )
            try:
                if method in ("harmonic", "cca"):
                    logits = np.asarray(arrays[method][split.test])
                    if method == "cca":
                        # Correlations are scores, not probability-calibrated logits.
                        logits = logits.astype(np.float64)
                else:
                    array = arrays["eeg" if method == "eegnet" else method]
                    model, stats = fit_model(
                        args, method, array, labels, split, directory, device, run
                    )
                    logits = predict(model, array, split.test, device, stats)
                rows = record_predictions(directory, manifest, split.test, logits, "test")
                for row in rows:
                    row.update({"protocol": protocol, "method": method, "seed": args.seed})
                    if method == "cca":
                        # Avoid suggesting raw correlations are calibrated probabilities.
                        row.pop("nll", None)
                        row.pop("ece", None)
                    run.summary[f"test/{row['dataset']}/accuracy"] = row["accuracy"]
                save_json(directory / "metrics_test.json", rows)
                if protocol == "pooled" and method in ("reve_mean", "reve_occ8"):
                    calibrate(
                        args, model, stats, array, labels, manifest, split, directory, device, run
                    )
                results.extend(rows)
                pd.DataFrame(results).to_csv(args.root / "summary.csv", index=False)
                save_json(
                    directory / "complete.json",
                    {
                        "results": rows,
                        "wandb_url": run.url,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(f"COMPLETE {protocol}/{method}: {rows}", flush=True)
                run.finish()
            except Exception as exc:
                failures.append({"protocol": protocol, "method": method, "error": repr(exc)})
                save_json(directory / "failure.json", failures[-1])
                run.finish(exit_code=1)
                raise
    save_json(
        args.root / "finished.json",
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "failures": failures,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
