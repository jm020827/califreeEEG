#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.eval_loop import _checkpoint_split_indices, _loader, load_evaluation_context
from cfeg.train_loop import _to_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source-only diagnostic: repeatedly optimize one fixed EEG batch."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--processed-dir", action="append", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        raise ValueError("steps and lr must be positive.")

    context = load_evaluation_context(
        {
            "data": {
                "processed_dirs": args.processed_dir,
                "split": "all",
                "batch_size": 64,
                "eval_batch_size": 64,
                "num_workers": 0,
                "persistent_hdf5_handles": True,
            }
        },
        args.ckpt,
    )
    indices = _checkpoint_split_indices(context, split="train")
    batch = next(iter(_loader(context, indices, training_role=True)))
    batch = _to_device(batch, context["device"])
    model = context["model"]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    trace = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["x"], batch["cond"], use_latent=False)
        loss = F.cross_entropy(output.logits, batch["y"])
        prediction = output.logits.argmax(dim=1)
        accuracy = prediction.eq(batch["y"]).float().mean()
        if step in {0, 1, 2, 5, 10, 20, 50, 100, args.steps}:
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "accuracy": float(accuracy.detach().cpu()),
                }
            )
        if step == args.steps:
            break
        loss.backward()
        optimizer.step()

    result = {
        "schema": "cfeg.single-batch-overfit-diagnostic.v1",
        "scope": "source_training_batch_only",
        "n_samples": int(batch["y"].numel()),
        "n_labels_present": int(batch["y"].unique().numel()),
        "steps": args.steps,
        "learning_rate": args.lr,
        "initial_loss": trace[0]["loss"],
        "final_loss": trace[-1]["loss"],
        "initial_accuracy": trace[0]["accuracy"],
        "final_accuracy": trace[-1]["accuracy"],
        "overfit_success": trace[-1]["accuracy"] >= 0.95,
        "trace": trace,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
