#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.eval_loop import collect_predictions, load_evaluation_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration-free inference on processed EEG.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out", default="outputs/predictions.csv")
    args = parser.parse_args()

    eval_cfg = {"data": {"processed_dirs": [args.processed_dir]}}
    context = load_evaluation_context(eval_cfg, args.ckpt)
    from cfeg.eval_loop import _loader

    indices = np.arange(len(context["dataset"]))
    y_true, logits, sample_ids = collect_predictions(
        context["model"], _loader(context, indices), context["device"]
    )
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predicted = probabilities.argmax(axis=1)
    class_map = context["checkpoint"].get("class_map") or {}
    frequencies = [
        (class_map.get(str(int(label))) or {}).get("stimulus_frequency_hz")
        for label in predicted
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_label": y_true,
            "predicted_label": predicted,
            "predicted_frequency_hz": frequencies,
            "confidence": probabilities.max(axis=1),
        }
    ).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
