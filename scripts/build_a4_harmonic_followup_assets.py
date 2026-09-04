#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

plt.switch_backend("Agg")


RUNS = [
    ("Fundamental only", "20260904_w2b_A4H_ablate_fundamental_only_s42_r2"),
    ("No channel corruption", "20260904_w2b_A4H_ablate_no_channel_corruption_s42_r2"),
    ("A2H clean", "20260904_w2b_A2H_clean_repro_s42_r2"),
    ("Fixed scale", "20260904_w2b_A4H_ablate_fixed_scale_s42_r2"),
    ("Inverse harmonics", "20260904_w2b_A4H_ablate_inverse_harmonics_s42_r2"),
    ("Three harmonics", "20260904_w2b_A4H_three_harmonics_s42"),
    ("Inverse square", "20260904_w2b_A4H_inverse_square_s42"),
]


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def _test_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics_test.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _effective_scale(run_dir: Path) -> float | None:
    path = run_dir / "best.pt"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", {})
    if "spectral_prior.log_scale" in state:
        return math.exp(float(state["spectral_prior.log_scale"].item()))
    config = checkpoint.get("config", {})
    spectral = config.get("model", {}).get("spectral_prior", {})
    if spectral.get("enabled"):
        return float(spectral.get("logit_scale", 1.0))
    return None


def _summary_rows(root: Path, reference: Path | None) -> list[dict]:
    rows = []
    candidates = list(RUNS)
    if reference is not None:
        candidates.insert(0, ("A4H reference", str(reference)))
    for label, name in candidates:
        run_dir = Path(name) if Path(name).is_absolute() else root / name
        history = _read_csv(run_dir / "metrics_val.csv")
        if history is None:
            continue
        best_index = history["val_accuracy"].astype(float).idxmax()
        best = history.loc[best_index]
        test = _test_metrics(run_dir)
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "best_epoch": int(best["epoch"]),
                "source_val_accuracy": float(best["val_accuracy"]),
                "target_accuracy": test.get("accuracy"),
                "target_balanced_accuracy": test.get("balanced_accuracy"),
                "target_macro_f1": test.get("macro_f1"),
                "target_nll": test.get("nll"),
                "target_ece": test.get("ece"),
                "target_itr_bits_per_min": test.get("itr_bits_per_min"),
                "effective_harmonic_scale": _effective_scale(run_dir),
            }
        )
    return rows


def _save_validation(root: Path, reference: Path | None, output: Path) -> Path | None:
    candidates = list(RUNS[:5])
    if reference is not None:
        candidates.insert(0, ("A4H reference", str(reference)))
    histories = []
    for label, name in candidates:
        run_dir = Path(name) if Path(name).is_absolute() else root / name
        frame = _read_csv(run_dir / "metrics_val.csv")
        if frame is not None:
            histories.append((label, frame))
    if not histories:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for label, frame in histories:
        axes[0].plot(frame["epoch"], 100 * frame["val_accuracy"], label=label, linewidth=2)
        axes[1].plot(frame["epoch"], frame["train_loss"], label=label, linewidth=2)
    axes[0].set(title="Wang validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].set(title="Training objective", xlabel="Epoch", ylabel="Composite loss")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = output / "overnight_validation_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_target(summary: pd.DataFrame, output: Path) -> Path | None:
    frame = summary.dropna(subset=["target_accuracy"])
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.8))
    values = 100 * frame["target_accuracy"].astype(float)
    bars = ax.bar(frame["label"], values, color="#4C78A8")
    ax.axhline(2.5, color="#777777", linestyle=":", label="Chance 2.5%")
    ax.axhline(47.79, color="#E45756", linestyle="--", label="Fixed harmonic 47.79%")
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set(title="BETA zero-shot accuracy", ylabel="Accuracy (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / "w2b_target_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_branches(summary: pd.DataFrame, output: Path) -> Path | None:
    records = []
    for row in summary.itertuples():
        run_dir = Path(row.run_dir)
        record = {"label": row.label, "Combined": row.target_accuracy}
        for branch, title in [("learned", "Learned"), ("spectral", "Harmonic")]:
            frame = _read_csv(run_dir / f"eval_{branch}.csv")
            if frame is not None and not frame.empty:
                record[title] = float(frame.iloc[0]["accuracy"])
        if len(record) > 2:
            records.append(record)
    if not records:
        return None
    frame = pd.DataFrame(records).set_index("label") * 100
    ax = frame.plot(kind="bar", figsize=(11, 4.8), width=0.8)
    ax.axhline(2.5, color="#777777", linestyle=":", linewidth=1)
    ax.set(title="BETA branch decomposition", ylabel="Accuracy (%)", xlabel="")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    path = output / "w2b_branch_decomposition.png"
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)
    return path


def _save_w2b_noise(summary: pd.DataFrame, output: Path) -> Path | None:
    records = []
    for row in summary.itertuples():
        frame = _read_csv(Path(row.run_dir) / "eval_noise_beta.csv")
        if frame is None:
            continue
        for item in frame.itertuples():
            scenario = str(item.scenario).removeprefix("saved_test_beta_")
            scenario = "clean" if scenario == "saved_test_beta" else scenario
            records.append({"label": row.label, "scenario": scenario, "accuracy": item.accuracy})
    if not records:
        return None
    frame = pd.DataFrame(records).pivot(index="scenario", columns="label", values="accuracy")
    order = ["clean", "gaussian_010", "gaussian_020", "ssvep_band_noise_010", "occipital_4"]
    frame = frame.reindex([item for item in order if item in frame.index]) * 100
    ax = frame.plot(marker="o", figsize=(11, 4.8))
    ax.set(title="BETA held-out robustness", ylabel="Accuracy (%)", xlabel="Perturbation")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    ax.figure.tight_layout()
    path = output / "w2b_noise_robustness.png"
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)
    return path


def _save_pooled(root: Path, output: Path) -> list[Path]:
    run_dir = root / "20260904_pooled_WB_A4H_capacity_noise_e25_s42_r2"
    paths = []
    noise = _read_csv(run_dir / "eval_noise_by_dataset.csv")
    if noise is not None:
        records = []
        for row in noise.itertuples():
            scenario = str(row.scenario)
            dataset = "wang" if scenario.startswith("saved_test_wang") else "beta"
            suffix = scenario.removeprefix(f"saved_test_{dataset}_")
            suffix = "clean" if suffix == f"saved_test_{dataset}" else suffix
            records.append({"dataset": dataset, "scenario": suffix, "accuracy": row.accuracy})
        frame = pd.DataFrame(records).pivot(index="scenario", columns="dataset", values="accuracy")
        order = ["clean", "gaussian_010", "gaussian_020", "ssvep_band_noise_010", "occipital_4"]
        frame = frame.reindex([item for item in order if item in frame.index]) * 100
        ax = frame.plot(marker="o", figsize=(8.5, 4.6))
        ax.set(title="Pooled held-out subject robustness", ylabel="Accuracy (%)", xlabel="Perturbation")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.25)
        ax.figure.tight_layout()
        path = output / "pooled_noise_robustness.png"
        ax.figure.savefig(path, dpi=180)
        plt.close(ax.figure)
        paths.append(path)
    calibration = _read_csv(run_dir / "eval_calibration_beta_0_1.csv")
    if calibration is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.plot(
            calibration["calibration_trials_per_class"],
            100 * calibration["accuracy"],
            marker="o",
            linewidth=2,
        )
        ax.set(
            title="Observed BETA calibration result (unmatched evaluation sets)",
            xlabel="Calibration trials per class",
            ylabel="Accuracy (%)",
            xticks=calibration["calibration_trials_per_class"].astype(int),
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output / "pooled_calibration_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _summary_rows(args.experiment_root, args.reference_run)
    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "overnight_summary.csv"
    summary.to_csv(summary_path, index=False)
    generated = [summary_path]
    for path in [
        _save_validation(args.experiment_root, args.reference_run, args.output_dir),
        _save_target(summary, args.output_dir),
        _save_branches(summary, args.output_dir),
        _save_w2b_noise(summary, args.output_dir),
    ]:
        if path is not None:
            generated.append(path)
    generated.extend(_save_pooled(args.experiment_root, args.output_dir))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
