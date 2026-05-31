# Calibration-Free EEG Decoding

This repository implements a calibration-free SSVEP EEG decoding pipeline where the model receives both EEG signals and acquisition-condition metadata. The goal is to reduce per-subject calibration when moving across subjects, sessions, hardware, and channel layouts.

```text
acquisition metadata -> ConditionEncoder -> prompt/condition vector
EEG window + prompt  -> TinyEEGTransformer or optional REVE wrapper
representation       -> adapter + latent nuisance branch + classifier
```

The default path is fully runnable without external models by using `TinyEEGTransformerBackbone`. REVE is optional and is loaded only from the local Hugging Face cache.

## GPU Pod Layout

Use this Kubernetes pod path for the project:

```bash
~/work/jm020827/califreeeeg
```

Move or clone this repository there, then initialize the environment:

```bash
mkdir -p "$HOME/work/jm020827"
cd "$HOME/work/jm020827"
# copy/clone the repository directory here as califreeeeg
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
```

The setup script creates and exports:

```bash
PROJECT_ROOT=$HOME/work/jm020827/califreeeeg
EEG_DATA_ROOT=$PROJECT_ROOT/.local/eeg_data
EEG_MODEL_ROOT=$PROJECT_ROOT/.local/eeg_models
HF_HOME=$EEG_MODEL_ROOT/huggingface
MNE_DATA=$EEG_DATA_ROOT/mne_data
```

## Asset Policy

Do not commit raw EEG, processed datasets, REVE weights, Hugging Face cache, OpenBCI recordings, checkpoints, TensorBoard/W&B logs, or HDF5/NPY/MAT/EDF/BDF files. In the pod, large assets live under `.local/`, which is ignored by Git:

```bash
source scripts/setup_gpu_pod.sh
```

Download never happens on import. Use explicit scripts only.

## Install

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional MOABB/OpenBCI support:

```bash
python -m pip install -e '.[moabb]'
python -m pip install -e '.[openbci]'
```

## Local Smoke Test

This uses generated non-human synthetic signals and writes only to ignored data paths.

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
python scripts/prepare_synthetic.py --out_dir data/processed/synthetic --n_subjects 8 --n_trials_per_class 20 --n_classes 4 --target_sfreq 200
python scripts/inspect_manifest.py --processed_dir data/processed/synthetic
python scripts/verify_assets.py --dataset synthetic --stage processed
python scripts/train.py --config configs/train/debug.yaml --dry-run
```

To run a short actual training smoke:

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
python scripts/train.py --config configs/train/debug.yaml train.epochs=1
python scripts/evaluate.py --config configs/eval/channel_stress.yaml --ckpt outputs/debug/best.pt
```

## REVE Setup

REVE weights are not redistributed here. On the GPU/storage server:

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
huggingface-cli login
python scripts/fetch_reve.py --probe-remote --model brain-bzh/reve-base --positions brain-bzh/reve-positions
python scripts/fetch_reve.py --model brain-bzh/reve-base --positions brain-bzh/reve-positions --cache-dir "$HF_HOME"
python scripts/verify_assets.py --model reve_base
```

Training with REVE uses local files only by default:

```bash
python scripts/train.py --config configs/train/ssvep_pretrain.yaml model.backbone.name=reve
```

If REVE is unavailable, use:

```bash
python scripts/train.py --config configs/train/ssvep_pretrain.yaml model.backbone.name=tiny_transformer
```

## Public Data Setup

BETA:

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
python scripts/fetch_dataset.py --dataset beta --probe-remote
python scripts/fetch_dataset.py --dataset beta
python scripts/verify_assets.py --dataset beta --stage raw
python scripts/prepare_dataset.py --dataset beta --raw_dir "$EEG_DATA_ROOT/raw/beta" --out_dir "$EEG_DATA_ROOT/processed/beta_v1" --config configs/data/beta.yaml
```

Wang2016:

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
python -m pip install -e '.[moabb]'
python scripts/fetch_dataset.py --dataset wang --method moabb
python scripts/prepare_dataset.py --dataset wang --raw_dir "$EEG_DATA_ROOT/raw/wang" --out_dir "$EEG_DATA_ROOT/processed/wang_v1" --config configs/data/wang.yaml
```

Wearable SSVEP is treated as manual/Figshare-first:

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
python scripts/fetch_dataset.py --dataset wearable
python scripts/verify_assets.py --dataset wearable --stage raw
python scripts/prepare_dataset.py --dataset wearable --raw_dir "$EEG_DATA_ROOT/raw/wearable" --out_dir "$EEG_DATA_ROOT/processed/wearable_v1" --config configs/data/wearable.yaml
```

Dataset-specific preparers include a conservative generic `.mat`/`.npz` adapter that finds numeric EEG arrays, infers trial/channel/time axes, and writes the common processed format. If a public source uses a different schema, the script fails with the available keys/shape context instead of copying raw data into the repo.

## OpenBCI Format

Private OpenBCI sessions should look like:

```text
$EEG_DATA_ROOT/raw/openbci/sub001_ses001/
  eeg.csv
  events.csv
  session_meta.json
```

Convert with:

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
python scripts/openbci_convert.py --raw_session_dir "$EEG_DATA_ROOT/raw/openbci/sub001_ses001" --out_dir "$EEG_DATA_ROOT/processed/openbci_v1" --target_sfreq 200
```

OpenBCI Cyton raw data is recorded at 250 Hz; REVE experiments resample processed windows to 200 Hz.

## W&B Monitoring

`requirements.txt` includes W&B. In the pod, log in once:

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
source .venv/bin/activate

# Interactive login
wandb login

# Or non-interactive login if the pod has the key
# export WANDB_API_KEY=<your-key>
# wandb login "$WANDB_API_KEY"
```

Start a monitored run by enabling W&B through config overrides:

```bash
python scripts/train.py --config configs/train/ssvep_pretrain.yaml \
  run_name=ssvep_pretrain_tiny_v1 \
  tracking.wandb.enabled=true \
  tracking.wandb.project=calibration-free-eeg \
  tracking.wandb.tags='["ssvep","tiny","wang-beta"]'
```

For a quick synthetic check:

```bash
python scripts/train.py --config configs/train/debug.yaml \
  run_name=debug_wandb \
  tracking.wandb.enabled=true \
  train.epochs=3
```

This logs epoch-level `train/loss`, `val/accuracy`, `val/nll`, learning rate, parameter counts, and train/val sample counts. Checkpoint upload is disabled by default; enable it only when you want W&B artifacts:

```bash
python scripts/train.py --config configs/train/ssvep_pretrain.yaml \
  tracking.wandb.enabled=true \
  tracking.wandb.log_model=true
```

If the pod has no outbound network during training, use offline mode and sync later:

```bash
python scripts/train.py --config configs/train/ssvep_pretrain.yaml \
  tracking.wandb.enabled=true \
  tracking.wandb.mode=offline

wandb sync "$WANDB_DIR"/wandb/offline-run-*
```

## Leakage Rules

`label`, `class_id`, `stimulus_frequency_hz`, `stimulus_phase_rad`, `trial_id`, `subject_id`, `session_id`, and `source_file` are never fed to `ConditionEncoder` by default. Subject/session/trial metadata is for splitting and evaluation only.

## Useful Commands

```bash
cd "$HOME/work/jm020827/califreeeeg"
source scripts/setup_gpu_pod.sh
python scripts/train.py --config configs/train/debug.yaml
python scripts/evaluate.py --config configs/eval/channel_stress.yaml --ckpt outputs/debug/best.pt
python scripts/run_ablation.py --config configs/train/ablation.yaml --only synthetic
python scripts/export_results_table.py --runs outputs/ablation_* --out outputs/summary/ablation_results.csv
```
