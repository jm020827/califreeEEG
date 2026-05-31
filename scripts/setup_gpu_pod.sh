#!/usr/bin/env bash
set -euo pipefail

# Source this file inside the Kubernetes pod:
#   cd "$HOME/work/jm020827/califreeEEG"
#   source scripts/setup_gpu_pod.sh

export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/work/jm020827/califreeEEG}"
export CFEG_HF_ROOT="${CFEG_HF_ROOT:-${CFEG_EXTERNAL_ROOT:-$HOME/nvme/cache/interns/hf}}"
export CFEG_EXTERNAL_ROOT="$CFEG_HF_ROOT"

if [[ "${CFEG_KEEP_EXISTING_ENV:-0}" == "1" ]]; then
  export EEG_DATA_ROOT="${EEG_DATA_ROOT:-$PROJECT_ROOT/.local/eeg_data}"
  export EEG_MODEL_ROOT="${EEG_MODEL_ROOT:-$CFEG_HF_ROOT/eeg_models}"
  export HF_HOME="${HF_HOME:-$CFEG_HF_ROOT}"
  export MNE_DATA="${MNE_DATA:-$EEG_DATA_ROOT/mne_data}"
  export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/.local/wandb}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_DIR/config}"
else
  export EEG_DATA_ROOT="$PROJECT_ROOT/.local/eeg_data"
  export EEG_MODEL_ROOT="$CFEG_HF_ROOT/eeg_models"
  export HF_HOME="$CFEG_HF_ROOT"
  export MNE_DATA="$EEG_DATA_ROOT/mne_data"
  export WANDB_DIR="$PROJECT_ROOT/.local/wandb"
  export WANDB_CACHE_DIR="$WANDB_DIR/cache"
  export WANDB_CONFIG_DIR="$WANDB_DIR/config"
fi

mkdir -p "$EEG_DATA_ROOT/raw" "$EEG_DATA_ROOT/processed" "$EEG_DATA_ROOT/mne_data"
mkdir -p "$EEG_MODEL_ROOT" "$HF_HOME"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
mkdir -p "$PROJECT_ROOT/data/processed" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/checkpoints"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CFEG_HF_ROOT=$CFEG_HF_ROOT"
echo "EEG_DATA_ROOT=$EEG_DATA_ROOT"
echo "EEG_MODEL_ROOT=$EEG_MODEL_ROOT"
echo "HF_HOME=$HF_HOME"
echo "MNE_DATA=$MNE_DATA"
echo "WANDB_DIR=$WANDB_DIR"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run with 'source scripts/setup_gpu_pod.sh' to keep these exports in your shell."
fi
