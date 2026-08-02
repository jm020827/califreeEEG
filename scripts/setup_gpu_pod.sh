#!/usr/bin/env bash
set -euo pipefail

# Source this file from any clone location. PROJECT_ROOT can still be overridden.
CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"

if [[ -z "${CFEG_HF_ROOT:-}" ]]; then
  if [[ -n "${HF_HOME:-}" ]]; then
    CFEG_HF_ROOT="$HF_HOME"
  elif [[ -n "${CFEG_EXTERNAL_ROOT:-}" ]]; then
    CFEG_HF_ROOT="$CFEG_EXTERNAL_ROOT"
  elif [[ -d "$HOME/nvme" ]]; then
    CFEG_HF_ROOT="$HOME/nvme/cache/interns/hf"
  else
    CFEG_HF_ROOT="$PROJECT_ROOT/.local/hf"
  fi
fi
export CFEG_HF_ROOT
export CFEG_EXTERNAL_ROOT="$CFEG_HF_ROOT"
export HF_HOME="${HF_HOME:-$CFEG_HF_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export CFEG_TMP_ROOT="${CFEG_TMP_ROOT:-$PROJECT_ROOT/.local/tmp}"
export TMPDIR="$CFEG_TMP_ROOT"
export TMP="$CFEG_TMP_ROOT"
export TEMP="$CFEG_TMP_ROOT"

export EEG_DATA_ROOT="${EEG_DATA_ROOT:-$PROJECT_ROOT/.local/eeg_data}"
export MNE_DATA="${MNE_DATA:-$EEG_DATA_ROOT/mne_data}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/.local/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_DIR/config}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.local/pip-cache}"
mkdir -p "$CFEG_TMP_ROOT" "$PIP_CACHE_DIR"
mkdir -p "$EEG_DATA_ROOT/raw" "$EEG_DATA_ROOT/processed" "$EEG_DATA_ROOT/mne_data"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
mkdir -p "$PROJECT_ROOT/data/processed" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/checkpoints"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CFEG_HF_ROOT=$CFEG_HF_ROOT"
echo "EEG_DATA_ROOT=$EEG_DATA_ROOT"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "MNE_DATA=$MNE_DATA"
echo "WANDB_DIR=$WANDB_DIR"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run with 'source scripts/setup_gpu_pod.sh' to keep these exports in your shell."
fi
