#!/usr/bin/env bash
set -euo pipefail

# Resolve all large assets to the current server's persistent SSD.
CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CFEG_SCRIPT_DIR/env_server.sh" >/dev/null

export TMPDIR="$CFEG_TMP_ROOT"
export TMP="$CFEG_TMP_ROOT"
export TEMP="$CFEG_TMP_ROOT"
mkdir -p "$CFEG_TMP_ROOT" "$PIP_CACHE_DIR"
mkdir -p "$EEG_DATA_ROOT/raw" "$EEG_DATA_ROOT/processed" "$EEG_DATA_ROOT/mne_data"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
mkdir -p "$CFEG_EXPERIMENT_ROOT"
mkdir -p "$PROJECT_ROOT/data/processed" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/checkpoints"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CFEG_HF_ROOT=$CFEG_HF_ROOT"
echo "EEG_DATA_ROOT=$EEG_DATA_ROOT"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "MNE_DATA=$MNE_DATA"
echo "WANDB_DIR=$WANDB_DIR"
echo "CFEG_EXPERIMENT_ROOT=$CFEG_EXPERIMENT_ROOT"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run with 'source scripts/setup_server.sh' to keep these exports in your shell."
fi
