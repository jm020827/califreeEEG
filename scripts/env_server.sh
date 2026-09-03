#!/usr/bin/env bash
# Source this profile on the current jm020827 multi-GPU server.
# Secrets are intentionally not stored here.

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"

CFEG_SERVER_ROOT="${CFEG_SERVER_ROOT:-/mnt/ssd3/jm020827/califreeEEG}"
CFEG_SERVER_CACHE_ROOT="${CFEG_SERVER_CACHE_ROOT:-/mnt/ssd3/jm020827/cache}"
export CFEG_SERVER_ROOT CFEG_SERVER_CACHE_ROOT

export CFEG_HF_ROOT="$CFEG_SERVER_CACHE_ROOT/huggingface"
export CFEG_EXTERNAL_ROOT="$CFEG_HF_ROOT"
export HF_HOME="$CFEG_HF_ROOT"
export HF_HUB_CACHE="$HF_HOME/hub"
export EEG_DATA_ROOT="$CFEG_SERVER_ROOT/eeg_data"
export MNE_DATA="$EEG_DATA_ROOT/mne_data"
export WANDB_DIR="$CFEG_SERVER_ROOT/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_CONFIG_DIR="$WANDB_DIR/config"
export WANDB_ENTITY="${CFEG_WANDB_ENTITY:-jm020827}"
export WANDB_PROJECT="${CFEG_WANDB_PROJECT:-calibration-free-eeg}"
export CFEG_EXPERIMENT_ROOT="$CFEG_SERVER_ROOT/experiments"
export CFEG_TMP_ROOT="$CFEG_SERVER_ROOT/tmp"
export PIP_CACHE_DIR="$CFEG_SERVER_CACHE_ROOT/pip/califreeEEG"

printf '%s\n' \
  "PROJECT_ROOT=$PROJECT_ROOT" \
  "HF_HOME=$HF_HOME" \
  "HF_HUB_CACHE=$HF_HUB_CACHE" \
  "EEG_DATA_ROOT=$EEG_DATA_ROOT" \
  "MNE_DATA=$MNE_DATA" \
  "WANDB_DIR=$WANDB_DIR" \
  "WANDB_ENTITY=$WANDB_ENTITY" \
  "WANDB_PROJECT=$WANDB_PROJECT" \
  "CFEG_EXPERIMENT_ROOT=$CFEG_EXPERIMENT_ROOT" \
  "CFEG_TMP_ROOT=$CFEG_TMP_ROOT" \
  "PIP_CACHE_DIR=$PIP_CACHE_DIR"
