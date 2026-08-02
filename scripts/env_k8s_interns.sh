#!/usr/bin/env bash
# Source this profile on the jm020827 interns Kubernetes pods.
# Secrets are intentionally not stored here.

CFEG_SERVER_HF_HOME="${CFEG_SERVER_HF_HOME:-/mnt/nvme/cache/interns/hf}"
CFEG_SERVER_PERSIST_ROOT="${CFEG_SERVER_PERSIST_ROOT:-/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage}"
CFEG_SERVER_FAST_ROOT="${CFEG_SERVER_FAST_ROOT:-/mnt/nvme/cache/interns}"

export CFEG_HF_ROOT="$CFEG_SERVER_HF_HOME"
export CFEG_EXTERNAL_ROOT="$CFEG_SERVER_HF_HOME"
export HF_HOME="$CFEG_SERVER_HF_HOME"
export HF_HUB_CACHE="$CFEG_SERVER_HF_HOME/hub"
export EEG_DATA_ROOT="$CFEG_SERVER_PERSIST_ROOT/eeg_data"
export MNE_DATA="$EEG_DATA_ROOT/mne_data"
export WANDB_DIR="$CFEG_SERVER_PERSIST_ROOT/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_CONFIG_DIR="$WANDB_DIR/config"
export CFEG_TMP_ROOT="$CFEG_SERVER_FAST_ROOT/tmp/jm020827/califreeEEG"
export PIP_CACHE_DIR="$CFEG_SERVER_FAST_ROOT/pip/jm020827/califreeEEG"

printf '%s\n' \
  "HF_HOME=$HF_HOME" \
  "HF_HUB_CACHE=$HF_HUB_CACHE" \
  "EEG_DATA_ROOT=$EEG_DATA_ROOT" \
  "MNE_DATA=$MNE_DATA" \
  "WANDB_DIR=$WANDB_DIR" \
  "CFEG_TMP_ROOT=$CFEG_TMP_ROOT" \
  "PIP_CACHE_DIR=$PIP_CACHE_DIR"
