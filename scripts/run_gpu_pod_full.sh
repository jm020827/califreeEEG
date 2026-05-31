#!/usr/bin/env bash
set -euo pipefail

# One-command GPU pod runner:
#   cd "$HOME/work/jm020827/califreeEEG"
#   bash scripts/run_gpu_pod_full.sh
#
# Non-interactive use is also supported by pre-setting env vars such as:
#   WANDB_API_KEY=... HF_TOKEN=... CFEG_TRAIN_CONFIG=configs/train/debug.yaml bash scripts/run_gpu_pod_full.sh

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/work/jm020827/califreeEEG}"
CFEG_HF_ROOT="${CFEG_HF_ROOT:-${CFEG_EXTERNAL_ROOT:-$HOME/nvme/cache/interns/hf}}"
CFEG_EXTERNAL_ROOT="$CFEG_HF_ROOT"

cd "$PROJECT_ROOT"

prompt_default() {
  local var_name="$1"
  local prompt="$2"
  local default_value="$3"
  local current_value="${!var_name:-}"
  if [[ -n "$current_value" ]]; then
    return
  fi
  if [[ -t 0 ]]; then
    local answer
    read -r -p "$prompt [$default_value]: " answer
    printf -v "$var_name" '%s' "${answer:-$default_value}"
  else
    printf -v "$var_name" '%s' "$default_value"
  fi
}

prompt_secret() {
  local var_name="$1"
  local prompt="$2"
  local current_value="${!var_name:-}"
  if [[ -n "$current_value" ]]; then
    return
  fi
  if [[ -t 0 ]]; then
    local answer
    read -r -s -p "$prompt (empty to skip): " answer
    echo
    printf -v "$var_name" '%s' "$answer"
  else
    printf -v "$var_name" '%s' ""
  fi
}

echo "Project: $PROJECT_ROOT"
echo "HF/model cache root: $CFEG_HF_ROOT"
echo
echo "Enter credentials and run options once. Tokens are not written to the repository."

prompt_secret HF_TOKEN "Hugging Face token"
prompt_secret WANDB_API_KEY "W&B API key"
prompt_default WANDB_PROJECT "W&B project" "calibration-free-eeg"
prompt_default WANDB_ENTITY "W&B entity/team; empty is okay" ""
prompt_default WANDB_MODE "W&B mode: online/offline/disabled" "online"
prompt_default CFEG_RUN_NAME "Run name" "cfeg_$(date +%Y%m%d_%H%M%S)"
prompt_default CFEG_TRAIN_CONFIG "Training config" "configs/train/debug.yaml"
prompt_default CFEG_BACKBONE "Backbone: tiny_transformer/reve" "tiny_transformer"
prompt_default CFEG_PREPARE_SYNTHETIC "Prepare synthetic data first? y/n" "y"
prompt_default CFEG_SYNTH_SUBJECTS "Synthetic subjects" "8"
prompt_default CFEG_SYNTH_TRIALS_PER_CLASS "Synthetic trials per class" "20"
prompt_default CFEG_TRAIN_EPOCHS "Epoch override; empty keeps config" ""
prompt_default CFEG_BATCH_SIZE "Batch size override; empty keeps config" ""

if [[ "$CFEG_BACKBONE" == "reve" ]]; then
  prompt_default CFEG_FETCH_REVE "Fetch REVE into HF_HOME before training? y/n" "y"
else
  prompt_default CFEG_FETCH_REVE "Fetch REVE into HF_HOME before training? y/n" "n"
fi

export PROJECT_ROOT
export CFEG_HF_ROOT
export CFEG_EXTERNAL_ROOT
export HF_TOKEN
export WANDB_API_KEY
export EEG_DATA_ROOT="$PROJECT_ROOT/.local/eeg_data"
export EEG_MODEL_ROOT="$CFEG_HF_ROOT/eeg_models"
export HF_HOME="$CFEG_HF_ROOT"
export MNE_DATA="$EEG_DATA_ROOT/mne_data"
export WANDB_DIR="$PROJECT_ROOT/.local/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_CONFIG_DIR="$WANDB_DIR/config"

mkdir -p "$EEG_DATA_ROOT/raw" "$EEG_DATA_ROOT/processed" "$MNE_DATA"
mkdir -p "$EEG_MODEL_ROOT" "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
mkdir -p "$PROJECT_ROOT/data/processed" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/checkpoints"

echo
echo "Resolved paths:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CFEG_HF_ROOT=$CFEG_HF_ROOT"
echo "  HF_HOME=$HF_HOME"
echo "  EEG_DATA_ROOT=$EEG_DATA_ROOT"
echo "  WANDB_DIR=$WANDB_DIR"

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

if [[ -n "${HF_TOKEN:-}" ]]; then
  python -c 'import os; from huggingface_hub import login; login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)'
fi

if [[ "$WANDB_MODE" == "online" && -n "${WANDB_API_KEY:-}" ]]; then
  python -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)'
elif [[ "$WANDB_MODE" != "disabled" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "W&B key was empty; switching to offline mode."
  WANDB_MODE="offline"
fi

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"cuda_device_{i}={torch.cuda.get_device_name(i)}")
PY

if [[ "$CFEG_PREPARE_SYNTHETIC" =~ ^[Yy]$ ]]; then
  python scripts/prepare_synthetic.py \
    --out_dir data/processed/synthetic \
    --n_subjects "$CFEG_SYNTH_SUBJECTS" \
    --n_trials_per_class "$CFEG_SYNTH_TRIALS_PER_CLASS" \
    --n_classes 4 \
    --target_sfreq 200
fi

if [[ "$CFEG_FETCH_REVE" =~ ^[Yy]$ ]]; then
  python scripts/fetch_reve.py \
    --model brain-bzh/reve-base \
    --positions brain-bzh/reve-positions \
    --cache-dir "$HF_HOME"
fi

train_args=(
  python scripts/train.py
  --config "$CFEG_TRAIN_CONFIG"
  "run_name=$CFEG_RUN_NAME"
  "tracking.wandb.project=$WANDB_PROJECT"
  "tracking.wandb.mode=$WANDB_MODE"
  "tracking.wandb.tags=[\"gpu-pod\"]"
)

if [[ "$WANDB_MODE" == "disabled" ]]; then
  train_args+=("tracking.wandb.enabled=false")
else
  train_args+=("tracking.wandb.enabled=true")
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  train_args+=("tracking.wandb.entity=$WANDB_ENTITY")
fi

if [[ "$CFEG_BACKBONE" == "reve" ]]; then
  train_args+=(
    "model.backbone.name=reve"
    "model.backbone.hf_model=brain-bzh/reve-base"
    "model.backbone.hf_positions=brain-bzh/reve-positions"
    "model.backbone.cache_dir=$HF_HOME"
    "model.backbone.trust_remote_code=true"
    "model.backbone.local_files_only=true"
    "model.backbone.freeze=true"
  )
else
  train_args+=("model.backbone.name=tiny_transformer")
fi

if [[ -n "$CFEG_TRAIN_EPOCHS" ]]; then
  train_args+=("train.epochs=$CFEG_TRAIN_EPOCHS")
fi

if [[ -n "$CFEG_BATCH_SIZE" ]]; then
  train_args+=("data.batch_size=$CFEG_BATCH_SIZE")
fi

echo
echo "Starting training:"
printf ' %q' "${train_args[@]}"
echo
"${train_args[@]}"
