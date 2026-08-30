#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

usage() {
  cat <<'EOF'
Calibration-Free EEG one-click commands

  bash scripts/cfeg.sh setup
  bash scripts/cfeg.sh assets synthetic
  bash scripts/cfeg.sh assets reve beta wang dong2023 wearable
  bash scripts/cfeg.sh migrate-labels [--apply]
  bash scripts/cfeg.sh smoke
  bash scripts/cfeg.sh train wang-to-beta|beta-to-wang|joint|synthetic
  bash scripts/cfeg.sh eval  wang-to-beta|beta-to-wang <checkpoint>
  bash scripts/cfeg.sh channel-stress <checkpoint> [processed-dir]
  bash scripts/cfeg.sh robustness <checkpoint> [processed-dir]
  bash scripts/cfeg.sh ablation [variant[,variant...]]
  bash scripts/cfeg.sh predict <checkpoint> <processed-dir> [output.csv] [val|test|all]
  bash scripts/cfeg.sh record-openbci <recording arguments...>

  Governed wearable outcome commands are intentionally absent. Physical reveal #2
  uses only the decision-receipt/tag-bound prepare|status|train|reveal orchestrator;
  reveal atomically publishes prediction, intervention, aggregate, and gate artifacts.

Environment:
  source scripts/env_k8s_interns.sh  # jm020827 interns cluster profile
  CFEG_BACKBONE=tiny_transformer|reve
  WANDB_API_KEY=... WANDB_MODE=online WANDB_PROJECT=calibration-free-eeg WANDB_ENTITY=...
  EEG_DATA_ROOT=/mnt/pvc/eeg HF_HOME=/mnt/pvc/hf HF_HUB_CACHE=/mnt/pvc/hf/hub
EOF
}

source_runtime() {
  source scripts/setup_gpu_pod.sh
  if [[ ! -x .venv/bin/python ]]; then
    echo "Runtime is missing. Run: bash scripts/cfeg.sh setup" >&2
    exit 1
  fi
  source .venv/bin/activate
}

tracking_args() {
  local mode="${WANDB_MODE:-}"
  if [[ -z "$mode" ]]; then
    mode="$([[ -n "${WANDB_API_KEY:-}" ]] && echo online || echo disabled)"
  fi
  TRACKING_ARGS=(
    "tracking.wandb.enabled=$([[ "$mode" == disabled ]] && echo false || echo true)"
    "tracking.wandb.mode=$mode"
    "tracking.wandb.project=${WANDB_PROJECT:-calibration-free-eeg}"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    TRACKING_ARGS+=("tracking.wandb.entity=$WANDB_ENTITY")
  fi
}

backbone_args() {
  BACKBONE_ARGS=("model.backbone.name=${CFEG_BACKBONE:-tiny_transformer}")
  if [[ "${CFEG_BACKBONE:-tiny_transformer}" == "reve" ]]; then
    BACKBONE_ARGS+=(
      "model.backbone.hf_model=brain-bzh/reve-base"
      "model.backbone.hf_positions=brain-bzh/reve-positions"
      "model.backbone.cache_dir=$HF_HUB_CACHE"
      "model.backbone.local_files_only=true"
      "model.backbone.freeze=true"
    )
  fi
}

run_train_preset() {
  local preset="$1"
  local config
  case "$preset" in
    wang-to-beta) config="configs/train/wang_to_beta.yaml" ;;
    beta-to-wang) config="configs/train/beta_to_wang.yaml" ;;
    wearable-development|wearable-loso|wearable-dry-to-wet|wearable-wet-to-dry)
      echo "This mutable wearable preset is retired; use the manifest-bound physical or confirmatory orchestrator." >&2
      exit 2
      ;;
    joint) config="configs/train/ssvep_pretrain.yaml" ;;
    synthetic) config="configs/train/debug.yaml" ;;
    *) echo "Unknown train preset: $preset" >&2; usage; exit 2 ;;
  esac
  tracking_args
  backbone_args
  python scripts/train.py --config "$config" "${TRACKING_ARGS[@]}" "${BACKBONE_ARGS[@]}"
}

run_eval_preset() {
  local preset="$1"
  local checkpoint="$2"
  local config
  case "$preset" in
    wang-to-beta) config="configs/eval/wang_to_beta.yaml" ;;
    beta-to-wang) config="configs/eval/beta_to_wang.yaml" ;;
    wearable-dry-to-wet|wearable-wet-to-dry)
      echo "This wearable evaluation preset is retired until registered as a governed post-primary action." >&2
      exit 2
      ;;
    *) echo "Unknown eval preset: $preset" >&2; usage; exit 2 ;;
  esac
  python scripts/evaluate.py --config "$config" --ckpt "$checkpoint"
}

run_stress_suite() {
  local checkpoint="$1"
  local processed_dir="$2"
  local output_dir="$3"
  local channel_override=()
  if [[ "$processed_dir" == *wearable* ]]; then
    channel_override+=("channel_sets=['all','wearable_native_8','occipital_4','occipital_2']")
  fi
  python scripts/evaluate.py --config configs/eval/channel_stress.yaml \
    --ckpt "$checkpoint" \
    "data.processed_dirs=['$processed_dir']" \
    "output_csv=$output_dir/channel_stress.csv" \
    "${channel_override[@]}"
  python scripts/evaluate.py --config configs/eval/robustness.yaml \
    --ckpt "$checkpoint" \
    "data.processed_dirs=['$processed_dir']" \
    "output_csv=$output_dir/robustness.csv"
}

command="${1:-help}"
shift || true
case "$command" in
  setup)
    CFEG_ENABLE_REVE="${CFEG_ENABLE_REVE:-1}" \
      CFEG_ENABLE_TRACKING="${CFEG_ENABLE_TRACKING:-1}" \
      bash scripts/bootstrap_k8s.sh
    ;;
  assets)
    source_runtime
    [[ "$#" -gt 0 ]] || { echo "Choose at least one asset." >&2; usage; exit 2; }
    bash scripts/prepare_k8s_assets.sh "$@"
    ;;
  migrate-labels)
    source_runtime
    python scripts/migrate_label_alignment.py \
      --processed-dir "$EEG_DATA_ROOT/processed/beta_v1" \
      --processed-dir "$EEG_DATA_ROOT/processed/wang_v1" "$@"
    ;;
  smoke)
    source_runtime
    bash scripts/prepare_k8s_assets.sh synthetic
    tracking_args
    python scripts/train.py --config configs/train/debug.yaml \
      "train.epochs=${CFEG_SMOKE_EPOCHS:-1}" \
      output_dir=outputs/smoke run_name=smoke "${TRACKING_ARGS[@]}"
    python scripts/evaluate.py --config configs/eval/channel_stress.yaml \
      --ckpt outputs/smoke/best.pt output_csv=outputs/smoke/eval/channel_stress.csv
    python scripts/evaluate.py --config configs/eval/robustness.yaml \
      --ckpt outputs/smoke/best.pt output_csv=outputs/smoke/eval/robustness.csv
    ;;
  train)
    source_runtime
    run_train_preset "${1:?train preset is required}"
    ;;
  eval)
    source_runtime
    run_eval_preset "${1:?eval preset is required}" "${2:?checkpoint is required}"
    ;;
  channel-stress)
    source_runtime
    checkpoint="${1:?checkpoint is required}"
    processed_dir="${2:-data/processed/synthetic}"
    channel_override=()
    if [[ "$processed_dir" == *wearable* ]]; then
      channel_override+=("channel_sets=['all','wearable_native_8','occipital_4','occipital_2']")
    fi
    python scripts/evaluate.py --config configs/eval/channel_stress.yaml \
      --ckpt "$checkpoint" "data.processed_dirs=['$processed_dir']" \
      "${channel_override[@]}"
    ;;
  robustness)
    source_runtime
    checkpoint="${1:?checkpoint is required}"
    processed_dir="${2:-data/processed/synthetic}"
    python scripts/evaluate.py --config configs/eval/robustness.yaml \
      --ckpt "$checkpoint" "data.processed_dirs=['$processed_dir']"
    ;;
  calibration)
    echo "Generic wearable calibration is retired until registered as a post-primary exploratory action." >&2
    exit 2
    ;;
  ablation)
    source_runtime
    if [[ -n "${1:-}" ]]; then
      python scripts/run_ablation.py --only "$1" --continue-on-error
    else
      python scripts/run_ablation.py --continue-on-error
    fi
    ;;
  controls)
    echo "Legacy prompt controls are retired; use the six-role manifest-bound physical orchestrator." >&2
    exit 2
    ;;
  predict)
    source_runtime
    python scripts/predict.py --ckpt "${1:?checkpoint is required}" \
      --processed-dir "${2:?processed-dir is required}" \
      --out "${3:-outputs/predictions.csv}" \
      --split "${4:-test}"
    ;;
  record-openbci)
    source_runtime
    python scripts/openbci_record.py "$@"
    ;;
  research)
    echo "The mutable legacy research shortcut is retired; use the canonical physical/confirmatory orchestrators." >&2
    exit 2
    ;;
  help|-h|--help) usage ;;
  *) echo "Unknown command: $command" >&2; usage; exit 2 ;;
esac
