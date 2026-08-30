#!/usr/bin/env bash
set -euo pipefail

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
source scripts/setup_gpu_pod.sh

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: bash scripts/bootstrap_k8s.sh" >&2
  exit 1
fi
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ "$#" -eq 0 ]]; then
  cat <<'USAGE'
No downloads started. Choose assets explicitly:
  bash scripts/prepare_k8s_assets.sh synthetic
  HF_TOKEN=... bash scripts/prepare_k8s_assets.sh reve
  bash scripts/prepare_k8s_assets.sh beta wang dong2023 wearable

This helper prepares and verifies full public cohorts only. For a development subset,
call scripts/fetch_dataset.py --subjects ... with separate raw/processed directories.
Set CFEG_FETCH_WORKERS=8 for parallel per-file downloads.
USAGE
  exit 0
fi

if [[ -n "${CFEG_BETA_SUBJECTS:-}${CFEG_WANG_SUBJECTS:-}${CFEG_DONG2023_SUBJECTS:-}${CFEG_WEARABLE_SUBJECTS:-}" ]]; then
  echo "prepare_k8s_assets.sh verifies full cohorts and does not accept CFEG_*_SUBJECTS." >&2
  echo "Use scripts/fetch_dataset.py --subjects ... with dedicated pilot directories." >&2
  exit 2
fi

processed_ready() {
  local root="$1"
  [[ -f "$root/signals.h5" && -f "$root/class_map.json" ]] && \
    [[ -f "$root/manifest.parquet" || -f "$root/manifest.jsonl" ]]
}

for asset in "$@"; do
  case "$asset" in
    synthetic)
      if ! processed_ready "$PROJECT_ROOT/data/processed/synthetic"; then
        "$PYTHON" scripts/prepare_synthetic.py \
          --out_dir data/processed/synthetic \
          --n_subjects "${CFEG_SYNTH_SUBJECTS:-8}" \
          --n_trials_per_class "${CFEG_SYNTH_TRIALS_PER_CLASS:-20}" \
          --n_classes "${CFEG_SYNTH_CLASSES:-4}" \
          --target_sfreq 200
      fi
      "$PYTHON" scripts/verify_assets.py --dataset synthetic --stage processed
      ;;
    reve)
      "$PYTHON" scripts/fetch_reve.py \
        --model brain-bzh/reve-base \
        --positions brain-bzh/reve-positions \
        --cache-dir "$HF_HUB_CACHE"
      "$PYTHON" scripts/verify_assets.py --model reve_base
      ;;
    beta)
      fetch_args=("$PYTHON" scripts/fetch_dataset.py --dataset beta --raw-dir "$EEG_DATA_ROOT/raw/beta" --workers "${CFEG_FETCH_WORKERS:-1}")
      "${fetch_args[@]}"
      "$PYTHON" scripts/prepare_dataset.py \
        --dataset beta \
        --raw_dir "$EEG_DATA_ROOT/raw/beta" \
        --out_dir "$EEG_DATA_ROOT/processed/beta_v1" \
        --config configs/data/beta.yaml
      "$PYTHON" scripts/verify_assets.py --dataset beta --stage processed
      ;;
    wang)
      fetch_args=("$PYTHON" scripts/fetch_dataset.py --dataset wang --raw-dir "$EEG_DATA_ROOT/raw/wang" --workers "${CFEG_FETCH_WORKERS:-1}")
      "${fetch_args[@]}"
      "$PYTHON" scripts/prepare_dataset.py \
        --dataset wang \
        --raw_dir "$EEG_DATA_ROOT/raw/wang" \
        --out_dir "$EEG_DATA_ROOT/processed/wang_v1" \
        --config configs/data/wang.yaml
      "$PYTHON" scripts/verify_assets.py --dataset wang --stage processed
      ;;
    dong2023)
      fetch_args=("$PYTHON" scripts/fetch_dataset.py --dataset dong2023 --raw-dir "$EEG_DATA_ROOT/raw/dong2023" --workers "${CFEG_FETCH_WORKERS:-1}")
      "${fetch_args[@]}"
      "$PYTHON" scripts/prepare_dataset.py \
        --dataset dong2023 \
        --raw_dir "$EEG_DATA_ROOT/raw/dong2023" \
        --out_dir "$EEG_DATA_ROOT/processed/dong2023_v1" \
        --config configs/data/dong2023.yaml
      "$PYTHON" scripts/verify_assets.py --dataset dong2023 --stage processed
      ;;
    wearable)
      fetch_args=("$PYTHON" scripts/fetch_dataset.py --dataset wearable --raw-dir "$EEG_DATA_ROOT/raw/wearable" --workers "${CFEG_FETCH_WORKERS:-1}")
      "${fetch_args[@]}"
      "$PYTHON" scripts/prepare_dataset.py \
        --dataset wearable \
        --raw_dir "$EEG_DATA_ROOT/raw/wearable" \
        --out_dir "$EEG_DATA_ROOT/processed/wearable_v3" \
        --config configs/data/wearable.yaml
      "$PYTHON" scripts/verify_assets.py --dataset wearable --stage processed
      ;;
    *)
      echo "Unknown asset: $asset (known: synthetic reve beta wang dong2023 wearable)" >&2
      exit 2
      ;;
  esac
done

du -sh "$HF_HUB_CACHE" "$EEG_DATA_ROOT" 2>/dev/null || true
