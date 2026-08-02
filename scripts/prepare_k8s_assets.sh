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
  CFEG_BETA_SUBJECTS=1,2 bash scripts/prepare_k8s_assets.sh beta
  CFEG_ENABLE_MOABB=1 bash scripts/bootstrap_k8s.sh
  CFEG_WANG_SUBJECTS=1,2 bash scripts/prepare_k8s_assets.sh wang

Use empty CFEG_BETA_SUBJECTS/CFEG_WANG_SUBJECTS for the full public dataset.
Wearable SSVEP remains a manual Figshare download.
USAGE
  exit 0
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
        --cache-dir "$HF_HOME"
      "$PYTHON" scripts/verify_assets.py --model reve_base
      ;;
    beta)
      fetch_args=("$PYTHON" scripts/fetch_dataset.py --dataset beta --raw-dir "$EEG_DATA_ROOT/raw/beta")
      if [[ -n "${CFEG_BETA_SUBJECTS:-}" ]]; then
        fetch_args+=(--subjects "$CFEG_BETA_SUBJECTS")
      fi
      "${fetch_args[@]}"
      "$PYTHON" scripts/prepare_dataset.py \
        --dataset beta \
        --raw_dir "$EEG_DATA_ROOT/raw/beta" \
        --out_dir "$EEG_DATA_ROOT/processed/beta_v1" \
        --config configs/data/beta.yaml
      "$PYTHON" scripts/verify_assets.py --dataset beta --stage processed
      ;;
    wang)
      if ! "$PYTHON" -c 'import moabb' >/dev/null 2>&1; then
        echo "MOABB is missing. Run CFEG_ENABLE_MOABB=1 bash scripts/bootstrap_k8s.sh" >&2
        exit 1
      fi
      fetch_args=("$PYTHON" scripts/fetch_dataset.py --dataset wang --method moabb --raw-dir "$EEG_DATA_ROOT/raw/wang")
      if [[ -n "${CFEG_WANG_SUBJECTS:-}" ]]; then
        fetch_args+=(--subjects "$CFEG_WANG_SUBJECTS")
      fi
      "${fetch_args[@]}"
      "$PYTHON" scripts/prepare_dataset.py \
        --dataset wang \
        --raw_dir "$EEG_DATA_ROOT/raw/wang" \
        --out_dir "$EEG_DATA_ROOT/processed/wang_v1" \
        --config configs/data/wang.yaml
      "$PYTHON" scripts/verify_assets.py --dataset wang --stage processed
      ;;
    wearable)
      if [[ ! -d "$EEG_DATA_ROOT/raw/wearable" ]]; then
        echo "Download wearable SSVEP manually into $EEG_DATA_ROOT/raw/wearable" >&2
        echo "https://figshare.com/articles/dataset/13560281" >&2
        exit 1
      fi
      "$PYTHON" scripts/prepare_dataset.py \
        --dataset wearable \
        --raw_dir "$EEG_DATA_ROOT/raw/wearable" \
        --out_dir "$EEG_DATA_ROOT/processed/wearable_v1" \
        --config configs/data/wearable.yaml
      "$PYTHON" scripts/verify_assets.py --dataset wearable --stage processed
      ;;
    *)
      echo "Unknown asset: $asset (known: synthetic reve beta wang wearable)" >&2
      exit 2
      ;;
  esac
done

du -sh "$HF_HOME" "$EEG_DATA_ROOT" 2>/dev/null || true
