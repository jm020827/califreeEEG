#!/usr/bin/env bash
set -uo pipefail

GPU_INDEX="${1:-2}"
STUDY_ROOT="${2:-/mnt/ssd3/jm020827/califreeEEG/experiments/20260905/gated_residual_study}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1
source scripts/env_server.sh
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
mkdir -p "$STUDY_ROOT"
printf 'physical_gpu=%s\nstarted_at=%s\n' "$GPU_INDEX" "$(date --iso-8601=seconds)" > "$STUDY_ROOT/queue_status.txt"

run_one() {
  local kind="$1"
  local seed="$2"
  local config="$3"
  local name="20260905_w2b_${kind}_s${seed}"
  local out="$STUDY_ROOT/$name"
  mkdir -p "$out"
  printf 'started_at=%s\nphysical_gpu=%s\nconfig=%s\nseed=%s\n' \
    "$(date --iso-8601=seconds)" "$GPU_INDEX" "$config" "$seed" > "$out/run_status.txt"

  set +e
  stdbuf -oL -eL .venv/bin/python scripts/train.py --config "$config" \
    "seed=$seed" "run_name=$name" "output_dir=$out" 2>&1 | tee "$out/train.log"
  local train_code=${PIPESTATUS[0]}
  set -e
  printf 'train_exit=%s\ntrain_finished_at=%s\n' "$train_code" "$(date --iso-8601=seconds)" >> "$out/run_status.txt"
  if [[ "$train_code" -ne 0 ]]; then
    return
  fi

  local branches=(combined spectral learned)
  if [[ "$kind" == "harmonic_gated" ]]; then
    branches+=(residual gated_residual)
  fi
  local branch
  for branch in "${branches[@]}"; do
    set +e
    .venv/bin/python scripts/evaluate.py \
      --config configs/eval/saved_split_by_dataset.yaml \
      --ckpt "$out/best.pt" \
      "split_csv=$out/split.csv" "test_datasets=['beta']" \
      "prediction_branch=$branch" "output_csv=$out/eval_${branch}.csv" \
      > "$out/eval_${branch}.log" 2>&1
    printf '%s\n' "$?" > "$out/eval_${branch}_status.txt"
    set -e
  done

  set +e
  .venv/bin/python scripts/evaluate.py \
    --config configs/eval/saved_split_noise.yaml \
    --ckpt "$out/best.pt" \
    "split_csv=$out/split.csv" "test_datasets=['beta']" \
    "prediction_branch=combined" "output_csv=$out/eval_noise.csv" \
    > "$out/eval_noise.log" 2>&1
  printf '%s\n' "$?" > "$out/eval_noise_status.txt"
  set -e
}

set -e
run_one harmonic_gated 42 configs/train/wang_to_beta_harmonic_gated.yaml
run_one harmonic_additive 123 configs/train/wang_to_beta_harmonic_additive.yaml
run_one harmonic_gated 123 configs/train/wang_to_beta_harmonic_gated.yaml
run_one harmonic_additive 456 configs/train/wang_to_beta_harmonic_additive.yaml
run_one harmonic_gated 456 configs/train/wang_to_beta_harmonic_gated.yaml
printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$STUDY_ROOT/queue_status.txt"
