#!/usr/bin/env bash
# Reuse deterministic, label-free features; refit every learned decoder per seed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env_server.sh
export CUDA_VISIBLE_DEVICES="${1:-2}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
base_root="$CFEG_EXPERIMENT_ROOT/20260906/decoder_diagnostics_v1"
repeat_root="$CFEG_EXPERIMENT_ROOT/20260906/decoder_diagnostics_repeats"
test -f "$base_root/cache/complete.json"
mkdir -p "$repeat_root"
exec 9>"$repeat_root/queue.lock"
flock -n 9 || { echo 'The diagnostic repeat queue is already running'; exit 1; }
exec > >(tee -a "$repeat_root/queue.log") 2>&1
trap 'code=$?; printf "exit_code=%s\nfinished_at=%s\n" "$code" "$(date --iso-8601=seconds)" > "$repeat_root/queue_status.txt"' EXIT
printf 'started_at=%s\nphysical_gpu=%s\n' "$(date --iso-8601=seconds)" "$CUDA_VISIBLE_DEVICES" > "$repeat_root/queue_status.txt"
for seed in 123 456; do
    run_root="$CFEG_EXPERIMENT_ROOT/20260906/decoder_diagnostics_s$seed"
    mkdir -p "$run_root"
    if [[ ! -e "$run_root/cache" && ! -L "$run_root/cache" ]]; then
        ln -s "$base_root/cache" "$run_root/cache"
    fi
    [[ "$(readlink -f "$run_root/cache")" == "$(readlink -f "$base_root/cache")" ]] || exit 1
    printf 'seed=%s started_at=%s\n' "$seed" "$(date --iso-8601=seconds)"
    .venv/bin/python scripts/run_decoder_diagnostics.py \
        --root "$run_root" --seed "$seed" --epochs 100 --patience 15
done
