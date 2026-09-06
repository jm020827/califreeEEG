#!/usr/bin/env bash
# Run inside tmux. Both extraction and training see exactly one physical GPU.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env_server.sh
gpu_index="${1:-2}"
study_root="${CFEG_EXPERIMENT_ROOT}/20260906/decoder_diagnostics_v1"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$study_root"
# A study-wide lock prevents a second queue from overwriting these results.
exec 9>"$study_root/queue.lock"
flock -n 9 || { echo 'This diagnostic study is already running'; exit 1; }
exec > >(tee -a "$study_root/queue.log") 2>&1
trap 'code=$?; printf "exit_code=%s\nfinished_at=%s\n" "$code" "$(date --iso-8601=seconds)" > "$study_root/queue_status.txt"' EXIT
printf 'started_at=%s\nphysical_gpu=%s\n' "$(date --iso-8601=seconds)" "$gpu_index" > "$study_root/queue_status.txt"
.venv/bin/python scripts/run_decoder_diagnostics.py --root "$study_root" --seed 42 --epochs 100 --patience 15
