#!/usr/bin/env bash
set -euo pipefail

# Run train.py while saving stdout/stderr to a timestamped log.
# Usage:
#   bash scripts/run_train_logged.sh --config configs/train/debug.yaml model.backbone.name=reve

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/work/jm020827/califreeEEG}"
cd "$PROJECT_ROOT"

log_dir="${CFEG_LOG_DIR:-$PROJECT_ROOT/outputs/logs}"
mkdir -p "$log_dir"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$log_dir/train_${timestamp}.log"

echo "Logging train output to: $log_file"

set +e
python scripts/train.py "$@" 2>&1 | tee "$log_file"
status="${PIPESTATUS[0]}"
set -e

echo "Exit code: $status" | tee -a "$log_file"
echo "Log saved to: $log_file"
exit "$status"
