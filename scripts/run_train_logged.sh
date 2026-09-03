#!/usr/bin/env bash
set -euo pipefail

# Run train.py while saving stdout/stderr to a timestamped log.
# Usage:
#   bash scripts/run_train_logged.sh --config configs/train/debug.yaml model.backbone.name=reve

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
source "$CFEG_SCRIPT_DIR/setup_server.sh" >/dev/null

log_dir="${CFEG_LOG_DIR:-$CFEG_EXPERIMENT_ROOT/logs}"
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
