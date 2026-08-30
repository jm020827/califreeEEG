#!/usr/bin/env bash
set -euo pipefail

# Run train.py while saving stdout/stderr to a timestamped log.
# Usage:
#   bash scripts/run_train_logged.sh --config configs/train/debug.yaml model.backbone.name=reve

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

log_dir="${CFEG_LOG_DIR:-$PROJECT_ROOT/outputs/logs}"
mkdir -p "$log_dir"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$log_dir/train_${timestamp}.log"
CFEG_PYTHON_BIN="${CFEG_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$CFEG_PYTHON_BIN" ]]; then
  echo "Project Python is missing: $CFEG_PYTHON_BIN. Run scripts/bootstrap_k8s.sh first." >&2
  exit 1
fi

echo "Logging train output to: $log_file"

set +e
"$CFEG_PYTHON_BIN" scripts/train.py "$@" 2>&1 | tee "$log_file"
status="${PIPESTATUS[0]}"
set -e

echo "Exit code: $status" | tee -a "$log_file"
echo "Log saved to: $log_file"
exit "$status"
