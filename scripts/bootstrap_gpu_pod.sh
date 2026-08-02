#!/usr/bin/env bash
set -euo pipefail

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"
export PROJECT_ROOT

echo "bootstrap_gpu_pod.sh now delegates to the portable Kubernetes bootstrap."
exec bash "$PROJECT_ROOT/scripts/bootstrap_k8s.sh"
