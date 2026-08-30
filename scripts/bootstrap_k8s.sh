#!/usr/bin/env bash
set -euo pipefail

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
source scripts/setup_gpu_pod.sh

CFEG_PYTHON="${CFEG_PYTHON:-/usr/bin/python3}"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to build the isolated CUDA environment." >&2
  exit 1
fi
if [[ "$("$CFEG_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  echo "CFEG_PYTHON must be Python 3.10; got $($CFEG_PYTHON --version 2>&1)." >&2
  exit 1
fi
if [[ ! -d .venv ]]; then
  uv venv .venv --python "$CFEG_PYTHON"
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Existing .venv is incomplete; move it aside and rerun bootstrap." >&2
  exit 1
fi
if grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' .venv/pyvenv.cfg; then
  echo "Existing .venv inherits ambient packages; move it aside and rerun bootstrap." >&2
  exit 1
fi
if [[ "$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  echo "Existing .venv is not Python 3.10; move it aside and rerun bootstrap." >&2
  exit 1
fi

extras=()
if [[ "${CFEG_ENABLE_REVE:-1}" == "1" ]]; then extras+=(reve); fi
if [[ "${CFEG_ENABLE_MOABB:-0}" == "1" ]]; then extras+=(moabb); fi
if [[ "${CFEG_ENABLE_TRACKING:-0}" == "1" ]]; then extras+=(tracking); fi
if [[ "${CFEG_ENABLE_OPENBCI:-0}" == "1" ]]; then extras+=(openbci); fi
editable_target="."
if (( ${#extras[@]} > 0 )); then
  editable_target=".[$(IFS=,; echo "${extras[*]}")]"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PIP_CACHE_DIR}/uv}"
uv pip install \
  --python .venv/bin/python \
  --torch-backend cu121 \
  --strict \
  -r requirements-cuda121.txt \
  -e "$editable_target"

.venv/bin/python scripts/verify_cuda.py

if [[ "${CFEG_RUN_TESTS:-1}" == "1" ]]; then
  .venv/bin/python -m pytest -q
fi

printf '
Kubernetes runtime ready. No model or EEG dataset was downloaded.
'
printf 'Prepare only the assets you need, for example:
'
printf '  HF_TOKEN=... bash scripts/prepare_k8s_assets.sh reve beta
'
