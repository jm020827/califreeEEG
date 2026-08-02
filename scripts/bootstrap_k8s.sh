#!/usr/bin/env bash
set -euo pipefail

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
source scripts/setup_gpu_pod.sh

CFEG_PYTHON="${CFEG_PYTHON:-python3}"
if [[ ! -d .venv ]]; then
  if "$CFEG_PYTHON" -c 'import torch' >/dev/null 2>&1; then
    "$CFEG_PYTHON" -m venv --system-site-packages .venv
  else
    "$CFEG_PYTHON" -m venv .venv
  fi
fi
source .venv/bin/activate

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.local/pip-cache}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

if [[ "${CFEG_ENABLE_REVE:-1}" == "1" ]]; then
  python -m pip install -e '.[reve]'
fi
if [[ "${CFEG_ENABLE_MOABB:-0}" == "1" ]]; then
  python -m pip install -e '.[moabb]'
fi
if [[ "${CFEG_ENABLE_TRACKING:-0}" == "1" ]]; then
  python -m pip install -e '.[tracking]'
fi
if [[ "${CFEG_ENABLE_OPENBCI:-0}" == "1" ]]; then
  python -m pip install -e '.[openbci]'
fi

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [[ "${CFEG_RUN_TESTS:-1}" == "1" ]]; then
  python -m pytest -q
fi

printf '
Kubernetes runtime ready. No model or EEG dataset was downloaded.
'
printf 'Prepare only the assets you need, for example:
'
printf '  HF_TOKEN=... bash scripts/prepare_k8s_assets.sh reve beta
'
