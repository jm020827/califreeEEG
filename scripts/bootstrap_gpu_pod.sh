#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-$HOME/work/jm020827/califreeEEG}"
source scripts/setup_gpu_pod.sh

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Bootstrap complete."
echo "Next time run:"
echo "  cd \"$PROJECT_ROOT\""
echo "  source scripts/setup_gpu_pod.sh"
echo "  source .venv/bin/activate"
