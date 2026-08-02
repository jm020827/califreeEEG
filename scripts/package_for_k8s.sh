#!/usr/bin/env bash
set -euo pipefail

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"
ARCHIVE_PATH="${1:-$(dirname -- "$PROJECT_ROOT")/califreeEEG-k8s.tar.gz}"
mkdir -p "$(dirname -- "$ARCHIVE_PATH")"

tar -C "$PROJECT_ROOT" -czf "$ARCHIVE_PATH" \
  --exclude='./.git' \
  --exclude='./.venv*' \
  --exclude='./.local' \
  --exclude='./data/raw/*' \
  --exclude='./data/processed/*' \
  --exclude='./data/manifests/*' \
  --exclude='./outputs/*' \
  --exclude='./checkpoints/*' \
  --exclude='./__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='./.pytest_cache' \
  .

printf 'Created %s\n' "$ARCHIVE_PATH"
du -h "$ARCHIVE_PATH"
