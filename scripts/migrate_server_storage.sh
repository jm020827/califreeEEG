#!/usr/bin/env bash
set -euo pipefail

CFEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$CFEG_SCRIPT_DIR/.." && pwd)}"

apply=0
case "${1:-}" in
  "") ;;
  --apply) apply=1 ;;
  -h|--help)
    cat <<'EOF'
Safely migrate the jm020827 server layout. The default is a dry run.

  bash scripts/migrate_server_storage.sh
  bash scripts/migrate_server_storage.sh --apply

Stop REVE downloads and training before --apply. Existing destinations are never overwritten,
and cross-filesystem moves are refused.
EOF
    exit 0
    ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

source "$PROJECT_ROOT/scripts/env_k8s_interns.sh"
legacy_hf_cache="${CFEG_LEGACY_HF_CACHE:-$HF_HOME}"
legacy_eeg_root="${CFEG_LEGACY_EEG_ROOT:-/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/.local/eeg_data}"

move_dir() {
  local label="$1"
  local source_path="$2"
  local target_path="$3"
  if [[ -e "$target_path" ]]; then
    if [[ -e "$source_path" ]]; then
      echo "SKIP $label: source and destination both exist; compare them manually."
      echo "  source=$source_path"
      echo "  target=$target_path"
    else
      echo "OK   $label: already migrated -> $target_path"
    fi
    return
  fi
  if [[ ! -e "$source_path" ]]; then
    echo "MISS $label: source not found -> $source_path"
    return
  fi
  if [[ "$apply" != 1 ]]; then
    echo "MOVE $label"
    echo "  source=$source_path"
    echo "  target=$target_path"
    return
  fi

  local target_parent
  target_parent="$(dirname -- "$target_path")"
  mkdir -p "$target_parent"
  local source_device target_device
  source_device="$(stat -c '%d' "$source_path")"
  target_device="$(stat -c '%d' "$target_parent")"
  if [[ "$source_device" != "$target_device" ]]; then
    echo "REFUSE $label: source and destination are on different filesystems." >&2
    echo "Copy and verify manually instead of using mv." >&2
    exit 1
  fi
  mv -- "$source_path" "$target_path"
  echo "DONE $label -> $target_path"
}

for repo_dir in models--brain-bzh--reve-base models--brain-bzh--reve-positions; do
  move_dir "$repo_dir" "$legacy_hf_cache/$repo_dir" "$HF_HUB_CACHE/$repo_dir"
  move_dir "$repo_dir locks" "$legacy_hf_cache/.locks/$repo_dir" "$HF_HUB_CACHE/.locks/$repo_dir"
done
move_dir "EEG data" "$legacy_eeg_root" "$EEG_DATA_ROOT"

if [[ "$apply" == 1 ]]; then
  echo
  echo "Migration finished. Verify with:"
  echo "  source scripts/env_k8s_interns.sh"
  echo "  .venv/bin/python scripts/fetch_reve.py --cache-dir \"$HF_HUB_CACHE\" --dry-run"
  echo "  .venv/bin/python scripts/verify_assets.py --dataset beta --stage processed"
  echo "  .venv/bin/python scripts/verify_assets.py --dataset wang --stage processed"
else
  echo
  echo "Dry run only. Stop downloads/training, then apply with:"
  echo "  bash scripts/migrate_server_storage.sh --apply"
fi
