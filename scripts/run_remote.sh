#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper retained from PR #1. New automation should call
# bootstrap_remote.sh and run_remote_full.sh directly.
PROFILE="${1:-}"
RUN_NAME="${2:-}"
WORKERS="${WORKERS:-4}"
ALLOW="${3:-}"
DRY_RUN="${4:-}"

if [[ "$PROFILE" != "full_stationary" && "$PROFILE" != "full_nonstationary" ]]; then
  echo "Usage: RL_RUN_CONTEXT=remote $0 {full_stationary|full_nonstationary} RUN_NAME --allow-full-run [--dry-run]" >&2
  exit 2
fi
if [[ -z "$RUN_NAME" || "$ALLOW" != "--allow-full-run" ]]; then
  echo "RUN_NAME and explicit --allow-full-run are required." >&2
  exit 2
fi
if [[ -n "$DRY_RUN" && "$DRY_RUN" != "--dry-run" ]]; then
  echo "Only optional fourth argument --dry-run is supported." >&2
  exit 2
fi

ARGS=(scripts/run_remote_full.sh \
  --allow-full-run \
  --config "configs/${PROFILE}.json" \
  --workers "$WORKERS" \
  --run-name "$RUN_NAME")
[[ -z "$DRY_RUN" ]] || ARGS+=(--dry-run)
exec "${ARGS[@]}"
