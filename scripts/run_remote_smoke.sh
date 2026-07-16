#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/remote_smoke.json"
WORKERS="2"
RUN_NAME="remote-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="results/remote_smoke"
DRY_RUN=0

usage() {
  echo "Usage: $0 [--config PATH] [--workers N] [--run-name NAME] [--output-root DIR] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/remote_common.sh"
remote_repo_root
PYTHON_BIN="${PYTHON_BIN:-python3}"
remote_export_resources
remote_validate_workers "$WORKERS"
[[ -f "$CONFIG" ]] || { echo "Smoke plan not found: $CONFIG" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid --run-name" >&2; exit 2; }
SUITE_DIR="$OUTPUT_ROOT/$RUN_NAME"

mapfile -t PLAN < <("$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment plan-runs --config "$CONFIG" --tsv)
if [[ "$DRY_RUN" -eq 0 ]]; then
  [[ ! -e "$SUITE_DIR" ]] || { echo "Refusing to overwrite $SUITE_DIR" >&2; exit 2; }
  mkdir -p "$SUITE_DIR"
fi
for line in "${PLAN[@]}"; do
  line="${line%$'\r'}"
  IFS=$'\t' read -r ID KIND RUN_CONFIG <<<"$line"
  if [[ "$KIND" == "core" ]]; then
    COMMAND=("$PYTHON_BIN" scripts/run_experiment.py --config "$RUN_CONFIG" --workers "$WORKERS" --output-dir "$SUITE_DIR" --run-name "$ID")
  else
    COMMAND=("$PYTHON_BIN" scripts/run_cross_experiment.py --config "$RUN_CONFIG" --workers "$WORKERS" --output-dir "$SUITE_DIR" --run-name "$ID")
  fi
  printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"
  echo "SMOKE_BATCH=$ID COMMAND=$COMMAND_TEXT"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${COMMAND[@]}"
    bash scripts/aggregate_remote.sh --config "$SUITE_DIR/$ID/config.json" --run-dir "$SUITE_DIR/$ID"
  fi
done
if [[ "$DRY_RUN" -eq 0 ]]; then
  "$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment aggregate-suite --suite-dir "$SUITE_DIR"
  echo "REMOTE_SMOKE_COMPLETE=$SUITE_DIR"
else
  echo "DRY_RUN_ONLY=true"
fi
