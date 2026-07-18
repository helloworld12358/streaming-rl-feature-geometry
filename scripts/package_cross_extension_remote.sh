#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR=""
RUN_NAME=""
OUTPUT_DIR=""
DRY_RUN=0

usage() {
  echo "Usage: RL_RUN_CONTEXT=remote $0 --suite-dir DIR --run-name NAME [--output-dir DIR] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${RL_RUN_CONTEXT:-}" != "remote" ]]; then
  echo "RL_RUN_CONTEXT=remote is required." >&2
  exit 2
fi
if [[ -z "$SUITE_DIR" || -z "$RUN_NAME" ]]; then
  usage >&2
  exit 2
fi

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
ARGS=(--suite-dir "$SUITE_DIR" --run-name "$RUN_NAME")
if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
COMMAND=("$PYTHON_BIN" -m streaming_rl_feature_geometry.cross_production package "${ARGS[@]}")
printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"
echo "COMMAND=$COMMAND_TEXT"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY_RUN_ONLY=true"
  exit 0
fi
"${COMMAND[@]}"
