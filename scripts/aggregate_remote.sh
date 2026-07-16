#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
RUN_DIR=""
SUITE_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: $0 (--suite-dir DIR | --run-dir DIR [--config CONFIG]) [--dry-run]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$SUITE_DIR" && -n "$RUN_DIR" ]]; then
  echo "Choose exactly one of --suite-dir or --run-dir." >&2
  exit 2
fi
if [[ -z "$SUITE_DIR" && -z "$RUN_DIR" ]]; then
  echo "One of --suite-dir or --run-dir is required." >&2
  exit 2
fi

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -n "$SUITE_DIR" ]]; then
  COMMAND=("$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment aggregate-suite --suite-dir "$SUITE_DIR")
else
  [[ -n "$CONFIG" ]] || CONFIG="$RUN_DIR/config.json"
  COMMAND=("$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment aggregate-run --run-dir "$RUN_DIR" --config "$CONFIG")
fi
printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"
echo "COMMAND=$COMMAND_TEXT"
[[ "$DRY_RUN" -eq 0 ]] || exit 0
"${COMMAND[@]}"
