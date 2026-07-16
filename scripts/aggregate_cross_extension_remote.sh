#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
RUN_DIR=""
SELECTED=""

usage() {
  echo "Usage: RL_RUN_CONTEXT=remote $0 --config PATH --run-dir DIR [--selected-learning-rates PATH]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --selected-learning-rates) SELECTED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${RL_RUN_CONTEXT:-}" != "remote" ]]; then
  echo "RL_RUN_CONTEXT=remote is required." >&2
  exit 2
fi
if [[ -z "$CONFIG" || -z "$RUN_DIR" ]]; then
  usage >&2
  exit 2
fi

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
ARGS=(--config "$CONFIG" --run-dir "$RUN_DIR")
if [[ -n "$SELECTED" ]]; then
  ARGS+=(--selected-learning-rates "$SELECTED")
fi
"$PYTHON_BIN" -m streaming_rl_feature_geometry.cross_production aggregate "${ARGS[@]}"
