#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR=""
OUTPUT_DIR=""
DRY_RUN=0

usage() {
  echo "Usage: $0 --suite-dir DIR [--output-dir DIR] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SUITE_DIR" ]] || { usage >&2; exit 2; }
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMMAND=("$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment package-suite --suite-dir "$SUITE_DIR")
[[ -z "$OUTPUT_DIR" ]] || COMMAND+=(--output-dir "$OUTPUT_DIR")
printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"
echo "COMMAND=$COMMAND_TEXT"
[[ "$DRY_RUN" -eq 0 ]] || exit 0
"${COMMAND[@]}"
