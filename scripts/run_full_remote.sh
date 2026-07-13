#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/full_stationary.json"
WORKERS="4"
RUN_NAME=""
ALLOW_FULL=0

usage() {
  echo "Usage: RL_RUN_CONTEXT=remote $0 --allow-full-run [--config PATH] [--workers N] --run-name NAME"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --allow-full-run) ALLOW_FULL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${RL_RUN_CONTEXT:-}" != "remote" ]]; then
  echo "Refusing full run: RL_RUN_CONTEXT must equal remote." >&2
  exit 2
fi
if [[ "$ALLOW_FULL" -ne 1 ]]; then
  echo "Refusing full run: pass --allow-full-run explicitly." >&2
  exit 2
fi
if [[ -z "$RUN_NAME" ]]; then
  echo "Refusing full run: --run-name is required." >&2
  exit 2
fi
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be a positive integer." >&2
  exit 2
fi

cd "$(dirname "$0")/.."
source .venv/bin/activate
AVAILABLE_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
SAFE_MAX=$(( AVAILABLE_CPUS > 1 ? AVAILABLE_CPUS - 1 : 1 ))
if (( WORKERS > SAFE_MAX )); then
  echo "Requested $WORKERS workers but safe maximum is $SAFE_MAX; choose a smaller value." >&2
  exit 2
fi

PROFILE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$CONFIG")"
LOG_DIR="results/$PROFILE/$RUN_NAME"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/remote_launcher.log"

set +e
python scripts/run_experiment.py \
  --config "$CONFIG" \
  --allow-full-run \
  --workers "$WORKERS" \
  --run-name "$RUN_NAME" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
exit "$STATUS"
