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
PYTHON_BIN="${PYTHON_BIN:-python3}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

ONLINE_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
AVAILABLE_CPUS="$ONLINE_CPUS"
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r CGROUP_QUOTA CGROUP_PERIOD < /sys/fs/cgroup/cpu.max
  if [[ "$CGROUP_QUOTA" != "max" \
      && "$CGROUP_QUOTA" =~ ^[0-9]+$ \
      && "$CGROUP_PERIOD" =~ ^[1-9][0-9]*$ ]]; then
    CGROUP_CPUS=$(( CGROUP_QUOTA / CGROUP_PERIOD ))
    if (( CGROUP_CPUS < 1 )); then
      CGROUP_CPUS=1
    fi
    if (( CGROUP_CPUS < AVAILABLE_CPUS )); then
      AVAILABLE_CPUS="$CGROUP_CPUS"
    fi
  fi
fi
SAFE_MAX=$(( AVAILABLE_CPUS > 1 ? AVAILABLE_CPUS - 1 : 1 ))
if (( WORKERS > SAFE_MAX )); then
  echo "Requested $WORKERS workers but safe maximum is $SAFE_MAX (available CPUs: $AVAILABLE_CPUS); choose a smaller value." >&2
  exit 2
fi

readarray -t CONFIG_METADATA < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["output_dir"])
print("cross" if "environments" in config else "core")
PY
)
OUTPUT_DIR="${CONFIG_METADATA[0]}"
RUNNER_KIND="${CONFIG_METADATA[1]}"
LOG_DIR="$OUTPUT_DIR/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/${RUN_NAME}.remote_launcher.log"

set +e
if [[ "$RUNNER_KIND" == "cross" ]]; then
  RUNNER="scripts/run_cross_experiment.py"
else
  RUNNER="scripts/run_experiment.py"
fi
"$PYTHON_BIN" "$RUNNER" \
  --config "$CONFIG" \
  --allow-full-run \
  --workers "$WORKERS" \
  --run-name "$RUN_NAME" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
if [[ -d "$LOG_DIR" ]]; then
  cp "$LOG_FILE" "$LOG_DIR/remote_launcher.log"
fi
exit "$STATUS"
