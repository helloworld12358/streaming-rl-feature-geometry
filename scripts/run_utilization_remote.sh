#!/usr/bin/env bash
set -euo pipefail

ALLOW=0
WORKERS=16
arguments=("$@")
for ((index=0; index < ${#arguments[@]}; index++)); do
  [[ "${arguments[$index]}" == "--allow-full-run" ]] && ALLOW=1
  if [[ "${arguments[$index]}" == "--workers" ]]; then
    WORKERS="${arguments[$((index + 1))]:-}"
  fi
done

source "$(dirname "$0")/remote_common.sh"
remote_repo_root >/dev/null
remote_prepare_repo_paths
remote_export_resources
remote_require_full_gates "$ALLOW"
remote_validate_workers "$WORKERS"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PYTHON_BIN" scripts/run_utilization_experiment.py "$@"
