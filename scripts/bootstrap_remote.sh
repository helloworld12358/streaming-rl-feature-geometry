#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 is required; found {sys.version.split()[0]}")
print(f"Python check passed: {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m pip install -e '.[dev]'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

"$PYTHON_BIN" -m pytest -q

SMOKE_NAME="remote-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
"$PYTHON_BIN" scripts/run_experiment.py \
  --config configs/smoke.json \
  --workers 1 \
  --run-name "$SMOKE_NAME"
"$PYTHON_BIN" scripts/validate_results.py \
  "results/smoke/$SMOKE_NAME" \
  --config configs/smoke.json

CROSS_SMOKE_NAME="cross-remote-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
"$PYTHON_BIN" scripts/run_cross_experiment.py \
  --config configs/cross_smoke.json \
  --workers 1 \
  --run-name "$CROSS_SMOKE_NAME"

echo "Bootstrap and smoke passed."
echo "Stationary full command:"
echo "RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/full_stationary.json --workers 16 --run-name full-stationary-<COMMIT_SHA>"
echo "Non-stationary full command:"
echo "RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/full_nonstationary.json --workers 16 --run-name full-nonstationary-<COMMIT_SHA>"
echo "Cross-environment full command:"
echo "RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name cross-full-<COMMIT_SHA>"
