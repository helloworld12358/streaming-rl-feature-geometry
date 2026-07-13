#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Python 3.11 is required; found {sys.version.split()[0]}")
print(f"Python check passed: {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q

SMOKE_NAME="remote-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/run_experiment.py \
  --config configs/smoke.json \
  --workers 1 \
  --run-name "$SMOKE_NAME"
python scripts/validate_results.py \
  "results/smoke/$SMOKE_NAME" \
  --config configs/smoke.json

echo "Bootstrap and smoke passed."
echo "Stationary full command:"
echo "RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/full_stationary.json --workers 4 --run-name full-stationary-<COMMIT_SHA>"
echo "Non-stationary full command:"
echo "RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/full_nonstationary.json --workers 4 --run-name full-nonstationary-<COMMIT_SHA>"
