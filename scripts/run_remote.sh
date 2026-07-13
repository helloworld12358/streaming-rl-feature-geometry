#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-full}"
RUN_NAME="${2:-${PROFILE}-$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS="${WORKERS:-4}"

if [[ "$PROFILE" != "validation" && "$PROFILE" != "full" ]]; then
  echo "profile must be validation or full" >&2
  exit 2
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt
python -m pytest -q
export RL_RUN_CONTEXT=remote
python scripts/run_experiment.py \
  --config "configs/${PROFILE}.json" \
  --workers "$WORKERS" \
  --run-name "$RUN_NAME"
