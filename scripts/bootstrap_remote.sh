#!/usr/bin/env bash
set -euo pipefail

BASE_PYTHON="${PYTHON_BIN:-python3}"
WORKERS="2"
RUN_NAME="bootstrap-$(date -u +%Y%m%dT%H%M%SZ)"
WHEELHOUSE=""
VENV_DIR=".venv"
SKIP_SMOKE=0
DRY_RUN=0

usage() {
  echo "Usage: $0 [--python PATH] [--venv DIR] [--workers N] [--run-name NAME] [--wheelhouse DIR] [--skip-smoke] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) BASE_PYTHON="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/remote_common.sh"
remote_repo_root
command -v git >/dev/null
command -v "$BASE_PYTHON" >/dev/null
remote_require_positive_integer "--workers" "$WORKERS"
if [[ "$DRY_RUN" -eq 0 && "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap_remote.sh requires Linux for actual installation." >&2
  exit 2
fi
if [[ "$DRY_RUN" -eq 1 && "$(uname -s)" != "Linux" ]]; then
  echo "WARNING: non-Linux host; validating commands only." >&2
fi
"$BASE_PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 is required; found {sys.version.split()[0]}")
print(f"Python check passed: {sys.version.split()[0]}")
PY

PIP_OPTIONS=()
if [[ -n "$WHEELHOUSE" ]]; then
  [[ -d "$WHEELHOUSE" ]] || { echo "Wheelhouse not found: $WHEELHOUSE" >&2; exit 2; }
  PIP_OPTIONS=(--no-index --find-links "$WHEELHOUSE")
fi
echo "COMMAND=$BASE_PYTHON -m venv $VENV_DIR"
echo "COMMAND=$VENV_DIR/bin/python -m pip install ${PIP_OPTIONS[*]} -r requirements.txt"
echo "COMMAND=$VENV_DIR/bin/python -m pip install ${PIP_OPTIONS[*]} -r requirements-dev.txt"
echo "COMMAND=$VENV_DIR/bin/python -m pip install ${PIP_OPTIONS[*]} --no-deps -e ."
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "COMMAND=$VENV_DIR/bin/python -m pytest -q"
  [[ "$SKIP_SMOKE" -eq 1 ]] || echo "COMMAND=PYTHON_BIN=$VENV_DIR/bin/python bash scripts/run_remote_smoke.sh --workers $WORKERS --run-name $RUN_NAME"
  exit 0
fi

"$BASE_PYTHON" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"
if [[ -z "$WHEELHOUSE" ]]; then
  "$VENV_PYTHON" -m pip install --upgrade pip
fi
"$VENV_PYTHON" -m pip install "${PIP_OPTIONS[@]}" -r requirements.txt
"$VENV_PYTHON" -m pip install "${PIP_OPTIONS[@]}" -r requirements-dev.txt
"$VENV_PYTHON" -m pip install "${PIP_OPTIONS[@]}" --no-deps -e .
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" -m pytest -q
if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  PYTHON_BIN="$VENV_PYTHON" bash scripts/run_remote_smoke.sh --workers "$WORKERS" --run-name "$RUN_NAME"
fi
echo "BOOTSTRAP_COMPLETE=true"
echo "NEXT=RL_RUN_CONTEXT=remote PYTHON_BIN=.venv/bin/python bash scripts/run_remote_suite.sh --allow-full-run --workers $WORKERS --run-name <RUN_NAME>"
