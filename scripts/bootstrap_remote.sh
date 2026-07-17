#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
WORKERS="2"
RUN_NAME="bootstrap-$(date -u +%Y%m%dT%H%M%SZ)"
WHEELHOUSE=""
SKIP_SMOKE=0

usage() {
  echo "Usage: $0 [--python /usr/bin/python3] [--workers N] [--run-name NAME] [--wheelhouse DIR] [--skip-smoke]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/remote_common.sh"
remote_repo_root >/dev/null
remote_prepare_repo_paths
remote_export_resources
remote_validate_workers "$WORKERS"
command -v git >/dev/null
[[ "$PYTHON_BIN" == "/usr/bin/python3" ]] || {
  echo "Production bootstrap requires /usr/bin/python3; received $PYTHON_BIN" >&2
  exit 2
}
command -v "$PYTHON_BIN" >/dev/null
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 is required; found {sys.version.split()[0]}")
print(f"Python check passed: executable={sys.executable} version={sys.version.split()[0]}")
PY

PIP_OPTIONS=()
if [[ -n "$WHEELHOUSE" ]]; then
  WHEELHOUSE="$(realpath -m "$WHEELHOUSE")"
  case "$WHEELHOUSE/" in
    "$REPO_ROOT"/*/) ;;
    *) echo "Wheelhouse must be inside repository: $WHEELHOUSE" >&2; exit 2 ;;
  esac
  [[ -d "$WHEELHOUSE" ]] || { echo "Wheelhouse not found: $WHEELHOUSE" >&2; exit 2; }
  PIP_OPTIONS=(--no-index --find-links "$WHEELHOUSE")
fi

"$PYTHON_BIN" -m pip install "${PIP_OPTIONS[@]}" -r requirements.txt -r requirements-dev.txt
"$PYTHON_BIN" -m pip install "${PIP_OPTIONS[@]}" --no-build-isolation --no-deps -e .
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -c 'import numpy,pandas,matplotlib,pytest,streaming_rl_feature_geometry; print(numpy.__version__, pandas.__version__, matplotlib.__version__)'
"$PYTHON_BIN" -m pytest -q
if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  "$PYTHON_BIN" scripts/run_cross_experiment.py \
    --config configs/cross_extension_smoke.json \
    --workers "$WORKERS" \
    --output-dir "$RESULTS_ROOT/bootstrap_smoke" \
    --run-name "$RUN_NAME"
fi
echo "BOOTSTRAP_COMPLETE=true"
echo "PYTHON_BIN=$PYTHON_BIN"
