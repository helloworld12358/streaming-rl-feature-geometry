#!/usr/bin/env bash
set -euo pipefail

WORKERS="16"
RUN_NAME=""
STAGE="all"
OUTPUT_ROOT="results/cross_extension"
PILOT_NAME=""
STORAGE_REPORT=""
WHEELHOUSE=""
EXPECTED_BRANCH=""
EXPECTED_COMMIT=""
ALLOW_FULL=0
SKIP_BOOTSTRAP=0
RESUME=0
RETRY_INVALID=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: RL_RUN_CONTEXT=remote bash scripts/remote_one_click.sh --allow-full-run \
  --workers N --run-name NAME [--stage fixed-full|lr-tune|lr-eval|norm-scaled|all] \
  [--output-root DIR] [--pilot-name NAME] [--storage-report PATH] \
  [--wheelhouse DIR] \
  [--expected-branch BRANCH] [--expected-commit SHA] [--skip-bootstrap] \
  [--resume] [--retry-invalid] [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --pilot-name) PILOT_NAME="$2"; shift 2 ;;
    --storage-report) STORAGE_REPORT="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --expected-branch) EXPECTED_BRANCH="$2"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="$2"; shift 2 ;;
    --allow-full-run) ALLOW_FULL=1; shift ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
    --resume) RESUME=1; shift ;;
    --retry-invalid|--retry-failed) RETRY_INVALID=1; RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/remote_common.sh"
remote_require_full_gates "$ALLOW_FULL"
remote_require_positive_integer "--workers" "$WORKERS"
if [[ -z "$RUN_NAME" || ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-name is required and must be filesystem-safe." >&2
  exit 2
fi
case "$STAGE" in
  fixed-full|lr-tune|lr-eval|norm-scaled|all) ;;
  *) echo "Unsupported --stage: $STAGE" >&2; exit 2 ;;
esac

remote_repo_root >/dev/null
remote_export_resources
remote_validate_workers "$WORKERS"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
[[ "$PYTHON_BIN" == "/usr/bin/python3" ]] || {
  echo "Production one-click requires /usr/bin/python3; received $PYTHON_BIN" >&2
  exit 2
}
[[ -n "$EXPECTED_BRANCH" ]] || EXPECTED_BRANCH="$(git branch --show-current)"
[[ -n "$EXPECTED_COMMIT" ]] || EXPECTED_COMMIT="$(git rev-parse HEAD)"
[[ -n "$PILOT_NAME" ]] || PILOT_NAME="storage-pilot-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"

echo "EXPECTED_BRANCH=$EXPECTED_BRANCH"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT"
echo "FORMAL_RESULT_ROOT=$PWD/$OUTPUT_ROOT/$RUN_NAME"

if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ "$SKIP_BOOTSTRAP" -eq 0 ]]; then
    BOOTSTRAP_DRY=(bash scripts/bootstrap_remote.sh --workers "$WORKERS" --run-name "${RUN_NAME}-bootstrap")
    [[ -n "$WHEELHOUSE" ]] && BOOTSTRAP_DRY+=(--wheelhouse "$WHEELHOUSE")
    printf 'DRY_RUN_BOOTSTRAP='
    printf '%q ' "${BOOTSTRAP_DRY[@]}"
    printf '\n'
  fi
  if [[ -z "$STORAGE_REPORT" ]]; then
    STORAGE_REPORT="$PWD/artifacts/storage_pilot/$PILOT_NAME/storage_projection.json"
    echo "DRY_RUN_STORAGE_PILOT=bash scripts/run_storage_pilot.sh --workers $WORKERS --run-name $PILOT_NAME"
  fi
  RL_RUN_CONTEXT=remote PYTHON_BIN="$PYTHON_BIN" bash scripts/run_cross_extension_remote.sh \
    --stage "$STAGE" --allow-full-run --workers "$WORKERS" --run-name "$RUN_NAME" \
    --output-root "$OUTPUT_ROOT" --storage-report "$STORAGE_REPORT" \
    --expected-branch "$EXPECTED_BRANCH" --expected-commit "$EXPECTED_COMMIT" --dry-run
  echo "DRY_RUN_ONLY=true"
  exit 0
fi

if [[ "$SKIP_BOOTSTRAP" -eq 0 ]]; then
  BOOTSTRAP_ARGS=(--workers "$WORKERS" --run-name "${RUN_NAME}-bootstrap")
  [[ -n "$WHEELHOUSE" ]] && BOOTSTRAP_ARGS+=(--wheelhouse "$WHEELHOUSE")
  bash scripts/bootstrap_remote.sh "${BOOTSTRAP_ARGS[@]}"
fi
if [[ -z "$STORAGE_REPORT" ]]; then
  bash scripts/run_storage_pilot.sh --workers "$WORKERS" --run-name "$PILOT_NAME"
  STORAGE_REPORT="$PWD/artifacts/storage_pilot/$PILOT_NAME/storage_projection.json"
fi
[[ -f "$STORAGE_REPORT" ]] || {
  echo "Storage report not found: $STORAGE_REPORT" >&2
  exit 2
}

FORMAL_ARGS=(
  --stage "$STAGE" --allow-full-run --workers "$WORKERS" --run-name "$RUN_NAME"
  --output-root "$OUTPUT_ROOT" --storage-report "$STORAGE_REPORT"
  --expected-branch "$EXPECTED_BRANCH" --expected-commit "$EXPECTED_COMMIT"
)
[[ "$RESUME" -eq 1 ]] && FORMAL_ARGS+=(--resume)
[[ "$RETRY_INVALID" -eq 1 ]] && FORMAL_ARGS+=(--retry-invalid)
echo "FORMAL_COMMAND=RL_RUN_CONTEXT=remote PYTHON_BIN=$PYTHON_BIN bash scripts/run_cross_extension_remote.sh ${FORMAL_ARGS[*]}"
RL_RUN_CONTEXT=remote PYTHON_BIN="$PYTHON_BIN" bash scripts/run_cross_extension_remote.sh "${FORMAL_ARGS[@]}"
echo "ONE_CLICK_COMPLETE=true"
