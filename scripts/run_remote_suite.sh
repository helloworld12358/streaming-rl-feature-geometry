#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/remote_full_suite.json"
WORKERS="16"
RUN_NAME=""
OUTPUT_ROOT="results/remote"
STAGES=""
SKIP_STAGES=()
ALLOW_FULL=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: RL_RUN_CONTEXT=remote bash scripts/run_remote_suite.sh --allow-full-run \
  --workers N --run-name NAME [--config PATH] [--output-root DIR] \
  [--stages core,cross_environment,production_extension,nonstationary] \
  [--skip-stage ID] [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --skip-stage) SKIP_STAGES+=("$2"); shift 2 ;;
    --allow-full-run) ALLOW_FULL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/remote_common.sh"
remote_require_full_gates "$ALLOW_FULL"
remote_require_positive_integer "--workers" "$WORKERS"
[[ -f "$CONFIG" ]] || { echo "Suite config not found: $CONFIG" >&2; exit 2; }
if [[ -z "$RUN_NAME" || ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-name is required and must be filesystem-safe." >&2
  exit 2
fi
for skipped in "${SKIP_STAGES[@]}"; do
  if [[ -z "$STAGES" ]]; then
    STAGES="core,cross_environment,production_extension,nonstationary"
  fi
  STAGES="$(awk -v stages="$STAGES" -v skip="$skipped" 'BEGIN {n=split(stages,a,","); out=""; for(i=1;i<=n;i++) if(a[i]!=skip && a[i]!="") out=out (out?",":"") a[i]; print out}')"
done
[[ -n "$STAGES" || ${#SKIP_STAGES[@]} -eq 0 ]] || { echo "All stages were skipped." >&2; exit 2; }

remote_repo_root
PYTHON_BIN="${PYTHON_BIN:-python3}"
remote_export_resources
remote_validate_workers "$WORKERS"
PLAN_ARGS=(--config "$CONFIG" --tsv)
[[ -z "$STAGES" ]] || PLAN_ARGS+=(--stages "$STAGES")
mapfile -t PLAN < <("$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment plan-suite "${PLAN_ARGS[@]}")
SUITE_DIR="$OUTPUT_ROOT/$RUN_NAME"
echo "SUITE_DIR=$SUITE_DIR BATCHES=${#PLAN[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  for line in "${PLAN[@]}"; do
    line="${line%$'\r'}"
    IFS=$'\t' read -r BATCH_ID STAGE KIND RUN_CONFIG SUBRUN EXTENSION_STAGE <<<"$line"
    echo "DRY_RUN_BATCH=$BATCH_ID STAGE=$STAGE KIND=$KIND"
    if [[ "$KIND" == "cross_extension" ]]; then
      RL_RUN_CONTEXT=remote PYTHON_BIN="$PYTHON_BIN" bash scripts/run_cross_extension_remote.sh \
        --allow-full-run --stage "$EXTENSION_STAGE" --run-name "$SUBRUN" \
        --output-root "$SUITE_DIR/$STAGE" --workers "$WORKERS" --dry-run
    else
      RL_RUN_CONTEXT=remote PYTHON_BIN="$PYTHON_BIN" bash scripts/run_remote_full.sh \
        --allow-full-run --config "$RUN_CONFIG" --workers "$WORKERS" \
        --run-name "$SUBRUN" --output-root "$SUITE_DIR/$STAGE" --dry-run
    fi
  done
  echo "DRY_RUN_ONLY=true"
  exit 0
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing formal suite execution from a dirty worktree." >&2
  exit 2
fi
INIT_ARGS=(--suite-dir "$SUITE_DIR" --run-name "$RUN_NAME" --config "$CONFIG" --workers "$WORKERS")
[[ -z "$STAGES" ]] || INIT_ARGS+=(--stages "$STAGES")
"$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment init-suite "${INIT_ARGS[@]}"
mkdir -p "$SUITE_DIR/logs"

for line in "${PLAN[@]}"; do
  line="${line%$'\r'}"
  IFS=$'\t' read -r BATCH_ID STAGE KIND RUN_CONFIG SUBRUN EXTENSION_STAGE <<<"$line"
  LOG_FILE="$SUITE_DIR/logs/$BATCH_ID.log"
  RESULT_DIR="$SUITE_DIR/$STAGE/$SUBRUN"
  set +e
  if [[ "$KIND" == "cross_extension" ]]; then
    RL_RUN_CONTEXT=remote PYTHON_BIN="$PYTHON_BIN" bash scripts/run_cross_extension_remote.sh \
      --allow-full-run --stage "$EXTENSION_STAGE" --run-name "$SUBRUN" \
      --output-root "$SUITE_DIR/$STAGE" --workers "$WORKERS" 2>&1 | tee "$LOG_FILE"
  else
    RL_RUN_CONTEXT=remote PYTHON_BIN="$PYTHON_BIN" bash scripts/run_remote_full.sh \
      --allow-full-run --config "$RUN_CONFIG" --workers "$WORKERS" \
      --run-name "$SUBRUN" --output-root "$SUITE_DIR/$STAGE" 2>&1 | tee "$LOG_FILE"
  fi
  STATUS=${PIPESTATUS[0]}
  set -e
  if [[ "$STATUS" -ne 0 ]]; then
    "$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment record-batch \
      --suite-dir "$SUITE_DIR" --batch-id "$BATCH_ID" --status failed \
      --result-dir "$RESULT_DIR" --error "exit-code-$STATUS"
    "$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment finalize-suite \
      --suite-dir "$SUITE_DIR" --status failed
    echo "Suite stopped after failed batch $BATCH_ID; evidence preserved." >&2
    exit "$STATUS"
  fi
  "$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment record-batch \
    --suite-dir "$SUITE_DIR" --batch-id "$BATCH_ID" --status ok --result-dir "$RESULT_DIR"
done

set +e
bash scripts/aggregate_remote.sh --suite-dir "$SUITE_DIR"
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  "$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment finalize-suite \
    --suite-dir "$SUITE_DIR" --status failed
  exit "$STATUS"
fi
"$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment finalize-suite \
  --suite-dir "$SUITE_DIR" --status ok
set +e
bash scripts/package_remote_results.sh --suite-dir "$SUITE_DIR"
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  "$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment finalize-suite \
    --suite-dir "$SUITE_DIR" --status failed
  exit "$STATUS"
fi
echo "REMOTE_SUITE_COMPLETE=$SUITE_DIR"
