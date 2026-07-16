#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
WORKERS="4"
RUN_NAME=""
OUTPUT_ROOT=""
SEED_START=""
SEED_END=""
ALLOW_FULL=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: RL_RUN_CONTEXT=remote bash scripts/run_remote_full.sh --allow-full-run \
  --config PATH --workers N --run-name NAME [--output-root DIR] \
  [--seed-start N] [--seed-end N] [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --seed-start) SEED_START="$2"; shift 2 ;;
    --seed-end) SEED_END="$2"; shift 2 ;;
    --allow-full-run) ALLOW_FULL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/remote_common.sh"
remote_require_full_gates "$ALLOW_FULL"
remote_require_positive_integer "--workers" "$WORKERS"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "--config must point to an existing executable JSON config." >&2
  exit 2
fi
if [[ -z "$RUN_NAME" || ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-name is required and may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
[[ -z "$SEED_START" ]] || [[ "$SEED_START" =~ ^-?[0-9]+$ ]] || { echo "--seed-start must be an integer" >&2; exit 2; }
[[ -z "$SEED_END" ]] || [[ "$SEED_END" =~ ^-?[0-9]+$ ]] || { echo "--seed-end must be an integer" >&2; exit 2; }

remote_repo_root
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v git >/dev/null
command -v "$PYTHON_BIN" >/dev/null
remote_export_resources
remote_validate_workers "$WORKERS"

PLAN_ARGS=(--config "$CONFIG" --lines)
[[ -z "$SEED_START" ]] || PLAN_ARGS+=(--seed-start "$SEED_START")
[[ -z "$SEED_END" ]] || PLAN_ARGS+=(--seed-end "$SEED_END")
readarray -t META < <("$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment plan-config "${PLAN_ARGS[@]}")
for index in "${!META[@]}"; do META[$index]="${META[$index]%$'\r'}"; done
KIND="${META[0]}"
PROFILE="${META[1]}"
SEED_CSV="${META[2]}"
EXPECTED_RUNS="${META[3]}"
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="${META[4]}"
RESULT_DIR="$OUTPUT_ROOT/$RUN_NAME"

if [[ "$KIND" == "cross" ]]; then
  RUNNER="scripts/run_cross_experiment.py"
  COMMAND=("$PYTHON_BIN" "$RUNNER" --config "$CONFIG" --allow-full-run --workers "$WORKERS" --output-dir "$OUTPUT_ROOT" --run-name "$RUN_NAME" --seed "$SEED_CSV")
else
  RUNNER="scripts/run_experiment.py"
  COMMAND=("$PYTHON_BIN" "$RUNNER" --config "$CONFIG" --allow-full-run --workers "$WORKERS" --output-dir "$OUTPUT_ROOT" --run-name "$RUN_NAME" --seeds "$SEED_CSV")
fi
printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"
echo "PROFILE=$PROFILE KIND=$KIND EXPECTED_RUNS=$EXPECTED_RUNS"
echo "RESULT_DIR=$RESULT_DIR"
echo "COMMAND=$COMMAND_TEXT"

if [[ "$DRY_RUN" -eq 1 ]]; then
  DRY_PLAN_ARGS=(--config "$CONFIG")
  [[ -z "$SEED_START" ]] || DRY_PLAN_ARGS+=(--seed-start "$SEED_START")
  [[ -z "$SEED_END" ]] || DRY_PLAN_ARGS+=(--seed-end "$SEED_END")
  "$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment plan-config "${DRY_PLAN_ARGS[@]}"
  echo "DRY_RUN_ONLY=true"
  exit 0
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing formal execution from a dirty worktree. Commit or reconcile changes first." >&2
  exit 2
fi
if [[ -e "$RESULT_DIR" ]]; then
  echo "Refusing to overwrite existing result directory: $RESULT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
LOG_FILE="$OUTPUT_ROOT/${RUN_NAME}.launcher.log"
START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
remote_print_inventory | tee "$LOG_FILE"
echo "COMMAND=$COMMAND_TEXT" | tee -a "$LOG_FILE"
set +e
"${COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
END_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STATUS_TEXT="ok"
[[ "$STATUS" -eq 0 ]] || STATUS_TEXT="failed"
if [[ ! -d "$RESULT_DIR" ]]; then
  RESULT_DIR="$OUTPUT_ROOT/failed_launches/${RUN_NAME}-${START_TIME//[:]/}"
  mkdir -p "$RESULT_DIR"
fi
cp "$LOG_FILE" "$RESULT_DIR/remote_launcher.log"
"$PYTHON_BIN" -m streaming_rl_feature_geometry.remote_deployment write-launch-manifest \
  --result-dir "$RESULT_DIR" --config "$CONFIG" --command "$COMMAND_TEXT" \
  --workers "$WORKERS" --start-time "$START_TIME" --end-time "$END_TIME" --status "$STATUS_TEXT"
if [[ "$STATUS" -ne 0 ]]; then
  echo "Full batch failed with exit code $STATUS; evidence preserved at $RESULT_DIR" >&2
  exit "$STATUS"
fi
bash scripts/aggregate_remote.sh --config "$RESULT_DIR/config.json" --run-dir "$RESULT_DIR"
echo "REMOTE_FULL_COMPLETE=$RESULT_DIR"
