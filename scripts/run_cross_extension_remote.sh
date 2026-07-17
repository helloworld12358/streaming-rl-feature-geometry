#!/usr/bin/env bash
set -euo pipefail

STAGE=""
RUN_NAME=""
WORKERS="16"
RESUME=0
RETRY_INVALID=0
SELECTED=""
OUTPUT_ROOT="results/cross_extension"
STORAGE_REPORT=""
EXPECTED_BRANCH="codex/fix-streaming-rl-production-root-causes"
EXPECTED_COMMIT=""
ALLOW_FULL=0

usage() {
  cat <<'EOF'
Usage: RL_RUN_CONTEXT=remote bash scripts/run_cross_extension_remote.sh \
  --stage fixed-full|lr-tune|lr-eval|norm-scaled|all \
  --allow-full-run --run-name NAME --storage-report PATH \
  --expected-commit SHA [--workers N] [--output-root DIR] \
  [--resume] [--retry-invalid] [--selected-learning-rates PATH]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --retry-invalid|--retry-failed) RETRY_INVALID=1; RESUME=1; shift ;;
    --selected-learning-rates) SELECTED="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --storage-report) STORAGE_REPORT="$2"; shift 2 ;;
    --expected-branch) EXPECTED_BRANCH="$2"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="$2"; shift 2 ;;
    --allow-full-run) ALLOW_FULL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$STAGE" in
  fixed-full|lr-tune|lr-eval|norm-scaled|all) ;;
  *) echo "--stage must be fixed-full, lr-tune, lr-eval, norm-scaled, or all" >&2; exit 2 ;;
esac
if [[ -z "$RUN_NAME" || ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-name is required and must be filesystem-safe" >&2
  exit 2
fi
if [[ -z "$STORAGE_REPORT" || -z "$EXPECTED_COMMIT" ]]; then
  echo "--storage-report and --expected-commit are required" >&2
  exit 2
fi

source "$(dirname "$0")/remote_common.sh"
remote_repo_root >/dev/null
remote_prepare_repo_paths
remote_export_resources
remote_require_full_gates "$ALLOW_FULL"
remote_validate_workers "$WORKERS"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
[[ "$PYTHON_BIN" == "/usr/bin/python3" ]] || {
  echo "Formal execution requires /usr/bin/python3; received $PYTHON_BIN" >&2
  exit 2
}
remote_print_inventory

STORAGE_REPORT="$(realpath -m "$STORAGE_REPORT")"
case "$STORAGE_REPORT/" in
  "$REPO_ROOT"/*/) ;;
  *) echo "Storage report escapes repository: $STORAGE_REPORT" >&2; exit 2 ;;
esac
[[ -f "$STORAGE_REPORT" ]] || { echo "Storage report not found: $STORAGE_REPORT" >&2; exit 2; }
PROJECTED_PEAK_BYTES="$($PYTHON_BIN - "$STORAGE_REPORT" "$EXPECTED_COMMIT" <<'PY'
import sys
from streaming_rl_feature_geometry.storage_budget import validated_production_peak_bytes
print(validated_production_peak_bytes(sys.argv[1], sys.argv[2]))
PY
)"

OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
case "$OUTPUT_ROOT/" in
  "$REPO_ROOT"/*/) ;;
  *) echo "Output root escapes repository: $OUTPUT_ROOT" >&2; exit 2 ;;
esac
LOG_ROOT="$LOGS_ROOT/cross_extension/$RUN_NAME"
ARTIFACT_ROOT="$ARTIFACTS_ROOT/cross_extension/$RUN_NAME"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$ARTIFACT_ROOT"

"$PYTHON_BIN" -m streaming_rl_feature_geometry.production_runtime \
  --workers "$WORKERS" \
  --output-root "$OUTPUT_ROOT" \
  --logs-root "$LOG_ROOT" \
  --artifacts-root "$ARTIFACT_ROOT" \
  --runtime-root "$REPO_ROOT/.runtime" \
  --projected-peak-bytes "$PROJECTED_PEAK_BYTES" \
  --expected-branch "$EXPECTED_BRANCH" \
  --expected-commit "$EXPECTED_COMMIT" \
  --report "$LOG_ROOT/production_preflight.json"

SUITE_DIR="$OUTPUT_ROOT/$RUN_NAME"
SELECTED_DEFAULT="$SUITE_DIR/lr-tune/selected_learning_rates.csv"
[[ -z "$SELECTED" ]] && SELECTED="$SELECTED_DEFAULT"

config_for_stage() {
  case "$1" in
    fixed-full) echo "configs/cross_extension_fixed_full.json" ;;
    lr-tune) echo "configs/cross_extension_lr_tune_full.json" ;;
    lr-eval) echo "configs/cross_extension_lr_eval_full.json" ;;
    norm-scaled) echo "configs/cross_extension_norm_scaled_full.json" ;;
  esac
}

write_stage_status() {
  local status_file="$1"
  local stage="$2"
  local config="$3"
  local command="$4"
  local started="$5"
  local ended="$6"
  local status="$7"
  "$PYTHON_BIN" - "$status_file" "$stage" "$config" "$command" "$started" "$ended" "$status" "$EXPECTED_COMMIT" <<'PY'
import json, pathlib, sys
path, stage, config, command, started, ended, status, commit = sys.argv[1:]
value = {
    "stage": stage,
    "command": command,
    "config": config,
    "commit": commit,
    "start_time": started,
    "end_time": ended,
    "exit_code": int(status),
    "resume_eligible": int(status) != 0,
}
target = pathlib.Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_stage() {
  local current_stage="$1"
  local config stage_dir log_file status_file started ended status command
  config="$(config_for_stage "$current_stage")"
  stage_dir="$SUITE_DIR/$current_stage"
  log_file="$LOG_ROOT/${current_stage}.log"
  status_file="$LOG_ROOT/${current_stage}.status.json"
  local args=(
    scripts/run_cross_experiment.py --config "$config" --allow-full-run
    --workers "$WORKERS" --output-dir "$SUITE_DIR" --run-name "$current_stage"
  )
  [[ "$RESUME" -eq 1 ]] && args+=(--resume)
  [[ "$RETRY_INVALID" -eq 1 ]] && args+=(--retry-invalid)
  if [[ "$current_stage" == "lr-eval" ]]; then
    [[ -f "$SELECTED" ]] || { echo "Selected-alpha file not found: $SELECTED" >&2; return 2; }
    args+=(--selected-learning-rates "$SELECTED")
  fi
  command="$PYTHON_BIN ${args[*]}"
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "STAGE=$current_stage"
  echo "COMMAND=$command"
  set +e
  RL_RUN_CONTEXT=remote "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$log_file"
  status=${PIPESTATUS[0]}
  set -e
  ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_stage_status "$status_file" "$current_stage" "$config" "$command" "$started" "$ended" "$status"
  if [[ "$status" -ne 0 ]]; then
    echo "STAGE_STATUS=$status status_file=$status_file" >&2
    return "$status"
  fi
  local aggregate_args=(--config "$stage_dir/config.json" --run-dir "$stage_dir")
  [[ "$current_stage" == "lr-eval" ]] && aggregate_args+=(--selected-learning-rates "$SELECTED")
  "$PYTHON_BIN" -m streaming_rl_feature_geometry.cross_production aggregate "${aggregate_args[@]}" 2>&1 | tee -a "$log_file"
  cp "$log_file" "$stage_dir/remote_launcher.log"
}

if [[ "$STAGE" == "all" ]]; then
  for current in fixed-full lr-tune lr-eval norm-scaled; do
    run_stage "$current"
  done
else
  run_stage "$STAGE"
fi

"$PYTHON_BIN" -m streaming_rl_feature_geometry.cross_production package \
  --suite-dir "$SUITE_DIR" --run-name "$RUN_NAME" --output-dir "$ARTIFACT_ROOT"
echo "FORMAL_RUN_COMPLETE=true"
echo "SUITE_DIR=$SUITE_DIR"
echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"
