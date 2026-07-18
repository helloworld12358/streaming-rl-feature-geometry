#!/usr/bin/env bash
set -euo pipefail

WORKERS="4"
RUN_NAME="storage-pilot-$(date -u +%Y%m%dT%H%M%SZ)"
DRY_RUN=0

usage() {
  echo "Usage: $0 [--workers N] [--run-name NAME] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/remote_common.sh"
remote_repo_root >/dev/null
remote_export_resources
remote_validate_workers "$WORKERS"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
[[ "$PYTHON_BIN" == "/usr/bin/python3" ]] || {
  echo "Storage pilot requires /usr/bin/python3; received $PYTHON_BIN" >&2
  exit 2
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY_RUN_COMMAND=$PYTHON_BIN scripts/run_cross_experiment.py --config configs/cross_extension_storage_pilot.json --workers $WORKERS --output-dir results/storage_pilot --run-name $RUN_NAME"
  echo "DRY_RUN_COMMAND=$PYTHON_BIN -m streaming_rl_feature_geometry.cross_production package --suite-dir results/storage_pilot/$RUN_NAME --run-name $RUN_NAME --output-dir artifacts/storage_pilot/$RUN_NAME/packages"
  echo "DRY_RUN_COMMAND=$PYTHON_BIN -m streaming_rl_feature_geometry.storage_budget --pilot-dir results/storage_pilot/$RUN_NAME --package-dir artifacts/storage_pilot/$RUN_NAME/packages --formal-config configs/cross_extension_fixed_full.json --formal-config configs/cross_extension_lr_tune_full.json --formal-config configs/cross_extension_lr_eval_full.json --formal-config configs/cross_extension_norm_scaled_full.json --output artifacts/storage_pilot/$RUN_NAME/storage_projection.json --inventory-csv artifacts/storage_pilot/$RUN_NAME/storage_pilot_inventory.csv"
  echo "DRY_RUN_ONLY=true"
  exit 0
fi
remote_prepare_repo_paths

PILOT_PARENT="$RESULTS_ROOT/storage_pilot"
PILOT_DIR="$PILOT_PARENT/$RUN_NAME"
LOG_DIR="$LOGS_ROOT/storage_pilot/$RUN_NAME"
PACKAGE_DIR="$ARTIFACTS_ROOT/storage_pilot/$RUN_NAME/packages"
REPORT_DIR="$ARTIFACTS_ROOT/storage_pilot/$RUN_NAME"
mkdir -p "$PILOT_PARENT" "$LOG_DIR" "$PACKAGE_DIR" "$REPORT_DIR"
LOG_FILE="$LOG_DIR/storage_pilot.log"
STATUS_FILE="$LOG_DIR/storage_pilot.status"

set +e
"$PYTHON_BIN" scripts/run_cross_experiment.py \
  --config configs/cross_extension_storage_pilot.json \
  --workers "$WORKERS" \
  --output-dir "$PILOT_PARENT" \
  --run-name "$RUN_NAME" 2>&1 | tee -a "$LOG_FILE"
RUN_STATUS=${PIPESTATUS[0]}
set -e
echo "RUN_STATUS=$RUN_STATUS"
printf '%s\n' "$RUN_STATUS" > "$STATUS_FILE"
if [[ "$RUN_STATUS" -ne 0 ]]; then
  echo "Storage pilot failed; see $LOG_FILE and $STATUS_FILE" >&2
  false
fi

"$PYTHON_BIN" -m streaming_rl_feature_geometry.cross_production package \
  --suite-dir "$PILOT_DIR" --run-name "$RUN_NAME" --output-dir "$PACKAGE_DIR"
"$PYTHON_BIN" -m streaming_rl_feature_geometry.storage_budget \
  --pilot-dir "$PILOT_DIR" \
  --package-dir "$PACKAGE_DIR" \
  --formal-config configs/cross_extension_fixed_full.json \
  --formal-config configs/cross_extension_lr_tune_full.json \
  --formal-config configs/cross_extension_lr_eval_full.json \
  --formal-config configs/cross_extension_norm_scaled_full.json \
  --output "$REPORT_DIR/storage_projection.json" \
  --inventory-csv "$REPORT_DIR/storage_pilot_inventory.csv"
echo "STORAGE_PILOT_DIR=$PILOT_DIR"
echo "STORAGE_REPORT=$REPORT_DIR/storage_projection.json"
echo "PACKAGE_DIR=$PACKAGE_DIR"
