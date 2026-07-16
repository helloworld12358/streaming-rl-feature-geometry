#!/usr/bin/env bash
set -euo pipefail

STAGE=""
RUN_NAME=""
WORKERS="16"
RESUME=0
RETRY_FAILED=0
DRY_RUN=0
SELECTED=""
OUTPUT_ROOT="results/cross_extension"
ALLOW_FULL=0
ENVIRONMENTS=()
CONDITIONS=()

usage() {
  cat <<'EOF'
Usage: RL_RUN_CONTEXT=remote bash scripts/run_cross_extension_remote.sh \
  --stage fixed-full|lr-tune|lr-eval|norm-scaled|all \
  --allow-full-run --run-name NAME [--workers N] [--output-root DIR] \
  [--resume] [--retry-failed] [--dry-run] \
  [--environment ID] [--condition NAME] [--selected-learning-rates PATH]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --retry-failed) RETRY_FAILED=1; RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --environment) ENVIRONMENTS+=("$2"); shift 2 ;;
    --condition) CONDITIONS+=("$2"); shift 2 ;;
    --selected-learning-rates) SELECTED="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
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
  echo "--run-name is required and may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be a positive integer" >&2
  exit 2
fi
if [[ "${RL_RUN_CONTEXT:-}" != "remote" ]]; then
  echo "RL_RUN_CONTEXT=remote is required for production full execution and dry-run." >&2
  exit 2
fi
if [[ "$ALLOW_FULL" -ne 1 ]]; then
  echo "--allow-full-run is required for production full execution and dry-run." >&2
  exit 2
fi

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMMIT="$(git rev-parse HEAD)"
echo "START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GIT_COMMIT=$COMMIT"
echo "GIT_BRANCH=$(git branch --show-current)"

if [[ -n "$(git status --porcelain)" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "WARNING: worktree is dirty; dry-run continues without executing experiments." >&2
  else
    echo "Refusing production run from a dirty worktree. Preserve or commit changes first." >&2
    exit 2
  fi
fi
command -v git >/dev/null
command -v "$PYTHON_BIN" >/dev/null
"$PYTHON_BIN" -c 'import numpy, pandas, matplotlib; import streaming_rl_feature_geometry'

AVAILABLE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
echo "DISK_AVAILABLE_KB=$AVAILABLE_KB"
if [[ "$DRY_RUN" -ne 1 && "$AVAILABLE_KB" -lt 10485760 ]]; then
  echo "At least 10 GiB free disk space is required." >&2
  exit 2
fi

ONLINE_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
AVAILABLE_CPUS="$ONLINE_CPUS"
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r CGROUP_QUOTA CGROUP_PERIOD < /sys/fs/cgroup/cpu.max
  if [[ "$CGROUP_QUOTA" != "max" && "$CGROUP_QUOTA" =~ ^[0-9]+$ && "$CGROUP_PERIOD" =~ ^[1-9][0-9]*$ ]]; then
    CGROUP_CPUS=$(( CGROUP_QUOTA / CGROUP_PERIOD ))
    if (( CGROUP_CPUS < 1 )); then CGROUP_CPUS=1; fi
    if (( CGROUP_CPUS < AVAILABLE_CPUS )); then AVAILABLE_CPUS="$CGROUP_CPUS"; fi
  fi
fi
SAFE_MAX=$(( AVAILABLE_CPUS > 1 ? AVAILABLE_CPUS - 1 : 1 ))
if (( WORKERS > SAFE_MAX )); then
  echo "Requested $WORKERS workers; safe maximum is $SAFE_MAX for this allocation." >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

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

run_stage() {
  local current_stage="$1"
  local config
  config="$(config_for_stage "$current_stage")"
  local stage_dir="$SUITE_DIR/$current_stage"
  "$PYTHON_BIN" - "$config" <<'PY'
import sys
from streaming_rl_feature_geometry.cross_experiment import load_cross_config, expected_cross_run_count
config = load_cross_config(sys.argv[1])
print(f"PROFILE={config['profile']} EXPECTED_RUNS={expected_cross_run_count(config)}")
PY
  local args=(
    scripts/run_cross_experiment.py --config "$config" --allow-full-run
    --workers "$WORKERS" --output-dir "$SUITE_DIR" --run-name "$current_stage"
  )
  [[ "$RESUME" -eq 1 ]] && args+=(--resume)
  [[ "$RETRY_FAILED" -eq 1 ]] && args+=(--retry-failed)
  for environment in "${ENVIRONMENTS[@]}"; do args+=(--environment "$environment"); done
  for condition in "${CONDITIONS[@]}"; do args+=(--condition "$condition"); done
  if [[ "$current_stage" == "lr-eval" ]]; then
    if [[ "$DRY_RUN" -ne 1 && ! -f "$SELECTED" ]]; then
      echo "Selected-alpha file not found: $SELECTED. Run lr-tune first." >&2
      exit 2
    fi
    args+=(--selected-learning-rates "$SELECTED")
  fi
  echo "STAGE=$current_stage"
  echo "COMMAND=$PYTHON_BIN ${args[*]}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$current_stage" == "lr-eval" && ! -f "$SELECTED" ]]; then
      echo "DRY_RUN_EXPECTED_SELECTED_ALPHA=$SELECTED"
      "$PYTHON_BIN" - "$config" <<'PY'
import sys
from streaming_rl_feature_geometry.cross_experiment import expected_cross_run_count, load_cross_config
print({"expected_runs": expected_cross_run_count(load_cross_config(sys.argv[1])), "selected_alpha_required": True})
PY
    else
      RL_RUN_CONTEXT=remote "$PYTHON_BIN" "${args[@]}" --dry-run
    fi
    return
  fi
  RL_RUN_CONTEXT=remote "$PYTHON_BIN" "${args[@]}" --dry-run
  mkdir -p "$SUITE_DIR/logs"
  local log="$SUITE_DIR/logs/${current_stage}.log"
  set +e
  "$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  [[ "$status" -eq 0 ]] || return "$status"
  cp "$log" "$stage_dir/remote_launcher.log"
  local aggregate_args=(--config "$stage_dir/config.json" --run-dir "$stage_dir")
  [[ "$current_stage" == "lr-eval" ]] && aggregate_args+=(--selected-learning-rates "$SELECTED")
  bash scripts/aggregate_cross_extension_remote.sh "${aggregate_args[@]}"
  RL_RUN_CONTEXT=remote "$PYTHON_BIN" "${args[@]}" --dry-run
}

if [[ "$STAGE" == "all" ]]; then
  for current in fixed-full lr-tune lr-eval norm-scaled; do
    run_stage "$current"
  done
else
  run_stage "$STAGE"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  bash scripts/package_cross_extension_remote.sh \
    --suite-dir "$SUITE_DIR" --run-name "$RUN_NAME" --output-dir "$SUITE_DIR/packages"
fi
echo "END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
