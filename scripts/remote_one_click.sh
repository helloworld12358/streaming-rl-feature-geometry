#!/usr/bin/env bash
set -euo pipefail

WORKERS="16"
RUN_NAME=""
CONFIG="configs/remote_full_suite.json"
WHEELHOUSE=""
ALLOW_FULL=0
DRY_RUN=0
FOREGROUND=0
SKIP_BOOTSTRAP=0

usage() {
  cat <<'EOF'
Usage: RL_RUN_CONTEXT=remote bash scripts/remote_one_click.sh --allow-full-run \
  --workers N --run-name NAME [--config PATH] [--wheelhouse DIR] \
  [--skip-bootstrap] [--foreground] [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --allow-full-run) ALLOW_FULL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
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
remote_repo_root
if [[ "$SKIP_BOOTSTRAP" -eq 0 ]]; then
  BOOTSTRAP_ARGS=(--workers "$WORKERS" --run-name "$RUN_NAME-bootstrap")
  [[ -z "$WHEELHOUSE" ]] || BOOTSTRAP_ARGS+=(--wheelhouse "$WHEELHOUSE")
  [[ "$DRY_RUN" -eq 0 ]] || BOOTSTRAP_ARGS+=(--dry-run)
  bash scripts/bootstrap_remote.sh "${BOOTSTRAP_ARGS[@]}"
fi
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ "$DRY_RUN" -eq 0 && ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at $PYTHON_BIN; run bootstrap or remove --skip-bootstrap." >&2
  exit 2
fi
SUITE_COMMAND=(bash scripts/run_remote_suite.sh --allow-full-run --config "$CONFIG" --workers "$WORKERS" --run-name "$RUN_NAME")
printf -v SUITE_COMMAND_TEXT '%q ' "${SUITE_COMMAND[@]}"
echo "FULL_COMMAND=RL_RUN_CONTEXT=remote PYTHON_BIN=$PYTHON_BIN $SUITE_COMMAND_TEXT"
echo "RESULT_DIR=$PWD/results/remote/$RUN_NAME"
echo "AGGREGATE_COMMAND=PYTHON_BIN=$PYTHON_BIN bash scripts/aggregate_remote.sh --suite-dir results/remote/$RUN_NAME"
echo "PACKAGE_COMMAND=PYTHON_BIN=$PYTHON_BIN bash scripts/package_remote_results.sh --suite-dir results/remote/$RUN_NAME"
echo "DOWNLOAD_COMMAND=scp -r <REMOTE_USER>@<REMOTE_HOST>:<REMOTE_PROJECT_DIR>/results/remote/$RUN_NAME/packages <LOCAL_DOWNLOAD_DIR>"
if [[ "$DRY_RUN" -eq 1 ]]; then
  RL_RUN_CONTEXT=remote PYTHON_BIN="$PYTHON_BIN" "${SUITE_COMMAND[@]}" --dry-run
  exit 0
fi
if [[ "$FOREGROUND" -eq 1 ]]; then
  export RL_RUN_CONTEXT=remote
  export PYTHON_BIN
  exec "${SUITE_COMMAND[@]}"
fi

BACKGROUND_DIR="results/remote/background"
mkdir -p "$BACKGROUND_DIR"
LOG_FILE="$BACKGROUND_DIR/$RUN_NAME.log"
if command -v tmux >/dev/null 2>&1; then
  SESSION="srl-${RUN_NAME:0:40}"
  printf -v BACKGROUND_COMMAND 'cd %q && RL_RUN_CONTEXT=remote PYTHON_BIN=%q %s > %q 2>&1' \
    "$PWD" "$PYTHON_BIN" "$SUITE_COMMAND_TEXT" "$PWD/$LOG_FILE"
  tmux new-session -d -s "$SESSION" "$BACKGROUND_COMMAND"
  echo "BACKGROUND=tmux SESSION=$SESSION"
  echo "RECONNECT=tmux attach -t $SESSION"
else
  nohup env RL_RUN_CONTEXT=remote PYTHON_BIN="$PYTHON_BIN" "${SUITE_COMMAND[@]}" >"$LOG_FILE" 2>&1 &
  PID=$!
  echo "$PID" > "$BACKGROUND_DIR/$RUN_NAME.pid"
  echo "BACKGROUND=nohup PID=$PID"
  echo "NOTE=nohup cannot survive destruction of the cloud container; prefer the platform job runner when available."
fi
echo "LOG=$PWD/$LOG_FILE"
echo "FOLLOW=tail -f $PWD/$LOG_FILE"
