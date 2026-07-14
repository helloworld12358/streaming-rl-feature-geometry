#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" || -z "$RUN_DIR" ]]; then
  echo "Usage: $0 --config CONFIG --run-dir RESULT_DIR" >&2
  exit 2
fi
if [[ "${RL_RUN_CONTEXT:-}" != "remote" ]]; then
  echo "RL_RUN_CONTEXT=remote is required for full-result aggregation." >&2
  exit 2
fi

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - "$CONFIG" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
missing = []
failed = []
if "environments" in config:
    axes = (
        (environment, condition, seed)
        for environment, spec in config["environments"].items()
        for condition in spec["conditions"]
        for seed in config["seeds"]
    )
else:
    axes = ((None, condition, seed) for condition in config["conditions"] for seed in config["seeds"])
count = 0
for environment, condition, seed in axes:
    relative = Path(environment) / condition if environment else Path(condition)
    run_dir = root / "runs" / relative / f"seed_{seed:03d}"
    label = f"{relative}/seed_{seed:03d}"
    count += 1
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        missing.append(label)
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("exit_status") != "ok":
        failed.append(f"{label}:{manifest.get('exit_status')}")
if missing:
    print("Missing seeds/runs:", *missing, sep="\n  ", file=sys.stderr)
if failed:
    print("Failed seeds/runs:", *failed, sep="\n  ", file=sys.stderr)
if missing or failed:
    raise SystemExit(1)
print(f"Completeness check passed: {count} runs")
PY

if "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    is_cross = "environments" in json.load(handle)
raise SystemExit(0 if is_cross else 1)
PY
then
  "$PYTHON_BIN" scripts/run_cross_experiment.py \
    --config "$CONFIG" \
    --allow-full-run \
    --aggregate-only \
    --output-dir "$(dirname "$RUN_DIR")" \
    --run-name "$(basename "$RUN_DIR")"
else
  "$PYTHON_BIN" scripts/run_experiment.py \
    --config "$CONFIG" \
    --allow-full-run \
    --aggregate-only \
    --output-dir "$RUN_DIR"
  "$PYTHON_BIN" scripts/validate_results.py "$RUN_DIR" --config "$CONFIG"
fi
tar -czf "${RUN_DIR%/}.tar.gz" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
echo "Validated aggregate and package: ${RUN_DIR%/}.tar.gz"
