#!/usr/bin/env bash
set -euo pipefail

echo "remote_one_click.sh is a foreground compatibility wrapper." >&2
echo "Run scripts/run_storage_pilot.sh first and provide --storage-report and --expected-commit." >&2
bash "$(dirname "$0")/run_cross_extension_remote.sh" --stage all "$@"
