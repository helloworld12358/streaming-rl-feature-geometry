#!/usr/bin/env bash
set -euo pipefail

remote_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
}

remote_require_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; received: $value" >&2
    return 2
  fi
}

remote_available_cpus() {
  local online available quota period cgroup_cpus
  online="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
  available="$online"
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [[ "$quota" != "max" && "$quota" =~ ^[0-9]+$ && "$period" =~ ^[1-9][0-9]*$ ]]; then
      cgroup_cpus=$(( quota / period ))
      (( cgroup_cpus < 1 )) && cgroup_cpus=1
      (( cgroup_cpus < available )) && available="$cgroup_cpus"
    fi
  fi
  echo "$available"
}

remote_validate_workers() {
  local workers="$1"
  local available safe_max
  remote_require_positive_integer "--workers" "$workers"
  available="$(remote_available_cpus)"
  safe_max=$(( available > 1 ? available - 1 : 1 ))
  if (( workers > safe_max )); then
    echo "Requested $workers workers, but the safe maximum is $safe_max for $available available CPUs." >&2
    return 2
  fi
  echo "CPU_AVAILABLE=$available CPU_SAFE_WORKERS=$safe_max CPU_REQUESTED_WORKERS=$workers"
}

remote_export_resources() {
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export RL_REMOTE_CPU_MODEL="${RL_REMOTE_CPU_MODEL:-$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    export RL_REMOTE_GPU_DETECTION="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' - || true)"
    [[ -n "$RL_REMOTE_GPU_DETECTION" ]] || export RL_REMOTE_GPU_DETECTION="nvidia-smi-present-no-readable-device"
  else
    export RL_REMOTE_GPU_DETECTION="nvidia-smi-unavailable"
  fi
}

remote_require_full_gates() {
  local allow_full="$1"
  if [[ "${RL_RUN_CONTEXT:-}" != "remote" ]]; then
    echo "Refusing full run: RL_RUN_CONTEXT=remote is required." >&2
    return 2
  fi
  if [[ "$allow_full" -ne 1 ]]; then
    echo "Refusing full run: --allow-full-run is required." >&2
    return 2
  fi
}

remote_print_inventory() {
  echo "HOSTNAME=$(hostname)"
  echo "GIT_BRANCH=$(git branch --show-current)"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "GIT_DIRTY=$([[ -n "$(git status --porcelain)" ]] && echo true || echo false)"
  echo "PYTHON_BIN=${PYTHON_BIN:-python3}"
  "${PYTHON_BIN:-python3}" --version
  echo "CPU_MODEL=${RL_REMOTE_CPU_MODEL:-unknown}"
  echo "GPU_DETECTION=${RL_REMOTE_GPU_DETECTION:-not-recorded}"
  echo "GPU_BACKEND_USED=none"
  df -h .
}
