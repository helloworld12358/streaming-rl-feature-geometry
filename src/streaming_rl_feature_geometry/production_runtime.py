"""Repository-contained production preflight and storage-budget checks."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .experiment import write_json


GIB = 1024**3
DEFAULT_SAFE_MARGIN_BYTES = 10 * GIB
DEFAULT_TARGET_PEAK_BYTES = 50 * GIB
RUNTIME_PATH_VARIABLES = (
    "TMPDIR",
    "TMP",
    "TEMP",
    "MPLCONFIGDIR",
    "XDG_CACHE_HOME",
    "PIP_CACHE_DIR",
)


def repository_root(start: str | Path | None = None) -> Path:
    """Locate the real Git worktree root and fail if the marker is ambiguous."""

    start_path = Path(start or Path.cwd()).resolve()
    probe = start_path if start_path.is_dir() else start_path.parent
    try:
        value = subprocess.check_output(
            ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot locate repository root from {probe}") from error
    root = Path(value).resolve()
    if not (root / ".git").exists() or not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"invalid repository root: {root}")
    return root


def contained_path(path: str | Path, root: str | Path, *, label: str) -> Path:
    """Resolve a path and require it to be strictly below the repository root."""

    repository = Path(root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(repository)
    except ValueError as error:
        raise ValueError(
            f"{label} escapes repository root: resolved={candidate} root={repository}"
        ) from error
    if relative == Path("."):
        raise ValueError(f"{label} must be below repository root, not the root itself: {candidate}")
    return candidate


def validate_contained_paths(
    paths: Mapping[str, str | Path], root: str | Path | None = None
) -> dict[str, Path]:
    repository = repository_root(root)
    return {
        label: contained_path(path, repository, label=label)
        for label, path in paths.items()
    }


def validate_runtime_environment(root: str | Path | None = None) -> dict[str, Path]:
    repository = repository_root(root)
    missing = [name for name in RUNTIME_PATH_VARIABLES if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "production runtime path variables are required and must point inside the repository: "
            + ", ".join(missing)
        )
    return validate_contained_paths(
        {name: os.environ[name] for name in RUNTIME_PATH_VARIABLES}, repository
    )


def _read_text(path: str) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        return candidate.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return None


def cgroup_cpu_limit() -> float | None:
    value = _read_text("/sys/fs/cgroup/cpu.max")
    if value:
        quota, period = value.split()[:2]
        if quota != "max" and float(period) > 0:
            return float(quota) / float(period)
    quota = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0 and int(period) > 0:
        return float(quota) / float(period)
    return None


def cgroup_memory_limit() -> int | None:
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        value = _read_text(path)
        if value and value != "max":
            number = int(value)
            if number > 0 and number < 2**60:
                return number
    return None


def safe_worker_limit() -> tuple[int, int, float | None]:
    online = int(os.cpu_count() or 1)
    quota = cgroup_cpu_limit()
    effective = min(online, max(1, math.floor(quota))) if quota is not None else online
    return max(1, effective - 1), effective, quota


def _git(*arguments: str, root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.STDOUT
    ).strip()


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("numpy", "pandas", "matplotlib", "streaming_rl_feature_geometry"):
        module = importlib.import_module(name)
        result[name] = str(getattr(module, "__version__", "installed"))
    return result


def _inode_free(path: Path) -> int | None:
    if not hasattr(os, "statvfs"):
        return None
    value = os.statvfs(path)
    return int(value.f_favail)


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    repository_root: str
    branch: str
    commit: str
    worktree_clean: bool
    requested_workers: int
    safe_max_workers: int
    effective_cpus: int
    cgroup_cpu_quota: float | None
    cgroup_memory_limit_bytes: int | None
    disk_free_bytes: int
    inode_free: int | None
    projected_peak_bytes: int
    required_safe_margin_bytes: int
    target_peak_bytes: int
    paths: dict[str, str]
    dependency_versions: dict[str, str]
    errors: tuple[str, ...]


def production_preflight(
    *,
    workers: int,
    output_root: str | Path,
    logs_root: str | Path,
    artifacts_root: str | Path,
    runtime_root: str | Path,
    projected_peak_bytes: int,
    expected_branch: str | None = None,
    expected_commit: str | None = None,
    safe_margin_bytes: int = DEFAULT_SAFE_MARGIN_BYTES,
    target_peak_bytes: int = DEFAULT_TARGET_PEAK_BYTES,
    root: str | Path | None = None,
) -> PreflightReport:
    repository = repository_root(root)
    paths = validate_contained_paths(
        {
            "output_root": output_root,
            "logs_root": logs_root,
            "artifacts_root": artifacts_root,
            "runtime_root": runtime_root,
        },
        repository,
    )
    branch = _git("branch", "--show-current", root=repository)
    commit = _git("rev-parse", "HEAD", root=repository)
    status_lines = [
        line
        for line in _git("status", "--porcelain", "--untracked-files=all", root=repository).splitlines()
        if line and not any(
            line[3:].replace("\\", "/").startswith(prefix)
            for prefix in (".local_diagnostics/", ".runtime/", "results/", "logs/", "artifacts/")
        )
    ]
    worktree_clean = not status_lines
    safe_workers, effective_cpus, quota = safe_worker_limit()
    disk = shutil.disk_usage(repository)
    memory_limit = cgroup_memory_limit()
    errors: list[str] = []
    if workers < 1 or workers > safe_workers:
        errors.append(
            f"requested workers={workers}, safe maximum={safe_workers}, effective CPUs={effective_cpus}"
        )
    if not worktree_clean:
        errors.append(f"Git worktree contains production-relevant changes: {status_lines[:10]}")
    if expected_branch and branch != expected_branch:
        errors.append(f"branch mismatch: expected={expected_branch} observed={branch}")
    if expected_commit and commit != expected_commit:
        errors.append(f"commit mismatch: expected={expected_commit} observed={commit}")
    if projected_peak_bytes > target_peak_bytes:
        errors.append(
            f"projected peak {projected_peak_bytes} exceeds target {target_peak_bytes}"
        )
    if projected_peak_bytes + safe_margin_bytes > disk.free:
        errors.append(
            "insufficient repository filesystem space: "
            f"projected={projected_peak_bytes} free={disk.free} safe_margin={safe_margin_bytes}"
        )
    inode_free = _inode_free(repository)
    if inode_free is not None and inode_free < 100_000:
        errors.append(f"insufficient free inodes: observed={inode_free} required=100000")
    if memory_limit is None:
        errors.append("cgroup memory limit is unavailable; refusing to use host-visible memory")
    elif memory_limit < 16 * GIB:
        errors.append(f"cgroup memory limit is too small: observed={memory_limit} required={16 * GIB}")
    dependencies = _dependency_versions()
    report = PreflightReport(
        ok=not errors,
        repository_root=str(repository),
        branch=branch,
        commit=commit,
        worktree_clean=worktree_clean,
        requested_workers=int(workers),
        safe_max_workers=safe_workers,
        effective_cpus=effective_cpus,
        cgroup_cpu_quota=quota,
        cgroup_memory_limit_bytes=memory_limit,
        disk_free_bytes=disk.free,
        inode_free=inode_free,
        projected_peak_bytes=int(projected_peak_bytes),
        required_safe_margin_bytes=int(safe_margin_bytes),
        target_peak_bytes=int(target_peak_bytes),
        paths={key: str(value) for key, value in paths.items()},
        dependency_versions=dependencies,
        errors=tuple(errors),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logs-root", required=True)
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--projected-peak-bytes", type=int, required=True)
    parser.add_argument("--expected-branch")
    parser.add_argument("--expected-commit")
    parser.add_argument("--report")
    arguments = parser.parse_args()
    report = production_preflight(
        workers=arguments.workers,
        output_root=arguments.output_root,
        logs_root=arguments.logs_root,
        artifacts_root=arguments.artifacts_root,
        runtime_root=arguments.runtime_root,
        projected_peak_bytes=arguments.projected_peak_bytes,
        expected_branch=arguments.expected_branch,
        expected_commit=arguments.expected_commit,
    )
    value = asdict(report)
    if arguments.report:
        target = contained_path(arguments.report, report.repository_root, label="preflight_report")
        write_json(target, value)
    print(json.dumps(value, sort_keys=True))
    if not report.ok:
        raise SystemExit("production preflight failed:\n- " + "\n- ".join(report.errors))


if __name__ == "__main__":
    main()
