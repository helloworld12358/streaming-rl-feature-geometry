import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("remote_environment", "extra_arguments"),
    [
        (False, []),
        (True, []),
        (False, ["--allow-full-run"]),
    ],
)
def test_full_guard_requires_both_independent_gates(
    tmp_path, remote_environment, extra_arguments
):
    environment = os.environ.copy()
    if remote_environment:
        environment["RL_RUN_CONTEXT"] = "remote"
    else:
        environment.pop("RL_RUN_CONTEXT", None)
    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            "configs/full_stationary.json",
            "--output-dir",
            str(tmp_path),
            *extra_arguments,
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode != 0
    assert "both RL_RUN_CONTEXT=remote and --allow-full-run are required" in (
        process.stderr + process.stdout
    )
    assert not any(tmp_path.iterdir())

