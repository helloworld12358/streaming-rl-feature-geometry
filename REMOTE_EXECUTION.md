# Remote execution

No remote host, SSH credential, or self-hosted runner is encoded in this repository.

## Direct server workflow

```bash
ssh <remote-host>
git clone <repository-url>
cd streaming-rl-feature-geometry
git checkout <exact-commit-sha>
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt
pytest -q
python scripts/run_experiment.py --config configs/smoke.json --workers 1
python scripts/run_experiment.py --config configs/validation.json --workers 2
RL_RUN_CONTEXT=remote python scripts/run_experiment.py --config configs/full.json --workers 4
python scripts/run_experiment.py --config configs/full.json --aggregate-only
```

Download results with:

```bash
rsync -av <remote-host>:<repo>/results/full/ ./results/full/
```

## GitHub Actions self-hosted runner

Install a self-hosted runner with labels `self-hosted`, `linux`, `x64`, and `rl-cloud`, then trigger `Full remote experiment` manually. Inputs allow profile, seeds, interactions, workers, run name, and ref.

If no runner is present, full remote experiments are blocked and must not be reported as completed.
