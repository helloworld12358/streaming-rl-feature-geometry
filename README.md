# Imposing Statistical and Geometric Structure on Predictive Features for Streaming Reinforcement Learning

This repository studies whether mathematical constraints on predictive features improve strictly streaming, non-deep linear control under partial observability.

## RL question

In a continuing partially observable T-maze, the cue is visible only at trial start and the junction observation is identical for left/right cues. The question is: for predictive agent state built from observation history, which statistical/geometric feature properties improve online linear SARSA control, and which improve representation diagnostics without preserving task-relevant hidden-state information?

## Streaming definition

Each transition is used once for online GVF TD, online causal feature transformation, and online SARSA(lambda). There is no replay, minibatch sampling, batch fitting, future covariance, neural network, target network, GPU training, or world model.

## Methods and baselines

* `observation`: observation-only controller.
* `oracle`: upper-bound controller with direct hidden cue input, isolated from main methods.
* `raw`: fixed mixed GVF predictions.
* `standardized`: causal online mean/variance standardization.
* `decorrelated`: causal covariance eigenbasis rotation without unit variance scaling.
* `whitened`: causal second-order isotropy/whitening using `(C + eps I)^(-1/2)`.
* `gaussian`: experimental signed-power moment-shaping after whitening; this is not a normalizing flow and is not claimed to guarantee `N(0,I)`.

The GVF implementation uses a fixed linear trace memory fallback, `m_t = rho m_{t-1} + B o_t`, with online TD predictions. This keeps the project focused on feature-property constraints rather than learned recurrent architecture.

## Commands

```bash
python -m pip install -e . -r requirements.txt
pytest -q
python scripts/run_experiment.py --config configs/smoke.json --workers 1
python scripts/run_experiment.py --config configs/validation.json --workers 2
RL_RUN_CONTEXT=remote python scripts/run_experiment.py --config configs/full.json --workers 4
# or, on a remote server only:
python scripts/run_experiment.py --config configs/full.json --allow-full-run --workers 4
```

Full profile is intentionally blocked unless `RL_RUN_CONTEXT=remote` or `--allow-full-run` is supplied.

## Outputs

Each run writes `results/<profile>/<run_id>/config.json`, `manifest.json`, `per_step_metrics.csv`, `per_trial_metrics.csv`, `representation_metrics.csv`, `summary.csv`, `stdout.log`, and `figures/`. Aggregates are written to `aggregate_summary.csv`, `aggregate_representation.csv`, and `figures/`.

## GitHub Actions

* `.github/workflows/ci-smoke.yml` runs dependencies, unit tests, smoke experiment, and result validation on GitHub-hosted Linux.
* `.github/workflows/full-remote.yml` is manual and targets `[self-hosted, linux, x64, rl-cloud]` for remote full/validation runs.

See `REMOTE_EXECUTION.md` for remote server instructions.
