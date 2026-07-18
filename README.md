# Prescribed Predictive-Feature Geometry for Streaming RL

## 1. Project overview

This project tests whether explicitly prescribed statistical and geometric properties of predictive features improve a non-deep, strictly streaming reinforcement-learning agent under partial observability.

## 2. RL question

Do prescribed statistical and geometric properties of a fixed predictive state improve online control, or can a transform improve generic geometry while erasing the task-relevant cue? This is an RL question because the representation is learned online from a single continuing control stream and is evaluated by downstream return and junction decisions.

## 3. Environment

`ContinuingTMaze` is a continuing T-maze with cue, delayed echo, corridor, junction, and one-step outcome phases. The left/right cue is unavailable at the junction. After the outcome, the environment enters the next trial rather than remaining terminal. The cross-environment extension adds an aliased Ringworld, an identity-by-phase two-loop task, bounded hidden-velocity control, and an independently registered informative hidden-velocity variant with sparse bounded impulse disturbances. Both velocity environments keep velocity out of the ordinary observation. The remote non-stationary profiles apply one E1 corridor-length change from 5 to 9 at interaction 150,000.

## 4. Streaming definition

Every transition is consumed once, in causal order, by online GVF TD, an online transform, and accumulating-trace semi-gradient SARSA(lambda). There is no replay, minibatch fitting, future covariance, deep learning, target network, or offline feature feedback. Analysis probes use deterministic held-out trial groups and never update the agent.

## 5. Predictive feature bank

The fixed mixed bank has ten semantic GVFs: observation/echo, junction, positive-outcome, and negative-outcome cumulants at two continuation horizons. A fixed 12-dimensional trace state supplies history; it is not a learned recurrent model.

## 6. Feature-property methods

- `raw`: unmodified GVF prediction vector.
- `rms_raw`: causal scalar RMS matching.
- `standardized`: online marginal centering and variance normalization.
- `decorrelated`: causal correlation-eigenbasis rotation with marginal scale reapplied.
- `whitened`: causal covariance whitening toward second-order isotropy.
- `gaussian_moment`: exploratory bounded skew/tail shaping after whitening; it does not guarantee a Gaussian distribution.
- `unit_sphere`: optional causal unit-norm projection.
- `sparse`: fixed top-k causal projection.
- `bounded`: causal elementwise bounded projection.
- `matched`: fixed predictive-semantic simplex, circular, block, or anisotropic prior selected by environment; it never receives latent labels.

## 7. Baselines

`observation_only` has no history, `trace_only` receives the fixed trace state directly, and `oracle` receives the hidden cue as an upper bound. A controller bias is appended explicitly and consistently.

## 8. Metrics

The runner records trial accuracy, cumulative reward, time to threshold, GVF TD errors, update and parameter norms, real non-finite/extreme counts, full-stream and true-decision held-out probes, covariance eigenvalues, effective rank, isotropy, condition number, correlation, skewness, kurtosis, transform drift, and non-stationary adaptation where applicable. Hidden velocity computes its authoritative cost decomposition every step and stores online summaries, final windows, strided diagnostics, and disturbance/recovery events. Condition summaries include median, IQM, bootstrap intervals, quantiles, extrema, and configured catastrophic-failure rates rather than relying only on mean ± SEM.

## 9. Local installation

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install --no-build-isolation --no-deps -e .
```

Python >= 3.10 is supported. Use the current Python environment; production cloud execution specifically uses `/usr/bin/python3` and does not create or activate a virtual environment. There are no external datasets, checkpoints, or model weights.

## 10. Unit tests

```powershell
python -m pytest -q
```

## 11. Smoke command

```powershell
python scripts\run_cross_experiment.py --config configs\cross_smoke.json --workers 2 --run-name local-smoke
```

The smoke profile uses seed 0, 3,000 interactions, and the observation-only, oracle, raw, and whitened conditions.

## 12. Pilot command

```powershell
python scripts\run_cross_experiment.py --config configs\cross_pilot.json --workers 3 --run-name preregistered-pilot
```

The pilot uses seeds 0, 1, and 2, 20,000 interactions, and all registered conditions. It is diagnostic evidence, not a final experiment.

## 13. Full remote guard

The 20-seed full profiles are blocked unless both safeguards are present: `RL_RUN_CONTEXT=remote` and `--allow-full-run`. Hostname guesses and either safeguard alone are intentionally insufficient.

Fixed nonlinear utilization adapters are registered in three guarded stages. Identity, residual 64-wide random Fourier features, and 8-tiling/512-entry tile coding are non-learnable maps applied after the complete legacy controller input `[observation, condition state, bias]` has been assembled. The exact matrices, strict fieldwise reuse rules, lightweight `adapter_summary_v1` storage, audit/aggregation tools, local smoke, and remote workflow are documented in [docs/UTILIZATION_ADAPTER_EXPERIMENT_GUIDE_ZH.md](docs/UTILIZATION_ADAPTER_EXPERIMENT_GUIDE_ZH.md). Formal execution remains CPU-only and remote-only; missing or incompatible baseline cells are reported and scheduled as new identity runs without shrinking the matrix.

## 14. Remote command

The authoritative remote procedure is [docs/PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md](docs/PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md). It uses `/usr/bin/python3`, foreground execution with `tee`, repository-contained temp/cache/results/logs/artifacts, a real storage pilot, cgroup-aware preflight, and both full-run gates. It never uses tmux, nohup, setsid, a scheduler, or a virtual environment.

## 15. Results directories

Each condition/seed is isolated in a run partition. Production compact v2 stores scalar/probe CSVs plus compressed strided, decision, disturbance, and model-state NPZ files. It never duplicates all raw traces into monolithic aggregate CSVs, and existing or failed run directories are never overwritten.

## 16. Proposal paths

- `docs/proposal_en.md`
- `docs/proposal_zh_explanation.md`
- `docs/research_specification.md`

## 17. Known limitations

The local smoke and three-seed pilot are too small for final claims. Linear off-policy GVF learning and a single T-maze family limit generality. Whitening can improve isotropy without improving control, moment shaping is exploratory, and fixed controller step sizes may interact with feature scale and transform drift.

## 18. Reproduction information

Every manifest records the exact command, config hash, seed, Git branch/commit/dirty state, Python and dependency versions, host/OS, times, and exit status. Reproduce a reported result by checking out its commit, using its recorded config, and choosing a new run name. Full results must also record the remote machine and retain failed-seed manifests.

## 19. Cross-environment extension

The environment-aware runner is separate from the verified core runner:

```powershell
python scripts\run_cross_diagnostics.py --run-name cross-diagnostics-unique
python scripts\run_cross_experiment.py --config configs\cross_smoke.json --workers 4 --run-name cross-smoke-unique
python scripts\run_cross_experiment.py --config configs\cross_pilot.json --workers 4 --run-name cross-pilot-unique
```

Local pilot evidence uses seeds 0-2 and a staged matrix. Same-budget compact/mixed and matched/short-horizon configs are separate so the project never runs a full local Cartesian product. The current local result is heterogeneous: E3 has positive conditioning/moment signals, E1 is null, E2 is uncertain, E4 often favors raw, and the E2 circular prior fails its real-stream phase-order property gate. See `docs/CROSS_ENV_RESULTS.md`; these are pilot observations, not final claims.

## 20. Cross-environment remote full

The 20-seed cross full suite is prepared but has not been run. On a real remote Linux machine only:

```bash
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name cross-full-<COMMIT_SHA>
```

Four environment-specific `cross_full_<environment>.json` configs and the selected `cross_full_compact.json`, `cross_full_short_horizon.json`, and `cross_full_nonstationary.json` suites remain separately guarded. Follow `docs/CROSS_ENV_REMOTE_RUN_GUIDE_ZH.md`.

## 21. Cross-environment documents

- `docs/CROSS_ENV_EXTENSION_SPEC.md`
- `docs/CROSS_ENV_IMPLEMENTATION_REPORT.md`
- `docs/CROSS_ENV_RESULTS.md`
- `docs/proposal_cross_environment_en.md`
- `docs/proposal_cross_environment_zh.md`

## 22. Production cross extension

The production extension separates fixed-alpha, condition-specific tuning/tuned evaluation, and causal norm-scaled alpha modes. Fixed mode preserves the original SARSA update. Tuning uses seeds 100–104 and the fixed multiplier grid `base × [0.125, 0.25, 0.5, 1, 2, 4]`; evaluation uses seeds 0–19 and only consumes the generated `selected_learning_rates.csv`. Design seeds 200–204 and smoke seeds 9000–9001 remain disjoint.

The four guarded profiles contain exactly 1000, 1500, 1000, and 700 runs. Do not run them locally. Run the actual storage pilot first, then pass its repository-contained report to `scripts/run_cross_extension_remote.sh`. Exact fresh, resume, retry-invalid, status, validation, package, and SHA-256 commands are in [docs/PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md](docs/PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md). Successful runs are reused only when config, commit, schema, identity, required files, and runtime validity all match.

The local 50-run storage calibration projected a 22.742 GiB peak for all 4200 formal runs, including the source tree, a full archive budgeted without assuming compression savings, and an analysis-core archive. The remote filesystem must reproduce a clean-final-commit passing pilot report before the guarded full command will start.

## 23. Remote deployment and reproducibility

The canonical production Linux entries are `scripts/run_storage_pilot.sh` and `scripts/run_cross_extension_remote.sh`. Formal execution always requires both `RL_RUN_CONTEXT=remote` and `--allow-full-run`; local work is limited to tests, diagnostics, smoke, validation, and finite pilots. Results stay under `results/`, logs under `logs/`, packages under `artifacts/`, and cache/temp state under `.runtime/`.

- Production root-cause and cloud guide: `docs/PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md`
- Resource strategy: `docs/REMOTE_RESOURCE_STRATEGY.md`
- Deployment reconciliation/report: `docs/REMOTE_DEPLOYMENT_RECONCILIATION.md`, `docs/REMOTE_DEPLOYMENT_REPORT.md`
- Remote plans: `configs/remote_smoke.json`, `configs/remote_full_core.json`, `configs/remote_full_cross_environment.json`, `configs/remote_full_nonstationary.json`, `configs/remote_full_suite.json`

Known deployment limitations: there is no CUDA backend and no background/scheduler integration. The required formal command runs in the current foreground shell, so the cloud terminal/container must remain alive.
