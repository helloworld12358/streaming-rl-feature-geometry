# Prescribed Predictive-Feature Geometry for Streaming RL

## 1. Project overview

This project tests whether explicitly prescribed statistical and geometric properties of predictive features improve a non-deep, strictly streaming reinforcement-learning agent under partial observability.

## 2. RL question

Do prescribed statistical and geometric properties of a fixed predictive state improve online control, or can a transform improve generic geometry while erasing the task-relevant cue? This is an RL question because the representation is learned online from a single continuing control stream and is evaluated by downstream return and junction decisions.

## 3. Environment

`ContinuingTMaze` is a continuing T-maze with cue, delayed echo, corridor, junction, and one-step outcome phases. The left/right cue is unavailable at the junction. After the outcome, the environment enters the next trial rather than remaining terminal. The non-stationary full profile changes corridor length once, from 5 to 9 at interaction 150,000.

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

## 7. Baselines

`observation_only` has no history, `trace_only` receives the fixed trace state directly, and `oracle` receives the hidden cue as an upper bound. A controller bias is appended explicitly and consistently.

## 8. Metrics

The runner records trial accuracy, cumulative reward, time to threshold, GVF TD errors, update and parameter norms, non-finite counts, cue decodability by position, cue separation, covariance eigenvalues, effective rank, isotropy, condition number, correlation, skewness, kurtosis, transform drift, and non-stationary adaptation where applicable. Figures show mean and standard error with sample count.

## 9. Local installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Python 3.11 is the supported runtime. Dependencies are intentionally limited to NumPy, pandas, Matplotlib, and pytest.

## 10. Unit tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 11. Smoke command

```powershell
.\scripts\run_smoke.ps1 -RunName local-smoke -Workers 1
```

The smoke profile uses seed 0, 3,000 interactions, and the observation-only, oracle, raw, and whitened conditions.

## 12. Pilot command

```powershell
.\scripts\run_pilot.ps1 -RunName preregistered-pilot -Workers 3
```

The pilot uses seeds 0, 1, and 2, 20,000 interactions, and all registered conditions. It is diagnostic evidence, not a final experiment.

## 13. Full remote guard

The 20-seed full profiles are blocked unless both safeguards are present: `RL_RUN_CONTEXT=remote` and `--allow-full-run`. Hostname guesses and either safeguard alone are intentionally insufficient.

## 14. Remote command

```bash
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/full_stationary.json --workers 4 --run-name full-stationary-<COMMIT_SHA>
```

See `docs/REMOTE_RUN_GUIDE_ZH.md` before launching the stationary 200,000-interaction or non-stationary 300,000-interaction suite.

## 15. Results directories

Each batch uses `results/<profile>/<run-name>/`; each condition/seed is isolated under `runs/<condition>/seed_NNN/`. Configs, manifests, per-trial/prediction/representation/hidden-state/update CSVs, summaries, logs, learned GVF arrays, and real figures are recorded. Existing run directories are never overwritten.

## 16. Proposal paths

- `docs/proposal_en.md`
- `docs/proposal_zh_explanation.md`
- `docs/research_specification.md`

## 17. Known limitations

The local smoke and three-seed pilot are too small for final claims. Linear off-policy GVF learning and a single T-maze family limit generality. Whitening can improve isotropy without improving control, moment shaping is exploratory, and fixed controller step sizes may interact with feature scale and transform drift.

## 18. Reproduction information

Every manifest records the exact command, config hash, seed, Git branch/commit/dirty state, Python and dependency versions, host/OS, times, and exit status. Reproduce a reported result by checking out its commit, using its recorded config, and choosing a new run name. Full results must also record the remote machine and retain failed-seed manifests.
