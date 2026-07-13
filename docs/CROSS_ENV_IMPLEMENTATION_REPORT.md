# Cross-Environment Implementation Report

## Scope and invariants

The extension asks whether prescribed representation properties transfer across four different latent-state geometries while the streaming linear agent remains fixed. The implementation remains non-deep and one-pass: each transition is consumed once by a fixed trace, a linear TD(0) predictive bank, one causal or fixed-semantic transform, and linear accumulating-trace SARSA(lambda). There is no replay, minibatch, target network, future-data fitting, world model, or analysis-to-agent feedback.

Oracle and latent state are exposed only by the environment's diagnostic interface. `_controller_state` is the single controller boundary: only `oracle` reads `oracle_features`; all predictive conditions receive transformed predictions, and `observation_only` receives no predictive state. A regression test uses an oracle property that raises on access to prove non-oracle paths do not read it.

## Environments

- E1 `tmaze`: an adapter around the verified continuing T-maze. The remote-only non-stationary profile applies one corridor-length change through the core trial-start guard.
- E2 `ringworld`: a continuing aliased cycle with two observationally identical decision phases that require different actions.
- E3 `two_loop`: continuing identity-by-phase dynamics with an aliased decision and product latent structure.
- E4 `hidden_velocity`: bounded continuous position/velocity dynamics; velocity is hidden from the ordinary observation.

All four implement the same observation/action/step contract and return diagnostic metadata separately. Reproducibility, continuing behavior, aliasing, bounds, and observation/oracle separation are tested.

## Predictive banks and representations

`CausalPredictiveBank` provides compact and mixed fixed-semantic banks under one API. It uses deterministic leaky traces and linear TD(0), enforces exactly one update per presented feature vector, and records every predictive definition and learned weight array.

Generic conditions are raw, RMS-matched raw, marginal standardized, decorrelated, whitened, exploratory `gaussian_moment`, unit sphere, fixed top-k sparse, and bounded. Whitening is described only as a second-order transform. `gaussian_moment` remains an online moment-shaping extension without distributional guarantees.

The fixed task-matched transforms are M1 simplex for E1, M2 circular for E2, M3 simplex-by-circle blocks for E3, and M4 prescribed anisotropic scaling for E4. Their API accepts predictive features and fixed semantic indices only; it has no latent-label argument.

## Runner, diagnostics, and artifacts

`cross_experiment.py` adds an environment-aware runner alongside the unchanged core runner. Each environment/condition/seed receives an isolated directory containing config, manifest, logs, predictive definitions, learned weights, online step/decision/prediction/update metrics, representation metrics, deterministic held-out task probes, summaries, and diagnostic samples. Aggregation requires the complete expected matrix, reports mean/SEM, and produces Matplotlib figures from saved data.

The figure set covers learning curves, final performance, matched versus generic conditions, rank/isotropy/decodability/moment relationships, update stability, eigenvalue spectra, property fulfillment, and E1-E4 geometry-specific diagnostics. A non-stationary adaptation figure is emitted only when an executed E1 change exists.

`run_cross_diagnostics.py` writes synthetic and real-stream metrics plus a machine-readable pass/fail table. The superseding v2 diagnostic deliberately exits nonzero because the real E2 circular representation meets the unit-radius constraint but fails the fixed phase-order threshold. This failed method evidence is retained.

## Profiles

- `cross_smoke.json`: E1-E4, one seed, observation/oracle/raw/whitened/matched.
- `cross_pilot.json`: E1-E4, seeds 0-2, core generic conditions plus matched, without a full Cartesian product.
- `cross_pilot_compact.json` and `cross_pilot_bank_mixed.json`: same-budget compact/mixed comparison under raw and matched only.
- `cross_pilot_short_horizon.json` and `cross_pilot_matched_horizon.json`: same-budget E1/E2 horizon comparison under raw and matched only.
- `cross_full*.json`: 20-seed cross-environment main suite, four independently runnable E1-E4 environment configs, compact, short-horizon, and E1 non-stationary suites. They are remote-only.

Both the Python runner and Linux launcher require `RL_RUN_CONTEXT=remote` and `--allow-full-run` for every full profile. The launcher keeps one CPU free, preserves logs before and after result-directory creation, and the aggregator checks every expected manifest before rebuilding figures and packaging.

## Verification status

- Full local suite after extension: 72 passed in 31.37 seconds.
- Formal cross smoke: 20/20 runs, top manifest `ok`, 15 real figures.
- Formal main pilot: 120/120 runs, top manifest `ok`.
- Compact/mixed bank ablation: 24/24 plus 24/24 runs, both `ok`.
- Short/matched horizon ablation: 12/12 plus 12/12 runs, both `ok`.
- Remote-full guard probes: both local launcher and aggregator refuse with exit code 2 and create no full result directory.
- Remote full experiments: prepared but not run.

The result interpretation, including negative and failed findings, is in `docs/CROSS_ENV_RESULTS.md`.
