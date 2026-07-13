# Requirements reconciliation

Status captured on 2026-07-13 after receiving the complete project specification. The active branch is `codex/predictive-feature-properties`; the checkout is intentionally dirty because the audited fixes are in progress. This document preserves the pre-specification work and identifies the remaining gap without resetting, cleaning, recloning, or discarding files.

## Already satisfied

- The repository is cloned at `D:\download\GitHub\streaming-rl-feature-geometry`; `origin` is the required GitHub URL and the active feature branch is correct.
- The repository-local `.venv` uses Python 3.11 and contains NumPy, pandas, matplotlib, and pytest.
- The continuing T-maze state machine now advances from outcome to the next cue trial, has no terminal training reset, and exposes the delayed cue echo only after the junction action.
- The junction observation has no cue bits; the oracle uses a separate controller-feature path.
- A fixed leaky trace state and linear TD(0) GVF bank exist, and controller reward is excluded from the shared observation-cumulant bank.
- Raw, standardized, decorrelated, whitened, and a preliminary signed-power Gaussian-inspired transform are causal: each output is computed from statistics through `t-1`, then statistics are updated.
- Semi-gradient linear SARSA(lambda) is strictly online and keeps eligibility traces across trial boundaries.
- Result directories use a unique run name and refuse overwrite; aggregation reads only the selected run root.
- The cue probe uses a deterministic held-out split and never feeds results back to the agent.
- Real CSV-backed matplotlib figures replaced PR #1 placeholder figures.
- The full-profile guard exists, unit tests currently pass (`7 passed`), a 4-condition smoke completed, and a preliminary 3-seed/6,000-interaction audit pilot completed. These runs predate the complete specification and are not the final required pilot.

## Partially satisfied

- The GVF bank has explicit observation-event definitions and multiple horizons, but its current dimension is 18 rather than the requested 8--16 and it lacks positive/negative outcome cumulants, per-GVF TD metrics, predictor update norms, and a trace-only controller baseline.
- Standardization and whitening have warm-up and numerical floors, but decorrelation currently rotates into the covariance eigenbasis without explicitly preserving marginal scale.
- The preliminary Gaussian condition is a fixed signed-power transform. It does not maintain exponentially weighted third/fourth moments or adapt a bounded parameter, so it is only a heuristic and cannot yet satisfy P5.
- Representation diagnostics include covariance, rank, moments, norm, and junction cue decodability, but they do not yet include position-specific cue probes, class separation, sparsity, transform drift, or maximum norm.
- Manifests contain core provenance, but not dirty status, dependency versions, config hash, exact per-condition command, and all required exit metadata.
- Current configs have smoke/pilot/full-like profiles, but smoke is below 2,000 interactions, pilot is below 20,000, and full lacks separate stationary/non-stationary profiles with the required two-part guard.
- Remote GitHub Actions and one generic shell script exist, but the three specifically required remote scripts, result completeness checks, packaging flow, and beginner Chinese guide do not.

## Not yet implemented

- P1 `rms_raw`, B2 `trace_only`, and at least one optional P6--P8 property.
- Online Gaussian-moment parameter adaptation with synthetic pass/fail diagnostics.
- Synthetic property-fulfillment suite and artifacts under `results/diagnostics/`.
- The single permitted non-stationarity: one midpoint corridor-length increase, with adaptation metrics.
- Cue-only versus mixed-bank and short-horizon ablations in a bounded remote design.
- Time-to-threshold, per-GVF prediction metrics, cue decodability by corridor position, cue margin, between/within-class ratio, predictor update norm, transform drift, sparsity, and divergence metrics.
- Required figure set, including feature histograms and accuracy-versus-isotropy/effective-rank views.
- The final 3-seed, 20,000--50,000-interaction stationary pilot.
- English proposal, Chinese explanation, implementation report, expanded research specification, paper/poster outlines, work log, and Chinese remote guide.
- Final review, diff checks, commit, push, and Draft PR.

## Conflicts between the current implementation and the complete specification

- Current controller features omit an explicit appended bias because the observation contains `bias_obs`; the specification explicitly requires `[observation, z, 1]`. The runner must append a separate controller bias and document the intentional redundancy or remove `bias_obs` from the controller path without changing environment observations.
- Current full guard accepts either `RL_RUN_CONTEXT=remote` or `--allow-full-run`; the specification requires both.
- Current full config uses 10 seeds and 50,000 interactions; the remote plan requires 20 seeds and 200,000 stationary interactions, plus a separate 300,000-interaction non-stationary profile.
- Current pilot has seven conditions and 6,000 interactions; the required final pilot must include `trace_only`, `rms_raw`, a diagnostic-qualified Gaussian condition, at most one optional property, three seeds, and at least 20,000 interactions.
- Current GVF observation-only cumulants ensure identical raw streams across policies, but the required semantic bank includes positive and negative outcome predictions. These can remain policy-independent by defining outcome correctness sensors in the post-action observation/transition metadata only if that does not leak into the pre-action controller; otherwise the design must explicitly document the fairness tradeoff.
- Current signed-power Gaussian transform has a fixed exponent and therefore cannot be reported as online moment matching.
- Current generic per-step CSV does not match the required separated prediction/update artifacts and saves more step-level data than necessary.

## Code to preserve

- The corrected `ContinuingTMaze` phase logic and delayed echo timing.
- The causal Welford statistics order and first-sample identity behavior.
- The shared fixed-trace GVF architecture and reward-exclusion fairness test.
- The online SARSA(lambda) implementation, held-out probe, result isolation, validation, multiprocessing, and real plotting foundation.
- The repository-local environment, ignore rules, CI hardening, and all completed audit results.

## Code to modify

- `env.py`: add one explicit midpoint corridor-length change interface and position metadata without adding any other drift.
- `gvf.py`: expose trace state, use 8--16 preregistered semantic GVFs, record per-GVF TD/predictor metrics, and support a bounded cue-only/short-horizon ablation.
- `transforms.py`: add `rms_raw`, scale-preserving decorrelation, bounded online Gaussian-moment adaptation, and one optional property; expose drift/property diagnostics.
- `experiment.py`, `metrics.py`, and `reporting.py`: add required baselines, separated outputs, richer metrics, stronger manifest, non-stationary summaries, exact profile guards, and remaining real figures.
- Configs, tests, workflows, scripts, README, and AGENTS instructions to match the complete specification.

## Placeholder or invalid behavior to replace

- Replace the fixed signed-square-root `gaussian` heuristic as the claimed P5 condition; retain its scientific idea only through an explicit bounded online moment objective and failure diagnostics.
- Replace the single generic `run_remote.sh` with the required bootstrap, full-run, and aggregation scripts; it may remain only if clearly documented as a convenience wrapper.
- Replace outdated top-level status documents that say implementation is complete or imply unrun experiments succeeded.

## Test history and causes

- PR #1 baseline: `2 failed, 1 passed`. The environment test had an incorrect phase count and exposed the broken outcome loop; the transform test showed first-sample standardization of `[10, -5]` becoming approximately `[316, -158]` because variance started at zero plus epsilon.
- After the initial audited repair: `7 passed`. This is not yet sufficient coverage for the complete specification; new environment, GVF, transform, Gaussian, controller, isolation, guard, and reproducibility tests are required.
- The preliminary smoke and audit pilot completed without runtime failure, but property diagnostics show covariance rank deficiency and therefore do not yet prove every named property is fulfilled.

## Execution order from this reconciliation

1. Freeze the full research specification and hypotheses before running the final pilot.
2. Complete environment non-stationarity, semantic GVFs, trace/RMS baselines, transforms, diagnostics, metrics, manifests, and profile guards.
3. Expand tests and run the full local suite.
4. Run synthetic diagnostics and retain pass/fail evidence.
5. Run a fresh compliant smoke, inspect its CSVs and figures, and fix only correctness defects.
6. Run the fixed 3-seed final pilot without per-condition tuning; aggregate and validate evidence.
7. Write proposals, reports, outlines, README/AGENTS, work log, and remote package from the frozen design and actual pilot.
8. Review causality, leakage, scale fairness, stale-result isolation, Gaussian claims, placeholders, secrets, and code/document consistency.
9. Run final tests, smoke/pilot validation, `git diff --check`, commit, push, and create a Draft PR if authentication permits.
