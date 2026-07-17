# Implementation Report

## Repository initial state

The remote `main` initially contained only a short README. Existing PR #1 (`codex/githubrl`, base commit `1d70ecf`) supplied a useful first implementation, which was fetched and audited rather than discarded. Development continued on `codex/predictive-feature-properties` without reinitializing Git, changing `origin`, or rewriting `main`.

## PR #1 audit and discovered defects

The baseline test suite reported **2 failed, 1 passed**. The continuing T-maze remained in the outcome phase instead of entering the next trial; the first standardized sample was divided by epsilon and amplified to roughly `[316, -158]`. Additional audit findings were a non-semantic predictive bank, incomplete baselines/transforms, in-sample cue probes, stale-output risk, placeholder figures, insufficient manifests, and a full-run guard that accepted either safeguard instead of requiring both.

## Modifications

- Repaired the cue → delayed echo → corridor → junction → outcome → next-cue state machine and added the single non-stationary corridor-change interface.
- Implemented a fixed 12-dimensional trace and ten semantic online GVFs with per-GVF cumulants, continuations, errors, weights, and definitions.
- Implemented isolated observation-only, oracle, trace-only, raw, RMS raw, standardized, decorrelated, whitened, Gaussian-inspired, and unit-sphere conditions.
- Implemented causal `OnlineMoments`, protected first-sample behavior, singular covariance handling, transform/drift diagnostics, and no generic hidden clipping.
- Implemented accumulating-trace semi-gradient SARSA(lambda), deterministic seeds, finite-update checks, and explicit bias handling.
- Replaced in-sample analysis with deterministic trial-grouped held-out ridge probes.
- Added unique condition/seed directories, complete configs/manifests/logs/CSVs, completeness validation, real Matplotlib plots, and strict full-run gating.
- Added local PowerShell launchers, Linux remote bootstrap/full/aggregate scripts, CI smoke, and manual self-hosted full workflows.

## Tests

The expanded suite covers environment transitions/continuing behavior/no cue leakage/reproducibility, GVF semantics and single-use updates, causal transforms and singular streams, Gaussian synthetic moments, controller traces and learnable sanity cases, result isolation/manifests, full guards, reproducibility, and held-out probes. The final full run reported **32 passed in 44.62 s**. A targeted post-pilot-validator regression run reported **2 passed in 21.38 s**. Bash and PowerShell launcher syntax checks also passed.

## Property fulfillment

The formal synthetic diagnostic `results/diagnostics/preregistered-20260713-v3` passed all 12 registered checks. RMS scaling reached global RMS 1.003 without changing correlation; standardization reached mean error 0.022 and variance mean 1.005; decorrelation reduced mean absolute correlation 0.977 → 0.005; whitening reduced isotropy error 2.408 → 0.016 and remained finite on repeated/constant streams; unit sphere reached norm 0.9992. Gaussian-inspired shaping reduced skew error on positive/negative-skew streams and kurtosis error on a heavy-tailed stream while retaining the synthetic bimodal cue probe, but it does not guarantee Gaussianity.

## Smoke

The compliant smoke `results/smoke/compliant-smoke-20260713-v2` completed 4/4 runs at 3,000 interactions. Final-window accuracy was observation-only 0.61, oracle 0.95, raw 0.59, and whitened 0.58. Whitening reduced real-stream isotropy error from 5.61 to 3.10. Nineteen non-placeholder figures were generated. The prior v1 batch is retained as failed because its validator incorrectly treated not-applicable transform fields as numerical divergence.

## Pilot

The final pilot `results/pilot/preregistered-pilot-20260713-v3` used seeds 0–2, 20,000 interactions, and 10 conditions. Its manifest records clean commit `e5f31cc7bbed5faf1a64d393708a0b5790a41a2a`; the identical tree was republished as `7687451` after an author-email-only rewrite required by GitHub privacy protection. It completed 30/30 runs with no duplicate condition/seed rows and zero NaN, Inf, or divergence counts. Oracle final accuracy was `0.985 ± 0.006`; all non-oracle conditions lay between `0.522 ± 0.026` and `0.568 ± 0.004`, with overlapping uncertainty. Whitening improved isotropy and effective rank but not control clearly. Trace-only perfectly decoded cue at the held-out junction yet achieved only `0.567 ± 0.009`, further separating diagnostic accessibility from online utility.

Failed pilot batches are preserved: v1 completed all child runs but an over-broad optional-field validator failed the batch; v2 was interrupted by the desktop command lifetime and remained incomplete. A fresh background v3 was used rather than editing either manifest.

## Failures and limitations

- The short pilot does not establish comparative performance or causality.
- Raw predictive features did not outperform observation-only; environment, trace, cumulants, continuation, feature non-constancy, cue probes, and controller wiring were checked rather than tuned post hoc.
- Real-stream covariance remains rank-deficient in some directions, so condition number and isotropy must be interpreted together.
- Whitening improved geometry without clear control benefit; possible explanations include small cue margins, transform drift, scale/step-size interaction, and categorical latent structure.
- Gaussian-inspired shaping is exploratory and environment-dependent.
- A single T-maze family and fixed linear controller limit external validity.
- Full stationary and non-stationary experiments have not been run.

## Remote readiness

`configs/full_stationary.json` and `configs/full_nonstationary.json` define 20-seed CPU studies at 200k and 300k interactions. Both environment variable `RL_RUN_CONTEXT=remote` and flag `--allow-full-run` are mandatory. The current production Linux workflow uses the existing `/usr/bin/python3` without venv/Conda, caps BLAS at one thread per process, enforces repository-contained temp/cache/results/logs/artifacts, reads cgroup CPU and memory limits, runs tests plus a real compact-storage pilot, and fails if the 4200-run peak projection exceeds the disk budget. Invalid runs preserve exact runtime evidence and cannot be resumed as completed. The supported procedure is `docs/PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md`; older single-batch guides are historical records only.

## Cross-environment extension handoff

The later extension preserves this core path and adds a separate environment-aware runner for E1-E4, compact/mixed causal predictive banks, sparse/bounded transforms, fixed-semantic simplex/circular/block/anisotropic priors, held-out task-specific probes, cross aggregation/figures, staged local profiles, and guarded 20-seed remote profiles. The current combined suite passes 72 tests. Implementation details and actual three-seed results are in `CROSS_ENV_IMPLEMENTATION_REPORT.md` and `CROSS_ENV_RESULTS.md`.

The extension does not turn the core pilot into a positive result. Its own results are also mixed: E3 shows positive signals, E1 is null, E2 is uncertain, E4 often favors raw, and E2 real-stream circular phase order fails its property gate. Cross-environment remote full experiments remain unrun.

## Hidden-velocity metrics, alpha controls, and production extension

The production follow-up keeps the verified core runner intact and extends only the cross-environment path. Both hidden-velocity environments now emit the exact reward components used by the environment; logging never reimplements the reward formula. Per-seed summaries include whole-run/final-window cost, RMSE, stabilization, boundary, settling, disturbance, and recovery metrics. Long-tailed condition evidence is summarized with median, IQM, bootstrap 95% intervals, 10/90% quantiles, extrema, and config-fixed catastrophic-failure rules.

The linear SARSA controller now has explicit `fixed` and default-off `norm_scaled` modes. Fixed mode retains the original numerical update. Norm scaling uses only the current/past feature-norm EMA and bounded effective alpha. A separate tuning module validates disjoint design/tuning/evaluation/smoke seeds, expands the six-alpha grid, applies the configured robust selection/tie break, writes `selected_learning_rates.csv`, and prevents evaluation-time reselection.

`hidden_velocity_informative` minimally increases inertia and velocity cost and adds seed-controlled bounded impulses while keeping velocity absent from observations. The local design pilot used only seeds 200–204 and four preregistered conditions. Oracle exceeded observation-only final reward in four of five seeds; predictive outcomes remained mixed and are preserved as such. Details and the small committed summary are in `HIDDEN_VELOCITY_INFORMATIVE_DESIGN.md`.

T-maze and Two-loop retain full-stream diagnostics and add true-decision, group-held-out probes with sample/group/split counts, status, failure reason, NaN for insufficient/not-applicable metrics, and T-maze corridor-position output. These probes remain offline analysis only.

Four remote-only profiles contain exactly 1000 fixed, 1500 tuning, 1000 tuned-evaluation, and 700 norm-scaled runs. The current production launcher runs the four stages in the foreground, requires the storage/commit/full-run gates, preserves failed attempts, and supports strict resume plus explicit retry-invalid. It has no dry-run, background, tmux, nohup, or scheduler fallback. Aggregation fails closed on incomplete/mixed/non-finite results and writes robust CSVs, figures, validation evidence, `analysis_manifest.json`, and a Chinese summary. Packaging produces verified full and analysis-core `.tar.gz` archives plus `.sha256`. The production-root-cause checkpoint passes 112 tests, a 70/70 compact production smoke, and all 13 original-horizon known-failure reruns; the final 50/50 storage pilot projects a 22.742 GiB peak for 4200 runs. The informative design pilot's historical `EmptyDataError` and mixed scientific result remain preserved. No formal 4200-run profile was run locally.
