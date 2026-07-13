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

`configs/full_stationary.json` and `configs/full_nonstationary.json` define 20-seed CPU studies at 200k and 300k interactions. Both environment variable `RL_RUN_CONTEXT=remote` and flag `--allow-full-run` are mandatory. Remote scripts validate Python, create `.venv`, run tests/smoke, enforce a CPU safety margin, preserve failed-seed manifests, verify completeness, regenerate aggregates/figures, and package artifacts. The exact user procedure is in `docs/REMOTE_RUN_GUIDE_ZH.md`.
