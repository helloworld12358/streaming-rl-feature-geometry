# Preregistered research specification

Created: 2026-07-13 (Asia/Shanghai), before the specification-compliant 20,000-interaction pilot. Git state at registration: branch `codex/predictive-feature-properties`, base/head inherited from PR #1 commit `1d70ecf`, working tree dirty with audited implementation fixes. A prior 6,000-interaction engineering run had already been observed; it is excluded from confirmatory claims and is retained only as an audit artifact.

## Research question

Do prescribed statistical and geometric properties of online predictive features improve hidden-state retention, numerical conditioning, update stability, sample efficiency, final control, or adaptation in strictly streaming reinforcement learning under partial observability?

The fixed causal pipeline is:

`partial observations -> fixed trace state -> online linear GVFs -> one prescribed transform -> online linear SARSA(lambda)`

The primary independent variable is the mathematical property imposed on the same mixed predictive-feature construction. GVF questions and horizons are fixed construction choices, not the main comparison.

## Setting and invariants

- Continuing partially observable T-maze; no agent reset at trial boundaries.
- The cue is visible only at trial start. Junction observations are identical across cues.
- A cue echo is observable only after the junction action.
- Each transition is consumed once by online TD and SARSA updates. There is no replay, minibatch, offline representation fitting, future-statistics use, neural network, target network, or GPU training.
- A transform applied at time `t` uses statistics available through `t-1`; the raw sample is incorporated only after its transformed output has been produced.
- All conditions share the same environment family, GVF definitions, initialization rules, action/reward interface, budgets, seeds, exploration, predictor rate, and controller rate.
- Offline probes and PCA are diagnostics only and never feed the agent.
- Failed seeds and negative results remain in every aggregate.

## Predictive feature construction

The hand-designed memory is a bounded fixed linear trace:

`m_t = rho * m_(t-1) + (1-rho) * B * o_t`

Each linear GVF predicts one of five semantic cumulants at one of two preregistered continuations: left cue echo, right cue echo, junction occupancy, positive outcome, or negative outcome. The intended mixed bank therefore has `d=10`, within the registered 8--16 range. The predictor input is `[o_t, m_t, 1]`; readout weights update once by TD(0). Fixed trace dynamics are hand-designed memory, while GVF predictions are learned online. Fully learned recurrence is out of scope.

The two permitted auxiliary ablations are reserved for remote full runs only: mixed versus cue-only bank, and matched versus deliberately short horizons. They will not be crossed into a large grid.

## Conditions and property definitions

- **B0 observation_only:** controller input `[o_t, 1]`.
- **B1 oracle:** separate upper-bound path `[o_t, latent_cue, 1]`; no main method receives this cue.
- **B2 trace_only:** `[o_t, m_t, 1]`, isolating hand-designed memory from learned predictions.
- **P0 raw:** `z_t = g_t`.
- **P1 rms_raw:** one causal scalar rescales `g_t` by the past root-mean-square; correlations are unchanged. Raw RMS, factor, and output RMS are recorded.
- **P2 standardized:** past running marginal mean/variance target zero mean and unit variance. Identity warm-up and fixed variance floor prevent first-sample blow-up.
- **P3 decorrelated:** a past-covariance correlation-whitening map removes off-diagonal covariance while reapplying past marginal scales; it is not full whitening.
- **P4 whitened:** symmetric `(C + epsilon I)^(-1/2)(g_t-mu)` targets second-order isotropy only. Matrix refresh interval, eigenvalue floor, warm-up, eigenvalues, and matrix drift are recorded.
- **P5 gaussian_moment:** exploratory bounded online sinh--arcsinh moment shaping after whitening. Exponentially weighted second through fourth moments update bounded skew/tail parameters without reward. Passing synthetic skew/kurtosis diagnostics is required before interpreting it as a property-fulfilling condition. Moment matching is not distribution matching and whitening is not Gaussianity.
- **P6 unit_sphere (only optional local/full property):** `g_t/(||g_t||+epsilon)` tests whether discarding magnitude removes useful horizon/proximity/confidence information.

No transforms are stacked except the explicitly defined whitening first stage inside exploratory P5. No per-condition learning-rate tuning is allowed.

## Preregistered hypotheses

- **H1 Scale hypothesis.** Marginal standardization or RMS matching may improve linear-controller learning speed and stability by reducing feature-scale mismatch.
- **H2 Redundancy hypothesis.** A mixed GVF bank may be highly correlated; decorrelation or whitening may reduce redundancy and covariance condition number.
- **H3 Isotropy hypothesis.** Better covariance isotropy may prevent online updates from remaining concentrated in a few feature directions.
- **H4 Gaussian-moment hypothesis.** Smaller skewness and kurtosis error may reduce extreme updates but need not improve control.
- **H5 Task-information hypothesis.** The binary hidden cue can make a bimodal representation useful; strong Gaussian-inspired shaping may weaken cue separation.
- **H6 Rank-not-enough hypothesis.** Higher effective rank or lower isotropy error need not yield higher trial accuracy.
- **H7 Mixed-bank hypothesis.** Property constraints are more likely to help a multidimensional redundant mixed bank than a low-dimensional cue-only bank.
- **H8 Adaptation hypothesis.** Better-conditioned features may recover faster after a corridor-length increase, while adaptive normalization itself may introduce drift.
- **H9 Magnitude-information hypothesis.** Unit-sphere geometry can remove useful time-distance, probability, confidence, or horizon magnitude and may reduce performance.

No hypothesis guarantees a positive result. Counterexamples are a planned contribution.

## Metrics

Control: per-trial and moving accuracy, cumulative reward, final-window accuracy, time to an accuracy threshold, condition mean and across-seed standard error. Prediction: per-GVF squared TD error, event-family errors, predictor update/parameter norm. Hidden-state information: deterministic held-out cue decoding by corridor position and junction, cue-conditioned distance, junction margin, and between/within variance. Representation: mean error, marginal variance imbalance, mean absolute correlation, covariance eigenvalues, regularized condition number, effective rank, normalized isotropy error, skewness/kurtosis error, feature norm mean/variance/maximum, near-zero fraction, active dimensions, and transform drift. Stability: TD-error variance, controller/predictor update norms, parameter norms, non-finite counts, and divergence flags.

## Frozen experiment profiles

- **Smoke:** seed 0, 3,000 interactions, `observation_only`, `oracle`, `raw`, and `whitened`, one worker. It must demonstrate complete trials, oracle superiority, nonconstant raw features, an improved covariance metric, isolated CSV/JSON outputs, real figures, and finite metrics.
- **Pilot:** seeds `[0,1,2]`, 20,000 interactions, stationary corridor length 5, conservative process workers. Conditions: `observation_only`, `oracle`, `trace_only`, `raw`, `rms_raw`, `standardized`, `decorrelated`, `whitened`, `gaussian_moment` only if diagnostics run, and `unit_sphere` as the sole optional property. Learning rates and transform settings are shared and frozen before this run.
- **Full stationary (remote only):** seeds 0--19, 200,000 interactions, diagnostic-qualified conditions, bounded mixed/cue-only and matched/short ablations.
- **Full non-stationary (remote only):** seeds 0--19, 300,000 interactions, one fixed midpoint change from corridor length 5 to 9; no other drift. Record the change point, pre-change performance, post-change minimum, recovery time, and final post-change performance.

Both full profiles require `RL_RUN_CONTEXT=remote` **and** `--allow-full-run`; either missing gate must fail without fallback.

## Decision rules and interpretation

First assess whether each transform fulfills its named property on synthetic data and on observed representations. Then report control utility separately. P5 is labeled failed/exploratory if it does not reduce the targeted synthetic moment errors without destroying a bimodal separation diagnostic. The final pilot is a minimal reproducible observation, not a final conclusion. No method or hyperparameter will be selected using the final pilot reward.

## Cross-environment registered extension

The serial extension is specified in `CROSS_ENV_EXTENSION_SPEC.md` and its frozen hypotheses/matrix. It adds E2 Ringworld, E3 identity × phase two-loop, and E4 hidden velocity while adapting E1 through the existing implementation. The scientific variable remains the representation property; predictive questions, controller class, causal order, and analysis isolation stay fixed within each staged comparison.

Additional registered representations are fixed top-k sparse, causal bounded, and the fixed-semantic M1-M4 simplex/circular/block/anisotropic priors. Whitening remains a second-order transform and `gaussian_moment` remains exploratory. The local matrix uses seeds 0-2, E1-E4, core generic representations, and matched priors. Compact/mixed and matched/short horizons are paired only under raw/matched at identical budgets. Remote full configs use 20 seeds and require both independent safeguards.

The local pilot has now been observed and is reported in `CROSS_ENV_RESULTS.md`; it must not be used to retune the frozen profiles. The superseding property diagnostic records one failed condition: E2's circular output has unit radius but does not meet the real-stream phase-order threshold. Remote full and non-stationary adaptation remain planned and unverified.

## Production validity and storage addendum

The production extension retains fixed-full/lr-tune/lr-eval/norm-scaled at exactly 1000/1500/1000/700 runs. A run is scientifically complete only when its config hash, Git commit, schema, identity, required compact files, finite/extreme checks, and `runtime_validity.json` all agree. Finite values above `1e12` are invalid rather than silently accepted; the threshold is not relaxed, and no clipping or fallback condition is permitted.

Whitening uses causal covariance shrinkage and smoothed matrix refresh as second-order estimator regularization. `gaussian_moment` uses causal exponential output moments on the same time scale as its changing shape parameters and remains an exploratory transform without a distributional guarantee. These changes address estimator-time-scale leverage without changing the controller update, deleting failures, or feeding diagnostics back to the agent.

Formal output uses compact v2 run partitions: online scalar/final-window summaries, strided typed traces, decision/event traces, optional explicitly selected full diagnostics, and model/transform state. Aggregation never materializes duplicate monolithic raw trace tables. A real all-environment/all-condition storage pilot calibrates the 4200-run result, full-package, analysis-core, and peak estimates before production execution.
