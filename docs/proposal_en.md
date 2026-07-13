# Project Proposal

## Working title

**Prescribed Statistical and Geometric Properties of Predictive Features for Strictly Streaming Reinforcement Learning**

## What do you want to understand?

The central question is: **Do prescribed statistical and geometric properties of predictive features improve a streaming RL agent under partial observability?** I will test whether causal RMS matching, marginal standardization, decorrelation, whitening, exploratory moment shaping, or unit-norm projection makes a fixed predictive state more useful for online linear control. The study separates property fulfillment from control utility: a representation may have attractive covariance statistics yet discard a small task-relevant direction.

## Why is this a reinforcement-learning question?

The agent acts in one continuing stream, learns predictions and a control policy online, and receives delayed rewards. Each transition is used once by linear GVF TD, a causal transform, and semi-gradient SARSA(lambda). There is no replay, batch fitting, deep network, future-statistics access, or analysis feedback. Representation quality is judged primarily by return and decisions, not reconstruction.

## Setting and environment

The environment is a continuing partially observable T-maze. A binary cue appears at trial start, is echoed once after a delay, disappears through a five-step corridor, and determines the correct junction action. The junction observation is identical under both cues. A one-step outcome is followed immediately by the next trial. A remote-only extension changes corridor length once from 5 to 9.

## Predictive feature construction

A fixed 12-dimensional trace memory drives ten semantic linear GVFs: observation/echo, junction, positive-outcome, and negative-outcome cumulants at two continuation horizons (0.6 and 0.9). The question bank and controller are held fixed across property conditions. Every GVF, cumulant, continuation, update, and prediction error is recorded.

## Feature-property conditions and baselines

The registered conditions are raw predictions, global RMS-matched raw, causal standardized, decorrelated, whitened, exploratory Gaussian-inspired moment shaping, and unit sphere. Whitening targets second-order isotropy only; it is not called Gaussian. Baselines are observation-only, fixed-trace-only, and an isolated hidden-cue oracle. The explicit controller bias and learning settings are shared.

## Hypotheses

H1–H4 predict that scale control, lower correlation, and better conditioning can stabilize updates. H5 predicts that second-order geometry alone is insufficient. H6 predicts that whitening or moment shaping can hurt a binary cue by weakening task-relevant separation. H7 predicts that cue decodability mediates control. H8 predicts possible adaptation benefits after a remote temporal change. H9 treats online Gaussian-inspired shaping as a high-risk exploratory extension, not an assumed success.

## Metrics

Primary metrics are final-window junction accuracy, cumulative reward, time to 0.8 accuracy, and update stability. Secondary metrics include per-GVF TD error, held-out cue decodability/margin by corridor position, covariance eigenvalues, effective rank, isotropy error, correlation, condition number, skewness/kurtosis error, feature norms, transform drift, and non-finite/divergence counts. Results report mean ± standard error across independent seeds.

## Minimal local pilot result

The preregistered local pilot used seeds 0–2, 20,000 interactions per condition, commit `e5f31cc7bbed5faf1a64d393708a0b5790a41a2a`, and `results/pilot/preregistered-pilot-20260713-v3`. All 30 runs completed with zero recorded NaN, Inf, or divergence events.

| Condition | Final accuracy (mean ± SEM) | Cumulative reward (mean ± SEM) |
|---|---:|---:|
| observation-only | 0.563 ± 0.009 | 29.3 ± 43.5 |
| oracle | 0.985 ± 0.006 | 2366.0 ± 19.0 |
| trace-only | 0.567 ± 0.009 | 120.0 ± 52.3 |
| raw | 0.552 ± 0.021 | 8.0 ± 18.5 |
| RMS raw | 0.522 ± 0.026 | -10.7 ± 24.3 |
| standardized | 0.538 ± 0.021 | 4.0 ± 32.5 |
| decorrelated | 0.558 ± 0.007 | 25.3 ± 23.6 |
| whitened | 0.562 ± 0.025 | 94.0 ± 72.9 |
| Gaussian-inspired | 0.550 ± 0.040 | 42.7 ± 61.2 |
| unit sphere | 0.568 ± 0.004 | 55.3 ± 36.8 |

These are pilot observations, not final conclusions. The oracle validates task learnability; no predictive condition clearly exceeds observation-only at this budget. Whitening improved mean isotropy error from 6.01 (raw) to 2.98 and effective rank from 2.59 to 5.88, yet did not improve final control beyond uncertainty. This is the intended geometry–utility counterexample. Held-out junction cue decoding remained above chance for most predictive conditions, showing that information accessibility and online control use are distinct.

## Full experiment plan and compute need

The stationary full study uses 20 seeds × 10 conditions × 200,000 interactions. The non-stationary study uses 20 seeds × 10 conditions × 300,000 interactions with a single midpoint corridor change. Limited cue-only and short-horizon ablations test question-bank and horizon dependence. These CPU process-level runs are remote-only and require both `RL_RUN_CONTEXT=remote` and `--allow-full-run`; no GPU is needed.

## Fallback and expected contribution

If moment shaping is unstable or fails real-stream property checks, it remains documented as unavailable while raw, RMS, standardized, decorrelated, and whitened comparisons continue. If predictive control remains near chance after full runs, the negative result is still informative: generic geometric regularity is not sufficient for task-useful predictive state. The expected contribution is a reproducible, causal evaluation framework and evidence about when representation geometry aligns—or conflicts—with online RL utility.
