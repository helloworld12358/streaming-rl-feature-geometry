# Proposal: Prescribed Representation Priors Across Latent-State Geometries in Streaming RL

## Motivation

Predictive features are often judged by rank, isotropy, marginal moments, or visual organization. A streaming controller, however, needs task-relevant information in a usable scale and geometry. This project asks whether prescribing those properties helps a fixed linear online agent across categorical, circular, product, and continuous latent structures.

## Research question

When observations are partially aliased and predictive questions/controller rules are held fixed, which causal representation priors improve conditioning, state information, update stability, learning speed, or control? When do apparently desirable geometric properties fail?

## Method

The agent is strictly streaming and non-deep: fixed leaky traces, linear TD(0) predictions, one prescribed transform, and linear SARSA(lambda). Each transition is processed once. There is no replay, minibatch, target network, future covariance, or diagnostic feedback.

The environments are continuing T-maze (binary cue), Ringworld (circular phase), two-loop (identity × phase), and hidden velocity (continuous state). Baselines isolate observation-only and oracle access. Generic properties include scale matching, marginal standardization, decorrelation, whitening as a second-order transform, exploratory online moment shaping, unit norm, sparsity, and boundedness. Task-matched priors are simplex, circle, block simplex × circle, and prescribed anisotropy.

## Hypotheses and falsification

Conditioning methods may transfer across tasks, but task-matched geometry should help only when the predictive bank actually recovers the needed semantics. Higher rank or lower isotropy error alone should be insufficient. Unit norm and sparsity may discard magnitude or redundancy. A method fails its interpretation gate if it does not fulfill its named property on synthetic and real-stream data.

## Current local pilot evidence

The completed 3-seed pilot is diagnostic, not final. E3 showed positive signals for standardization, whitening, exploratory moment shaping, and the block prior. E1 was largely null; E2 effects were uncertain; E4 often favored raw features. The circular prior achieved unit radius but failed the real-stream phase-order threshold. These negative and failed results are retained rather than tuned away.

## Planned remote confirmation

Prepared remote-only suites use 20 seeds for the main E1-E4 matrix, limited compact/mixed and matched/short ablations, and one E1 corridor-length change. Both `RL_RUN_CONTEXT=remote` and `--allow-full-run` are mandatory. No remote full result currently exists, so adaptation and confirmatory cross-environment conclusions remain unverified.

## Expected contribution

The contribution is a controlled map from property fulfillment to predictive information and online control—not a claim that one geometry is universally best. Counterexamples where rank, isotropy, unit radius, or moment shaping fail to help are first-class outcomes.

## Production analysis addendum

The confirmatory plan now reports long-tailed seed distributions and preregistered catastrophic failures, separates fixed-alpha from tuning-seed-selected alpha and causal norm-scaled alpha, and adds a minimally modified informative hidden-velocity environment. T-maze and Two-loop interpretations include group-held-out probes at true decision times. These additions diagnose optimization confounding and information availability; they do not authorize post-evaluation tuning or change the linear streaming hypothesis.
