# Poster Outline

## RL question

**Do prescribed statistical and geometric properties of predictive features improve strictly streaming RL under partial observability?**

## Why it matters

Predictive features are often judged by loss or covariance summaries, but an online control agent needs small latent directions to remain accessible and learnable.

## Environment

Continuing T-maze diagram: transient cue, delayed echo, five aliased corridor states, identical junction observation, cue-dependent action, one-step outcome, immediate next trial.

## Method

Single transition stream → fixed trace → ten linear GVFs → causal property transform → linear SARSA(lambda). Add a bold “no replay, no deep learning, no future statistics” strip.

## Feature properties

Raw, RMS-matched, standardized, decorrelated, whitened, exploratory moment shaping, and unit sphere; show compact equations and explicitly label whitening as second-order only.

## Baselines

Observation-only, fixed trace-only, and hidden-cue oracle.

## Main plots

1. Property diagnostic before/after panel.
2. Final accuracy with mean ± SEM and n.
3. Learning curves.
4. Isotropy versus control scatter.
5. Cue decodability by corridor position.
6. Covariance eigenvalue spectra.

## Result or failure

Pilot: oracle learns; predictive conditions remain near chance within uncertainty. Whitening improves isotropy/effective rank without a clear control gain. This is a candidate counterexample, not a final conclusion.

## Interpretation

Generic geometric regularity can differ from task-aligned accessibility and online optimization. Perfect offline cue decoding for trace-only does not guarantee that the fixed online controller exploits it at the tested budget.

## Reproducibility

Show branch/commit, 3 seeds × 20k pilot, result path, 30/30 manifests, full command guard, tests count, and QR/link to repository. Mark 20-seed remote experiments as planned/unrun.
