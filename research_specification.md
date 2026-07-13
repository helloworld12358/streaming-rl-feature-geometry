# Research specification

## Research question

For streaming RL with predictive features built from partial-observation history, do imposed feature properties such as standardization, decorrelation, covariance isotropy, or approximate Gaussian moments improve hidden-state retention, online update stability, sample efficiency, and control performance?

## Pre-registered hypotheses

* H1: Standardization can reduce feature scale mismatch.
* H2: Decorrelation and whitening can lower redundancy and covariance condition number.
* H3: Better covariance isotropy may make linear-controller online updates more stable.
* H4: More Gaussian-like moments may reduce extreme or skewed updates, but this is not guaranteed.
* H5: High effective rank or low isotropy error does not guarantee high reward.
* H6: Pursuing Gaussian structure may damage the bimodal structure induced by a binary cue.
* H7: Property constraints should help mixed/redundant GVF banks more than very low-dimensional cue-only representations.

Negative results must be retained and analyzed.

## Fairness and leakage checks

Junction observations omit cue bits; cue echo appears only during outcome after the junction action; oracle is a separate baseline; transforms use pre-update online statistics; probes are offline analysis only; all property conditions share the same GVF generator; no replay is implemented; trial boundaries do not reset learned parameters; full profile is guarded for remote execution.
