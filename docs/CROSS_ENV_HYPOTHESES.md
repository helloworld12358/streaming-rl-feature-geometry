# Preregistered Cross-Environment Hypotheses

**Timestamp:** 2026-07-13, Asia/Shanghai

**Status before first cross-environment multi-seed pilot:** working tree dirty only with the normative extension specification and preregistration documents; no cross-environment pilot has run.

**Core base:** `f880b4977278016c25f3180b10a2206115be99f2`

**Frozen implementation checkpoint before pilot:** `f40284c` with clean working tree; no cross-environment multi-seed result had been generated.
**Pre-pilot property gate:** `results/diagnostics/cross_environment/preregistered-20260713-v1`, 17/17 synthetic and real-stream checks passed.

These hypotheses are directional research expectations, not acceptance tests. Environment correctness, property fulfillment, uncertainty, and negative results take priority over confirmation.

## H1. Universal conditioning hypothesis

RMS matching, marginal standardization, or whitening may reduce update-scale variation and improve linear online optimization across multiple environments. This is not expected to guarantee better return.

## H2. Geometry-matching hypothesis

A prior aligned with the latent-state geometry should outperform mismatched generic priors more consistently than any one universal prior: simplex for binary categories, circle for ring phase, blocks for identity × phase, and anisotropy for position × velocity.

## H3. Binary/multimodal hypothesis

Gaussian-inspired moment shaping may reduce categorical separation or confidence magnitude in E1, so improved moment errors may coexist with worse cue margin or control.

## H4. Circular hypothesis

The causal circular representation should reduce held-out phase error and improve neighborhood preservation/action accuracy in E2 relative to generic whitening or Gaussian-inspired shaping.

## H5. Product-structure hypothesis

In E3, a blockwise categorical × circular representation should preserve loop identity and within-loop phase better than one global covariance prior, improving joint-state decoding and control.

## H6. Continuous-state hypothesis

Gaussian-inspired or prescribed anisotropic covariance priors are more plausible in E4 than in categorical tasks; M4 should preserve velocity-sensitive directions better than full isotropic whitening.

## H7. Rank-not-enough hypothesis

Across conditions/environments, higher effective rank will not be sufficient for better final-window return or accuracy.

## H8. Isotropy-not-enough hypothesis

Lower isotropy error will not be sufficient for better task-relevant decodability or control; E1 is already a pilot counterexample candidate.

## H9. Magnitude-information hypothesis

Unit sphere may discard event proximity, confidence, or hidden-velocity magnitude, harming at least one continuous or phase-dependent task despite constant norm.

## H10. Sparsity hypothesis

A fixed top-k sparse transform may reduce interference in categorical E1/E3 but harm smooth circular phase and hidden-velocity interpolation in E2/E4.

## H11. Blockwise-prior hypothesis

Different predictive subspaces require different geometry; global whitening should be less reliable than semantic blockwise transformation in E3.

## H12. Adaptation hypothesis

Well-conditioned representations may recover faster after E1's remote-only corridor change, but normalization/transform drift may delay recovery. This hypothesis is evaluated only by remote full runs.

## Frozen interpretation rules

- A condition is interpreted only after its synthetic and real-stream property checks pass.
- Oracle must learn and observation-only must remain meaningfully worse before an environment enters the scientific matrix.
- Probes use deterministic train/test splits and never update the agent.
- Three-seed local results are pilot observations with mean ± SEM, not final inference.
- No parameter is changed after viewing cross-environment pilot outcomes to manufacture a positive result.
- E5 and low-priority ablations cannot delay E1–E4 required work.
