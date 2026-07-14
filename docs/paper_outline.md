# Paper Outline

## 1. Introduction

Motivate predictive state as an online RL problem under partial observability. State the main question: can prescribed task-agnostic feature properties improve control, and when do geometry metrics mislead?

## 2. Streaming agent-state construction

Define single-use transitions, causal update order, fixed trace memory, no replay/future statistics, and the relationship between predictive state and SARSA(lambda).

## 3. GVFs and predictive features

Define the ten semantic GVFs, two horizons, fixed question bank, cumulants/continuations, and why prediction accuracy alone is insufficient.

## 4. Prescribed feature properties

Give equations and causal estimators for raw, RMS, standardized, decorrelated, whitened, Gaussian-inspired moment shaping, and unit sphere. Separate second-order isotropy from Gaussianity.

## 5. Experimental setup

Describe continuing T-maze, isolated baselines, controller, seeds/budgets, stationary/non-stationary protocols, preregistered hypotheses, held-out probes, uncertainty, and full-run guard.

## 6. Results

Report property diagnostics first, then prediction, control, stability, and adaptation. Treat the three-seed pilot as engineering evidence and reserve inference for the 20-seed full study.

## 7. Property–performance relationship

Relate effective rank, isotropy, correlation, cue decoding/margin, transform drift, and update norms to return. Test whether geometry changes mediate or merely accompany control changes.

## 8. Counterexamples

Foreground cases where whitening improves isotropy without improving accuracy, trace/cue information is decodable but not exploited, unit norm removes magnitude, or Gaussian-inspired shaping conflicts with categorical state.

## 9. Limitations

Discuss fixed linear questions/controller, one environment family, finite horizons, off-policy risk, small pilot, multiple metrics, step-size interactions, and absence of remote results until actually run.

## 10. Conclusion

Summarize what properties were achieved, which helped optimization/control, and why task-relevant information—not generic geometry alone—must guide streaming agent-state design.
