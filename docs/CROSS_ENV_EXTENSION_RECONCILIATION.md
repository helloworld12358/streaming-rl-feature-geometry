# Cross-Environment Extension Reconciliation

**Created:** 2026-07-13 (Asia/Shanghai)  
**Extension branch:** `codex/cross-environment-representation-priors`  
**Core base commit:** `f880b4977278016c25f3180b10a2206115be99f2`  
**Normative extension specification:** `docs/CROSS_ENV_EXTENSION_SPEC.md` (SHA-256 `2F4C49A1C029EADA13C5F1D2E0FAE158C4D4412304383673EC0FFEAA0F36CE5B`)

## Current core implementation status

The core project is complete and pushed on `codex/predictive-feature-properties`. It provides a strictly streaming, non-deep continuing T-maze study; fixed trace memory; a semantic linear GVF bank; accumulating-trace semi-gradient SARSA(lambda); causal transforms; held-out probes; isolated results/manifests; real Matplotlib figures; local tests/smoke/pilot; and guarded remote full tooling. The final core test gate was 32 passed. Full remote experiments remain unrun, and no Draft PR was created because GitHub CLI is unavailable.

## Currently implemented environments

- **E1 continuing T-maze:** implemented and verified, including stationary length 5 and a remote-only one-time 5→9 corridor change.
- **E2 Ringworld:** missing.
- **E3 two-loop aliased POMDP:** missing.
- **E4 hidden-velocity control:** missing.
- **E5 switching context:** not implemented and deferred until all required E1–E4 work completes.

The current runner is T-maze-specific; it does not yet expose a common registered environment interface.

## Currently implemented representation transforms

- Task-agnostic: raw (R0), RMS-matched raw (R1), marginal standardized (R2), decorrelated (R3), whitened/second-order isotropic (R4), exploratory Gaussian-inspired moment shaping (R5), and unit sphere (R6).
- Missing task-agnostic: sparse (R7) and bounded (R8).
- Missing task-matched: categorical/simplex (M1), circular (M2), block categorical × circular (M3), and prescribed anisotropic covariance (M4).

All existing transforms are causal and non-neural. Core diagnostics passed 12/12 registered synthetic checks, but Gaussian-inspired shaping remains exploratory.

## Currently implemented experiment profiles

- Local: `smoke` (seed 0, 3k), `validation` (3 seeds, 10k), and `pilot` (seeds 0–2, 20k).
- Remote-only: `full_stationary` (20 seeds, 200k) and `full_nonstationary` (20 seeds, 300k), plus limited cue/horizon ablations.
- Guard: both `RL_RUN_CONTEXT=remote` and `--allow-full-run` are required.
- Missing: environment-aware smoke/pilot/full suite configs, bank variants across E1–E4, cross-environment aggregation, and environment-specific figures.

## Current test status

The stage-A final full suite reported **32 passed in 44.62 s**. Coverage includes E1 transitions/leakage/reproducibility, GVF online updates, causal/numerical transform behavior, controller sanity, isolation/manifests, full guards, same-seed reproducibility, and train/test-separated probes. No E2–E4 or matched-prior tests exist yet.

## Current local pilot status

`results/pilot/preregistered-pilot-20260713-v3` completed 30/30 runs (3 seeds × 10 conditions × 20k interactions) with zero recorded NaN, Inf, or divergence. Oracle accuracy was `0.985 ± 0.006`; non-oracle final accuracy remained about 0.52–0.57. Whitening improved isotropy/effective rank without a clear control gain. The pilot manifest records pre-publication SHA `e5f31cc`; its code tree is identical to published no-reply rewrite `7687451`.

## Requirements already satisfied and preserved

- Strict streaming causality; no replay, batch agent fitting, deep learning, future leakage, or analysis feedback.
- E1, fixed trace + online GVFs, linear controller, isolated oracle and observation/trace baselines.
- R0–R6 plus R3 tests and real property diagnostics.
- Deterministic seed handling, unique result paths, configs/manifests/logs, completeness validation, mean/SEM figures.
- Two-factor remote guard, CPU process parallelism, missing-seed detection, packaging, and Chinese remote instructions.
- Negative-result preservation and explicit distinction between whitening and Gaussianity.

## Missing requirements from the extension specification

1. A common environment protocol and registry with diagnostic latent state and oracle-only features.
2. E2 aliased Ringworld with a non-observation-solvable phase-dependent decision.
3. E3 aliased two-loop task with identity × phase diagnostics.
4. E4 bounded hidden-velocity control with continuous latent diagnostics.
5. Environment-specific compact and mixed predictive banks under one interface.
6. R7 sparse and R8 bounded transforms.
7. M1 simplex, M2 circular, M3 block, and M4 anisotropic task-matched priors without latent-label training leakage.
8. Unified reward metrics and E2/E3/E4 held-out probes.
9. Cross-environment property diagnostics and machine-readable pass/fail summary.
10. E1–E4 smoke, three-seed staged local pilot, cross-environment figures, revised proposals/results, and extension remote suite.

## Planned minimal changes

- Add a small environment protocol/registry and adapt E1 through an adapter rather than rewriting the verified class.
- Add `ringworld.py`, `two_loop.py`, and `hidden_velocity.py` with compact observation vectors, explicit diagnostic state, bounded dynamics, and reproducibility tests.
- Generalize predictive construction through fixed causal trace/event predictors while retaining the existing E1 GVF bank for backward compatibility.
- Extend `FeatureTransform` with sparse, bounded, simplex, circular, block, and anisotropic modes using only fixed semantics or causal predictive statistics.
- Add an environment-aware runner and suite aggregator alongside the existing runner; preserve every stage-A command.
- Use short oracle/observation sanity runs before any multi-seed comparison; do not proceed scientifically on an invalid task.
- Use a staged condition matrix and conservative budgets rather than a full Cartesian product.

## Scientific risks

- A task-matched representation may accidentally encode true latent labels; every matched transform needs explicit no-leakage tests and agent-input audits.
- Fixed predictive banks may be insufficient to recover phase/velocity even when an oracle succeeds.
- Offline decodability may not translate to online control, as already observed in E1.
- Gaussian-inspired shaping may be unsuitable for categorical/circular tasks; it may be disabled for interpretation if real-stream property checks fail.
- Controller step-size/feature-scale interactions can confound property comparisons; RMS control and update-norm reporting are mandatory.
- Three seeds and short budgets support engineering/pilot observations only, not final claims.

## Engineering risks

- Generalizing the runner could silently break stage-A paths or manifests; backward-compatibility tests are required.
- Environment-specific action/reward semantics need a common controller contract without hidden branching that favors one method.
- Cross-environment runs can exceed the 60-minute local target; reduce interaction budgets before seeds or required environments/conditions.
- Process-level parallel workers must not mix run directories or failed seeds.
- Rank-deficient covariance, circular angle wrap, and continuous-state bounds require explicit numerical tests.

## Execution order

1. Commit this specification/reconciliation/hypothesis checkpoint.
2. Add common interfaces plus E2, E3, and E4 in separate verified checkpoints.
3. Add task-matched priors and property diagnostics.
4. Add unified staged configs/runner/aggregation/figures.
5. Run E1–E4 smoke, then the preregistered three-seed pilot.
6. Write results/proposals/remote package, run final tests/review, push extension branch, and attempt Draft PR only if authentication tooling exists.
