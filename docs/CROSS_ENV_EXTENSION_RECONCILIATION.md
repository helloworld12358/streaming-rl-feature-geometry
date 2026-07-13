# Cross-Environment Extension Reconciliation

**Created:** 2026-07-13 (Asia/Shanghai)

**Updated after local execution:** 2026-07-14 (Asia/Shanghai)

**Extension branch:** `codex/cross-environment-representation-priors`

**Core base:** `f880b4977278016c25f3180b10a2206115be99f2`

**Normative specification:** `docs/CROSS_ENV_EXTENSION_SPEC.md`

## Preserved core

The pushed core branch remains intact: continuing T-maze, fixed trace plus semantic linear GVFs, linear SARSA(lambda), isolated baselines/oracle, causal transforms, deterministic held-out probes, manifests/validation/figures, and dual-guarded remote profiles. Historical failed and null Stage A evidence was not deleted or relabeled. The extension uses a separate runner and does not break the original configs or commands.

## Completed extension implementation

- Common continuing environment protocol and registry.
- E1 adapter, E2 aliased Ringworld, E3 aliased two-loop identity × phase, and E4 bounded hidden velocity. Optional E5 remains deliberately deferred.
- Compact/mixed fixed-trace linear predictive banks with exactly-once TD updates.
- Generic raw/RMS/standardized/decorrelated/whitened/exploratory-moment/sphere/sparse/bounded representations.
- Fixed-semantic M1 simplex, M2 circle, M3 simplex × circle blocks, and M4 anisotropy with no latent-label transform input.
- Controller-boundary oracle isolation and deterministic train/test-separated task probes.
- Environment-specific control, prediction, representation, stability, and task-information metrics.
- Isolated environment/condition/seed artifacts, completeness validation, mean/SEM aggregation, and 15-data-figure reporting.
- Machine-readable synthetic and real-stream property diagnostics.
- Cross smoke, main three-seed pilot, and same-budget bank/horizon ablations.
- Four 20-seed remote full configs, a single E1 non-stationary change, reusable Linux launch/aggregate/package scripts, and a Chinese remote guide.

## Verification evidence

- Combined full test suite: 72 passed in 31.37 seconds.
- Cross smoke: `results/cross_smoke/preregistered-cross-smoke-20260713-v1`, 20/20, `ok`, clean commit `f18cc36`.
- Main pilot: `results/cross_pilot/preregistered-cross-pilot-20260713-v1`, 120/120, `ok`, clean commit `408dcf6`.
- Bank pairs: compact 24/24 and same-budget mixed 24/24, both `ok`.
- Horizon pairs: short 12/12 and same-budget matched 12/12, both `ok`.
- Guard probes: local full launcher and aggregator both exit 2 without creating a full result directory.
- Git-Bash syntax check passes for bootstrap/full/aggregate scripts.

## Failed and negative evidence

Diagnostics v1 passed its then-present 24 checks but omitted the required circular phase-order test. It is retained, not presented as final. The superseding v2 batch passes 29/30 and exits nonzero: E2 real-stream circular alignment is 0.2406 against shuffled 0.0098, below the fixed 0.3 threshold. Unit radius alone is therefore insufficient.

The main pilot is heterogeneous. E3 has positive signals for standardization, whitening, moment shaping, and the block prior; E1 is null; E2 is uncertain; E4 often favors raw. Task-matched priors are not universally beneficial. These outcomes were documented without post-outcome tuning.

## Remote status

`cross_full.json`, `cross_full_compact.json`, `cross_full_short_horizon.json`, and `cross_full_nonstationary.json` are prepared and parse successfully. Both `RL_RUN_CONTEXT=remote` and `--allow-full-run` are mandatory in the shell and Python layers. No remote full experiment has been run locally or remotely during this goal; adaptation and confirmatory conclusions remain unverified.

## Remaining publication gate

Only final repository QA, documentation reconciliation, branch push, and Draft PR creation remain after this update. Exact final commit/remote/PR identifiers belong in `COMPLETE_TASK_RECONCILIATION.md` after those actions succeed.
