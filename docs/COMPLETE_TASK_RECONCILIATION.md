# Complete Task Reconciliation

**Goal archive:** `docs/COMPLETE_OVERNIGHT_MASTER_GOAL.md`

**Project:** `D:\download\GitHub\streaming-rl-feature-geometry`

**Execution branch:** `codex/cross-environment-representation-priors`

This document reconciles the full attached goal with the actual repository, ignored evidence directories, Git history, local executions, and remote status. No prior valid change or failed/null result was deleted, overwritten, or relabeled.

## Exact goal archive

| Field | Value |
|---|---|
| Source | `C:\Users\27768\Downloads\codex_complete_overnight_master_goal.md` |
| Destination | `docs/COMPLETE_OVERNIGHT_MASTER_GOAL.md` |
| Lines | 1,657 |
| Bytes | 30,161 |
| SHA-256 | `8B23DB2815F7E7A95DC37E6AAF70D1A931A5CC00AA9338AE08C6E6ECDF3C3FF4` |
| First heading | `# 今晚统一总目标：Streaming RL Predictive Representation Properties` |
| Last heading | `# 32. 立即执行方式` |

The copy was byte-for-byte verified against the source. It remains the normative completion specification.

## Starting audit and preservation

- Starting HEAD: `f880b4977278016c25f3180b10a2206115be99f2` on the requested extension branch.
- Starting uncommitted file: `docs/CROSS_ENV_EXTENSION_SPEC.md`, preserved and committed.
- Core branch `codex/predictive-feature-properties` was already pushed at the same base.
- The existing public PR #1 was an older non-draft `codex/githubrl` PR; no extension PR existed.
- All maintained source, tests, configs, scripts, workflows, documentation, Git status/history/remotes, and ignored result batches were audited before controlled writes.
- No clone, `git reset --hard`, `git clean -fd`, force push, history rewrite, credential write, result deletion, or destructive cleanup was performed.

## Stage A reconciliation

Stage A was complete and checkpointed before Stage B. It provides the continuing T-maze, isolated observation/trace/oracle baselines, fixed causal trace plus semantic linear GVFs, linear SARSA(lambda), raw/RMS/standardized/decorrelated/whitened/exploratory-moment/unit-sphere representations, held-out probes, manifests, validation, figures, and dual remote guard.

Historical evidence is preserved:

- Core diagnostics v1 and v2: 11/12, failed and retained; v3: 12/12 passing.
- Core smoke v1: historical top failure retained; v2 and final smoke: 4/4 passing.
- Core main pilot v3: 30/30, seeds 0-2 × 10 conditions × 20,000 interactions.
- Core pilot result: oracle `0.985 ± 0.006`; non-oracle conditions approximately 0.52-0.57. Whitening improved second-order geometry without clear control benefit.
- Remote core full stationary/non-stationary experiments were not run.

Sparse and bounded were named as shared properties but were not in the historical core checkpoint. They were added causally and tested on the extension branch rather than rewriting Stage A history.

## Stage B implementation

Completed components:

1. Common continuing environment protocol/registry with diagnostic-only latent state.
2. E1 adapter, E2 aliased Ringworld, E3 aliased two-loop, and E4 bounded hidden velocity; optional E5 deferred.
3. Compact/mixed fixed-trace linear TD predictive banks with exactly-once update guards.
4. R0-R8 generic representations, including sparse and bounded.
5. M1-M4 simplex/circular/block/anisotropic fixed-semantic priors with no latent-label argument.
6. Linear controller integration with explicit non-oracle boundary tests.
7. Deterministic grouped held-out phase/identity/joint-state/velocity probes that never update the agent.
8. Unified online control/prediction/stability/representation metrics and environment-specific metrics.
9. Isolated environment/condition/seed configs, manifests, logs, CSVs, weights, definitions, diagnostic samples, validation, aggregation, and Matplotlib figures.
10. Synthetic and real-stream machine-readable property diagnostics.
11. Cross smoke, three-seed main pilot, and paired same-budget bank/horizon ablations.
12. 20-seed main/compact/short/non-stationary remote configs, reusable guarded scripts, packaging, and Chinese instructions.

The original core runner and commands remain available; Stage B uses `run_cross_experiment.py` alongside them.

## Local validation and result status

### Tests and guards

- Final pre-document full suite: 72 passed in 31.37 seconds.
- Cross-runner/config/environment/guard targeted gate: 30 passed.
- Git-Bash syntax check: `bootstrap_remote.sh`, `run_full_remote.sh`, and `aggregate_remote.sh` passed.
- Local `run_full_remote.sh` and `aggregate_remote.sh` probes both refused with exit code 2; no full output directory was created.

### Cross diagnostics

- v1: 24/24 checks then present, top `ok`, but lacked the required phase-order check and is not the final diagnostic.
- v2: 29/30, top `failed`, retained. E2 unit-circle radius passes, but real-stream phase alignment `0.2406` is below the fixed `0.3` threshold (shuffled `0.0098`). No threshold was lowered.

### Cross smoke and pilots

- Smoke: 20/20, top `ok`, 15 real figures, clean commit `f18cc36`.
- Main pilot: 120/120, top `ok`, clean commit `408dcf6`.
- Compact/mixed paired bank batches: 24/24 and 24/24, both `ok`.
- Short/matched paired horizon batches: 12/12 and 12/12, both `ok`.

Main pilot finding: E3 shows positive local signals for standardization, whitening, exploratory moment shaping, and block matched prior; E1 is null; E2 is uncertain; E4 often favors raw. Matched geometry is not universally beneficial. Details and mean ± SEM tables are in `CROSS_ENV_RESULTS.md`.

These are three-seed pilot observations. They were documented after frozen execution without reward-driven retuning.

## Remote package and status

Prepared configs:

- `configs/cross_full.json`
- `configs/cross_full_tmaze.json`
- `configs/cross_full_ringworld.json`
- `configs/cross_full_two_loop.json`
- `configs/cross_full_hidden_velocity.json`
- `configs/cross_full_compact.json`
- `configs/cross_full_short_horizon.json`
- `configs/cross_full_nonstationary.json`

Exact main command on a real remote Linux machine:

```bash
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 4 --run-name cross-full-<COMMIT_SHA>
```

The shell launcher and Python runner both require the environment variable and flag. Full runs have **not** been executed locally or remotely during this goal. The E1 adaptation hypothesis therefore remains unverified. See `docs/CROSS_ENV_REMOTE_RUN_GUIDE_ZH.md`.

## Documentation map

- Normative goal/spec: `COMPLETE_OVERNIGHT_MASTER_GOAL.md`, `CROSS_ENV_EXTENSION_SPEC.md`.
- Frozen hypotheses/matrix: `CROSS_ENV_HYPOTHESES.md`, `CROSS_ENV_EXPERIMENT_MATRIX.md`.
- Implementation/results: `CROSS_ENV_IMPLEMENTATION_REPORT.md`, `CROSS_ENV_RESULTS.md`.
- Proposals: `proposal_cross_environment_en.md`, `proposal_cross_environment_zh.md`.
- Remote instructions: `CROSS_ENV_REMOTE_RUN_GUIDE_ZH.md`.
- Manuscript planning: `cross_environment_paper_outline.md`, `cross_environment_poster_outline.md`.

README, research specification, the core implementation report, extension reconciliation, proposals, configs, executable commands, and remote status are synchronized.

## Git checkpoints completed before the final publication gate

- `19aaff8`: preregister extension specification.
- `d9097f5`: common E1-E4 environment family.
- `56ee135`: predictive banks and matched priors.
- `68c9ff2`: exact master-goal archive and initial reconciliation.
- `22c63ff`: cross-environment runner.
- `c65561b`, `1429b49`: empty-table, oracle-isolation, and causal analysis fixes.
- `f40284c`, `408dcf6`: diagnostics and frozen pilot hypotheses.
- `fbb3e3e`: staged remote package and E1 non-stationarity.
- `f18cc36`: sparse/bounded validation fix.
- `55fb480`: same-budget bank/horizon pairing.
- `ca9e884`: circular phase-order audit.

Final QA, documentation commit(s), push, and Draft PR are reported at handoff only after each action actually succeeds.
