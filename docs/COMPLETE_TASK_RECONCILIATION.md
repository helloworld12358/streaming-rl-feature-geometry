# Complete Task Reconciliation

**Captured:** 2026-07-13 (Asia/Shanghai)

**Repository:** `D:\download\GitHub\streaming-rl-feature-geometry`

**Origin:** `https://github.com/helloworld12358/streaming-rl-feature-geometry.git`

**Normative goal:** `docs/COMPLETE_OVERNIGHT_MASTER_GOAL.md`

**Normative goal SHA-256:** `8B23DB2815F7E7A95DC37E6AAF70D1A931A5CC00AA9338AE08C6E6ECDF3C3FF4`

This reconciliation records the actual repository state after a read-only audit of the complete attached goal, all maintained code, tests, configs, scripts, workflows, documents, ignored result batches, Git history, remote refs, and public pull-request metadata. It does not discard or relabel prior work or failed evidence.

## Goal-file copy verification

| Field | Value |
|---|---|
| Source | `C:\Users\27768\Downloads\codex_complete_overnight_master_goal.md` |
| Destination | `docs/COMPLETE_OVERNIGHT_MASTER_GOAL.md` |
| Lines | 1,657 |
| Bytes | 30,161 |
| SHA-256 | `8B23DB2815F7E7A95DC37E6AAF70D1A931A5CC00AA9338AE08C6E6ECDF3C3FF4` |
| First heading | `# 今晚统一总目标：Streaming RL Predictive Representation Properties` |
| Last heading | `# 32. 立即执行方式` |

The source and destination hashes were compared after a byte-for-byte copy and are identical.

## Git state at the start of this goal

- Starting branch: `codex/cross-environment-representation-priors`.
- Starting HEAD: `f880b4977278016c25f3180b10a2206115be99f2` (`Record core branch publication`).
- Starting uncommitted file: `docs/CROSS_ENV_EXTENSION_SPEC.md`, untracked and preserved.
- Core branch: `codex/predictive-feature-properties` at `f880b4977278016c25f3180b10a2206115be99f2`.
- Remote core ref: `origin/codex/predictive-feature-properties` at the same SHA.
- Remote `main`: `ab4f9a1998efb2736611e50be1c2e31b94d7e8f0`.
- The required origin URL is unchanged.
- No clone, `git reset --hard`, `git clean -fd`, force push, history deletion, or user-file deletion was performed.

During the audit, already-active extension work advanced the same branch through three clean checkpoints. Those changes were inspected and preserved rather than overwritten:

1. `19aaff8e268e2daac5d0358bf4d52b62bd87b8e1` — extension specification, reconciliation, hypotheses, and experiment matrix;
2. `d9097f5` — common environment protocol plus E2 Ringworld, E3 two-loop, E4 hidden velocity, and tests;
3. `56ee135` — common causal predictive bank, task-matched priors, sparse/bounded transforms, and tests.

The exact full SHA values of later checkpoints will be recorded after the final Git gate.

## Public branch and PR state

- The core branch is pushed.
- The extension branch was not present in `git ls-remote --heads origin` at the initial audit and therefore had not yet been pushed.
- GitHub's public API reported one open, non-draft PR: PR #1, head `codex/githubrl`, base `main`, URL `https://github.com/helloworld12358/streaming-rl-feature-geometry/pull/1`.
- There was no Draft PR for the core or extension branch at the audit point.
- GitHub CLI is not installed. This is not treated as success; browser/API alternatives will be attempted after the final pushed checkpoint.

## Stage A: actual implementation status

The stage-A T-maze implementation is present and its core checkpoint was created before the extension branch:

- continuing cue → corridor → junction → outcome → next-trial state machine;
- no training reset at trial boundaries;
- identical junction observations for both cues;
- delayed cue echo only after the junction action;
- isolated observation-only, trace-only, predictive, and oracle controller paths;
- fixed causal trace plus ten semantic linear TD(0) GVFs;
- linear accumulating-trace semi-gradient SARSA(lambda);
- raw, RMS raw, standardization, decorrelation, whitening/second-order isotropy, exploratory Gaussian-moment shaping, and unit sphere;
- deterministic held-out, trial-grouped diagnostic probes that do not update the agent;
- unique run directories, manifests, configs, CSVs, logs, validation, aggregation, and real Matplotlib figures;
- dual remote-full guard requiring both `RL_RUN_CONTEXT=remote` and `--allow-full-run`;
- Linux remote bootstrap/full/aggregation scripts and Chinese remote guide.

The master goal describes sparse and bounded transforms as properties shared by stages A and B. They were missing from the historical core checkpoint, but are now implemented and tested in extension checkpoint `56ee135`; this historical ordering discrepancy is preserved and documented rather than hidden by rewriting the core history.

## Stage A: tests and evidence

- Historical final core suite: 32 passed in 44.62 s.
- Current audit suite after the first E2–E4 additions: 42 passed in 24.39 s.
- Synthetic property diagnostics:
  - `results/diagnostics/preregistered-20260713`: 11/12, retained failed;
  - `results/diagnostics/preregistered-20260713-v2`: 11/12, retained failed;
  - `results/diagnostics/preregistered-20260713-v3`: 12/12, formal passing evidence.
- Smoke evidence:
  - `compliant-smoke-20260713-v1`: top manifest remains failed, although the corrected current validator can read all four child runs;
  - `compliant-smoke-20260713-v2`: 4/4, top manifest `ok`;
  - `final-core-smoke-b559734`: 4/4, top manifest `ok`;
  - `local-smoke-20260713`: older pre-final engineering evidence with a legacy manifest.
- Pilot evidence:
  - `local-pilot-20260713`: older 21-run engineering pilot;
  - `preregistered-pilot-20260713-v1`: 30 child runs but top manifest `failed`; retained as failed evidence;
  - `preregistered-pilot-20260713-v2`: actual on-disk manifest now says `ok` with 30/30 child runs;
  - `preregistered-pilot-20260713-v3`: formal 30/30, 3 seeds × 10 conditions × 20,000 interactions, top manifest `ok`.

The current validator successfully checks all three preregistered pilot directories, but this does not change v1's historical failed top manifest. Result directories remain ignored and are not mixed or overwritten.

## Stage A pilot finding

The formal v3 pilot supports only a limited observation: oracle control learned (`0.985 ± 0.006` final-window accuracy), while non-oracle conditions overlapped at roughly 0.52–0.57. Whitening improved second-order isotropy/effective rank without a clear control improvement. Trace-only cue decodability did not guarantee that the online controller used the information. These are preserved negative/null pilot findings, not final conclusions.

## Known stage-A documentation or evidence issues

1. `docs/work_log.md` says pilot v2 remained incomplete, but the current on-disk v2 top manifest is `ok`, has an end time, and has 30/30 successful child manifests. The final report must distinguish the command interruption from the eventual batch completion.
2. Older failed batches pass parts of the corrected current validator. Their original failed top manifests must not be rewritten as successful.
3. The core branch has no Draft PR. The earlier work log accurately reported the absence of `gh`, but the final goal still requires another safe Draft-PR attempt after push.
4. Remote full stationary and non-stationary experiments have not been run and must remain explicitly unrun locally.
5. Three-seed results are pilot evidence only.

## Stage B: implemented at this reconciliation point

- Normative extension specification and preregistered hypotheses/matrix.
- A common `StreamingEnvironment` protocol and registry with diagnostic-only latent/oracle fields.
- E1 adapter preserving the verified T-maze implementation.
- E2 aliased continuing Ringworld.
- E3 continuing aliased two-loop identity × phase task.
- E4 bounded hidden-velocity continuing control.
- A common compact/mixed causal fixed-trace linear predictive bank.
- R7 fixed top-k sparse and R8 causal bounded transforms.
- M1 simplex, M2 circular, M3 block simplex × circle, and M4 prescribed anisotropic transforms with no latent-label transform argument.
- Reproducibility, continuing, finiteness, aliasing, one-pass predictive-update, property, and no-latent-argument tests.
- E5 remains correctly deferred.

## Stage B: missing or not yet verified

1. Environment-aware runner/controller integration that preserves the stage-A commands.
2. End-to-end proof that predictive/matched conditions receive no latent/oracle input.
3. Oracle-learnability and observation-only partial-observability sanity checks for E2–E4.
4. Unified and environment-specific held-out metrics for phase, identity, joint state, velocity, cost, stabilization, and neighborhood structure.
5. Cross-environment result validation, isolation, aggregation, and real figure generation.
6. Machine-readable synthetic and real-stream property-fulfillment diagnostics for R7/R8 and M1–M4.
7. Cross-environment smoke for E1–E4.
8. Fixed three-seed staged local pilot for E1–E4 within the local runtime limit.
9. Cross-environment implementation/results reports, English/Chinese proposals, paper/poster outlines, and remote Chinese guide.
10. Cross-environment full configs/scripts with the existing two-factor guard.
11. Final full tests, diff/secret/result-size checks, commits, push, and Draft PR.

## Implementations to preserve

- All stage-A causal update ordering and baseline isolation.
- Existing public core history and the core commit-to-pilot mapping.
- Every failed, incomplete, null, and successful local result directory.
- The exact master-goal copy and existing extension specification.
- Backward-compatible stage-A configs, scripts, and result schema unless a tested compatibility layer requires a minimal extension.
- Current common environment, predictive-bank, and matched-prior work subject to scientific tests.

## Implementations to modify or extend

- Generalize orchestration through a separate environment-aware layer instead of destabilizing the verified T-maze path.
- Add exact semantic-index metadata rather than relying on unexplained positional assumptions for matched blocks.
- Add end-to-end causal/no-latent tests at the controller input boundary.
- Add environment-specific diagnostic probes with deterministic train/test grouping.
- Add staged configs, aggregation, figures, diagnostics, and guarded remote packaging.
- Reconcile work-log and final-report statements with actual manifests without editing historical manifests.

## Scientific risks

- Oracle success does not prove the predictive bank can recover the required state.
- A current observation or event definition can accidentally leak decision identity.
- A matched prior can encode semantics incorrectly even without accepting latent labels.
- Better rank, isotropy, moments, sparsity, or circular/simplex checks may not improve control.
- Feature scale and a shared controller step size can confound comparisons.
- Short local pilots and three seeds do not support final inference.
- E4 reward scale is not directly comparable with categorical-task accuracy and must be faceted or normalized transparently.

## Engineering risks

- Concurrent changes in the same checkout can absorb unrelated files into a checkpoint; status and diff must be checked before every commit.
- Circular imports between environment modules and the common metadata type need explicit import tests.
- Fixed semantic indices can silently mismatch a bank when event ordering, continuation ordering, or bank type changes.
- Result aggregation must not pool incompatible environments or stale directories.
- Local runtime must be reduced through interactions before seeds or required environments are removed.

## Required serial execution from here

1. Commit the exact master-goal copy and this reconciliation without absorbing unrelated work.
2. Finish and test the common runner/controller path and causal isolation.
3. Verify each E1–E4 environment with oracle and observation-only sanity runs.
4. Finish metrics, property diagnostics, isolation, aggregation, and real figures.
5. Run the cross-environment smoke only after the tests pass.
6. Freeze configs/hypotheses, then run the three-seed local pilot without outcome-driven tuning.
7. Complete proposals, reports, outlines, and the guarded remote package.
8. Run final tests and validations, inspect diffs/secrets/result sizes, commit, push, and attempt a Draft PR.
9. Keep remote full explicitly unrun on this local machine.
