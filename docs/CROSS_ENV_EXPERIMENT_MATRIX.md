# Cross-Environment Experiment Matrix

## Frozen environment/latent geometry map

| ID | Environment | Latent structure | Task-matched prior | Primary task metrics |
|---|---|---|---|---|
| E1 | Continuing T-maze | binary categorical cue | M1 simplex | trial accuracy, cue decoding/margin |
| E2 | Aliased Ringworld | circular phase S¹ | M2 circular | phase error/correlation, neighborhood, decision accuracy |
| E3 | Aliased two-loop | identity × circular phase | M3 block simplex × circle | identity/phase/joint decoding, decision accuracy |
| E4 | Hidden velocity | continuous anisotropic state | M4 prescribed covariance | velocity/state decoding error, cost, stabilization |

E5 is deferred until every required E1–E4 deliverable passes.

## Local smoke

One seed per environment with a short budget. Conditions: observation-only, oracle, raw, whitened, and the environment's matched prior. Each smoke must verify continuing transitions, oracle learnability, observation aliasing, finite outputs, isolated artifacts, and real figures.

## Local pilot (staged; no Cartesian product)

Seeds are frozen to `[0, 1, 2]`. Initial budgets are 12,000 interactions for E1–E3 and 20,000 for E4; they may be reduced uniformly before execution only if projected wall time exceeds 60 minutes. Required task-agnostic conditions are raw, RMS, standardized, whitened, unit sphere, sparse, and Gaussian-inspired only when diagnostics pass. Each environment also includes observation-only, oracle, and its matched prior. R3 decorrelation and R8 bounded remain tested but are not required in every pilot.

## Representation-bank ablation

For each environment compare compact task-relevant versus mixed redundant banks only under raw and the matched prior. Do not cross all banks with all transforms locally.

## Horizon ablation

Use E1 and E2 only: matched horizons versus deliberately short horizons, under raw and the matched prior. Defer to remote execution if local runtime threatens the principal pilot.

## Remote full

Prepare 20 seeds for E1–E4, Stage A task-agnostic conditions, Stage B matched priors, selected bank/horizon ablations, and E1 non-stationarity. Both `RL_RUN_CONTEXT=remote` and `--allow-full-run` remain mandatory. Do not run locally.

## Aggregation rules

Every environment/condition/bank/horizon/seed has a unique manifest. Aggregation occurs only after completeness checks. Failed or missing seeds remain explicit. Report mean, SEM, and n; do not pool incompatible reward units across environments without normalization or faceting.
