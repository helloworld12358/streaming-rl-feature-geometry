# Cross-Environment Local Results

## Evidence boundary

These are local three-seed pilot results, not final inference. No hyperparameter was changed after reading pilot performance. E4 reward is not pooled with categorical-task accuracy. Remote 20-seed full experiments have not been run.

## Executed batches

| Batch | Result path | Runs | Manifest | Clean commit |
|---|---|---:|---|---|
| Property diagnostics v1 | `results/diagnostics/cross_environment/preregistered-cross-20260713-v1` | 28/28 checks | `ok`, but phase-order check was not yet present | `f40284c` |
| Property diagnostics v2 | `results/diagnostics/cross_environment/preregistered-cross-20260713-v2` | 29/30 checks | `failed`, retained | `ca9e884` |
| Cross smoke | `results/cross_smoke/preregistered-cross-smoke-20260713-v1` | 20/20 | `ok` | `f18cc36` |
| Main pilot | `results/cross_pilot/preregistered-cross-pilot-20260713-v1` | 120/120 | `ok` | `408dcf6` |
| Compact bank | `results/cross_pilot_compact/preregistered-cross-pilot-compact-20260713-v1` | 24/24 | `ok` | `f18cc36` |
| Same-budget mixed bank | `results/cross_pilot_bank_mixed/preregistered-cross-pilot-bank-mixed-20260713-v1` | 24/24 | `ok` | `55fb480` |
| Short horizon | `results/cross_pilot_short_horizon/preregistered-cross-pilot-short-horizon-20260713-v1` | 12/12 | `ok` | `f18cc36` |
| Same-budget matched horizon | `results/cross_pilot_matched_horizon/preregistered-cross-pilot-matched-horizon-20260713-v1` | 12/12 | `ok` | `55fb480` |

All result directories are ignored by Git and remain isolated. Re-aggregation with the current validator found the expected number of unique environment/condition/seed summaries.

The main pilot's recorded command was:

```powershell
.\.venv\Scripts\python.exe scripts\run_cross_experiment.py --config configs\cross_pilot.json --workers 4 --run-name preregistered-cross-pilot-20260713-v1
```

Its top-level manifest reports 466.74 seconds elapsed (7 minutes 46.74 seconds), clean commit `408dcf6`, 120 unique runs, and 1,680,000 total interactions.

## Property fulfillment

The superseding v2 diagnostic passed finite-output checks and the named RMS, standardization, decorrelation, whitening, unit-sphere, top-k sparsity, boundedness, Gaussian-moment, simplex, block, and anisotropic checks. Whitening reduced synthetic second-order isotropy error from 5.1288 to 0.0741. The exploratory moment transform reduced positive-skew error from 2.0089 to 0.9470 and heavy-tail kurtosis error from 30.2110 to 3.1385; this is not a distributional guarantee.

The circular transform passed synthetic unit radius and phase order (`0.999946`) and real-stream unit radius, but failed real E2 phase order: alignment was `0.2406` versus `0.0098` after deterministic shuffling, below the fixed `0.3` threshold. Therefore M2 fulfilled the circle constraint but not the intended real-stream ordering property and is a failed/partial method in this pilot.

## Smoke

The smoke completed 20/20 runs and generated 15 figures. Oracle-versus-observation sanity checks passed in every environment. Final accuracy was oracle/observation `0.955/0.520` for E1, `0.950/0.510` for E2, and `0.970/0.555` for E3. For E4, the configured mean-reward sanity metric was oracle `-0.174` versus observation `-0.311`; larger is better.

## Main pilot final performance

Values are mean ± SEM across seeds 0-2. E1-E3 report final-window accuracy; E4 reports final-window reward, where larger/less negative is better.

| Condition | E1 T-maze | E2 Ringworld | E3 two-loop | E4 hidden velocity |
|---|---:|---:|---:|---:|
| observation_only | 0.5053 ± 0.0123 | 0.5007 ± 0.0041 | 0.4713 ± 0.0033 | -0.0787 ± 0.0030 |
| oracle | 0.9793 ± 0.0035 | 0.9753 ± 0.0013 | 0.9767 ± 0.0037 | -0.0854 ± 0.0093 |
| raw | 0.5060 ± 0.0092 | 0.4933 ± 0.0029 | 0.4953 ± 0.0041 | -0.0634 ± 0.0177 |
| rms_raw | 0.5047 ± 0.0007 | 0.4927 ± 0.0047 | 0.5007 ± 0.0135 | -0.1169 ± 0.0127 |
| standardized | 0.4980 ± 0.0070 | 0.6220 ± 0.0859 | 0.6133 ± 0.0157 | -0.1214 ± 0.0421 |
| whitened | 0.5060 ± 0.0110 | 0.6133 ± 0.1194 | 0.6967 ± 0.0112 | -0.0867 ± 0.0206 |
| gaussian_moment | 0.4973 ± 0.0207 | 0.6100 ± 0.1845 | 0.7973 ± 0.0681 | -0.0878 ± 0.0141 |
| unit_sphere | 0.5033 ± 0.0103 | 0.5027 ± 0.0029 | 0.4707 ± 0.0044 | -0.0842 ± 0.0189 |
| sparse | 0.4873 ± 0.0198 | 0.4960 ± 0.0111 | 0.4713 ± 0.0066 | -0.1006 ± 0.0127 |
| matched | 0.4973 ± 0.0154 | 0.5007 ± 0.0044 | 0.6367 ± 0.0388 | -0.0977 ± 0.0105 |

The paired mean differences versus raw were near zero in E1. E2 showed positive but highly uncertain differences for standardization (`+0.129 ± 0.085`), whitening (`+0.120 ± 0.117`), and moment shaping (`+0.117 ± 0.182`). E3 showed positive paired differences for standardization (`+0.118 ± 0.012`), whitening (`+0.201 ± 0.007`), moment shaping (`+0.302 ± 0.069`), and matched (`+0.141 ± 0.035`). In E4, all listed transforms were below raw in final reward; matched was `-0.034 ± 0.022` relative to raw.

E4 deserves caution: over the whole run, oracle mean reward (`-0.110 ± 0.008`) was slightly better than observation-only (`-0.119 ± 0.007`), while raw had the best final window. This does not support a claim that the hidden-velocity oracle uniformly dominates at the pilot budget.

## Bank and horizon ablations

The comparisons below use paired seeds and identical interaction budgets.

Compact minus mixed was small for E1-E3 raw (`+0.0100 ± 0.0029`, `+0.0050 ± 0.0029`, `+0.0058 ± 0.0046`) and exactly zero for their matched outputs in this short run. E4 was uncertain: raw `-0.0157 ± 0.0184`, matched `+0.0818 ± 0.0919`. The pilot does not show a robust compact-bank advantage.

Short minus matched horizon was E1 raw `+0.0083 ± 0.0169`, E1 matched `+0.0058 ± 0.0022`, E2 raw `+0.0058 ± 0.0046`, and E2 matched `-0.1408 ± 0.1396`. Only the E2 matched condition suggests a potentially harmful short horizon, with uncertainty comparable to the effect.

## Interpretation against hypotheses

- Universal conditioning: not supported universally. E3 benefited, E1 was null, E2 uncertain, and E4 often worsened.
- Geometry matching: not supported as a universal rule. M3 helped E3 relative to raw, while M1/M2/M4 did not reliably help their tasks.
- Rank/isotropy not enough: supported as a pilot counterexample; better geometric diagnostics did not consistently produce better control.
- Magnitude information and sparsity tradeoff: unit sphere and sparse were neutral or harmful in several environments.
- Circular hypothesis: failed at the real-stream phase-order property gate despite exact unit radius.
- Adaptation: unverified; the remote non-stationary suite is prepared but unrun.

The strongest local positive signal is specific to E3, not a cross-environment law. The strongest scientific outcome is heterogeneity: fulfilling a generic or matched geometric constraint is separable from recovering the task-relevant predictive structure and from improving streaming control.
