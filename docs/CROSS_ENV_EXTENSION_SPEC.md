# Codex Goal Specification
## Cross-Environment Predictive Representation Priors for Streaming RL

This file is the normative specification for the next Codex goal.

The project repository already contains a core implementation centered on a continuing partially observable T-maze, online predictive features/GVFs, linear control, representation transforms, local smoke/pilot profiles, and remote full-run tooling.

The purpose of this goal is to extend that existing project across multiple task environments with different latent-state structures, while preserving the original scientific question:

> In strict streaming reinforcement learning under partial observability, do prescribed statistical, distributional, geometric, temporal, or algebraic properties of predictive representations improve agent-state construction and downstream online control?

Do not replace the existing project, discard valid work, or redesign the project around deep learning, world models, replay, or offline representation fitting.

---

# 0. Required execution behavior

Work continuously until all completion criteria in this file are satisfied or a genuine external blocker is encountered.

Do not stop after producing a plan.

For every phase:

1. inspect the current implementation;
2. make the smallest necessary changes;
3. run the relevant tests;
4. diagnose failures;
5. fix implementation errors;
6. record negative scientific results rather than hiding them;
7. create a Git checkpoint after the phase is verified.

Do not use destructive Git commands.

Forbidden Git operations include:

- `git reset --hard`;
- `git clean -fd`;
- `git push --force`;
- rewriting `main`;
- deleting `.git`;
- replacing the existing `origin`;
- overwriting unknown user files.

Do not read, print, or commit passwords, tokens, SSH private keys, browser cookies, or other credentials.

Do not run the remote full experiment on the local machine.

---

# 1. Repository and branch handling

First run:

```text
git status
git branch --show-current
git remote -v
git log -1 --oneline
git diff --stat
```

Verify that the repository is:

```text
https://github.com/helloworld12358/streaming-rl-feature-geometry.git
```

Read all relevant existing files, including:

- `README.md`;
- `AGENTS.md`;
- existing research specifications;
- existing proposal documents;
- source code;
- tests;
- configs;
- scripts;
- result summaries;
- workflows;
- implementation reports.

Do not assume the existing implementation is correct merely because tests pass.

Audit scientific validity, streaming causality, result isolation, and documentation consistency.

Use the existing feature branch if the current core task created one and it is the intended development branch.

Otherwise create:

```text
codex/cross-environment-representation-priors
```

If this goal runs in a Git worktree, preserve the worktree setup and use its assigned branch.

Before substantive changes, create:

```text
docs/CROSS_ENV_EXTENSION_RECONCILIATION.md
```

It must state:

- the current core implementation status;
- the core commit SHA;
- currently implemented environments;
- currently implemented representation transforms;
- currently implemented experiment profiles;
- current test status;
- current local pilot status;
- missing requirements from this specification;
- planned minimal changes;
- scientific risks;
- engineering risks.

Continue implementation after writing this document.

---

# 2. Fixed scientific constraints

The project remains a Core RL streaming-state-construction study.

Hard constraints:

- strict streaming updates;
- no replay buffer;
- no experience replay;
- no minibatch training;
- no offline fitting of the agent representation;
- no batch least-squares control;
- no future-statistics leakage;
- no future-observation leakage;
- no deep learning;
- no neural networks;
- no PyTorch;
- no TensorFlow;
- no JAX;
- no learned deep world model;
- no autoencoder;
- no JEPA neural architecture;
- no target network;
- no GPU training;
- no episodic retrieval memory;
- no model-based planning;
- no large benchmark suite.

Use:

- Python;
- NumPy;
- SciPy when mathematically necessary;
- pandas;
- matplotlib;
- pytest;
- Python standard library.

Predictors and controllers must remain linear or small explicitly defined online recursions.

Every transition may be consumed once by the agent update.

Diagnostic data may be saved for post-run analysis, but it must not feed back into agent learning.

---

# 3. Refined cross-environment research question

The expanded research question is:

> How does the match or mismatch between latent-state structure and prescribed predictive-representation geometry affect strict streaming RL performance under partial observability?

Study the interaction:

```text
latent-state structure
× predictive feature bank
× prescribed representation property
→ property fulfillment
→ hidden-state accessibility
→ optimization stability
→ control performance
```

The project must distinguish:

1. task-agnostic priors;
2. task-matched priors;
3. property fulfillment;
4. predictive accuracy;
5. hidden-state decodability;
6. online-control utility.

Do not infer that a representation is useful merely because it has:

- lower covariance condition number;
- higher effective rank;
- lower isotropy error;
- lower skewness;
- Gaussian kurtosis;
- constant norm;
- greater sparsity.

Actively search for counterexamples.

---

# 4. Required environment family

Keep the existing continuing T-maze as Environment E1.

Add E2, E3, and E4.

E5 is optional and may be implemented only after E1–E4, tests, smoke runs, and required pilots are complete.

All environments must have:

- continuing interaction;
- partial observability;
- no training reset;
- lightweight CPU execution;
- deterministic reproducibility under a fixed seed;
- explicit latent variables available only to diagnostics and oracle baselines;
- a clear control objective;
- a clear reason memory or predictive state is required.

## E1. Continuing partially observable T-maze

Preserve and verify the existing implementation.

Latent structure:

```text
binary categorical cue
```

Expected geometry:

```text
two clusters / bimodal / categorical belief
```

Required variants:

- stationary corridor length;
- one remote-only corridor-length change.

Required baselines:

- observation-only;
- oracle cue;
- trace-only;
- raw predictive features.

## E2. Aliased continuing Ringworld

Implement a continuing cycle with positions:

```text
s_t ∈ {0, …, N-1}
```

Actions may include:

- clockwise;
- counterclockwise;
- optional stay.

Most positions must share aliased observations.

A small number of landmarks may be distinguishable.

The control objective must require inferring hidden phase or position from observation history.

Possible control objective:

- choose an action at designated decision positions based on hidden phase;
- navigate toward a moving or fixed rewarding region whose observation is aliased;
- maintain a phase-dependent action pattern.

Choose the simplest objective that:

- is not solvable from the current observation alone;
- supports a clear oracle;
- supports linear streaming control;
- yields a measurable accuracy or reward signal.

Latent structure:

```text
circular phase on S^1
```

Required task-matched prior:

```text
circular representation
z_phase = [cos(theta_hat), sin(theta_hat)]
```

The circular representation must be derived from online predictive features without using true latent phase in the agent.

True phase may be used only for diagnostics and oracle evaluation.

## E3. Continuing two-loop aliased POMDP

Implement two recurrent loops with:

- different loop identities;
- different loop lengths or transition rhythms;
- substantial observation aliasing;
- a brief loop-identity cue or a transition pattern that must be remembered;
- a decision whose correct action depends on loop identity, phase, or both.

Latent structure:

```text
discrete loop identity × circular phase
```

The environment must support separate diagnostics for:

- identity decodability;
- phase estimation;
- control accuracy.

Required task-matched prior:

```text
block-structured representation
[categorical/simplex identity block,
 circular phase block]
```

The identity block must not use the true identity label during agent training.

The phase block must not use the true phase during agent training.

Latent labels may be used only for diagnostics and oracle baselines.

## E4. Hidden-velocity continuing control

Implement a lightweight one-dimensional partially observed linear dynamical system.

Suggested dynamics:

```text
x_(t+1) = x_t + v_t
v_(t+1) = rho * v_t + a_t + noise_t
```

The agent observes position but not velocity.

Use a small discrete action set such as:

```text
{-a, 0, +a}
```

Use a continuing quadratic-style reward or a discretized control objective that is compatible with linear online control.

Example:

```text
r_t = -x_t^2 - eta * v_t^2 - xi * a_t^2
```

The environment must remain numerically bounded through explicit, documented environment dynamics, not hidden clipping inside the agent.

Latent structure:

```text
continuous approximately ellipsoidal/Gaussian hidden state
```

Required task-matched prior:

```text
anisotropic Gaussian / prescribed covariance representation
```

This condition must distinguish:

- good conditioning;
- full covariance isotropy;
- task-relevant anisotropy.

## E5. Optional switching hidden-context task

Implement only if all required work for E1–E4 is complete.

Use a small number of hidden modes:

```text
c_t ∈ {1, …, K}
```

Each mode changes a simple transition or reward mapping.

Modes persist for a while and then switch.

Latent structure:

```text
mixture / piecewise stationary categorical context
```

Required task-matched prior:

```text
online mixture or simplex representation
```

Do not let E5 delay the required deliverables.

---

# 5. Predictive feature construction

Preserve the existing fixed-trace plus online-GVF approach unless the repository already contains a simpler correct abstraction.

All environments must expose a common interface for:

- observation;
- action;
- reward;
- diagnostic latent state;
- phase or trial metadata;
- event cumulants;
- continuation values;
- oracle features.

The agent representation must be generated only from causal observation/action/reward history and online predictive learning.

For each environment define a small fixed predictive bank.

The bank may include:

- future landmark occupancy;
- future event indicators;
- future cue echo;
- future rewarding-event occupancy;
- future negative-event occupancy;
- different fixed horizons;
- phase-related sinusoidal event predictions;
- outcome predictions.

Do not design a different feature dimension solely to favor one transform.

Record each GVF or prediction:

- name;
- cumulant;
- continuation;
- effective horizon;
- interpretation;
- prediction error.

Provide:

- a compact task-relevant bank;
- a mixed redundant bank.

The mixed bank is used to test correlation, redundancy, scale mismatch, and rank collapse.

---

# 6. Representation families

All task-agnostic conditions must use the same raw predictive vector within a given environment.

Let:

```text
g_t = raw predictive vector
z_t = transformed control representation
```

Required task-agnostic conditions:

## R0. Raw

```text
z_t = g_t
```

## R1. RMS-matched raw

Causally match global feature scale without changing correlation structure.

## R2. Marginal standardization

Causal zero-mean/unit-variance transform with documented warm-up.

## R3. Decorrelation

Reduce off-diagonal covariance while distinguishing this condition from whitening.

## R4. Whitening / second-order isotropy

Causally target:

```text
mean(z) ≈ 0
Cov(z) ≈ I
```

Do not call this full Gaussianity.

## R5. Gaussian-inspired online moment matching

Attempt to reduce:

- mean error;
- covariance error;
- skewness error;
- kurtosis error.

Use one simple, low-parameter, monotonic, online non-neural transform.

Do not claim exact Gaussianity.

## R6. Unit sphere

```text
z_t = g_t / (||g_t|| + epsilon)
```

## R7. Sparse

Use one transparent fixed online rule:

- top-k;
- soft threshold;
- fixed threshold.

## R8. Bounded

Use one transparent bounded transform such as:

```text
tanh(alpha * standardized_g)
```

Required task-matched conditions:

## M1. Categorical simplex / mixture-like representation

For E1 and optionally E5.

Requirements:

- nonnegative coordinates;
- normalized or explicitly interpretable categorical scores;
- no use of latent labels during agent learning;
- online updates only.

A streaming unsupervised mixture approximation is allowed if simple and stable.

Do not introduce a complex EM system if it blocks the project.

## M2. Circular representation

For E2.

Use a two-dimensional circular block inferred from predictive features.

Do not use the true phase as an agent input.

## M3. Block categorical × circular representation

For E3.

Use distinct representation blocks for identity and phase.

Apply a matched property to each block rather than one global prior.

## M4. Prescribed anisotropic covariance

For E4.

Compare:

- raw;
- standardized;
- isotropic whitening;
- a fixed or causally estimated anisotropic target covariance.

The anisotropic target must be justified without using final control reward to tune it.

Possible justifications:

- feature semantics;
- stable dynamics scale;
- predeclared variance ratio;
- analytic linear-system structure.

---

# 7. Experiment matrix

Do not run a full Cartesian product.

Use a staged matrix.

## Stage A. Core cross-environment comparison

Required environments:

- E1;
- E2;
- E3;
- E4.

Required representations:

- R0 raw;
- R1 RMS-matched;
- R2 standardized;
- R4 whitened;
- R5 Gaussian-inspired if diagnostics pass;
- R6 unit sphere;
- R7 sparse.

R3 decorrelation and R8 bounded must remain implemented and tested, but may be omitted from every local pilot if runtime is excessive.

## Stage B. Task-matched prior comparison

Compare the best-controlled task-agnostic baselines against:

- E1: M1 categorical/simplex;
- E2: M2 circular;
- E3: M3 block categorical × circular;
- E4: M4 anisotropic covariance.

## Stage C. Limited representation-bank ablation

For each environment, compare only:

- compact task-relevant bank;
- mixed redundant bank.

Do not cross every bank with every transform locally.

## Stage D. Limited horizon ablation

For at most two environments, compare:

- matched horizon;
- deliberately short horizon.

Choose environments where horizon mismatch has a clear interpretation.

## Stage E. Non-stationary extension

Remote-only:

- E1 corridor-length change;
- E5 hidden-context switch if implemented.

---

# 8. Scientific hypotheses

Before running new multi-seed pilots, create:

```text
docs/CROSS_ENV_HYPOTHESES.md
```

Include:

## H1. Universal conditioning hypothesis

RMS matching, standardization, or whitening may improve linear online optimization across multiple environments.

## H2. Geometry-matching hypothesis

A prior aligned with latent-state geometry should outperform a mismatched generic prior more consistently than a single universal prior.

## H3. Binary/multimodal hypothesis

Gaussianization may harm T-maze or hidden-context tasks by reducing categorical separation.

## H4. Circular hypothesis

Circular representation should preserve phase relationships better than a generic Gaussian prior on Ringworld.

## H5. Product-structure hypothesis

A block categorical × circular representation should outperform a single global prior on the two-loop task.

## H6. Continuous-state hypothesis

Gaussian-inspired or anisotropic covariance priors should be more plausible in hidden-velocity control than in categorical tasks.

## H7. Rank-not-enough hypothesis

Higher effective rank does not imply better control.

## H8. Isotropy-not-enough hypothesis

Lower isotropy error does not imply better task-relevant decodability.

## H9. Magnitude-information hypothesis

Unit sphere may discard event proximity, confidence, or velocity magnitude.

## H10. Sparsity hypothesis

Sparsity may reduce interference in categorical tasks but harm smooth interpolation in continuous or phase tasks.

## H11. Blockwise-prior hypothesis

Different predictive subspaces may require different geometry.

## H12. Adaptation hypothesis

Well-conditioned representations may adapt faster after a temporal change, but online normalization drift may itself slow adaptation.

Timestamp the document and record the Git commit or dirty status before the first new multi-seed pilot.

---

# 9. Metrics

Provide unified metrics and environment-specific metrics.

Unified control metrics:

- cumulative reward;
- moving-average reward;
- final-window reward;
- time to threshold;
- across-seed standard error;
- parameter norm;
- update norm;
- TD-error variance;
- divergence count.

Unified predictive metrics:

- per-GVF TD error;
- mean squared prediction error;
- event prediction error;
- calibration when meaningful.

Unified representation metrics:

- mean error;
- variance imbalance;
- mean absolute off-diagonal correlation;
- covariance eigenvalues;
- condition number;
- effective rank;
- isotropy error;
- skewness error;
- kurtosis error;
- feature-norm statistics;
- sparsity;
- transform drift.

Task-information metrics:

E1:

- cue decodability;
- cue margin;
- trial accuracy.

E2:

- phase decoding error;
- circular correlation;
- action accuracy or reward;
- neighborhood preservation on the ring.

E3:

- loop-identity decodability;
- phase error;
- joint-state decodability;
- control accuracy.

E4:

- hidden-velocity linear decoding error;
- state-estimation error;
- control cost;
- stabilization rate;
- action/update variance.

E5:

- context decodability;
- switch detection delay;
- recovery time.

Offline probes:

- are diagnostics only;
- use train/test separation;
- do not feed the agent;
- use deterministic splits;
- report generalization, not training-set fit.

---

# 10. Property-fulfillment diagnostics

Before an experimental condition may be interpreted, verify that it achieves its stated property.

Create synthetic tests and real-stream diagnostics.

Examples:

- standardization lowers mean/variance error;
- decorrelation lowers off-diagonal correlation;
- whitening lowers second-order isotropy error;
- Gaussian-inspired transform lowers skewness/kurtosis error on suitable synthetic streams;
- circular representation stays near the unit circle and preserves phase ordering;
- simplex representation is nonnegative and normalized;
- block representation maintains block semantics;
- anisotropic transform approaches its prescribed covariance;
- sparse representation lowers active-dimension count;
- bounded representation limits norms.

Save diagnostics to:

```text
results/diagnostics/cross_environment/
```

Generate a machine-readable pass/fail summary.

A failed property condition remains documented but must not be presented as a successful method.

---

# 11. Local and remote execution profiles

## Local smoke profile

For every required environment:

- one seed;
- short interaction budget;
- observation-only;
- oracle;
- raw;
- whitened;
- task-matched prior.

Purpose:

- environment transition validation;
- no leakage;
- finite outputs;
- full pipeline;
- result files;
- figures.

## Local cross-environment pilot

Use:

- seeds `[0, 1, 2]`;
- conservative interaction budgets;
- E1–E4;
- Stage A representations;
- Stage B matched prior;
- no remote-only non-stationarity;
- no full Cartesian product.

Target local runtime:

```text
no more than approximately 60 minutes
```

If estimated runtime is excessive, reduce interaction budgets before removing seeds.

Never reduce the principal pilot to one seed.

## Remote full profile

Prepare, but do not execute locally.

Suggested:

- seeds `0–19`;
- E1–E4;
- Stage A;
- Stage B;
- selected Stage C/D;
- E1 non-stationary;
- optional E5;
- CPU process-level parallelism.

Require both:

```text
RL_RUN_CONTEXT=remote
```

and:

```text
--allow-full-run
```

Missing either must produce an explicit error.

---

# 12. Implementation architecture

Extend existing abstractions rather than duplicating the project.

Create or refine common interfaces for:

- environment;
- predictive bank;
- representation transform;
- linear controller;
- config;
- runner;
- aggregation;
- plotting;
- probes.

Prefer environment registration and config-driven experiments.

Do not create a separate ad hoc runner for every environment.

Do not place all logic in one file.

Do not rename existing files unless required by a proven collision.

Maintain backward compatibility for the existing T-maze commands.

Add tests for:

- each new environment;
- environment reproducibility;
- no latent leakage;
- continuing transitions;
- action/reward validity;
- each task-matched transform;
- common runner compatibility;
- result isolation;
- seed reproducibility;
- remote full guard;
- diagnostic train/test splits.

---

# 13. Result structure

Use unique run IDs.

Suggested path:

```text
results/<profile>/<environment>/<run_id>/
```

Each run must include:

- config;
- manifest;
- commit SHA;
- branch;
- dirty status;
- exact command;
- environment;
- representation;
- bank;
- horizon;
- seed;
- interaction budget;
- system metadata;
- metrics;
- exit status;
- logs.

Do not mix stale results.

Do not overwrite existing runs silently.

Aggregate across seeds only after verifying completeness.

Do not treat failed seeds as successful.

---

# 14. Figures

Use matplotlib only.

No seaborn.

All figures must derive from actual saved metrics.

Required cross-environment figures:

1. control learning curves by environment;
2. final performance by representation and environment;
3. task-matched prior versus generic priors;
4. effective rank versus control performance;
5. isotropy error versus control performance;
6. decodability versus control performance;
7. skewness/kurtosis error versus control performance;
8. update-norm stability;
9. covariance eigenvalue spectra;
10. property-fulfillment summary;
11. environment-specific latent-geometry visualization;
12. E1 cue clusters;
13. E2 circular phase plot;
14. E3 block representation plot;
15. E4 hidden-velocity prediction plot;
16. remote-only adaptation plot template.

Report seed count and uncertainty.

---

# 15. Required documents

Create:

```text
docs/CROSS_ENV_EXTENSION_RECONCILIATION.md
docs/CROSS_ENV_HYPOTHESES.md
docs/CROSS_ENV_EXPERIMENT_MATRIX.md
docs/CROSS_ENV_IMPLEMENTATION_REPORT.md
docs/CROSS_ENV_RESULTS.md
docs/proposal_cross_environment_en.md
docs/proposal_cross_environment_zh.md
docs/CROSS_ENV_REMOTE_RUN_GUIDE_ZH.md
docs/cross_environment_paper_outline.md
docs/cross_environment_poster_outline.md
```

Do not overwrite the original proposal documents unless explicitly necessary.

The revised proposal must foreground:

> prescribed representation properties across different latent-state geometries in streaming RL

It must not claim final full results before remote experiments run.

Distinguish:

- hypotheses;
- local pilot observations;
- unverified extensions;
- remote full plan.

---

# 16. Remote execution package

Extend the existing remote scripts and configs.

Provide:

- environment-specific full configs;
- a cross-environment suite config;
- CPU worker control;
- resume-safe result isolation;
- aggregation scripts;
- log inspection;
- missing-seed detection;
- artifact packaging.

Update the Chinese remote guide with exact commands for:

- checkout of the extension commit;
- environment bootstrap;
- tests;
- smoke suite;
- stationary full suite;
- optional non-stationary suite;
- tmux;
- log inspection;
- result aggregation;
- packaging;
- `scp` download to Windows.

Do not include secrets or server-specific credentials.

---

# 17. Overnight autonomy and failure policy

This goal must not stop merely because an optional method fails.

Failure policy:

1. record the exact failure;
2. preserve logs;
3. add or improve a test;
4. attempt a bounded number of principled fixes;
5. if still failing, mark the optional condition unavailable;
6. continue required environments and methods;
7. do not silently substitute another method;
8. do not hide negative results.

Required work has priority over optional E5 and optional extra ablations.

If local pilot runtime exceeds the target:

- reduce interaction budgets;
- preserve three seeds;
- preserve E1–E4;
- preserve raw, RMS, standardized, whitened, Gaussian-inspired if valid, sphere, sparse, and matched prior;
- defer low-priority conditions to remote configs.

If a new environment is not learnable by the oracle baseline:

- stop that environment's scientific comparison;
- fix the environment before continuing.

If observation-only unexpectedly solves a task:

- inspect leakage and task design;
- do not continue until the partial-observability requirement is restored.

If task-matched prior uses latent labels during agent training:

- treat it as invalid;
- redesign it causally.

---

# 18. Git checkpoints and review

After each verified phase:

- run targeted tests;
- inspect `git diff`;
- run `git diff --check`;
- create a descriptive local commit.

Suggested phases:

1. specification and reconciliation;
2. environment framework;
3. E2 Ringworld;
4. E3 two-loop;
5. E4 hidden velocity;
6. task-matched priors;
7. experiment matrix;
8. diagnostics;
9. local smoke;
10. local pilot;
11. documents;
12. remote package.

Before pushing:

- run full tests;
- run all local smoke suites;
- verify the local pilot outputs;
- scan for secrets;
- inspect result sizes;
- verify `.gitignore`;
- verify docs match code.

Push the extension branch to the existing origin.

Create a Draft PR if authentication permits.

Do not merge automatically.

The PR must state:

- core commit used as base;
- new environments;
- new representation priors;
- local tests;
- local smoke results;
- local pilot results;
- property-fulfillment failures;
- negative findings;
- remote full status;
- exact commands.

---

# 19. Definition of done

This goal is complete only when all required items are addressed:

- [ ] core project audited;
- [ ] extension reconciliation written;
- [ ] E1 verified;
- [ ] E2 implemented and tested;
- [ ] E3 implemented and tested;
- [ ] E4 implemented and tested;
- [ ] E5 either implemented after required work or explicitly deferred;
- [ ] common environment interface works;
- [ ] common predictive-bank interface works;
- [ ] task-agnostic transforms work;
- [ ] M1 simplex/mixture-like prior works;
- [ ] M2 circular prior works;
- [ ] M3 block prior works;
- [ ] M4 anisotropic prior works;
- [ ] no latent leakage;
- [ ] property diagnostics complete;
- [ ] local smoke complete for E1–E4;
- [ ] three-seed local pilot complete;
- [ ] real figures generated;
- [ ] revised proposal written;
- [ ] Chinese explanation written;
- [ ] remote configs and scripts complete;
- [ ] remote Chinese guide complete;
- [ ] tests pass;
- [ ] Git checkpoints created;
- [ ] branch pushed or exact blocker recorded;
- [ ] Draft PR created or exact blocker recorded;
- [ ] remote full experiment explicitly marked unrun unless actually run remotely.

---

# 20. Final report format

At completion report:

1. repository path;
2. base commit;
3. extension branch;
4. final commit SHA;
5. push status;
6. Draft PR URL;
7. environments implemented;
8. representation families implemented;
9. task-matched priors implemented;
10. tests and exact counts;
11. local smoke commands and results;
12. local pilot command and runtime;
13. pilot result paths;
14. key property-fulfillment findings;
15. key control findings;
16. negative findings;
17. deferred items;
18. proposal paths;
19. figure paths;
20. remote guide path;
21. exact remote full command;
22. remote full status;
23. remaining blockers.

Do not claim unexecuted remote results.

Start by reading this file in full, creating the reconciliation document, and then continue actual implementation until the definition of done is satisfied or an external blocker is reached.

## Production extension addendum

The later production addendum registers `hidden_velocity_informative` without changing the original four environments, adds environment-authored hidden-velocity cost diagnostics, true-decision grouped probes, robust/failure aggregation, and three explicitly separated controller-alpha analyses (`fixed`, tuned evaluation, and causal `norm_scaled`). Formal profiles use disjoint design 200–204, tuning 100–104, evaluation 0–19, and smoke 9000–9001 seeds. Exact executable configs and completion rules are documented in `CROSS_EXTENSION_REMOTE_RUN_GUIDE_ZH.md`; they supersede no core causal/isolation constraint in this specification.
