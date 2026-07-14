# Agent instructions

The primary research question is whether prescribed statistical and geometric properties of predictive features improve a strictly streaming reinforcement-learning agent under partial observability. The feature property is the experimental variable; GVF questions and the controller are held fixed unless a preregistered ablation explicitly changes them.

- Keep the implementation non-deep and linear. Do not add neural networks, replay, minibatches, target networks, world models, or future-data fitting.
- Preserve streaming causality: process each transition once, update transforms only from present/past samples, and never feed analysis probes back into the agent.
- Keep observation-only, fixed-trace, and oracle baselines isolated. Oracle state must never enter predictive-feature methods.
- Full profiles may run only when both `RL_RUN_CONTEXT=remote` and `--allow-full-run` are present. Local work is limited to tests, diagnostics, smoke, validation, and the three-seed pilot.
- Call whitening a second-order transform, not a Gaussian transform. Treat `gaussian_moment` as an exploratory, online moment-shaping extension without distributional guarantees.
- Preserve negative, null, and failed results. Do not tune after seeing pilot outcomes merely to obtain a positive result.
- Make the smallest maintainable change. Do not rename existing files without a concrete need, rewrite history, delete prior evidence, or mix stale result directories.
- Add or update tests for causal order, finiteness, reproducibility, isolation, and guards. Run the full suite plus smoke before handoff.
- Keep README, research specification, implementation report, proposals, and remote guide consistent with the executable configs and commands.
- Never store credentials or large raw result artifacts in Git.

## Collaboration workflow

Web ChatGPT handles research analysis, requirement decomposition, cloud-operation explanation, log/result analysis, and GitHub/PR/commit review, then turns them into explicit Codex tasks. Local Codex reads and edits the repository, updates scripts/configuration/documentation, runs explicitly requested checks, commits and pushes, and returns changed files, test evidence, and the commit SHA. Operational instructions belong in repository run guides and the final structured handoff. Do not assume access to the user's web ChatGPT session; exchange state through the GitHub branch, repository documents, and the handoff while preserving every research constraint above.

## Pre-existing worktree changes

- A dirty worktree is not by itself a reason to stop.
- First inspect `git status`, `git diff`, and `git diff --cached`, and classify the existing changes.
- Changes clearly left by an earlier Codex execution of the same Goal, or explicitly identified by the user as prior Codex work, are authorized for in-place continuation.
- Preserve valid content, reconcile it with the current Goal, test it, and include it in the intended commit.
- Do not repeatedly request confirmation for the same already-authorized Codex leftovers.
- Leave clearly unrelated changes untouched and exclude them from the task commit.
- Stop only when provenance is genuinely ambiguous, merge conflicts exist, credentials are present, or continuing would require destructive Git operations.
- Never use `git reset --hard`, `git clean`, force push, or discard unknown changes without explicit user authorization.
