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
