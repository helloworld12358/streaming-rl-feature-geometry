from dataclasses import dataclass
import numpy as np

A_LEFT, A_RIGHT = 0, 1

@dataclass
class StepInfo:
    trial: int; phase: str; cue: int; junction: bool; outcome: bool; correct: bool | None

class ContinuingTMaze:
    """Continuing partially observable T-maze with delayed cue echo after action."""
    names = ["bias_obs","cue_left","cue_right","corridor","junction","outcome","echo_left","echo_right"]
    def __init__(self, corridor_length=4, seed=0):
        self.corridor_length = corridor_length; self.rng = np.random.default_rng(seed); self.trial = -1; self.t = 0; self.reset_stream()
    @property
    def obs_dim(self): return len(self.names)
    def reset_stream(self):
        self.trial += 1; self.phase_i = 0; self.cue = int(self.rng.choice([-1,1])); self.last_correct = None; return self._obs()
    def _phase(self):
        if self.phase_i == 0: return "cue"
        if 1 <= self.phase_i <= self.corridor_length: return "corridor"
        if self.phase_i == self.corridor_length + 1: return "junction"
        return "outcome"
    def _obs(self):
        o = np.zeros(self.obs_dim, dtype=float); o[0] = 1.0; ph = self._phase()
        if ph == "cue": o[1 if self.cue < 0 else 2] = 1.0
        elif ph == "corridor": o[3] = 1.0
        elif ph == "junction": o[4] = 1.0  # identical for left/right cue; no cue leakage
        else:
            o[5] = 1.0; o[6 if self.cue < 0 else 7] = 1.0
        return o
    def step(self, action):
        ph = self._phase(); reward = 0.0; correct = None
        if ph == "junction":
            correct = (action == (A_LEFT if self.cue < 0 else A_RIGHT)); self.last_correct = correct; reward = 1.0 if correct else -1.0
        info = StepInfo(self.trial, ph, self.cue, ph == "junction", ph == "outcome", correct)
        self.phase_i += 1; self.t += 1
        if self._phase() == "outcome":
            return self._obs(), reward, False, info
        if self.phase_i > self.corridor_length + 2:
            return self.reset_stream(), reward, False, info
        return self._obs(), reward, False, info
