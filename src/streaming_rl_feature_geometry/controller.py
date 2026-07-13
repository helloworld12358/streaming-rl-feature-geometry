import numpy as np
class SarsaLambda:
    def __init__(self, n_actions, feat_dim, seed=0, alpha=0.03, gamma=0.98, lam=0.8, epsilon=0.08):
        self.rng=np.random.default_rng(seed); self.n_actions=n_actions; self.w=np.zeros((n_actions,feat_dim)); self.e=np.zeros_like(self.w); self.alpha=alpha; self.gamma=gamma; self.lam=lam; self.epsilon=epsilon
    def q(self,phi): return self.w @ phi
    def act(self,phi):
        if self.rng.random()<self.epsilon: return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q(phi)))
    def update(self,phi,a,r,next_phi,next_a):
        delta=r + self.gamma*self.q(next_phi)[next_a] - self.q(phi)[a]
        self.e *= self.gamma*self.lam; self.e[a] += phi
        upd=self.alpha*delta*self.e; self.w += upd
        return float(delta), float(np.linalg.norm(upd)), float(np.linalg.norm(self.w))
