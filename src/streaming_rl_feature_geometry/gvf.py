import numpy as np

class TraceGVFBank:
    """Fixed trace-state GVF fallback: m_t=rho m_{t-1}+B o_t, with online TD predictions."""
    def __init__(self, obs_dim, seed=0, alpha=0.04, gammas=(0.45,0.70,0.88)):
        self.rng = np.random.default_rng(seed); self.gammas = np.array(gammas); self.alpha = alpha
        self.targets = [6,7,4,"pos","neg"]
        self.defs = [(target, gamma) for gamma in self.gammas for target in self.targets]
        self.d = len(self.defs); self.rhos = np.linspace(0.35,0.92,self.d)
        self.B = self.rng.normal(0,0.35,size=(self.d, obs_dim)); self.w = np.zeros((self.d, self.d + obs_dim + 1)); self.m = np.zeros(self.d); self.prev_x = None; self.prev_g = np.zeros(self.d)
    def features(self, obs):
        self.m = self.rhos * self.m + self.B @ obs
        x = np.r_[obs, self.m, 1.0]
        g = self.w @ x
        self.prev_x, self.prev_g = x, g.copy()
        return g
    def update(self, next_obs, reward):
        if self.prev_x is None: return 0.0
        next_m = self.rhos * self.m + self.B @ next_obs
        next_x = np.r_[next_obs, next_m, 1.0]
        next_g = self.w @ next_x
        errs=[]
        for i,(target,gamma) in enumerate(self.defs):
            cumulant = float(next_obs[target]) if isinstance(target,int) else (1.0 if (target=="pos" and reward>0) or (target=="neg" and reward<0) else 0.0)
            delta = cumulant + gamma * next_g[i] - self.prev_g[i]
            self.w[i] += self.alpha * delta * self.prev_x; errs.append(delta)
        return float(np.mean(np.square(errs)))
