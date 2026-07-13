import numpy as np

class OnlineMoments:
    def __init__(self, d, eps=1e-4):
        self.d=d; self.eps=eps; self.n=0; self.mean=np.zeros(d); self.M2=np.zeros((d,d)); self.m2=np.zeros(d); self.m3=np.zeros(d); self.m4=np.zeros(d)
    def cov(self): return self.M2 / max(1,self.n-1)
    def var(self): return self.m2 / max(1,self.n-1)
    def update(self,x):
        x=np.asarray(x,float); self.n+=1; delta=x-self.mean; old=self.mean.copy(); self.mean += delta/self.n; delta2=x-self.mean
        self.M2 += np.outer(delta,delta2)
        # stable enough online raw central recompute approximation via EW-free recurrences for diagnostics
        self.m2 += delta*delta2; self.m3 += delta2**3; self.m4 += delta2**4

class FeatureTransform:
    def __init__(self, kind, d, eps=1e-3, update_every=10):
        self.kind=kind; self.stats=OnlineMoments(d, eps); self.eps=eps; self.update_every=update_every; self.W=np.eye(d)
    def _refresh(self):
        C=self.stats.cov()+self.eps*np.eye(self.stats.d); vals,vecs=np.linalg.eigh(C); vals=np.maximum(vals,self.eps)
        if self.kind=="decorrelated": self.W=vecs.T
        else: self.W=(vecs @ np.diag(1/np.sqrt(vals)) @ vecs.T)
    def transform(self,g):
        if self.kind in ("observation", "oracle"): return np.zeros(0)
        if self.kind=="raw": z=g.copy()
        elif self.kind=="standardized": z=(g-self.stats.mean)/np.sqrt(self.stats.var()+self.eps)
        elif self.kind in ("decorrelated","whitened","gaussian"):
            if self.stats.n % self.update_every == 0: self._refresh()
            z=self.W @ (g-self.stats.mean)
            if self.kind=="gaussian": z=np.sign(z)*np.sqrt(np.abs(z)+self.eps)  # monotone signed power; not full Gaussianization
        else: raise ValueError(self.kind)
        self.stats.update(g); return np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

def rep_metrics(Z):
    Z=np.asarray(Z,float)
    if Z.ndim!=2 or Z.shape[0]<3 or Z.shape[1]==0: return dict(mean_error=0, variance_imbalance=0, mean_abs_corr=0, effective_rank=0, isotropy_error=0, condition_number=0, skewness_error=0, kurtosis_error=0, norm_mean=0, norm_var=0)
    mu=Z.mean(0); C=np.cov(Z,rowvar=False)+1e-9*np.eye(Z.shape[1]); vals=np.linalg.eigvalsh(C); vals=np.maximum(vals,1e-12); p=vals/vals.sum(); er=float(np.exp(-(p*np.log(p)).sum()))
    std=Z.std(0)+1e-9; corr=np.corrcoef(Z,rowvar=False); off=corr[~np.eye(Z.shape[1],dtype=bool)]
    skew=np.mean(((Z-mu)/std)**3,0); kurt=np.mean(((Z-mu)/std)**4,0); norms=np.linalg.norm(Z,axis=1)
    return dict(mean_error=float(np.linalg.norm(mu)), variance_imbalance=float(vals.max()/vals.min()), mean_abs_corr=float(np.mean(np.abs(off))), effective_rank=er, isotropy_error=float(np.linalg.norm(C-np.eye(Z.shape[1]),ord='fro')), condition_number=float(vals.max()/vals.min()), skewness_error=float(np.mean(np.abs(skew))), kurtosis_error=float(np.mean(np.abs(kurt-3))), norm_mean=float(norms.mean()), norm_var=float(norms.var()))
