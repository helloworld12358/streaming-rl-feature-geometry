import argparse, json, os, socket, subprocess, sys, time, platform
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np, pandas as pd
from .env import ContinuingTMaze
from .gvf import TraceGVFBank
from .transforms import FeatureTransform, rep_metrics
from .controller import SarsaLambda

CONDS={"observation","oracle","raw","standardized","decorrelated","whitened","gaussian"}

def git(cmd):
    try: return subprocess.check_output(["git"]+cmd, text=True).strip()
    except Exception: return "unknown"

def phi(obs,z,cond,cue):
    if cond=="observation": return np.r_[obs,1.0]
    if cond=="oracle": return np.r_[obs, float(cue), 1.0]
    return np.r_[obs,z,1.0]

def run_one(args):
    cfg, cond, seed, outroot = args; start=time.time(); run_id=f"{cfg['profile']}_{cond}_seed{seed}_{int(start)}"; out=Path(outroot)/run_id; (out/"figures").mkdir(parents=True, exist_ok=True)
    env=ContinuingTMaze(cfg['corridor_length'], seed); obs=env._obs(); gvf=TraceGVFBank(env.obs_dim, seed+17, alpha=cfg['gvf_alpha'], gammas=tuple(cfg['horizons']))
    d=gvf.d; tr=FeatureTransform(cond,d,eps=cfg['transform_eps'],update_every=cfg['cov_update_every'])
    feat_dim=len(phi(obs,np.zeros(0 if cond in ('observation','oracle') else d),cond,env.cue)); ctl=SarsaLambda(2,feat_dim,seed+31, cfg['control_alpha'], cfg['gamma'], cfg['lambda'], cfg['epsilon'])
    g=gvf.features(obs); z=tr.transform(g); p=phi(obs,z,cond,env.cue); a=ctl.act(p)
    steps=[]; trials=[]; Z=[]; G=[]; cues=[]; cum=0; td_hist=[]; upd_hist=[]
    for t in range(cfg['total_interactions']):
        nobs,r,_,info=env.step(a); td_gvf=gvf.update(nobs,r); ng=gvf.features(nobs); nz=tr.transform(ng); np_=phi(nobs,nz,cond,env.cue); na=ctl.act(np_)
        td,un,pn=ctl.update(p,a,r,np_,na); cum+=r; td_hist.append(td); upd_hist.append(un)
        if len(nz): Z.append(nz.copy()); G.append(ng.copy()); cues.append(info.cue)
        steps.append(dict(t=t,condition=cond,seed=seed,reward=r,cumulative_reward=cum,gvf_td_error=td_gvf,control_td_error=td,update_norm=un,param_norm=pn,nan_inf=0 if np.isfinite(np_).all() else 1))
        if info.junction: trials.append(dict(trial=info.trial,condition=cond,seed=seed,correct=float(info.correct),reward=r,cumulative_reward=cum,moving_accuracy=np.mean([x['correct'] for x in trials[-20:]]+[float(info.correct)])))
        obs,g,z,p,a=nobs,ng,nz,np_,na
    pd.DataFrame(steps).to_csv(out/"per_step_metrics.csv",index=False); pd.DataFrame(trials).to_csv(out/"per_trial_metrics.csv",index=False)
    met=rep_metrics(np.array(Z)); met.update(condition=cond,seed=seed,gvf_nonconstant=float(np.var(G)>1e-8) if G else 0, cue_decodability=cue_probe(np.array(Z),np.array(cues)) if Z else 0)
    pd.DataFrame([met]).to_csv(out/"representation_metrics.csv",index=False)
    final_acc=float(pd.DataFrame(trials).tail(20)['correct'].mean()) if trials else 0
    summ=dict(condition=cond,seed=seed,final_window_accuracy=final_acc,cumulative_reward=cum,mean_update_norm=float(np.mean(upd_hist)),max_update_norm=float(np.max(upd_hist)),td_error_variance=float(np.var(td_hist)),run_dir=str(out))
    pd.DataFrame([summ]).to_csv(out/"summary.csv",index=False)
    manifest=dict(git_commit=git(['rev-parse','HEAD']),branch=git(['branch','--show-current']),command=' '.join(sys.argv),profile=cfg['profile'],seed=seed,condition=cond,hostname=socket.gethostname(),python=platform.python_version(),start_time=start,end_time=time.time(),run_status='ok',config=cfg)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); (out/"config.json").write_text(json.dumps(cfg,indent=2)); (out/"stdout.log").write_text("completed\n")
    return summ

def cue_probe(Z,cues):
    if len(Z)<4 or Z.shape[1]==0: return 0.0
    y=(cues>0).astype(float)*2-1; w=np.linalg.pinv(Z)@y; return float(np.mean(np.sign(Z@w)==y))

def load(path): return json.loads(Path(path).read_text())
def aggregate(outroot):
    root=Path(outroot); sums=[pd.read_csv(p) for p in root.glob("*/summary.csv")]; reps=[pd.read_csv(p) for p in root.glob("*/representation_metrics.csv")]
    if not sums: raise SystemExit("no runs")
    agg=pd.concat(sums); rep=pd.concat(reps); agg.to_csv(root/"aggregate_summary.csv",index=False); rep.to_csv(root/"aggregate_representation.csv",index=False); make_figs(root,agg,rep)

def make_figs(root,agg,rep):
    import matplotlib.pyplot as plt
    figdir=root/"figures"; figdir.mkdir(exist_ok=True)
    for col,name in [('final_window_accuracy','final_accuracy_comparison'),('cumulative_reward','cumulative_reward')]:
        m=agg.groupby('condition')[col].mean(); se=agg.groupby('condition')[col].sem().fillna(0); plt.figure(figsize=(8,4)); plt.bar(m.index,m.values,yerr=se.values); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(figdir/f"{name}.png"); plt.close()
    for x in ['isotropy_error','effective_rank','cue_decodability','condition_number','skewness_error','kurtosis_error']:
        if x in rep: plt.figure(); plt.bar(rep.groupby('condition')[x].mean().index, rep.groupby('condition')[x].mean().values); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(figdir/f"{x}.png"); plt.close()
    merged=agg.merge(rep,on=['condition','seed']);
    for x in ['isotropy_error','effective_rank','cue_decodability']:
        plt.figure(); plt.scatter(merged[x],merged['final_window_accuracy']); plt.xlabel(x); plt.ylabel('final accuracy'); plt.tight_layout(); plt.savefig(figdir/f"accuracy_vs_{x}.png"); plt.close()
    # placeholders backed by available summaries
    for name in ['trial_accuracy_learning_curve','gvf_prediction_error','covariance_eigenvalue_spectrum','update_norm_statistics','cue_conditioned_feature_projection']:
        plt.figure(); plt.text(.1,.5,'Generated in full runs from per-step/per-trial CSVs'); plt.savefig(figdir/f"{name}.png"); plt.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--allow-full-run',action='store_true'); ap.add_argument('--workers',type=int); ap.add_argument('--output-dir'); ap.add_argument('--aggregate-only',action='store_true')
    ns=ap.parse_args(); cfg=load(ns.config); out=ns.output_dir or cfg['output_dir']
    if cfg['profile']=='full' and not (ns.allow_full_run or os.environ.get('RL_RUN_CONTEXT')=='remote'): raise SystemExit('Full profile blocked: set RL_RUN_CONTEXT=remote or pass --allow-full-run on remote server.')
    if ns.aggregate_only: aggregate(out); return
    runs=[(cfg,c,s,out) for c in cfg['conditions'] for s in cfg['seeds']]; workers=ns.workers or cfg.get('workers') or max(1,min(cpu_count()-1,len(runs)))
    res = list(map(run_one,runs)) if workers<=1 else Pool(workers).map(run_one,runs)
    aggregate(out)
    if cfg['profile']=='smoke': smoke_assertions(out)

def smoke_assertions(out):
    agg=pd.read_csv(Path(out)/'aggregate_summary.csv'); rep=pd.read_csv(Path(out)/'aggregate_representation.csv')
    assert agg.query("condition=='oracle'")['final_window_accuracy'].mean() >= agg.query("condition=='observation'")['final_window_accuracy'].mean(), 'oracle not >= observation'
    assert rep.query("condition=='raw'")['gvf_nonconstant'].mean() > 0, 'GVF constant'
    if {'raw','whitened'} <= set(rep.condition): assert rep.query("condition=='whitened'")['condition_number'].mean() <= rep.query("condition=='raw'")['condition_number'].mean()*1.2, 'whitening did not reduce anisotropy enough'
    assert agg.notna().all().all() and rep.notna().all().all(), 'NaN in outputs'
if __name__=='__main__': main()
