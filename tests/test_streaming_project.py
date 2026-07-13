import os, subprocess, sys
import numpy as np
from streaming_rl_feature_geometry.env import ContinuingTMaze
from streaming_rl_feature_geometry.transforms import FeatureTransform

def test_junction_no_cue_and_echo_after_action():
    env=ContinuingTMaze(corridor_length=2, seed=0); obs=env._obs(); assert obs[1]+obs[2]==1
    for _ in range(2): obs,_,_,_=env.step(0)
    assert obs[4]==1 and obs[1]==obs[2]==obs[6]==obs[7]==0
    obs,r,_,info=env.step(0); assert info.junction and obs[5]==1 and (obs[6]+obs[7])==1

def test_transform_is_causal_updates_after_transform():
    tr=FeatureTransform('standardized',2); z=tr.transform(np.array([10.,-5.])); assert tr.stats.n==1; assert np.allclose(z, [10.,-5.], atol=1e-1)

def test_full_guard_blocks_without_remote():
    env=os.environ.copy(); env.pop('RL_RUN_CONTEXT',None)
    p=subprocess.run([sys.executable,'scripts/run_experiment.py','--config','configs/full.json','--output-dir','/tmp/blocked'],env=env,text=True,capture_output=True)
    assert p.returncode != 0 and 'Full profile blocked' in (p.stderr+p.stdout)
