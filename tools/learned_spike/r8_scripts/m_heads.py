"""Arm M (round 8): frame-level log loss of each head of the scored members on the left-out tests. Not part of the submission.

    # pf venv, from the repository root
    python tools/learned_spike/r8_scripts/m_heads.py ~/mtah2027/r8/m/score_F6b

Reads the member probabilities cached by m_score.py (OUT/probs/<member dir>/<game>.npy) and the
references of pf_eval.refs: team head = log loss and accuracy of p(H) / (p(H) + p(A)) on frames
the reference labels H or A; dead head = log loss of p(D) on graded H/A/D frames. Means over the
matches of each test.
"""
import sys, os, glob, numpy as np
sys.path[:0]=['.','tools']
from tools.learned_spike import pf_eval as pe, r4_screen as rs, folds
root=sys.argv[1]; dirs=sorted(glob.glob(root+'/probs/*'))
for t in ['T-M','T-D','T-S']:
    out={}
    for g in pe.games_of(t):
        ref,_=pe.refs(t,g)
        alive=ref<=1
        for d in dirs:
            if not os.path.isfile(os.path.join(d,g+'.npy')): continue
            P=np.load(os.path.join(d,g+'.npy'))
            ph=P[:,0]/np.maximum(P[:,0]+P[:,1],1e-12)
            y=(ref[alive]==0)
            p=np.clip(ph[alive],1e-6,1-1e-6)
            ll=-(y*np.log(p)+(1-y)*np.log(1-p)).mean()
            acc=((p>0.5)==y).mean()
            pdd=np.clip(P[:,2],1e-6,1-1e-6); yd=(ref==2); m=ref<=2
            lld=-(yd[m]*np.log(pdd[m])+(1-yd[m])*np.log(1-pdd[m])).mean()
            out.setdefault(os.path.basename(d).split('_')[-2],[]).append((ll,acc,lld))
    for k,v in out.items():
        v=np.array(v); print(t,k,'team logloss %.4f acc %.4f | dead logloss %.4f'%tuple(v.mean(0)))
