"""
De-biasing KNOB: interpolate raw (keeps pedigree) <-> fully neutralised.
  score_lambda = (1-l)*raw_pred + l*neutralised_pred     (both on the post_hire scale)
lambda=0  -> raw model (dev ~0.63, the 'sanity' number; keeps the old bias)
lambda=1  -> fully de-biased (what the debrief prescribes)
Prints the dev P@150/NDCG/MAP curve so we can pick empirically on the leaderboard.
No dev tuning inside training; dev only scored to display the curve.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import numpy as np, pandas as pd, lightgbm as lgb, common as C
from sklearn.linear_model import Ridge
SEEDS=(42,43,44); REMOVED=C.PEDIGREE+["old_boys"]; CLIP=60.0
LGB=dict(objective="regression",learning_rate=0.03,num_leaves=15,min_data_in_leaf=80,feature_fraction=0.8,
         bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,num_threads=2,deterministic=True,force_row_wise=True,verbose=-1)
def favg(X,y,cols):
    ms=[lgb.train(dict(LGB,seed=s),lgb.Dataset(X[cols],y),num_boost_round=900) for s in SEEDS]
    return lambda F:np.mean([m.predict(F[cols]) for m in ms],axis=0)

t0=time.time()
tr,dv,te=C.parse(C.load(".")[0]),C.parse(C.load(".")[1]),C.parse(C.load(".")[3]); dw=C.load(".")[2]
y=tr.post_hire_score.map(C.num).astype(float); y_fit=np.clip(y,CLIP,None)
fb=C.FeatureBuilder().fit(tr,[dv,te]); Xtr,Xdv,Xte=fb.transform(tr),fb.transform(dv),fb.transform(te)
allc=list(Xtr.columns); pa=favg(Xtr,y_fit,allc)
nmap={c:float(Xtr[c].mode().iloc[0]) for c in REMOVED}
raw_dv,neu_dv=pa(Xdv),pa(Xdv.assign(**nmap))
raw_te,neu_te=pa(Xte),pa(Xte.assign(**nmap))
print("de-biasing knob (lambda: 0=raw/biased ... 1=fully de-biased)\n")
print("%-8s %-9s %-10s %-9s"%("lambda","P@150","NDCG@150","MAP@150"))
for l in (0.0,0.25,0.5,0.75,1.0):
    s=(1-l)*raw_dv + l*neu_dv
    m=C.dev_metrics(dv,s,dw)
    print("%-8.2f %-9.3f %-10.3f %-9.3f"%(l,m["P@150"],m["NDCG@150"],m["MAP@150"]))
print("\n[%.1fs] raw dev = the friend's 0.6; de-biased = what the Vault should reward."%(time.time()-t0))
print("Re-uploads are free: submit lambda=1 (Track D, principled) first, then try 0.5 to test if the Vault rewards pedigree.")
