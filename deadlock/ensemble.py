"""
Deadlock ensemble — leakage-free 5-model rank fusion on the enhanced base.
Components (all de-biased/neutralised, trained on train only; dev scored once):
  1 LGB regression (all feats, pedigree neutralised)   2 LGB regression (de-biased, no pedigree)
  3 LGB LambdaMART (de-biased relevance)               4 HistGBM (grid-searched on train CV)
  5 ExtraTrees (de-biased)                             -> fused by percentile rank.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import common as C

T0=time.time(); SEEDS=(42,43,44); REMOVED=C.PEDIGREE+["old_boys"]; CLIP=60.0
LGB=dict(learning_rate=0.03,num_leaves=15,min_data_in_leaf=80,feature_fraction=0.8,bagging_fraction=0.8,
         bagging_freq=1,lambda_l2=1.0,num_threads=2,deterministic=True,force_row_wise=True,verbose=-1)
def log(m): print("[%5.1fs] %s"%(time.time()-T0,m),flush=True)
def pr(a,idx=None): return (pd.Series(a,index=idx) if idx is not None else pd.Series(a)).rank(pct=True)
def favg(X,y,cols,**kw):
    ms=[lgb.train(dict(LGB,objective="regression",seed=s,**kw),lgb.Dataset(X[cols],y),num_boost_round=900) for s in SEEDS]
    return lambda F:np.mean([m.predict(F[cols]) for m in ms],axis=0)
def ndcg(order,win,k=C.K_DEV):
    h=win[order[:k]].astype(float); disc=1/np.log2(np.arange(2,len(h)+2)); ide=disc[:min(k,int(win.sum()))].sum()
    return (h*disc).sum()/ide if ide else 0
def grid_hgb(Xtr,yb,cols,med):
    kf=KFold(3,shuffle=True,random_state=42); X=Xtr[cols].fillna(med).to_numpy(); yv=yb.to_numpy(); best=None; bs=-1
    for g in [dict(max_leaf_nodes=n,learning_rate=lr,min_samples_leaf=150) for n in (8,15) for lr in (0.05,0.03)]:
        sc=[]
        for a,b in kf.split(X):
            m=HistGradientBoostingRegressor(max_iter=400,random_state=42,**g).fit(X[a],yv[a]); p=m.predict(X[b])
            sc.append(ndcg(np.argsort(-p), yv[b]>=np.quantile(yv[b],0.95)))
        if np.mean(sc)>bs: bs,best=np.mean(sc),g
    log("grid HGB best=%s cv_ndcg=%.3f"%(best,bs)); return best

def rank_lgb(Xtr,rel,cols,gsz=1000):
    perm=np.random.RandomState(42).permutation(len(Xtr)); Xs,rs=Xtr[cols].iloc[perm],np.asarray(rel)[perm]
    n=len(Xtr); groups=[gsz]*(n//gsz)+([n%gsz] if n%gsz else [])
    ms=[lgb.train(dict(LGB,objective="lambdarank",metric="ndcg",num_leaves=7,min_data_in_leaf=150,seed=s),
                  lgb.Dataset(Xs,label=rs,group=groups),num_boost_round=1200) for s in SEEDS]
    return lambda F:np.mean([m.predict(F[cols]) for m in ms],axis=0)

tr,dv,te=C.parse(C.load(".")[0]),C.parse(C.load(".")[1]),C.parse(C.load(".")[3]); dw=C.load(".")[2]
y=tr.post_hire_score.map(C.num).astype(float); yf=np.clip(y,CLIP,None)
fb=C.FeatureBuilder().fit(tr,[dv,te]); Xtr,Xdv,Xte=fb.transform(tr),fb.transform(dv),fb.transform(te)
log("features %d"%Xtr.shape[1])
coef=pd.Series(Ridge(alpha=1.0).fit(Xtr.fillna(Xtr.median()),yf).coef_,index=Xtr.columns)
allc=list(Xtr.columns); colsb=[c for c in allc if c not in REMOVED]
nmap={c:float(Xtr[c].mode().iloc[0]) for c in REMOVED}; med=Xtr[colsb].median()
yb=yf-(Xtr[REMOVED].fillna(0)*coef[REMOVED]).sum(1)
pa=favg(Xtr,yf,allc); pb=favg(Xtr,yb,colsb)
rel=np.digitize(yb,[yb.quantile(.85),yb.quantile(.95),yb.quantile(.99)]); prk=rank_lgb(Xtr,rel,allc)
g=grid_hgb(Xtr,yb,colsb,med); hgb=HistGradientBoostingRegressor(max_iter=600,random_state=42,**g).fit(Xtr[colsb].fillna(med),yb)
et=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=20,random_state=42,n_jobs=2).fit(Xtr[colsb].fillna(med),yb)
log("trained 5 components")
def comps(F): return [pa(F.assign(**nmap)),pb(F),prk(F.assign(**nmap)),hgb.predict(F[colsb].fillna(med)),et.predict(F[colsb].fillna(med))]
def fuse(F): return sum(pr(c,F.index) for c in comps(F))/5
for nm,F in [("dev",Xdv)]:
    d=dv; cs=comps(F)
    for i,c in enumerate(cs,1): log("  comp%d: %s"%(i,C.fmt(C.dev_metrics(d,c,dw))))
    log("  DEADLOCK ENSEMBLE: %s"%C.fmt(C.dev_metrics(d,fuse(F),dw)))
