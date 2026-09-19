"""Ablation: incremental dev impact of each cleaning/feature fix (Track A scorers)."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/deadlock"); import numpy as np, pandas as pd, lightgbm as lgb, common as C
from sklearn.linear_model import Ridge
SEEDS=(42,43,44); REMOVED=C.PEDIGREE+["old_boys"]; CLIP=60.0
LGB=dict(objective="regression",learning_rate=0.03,num_leaves=7,min_data_in_leaf=150,feature_fraction=0.8,
         bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,num_threads=2,deterministic=True,force_row_wise=True,verbose=-1)
def favg(X,y,cols):
    ms=[lgb.train(dict(LGB,seed=s),lgb.Dataset(X[cols],y),num_boost_round=1500) for s in SEEDS]
    return lambda F:np.mean([m.predict(F[cols]) for m in ms],axis=0)

raw={n:C.load(".")[i] for n,i in [("tr",0),("dv",1),("te",3)]}; dw=C.load(".")[2]

def build(apt,rating,factors):
    def parse(df):
        d=C.parse(df)                     # enhanced parse already applies apt+rating fixes
        if not apt:  d["c_apt"]=df.aptitude_score.map(C.num)
        if not rating: d["c_rating"]=df.last_rating.map(C.num)
        return d
    tr,dv,te=parse(raw["tr"]),parse(raw["dv"]),parse(raw["te"])
    y=np.clip(tr.post_hire_score.map(C.num).astype(float),CLIP,None)
    fb=C.FeatureBuilder();
    if not factors: fb.majors_off=True
    fb=fb.fit(tr,[dv,te])
    Xtr,Xdv=fb.transform(tr),fb.transform(dv)
    if not factors:
        keep=[c for c in Xtr.columns if not (c.startswith("major_") or c.startswith("ctype_") or c=="ectc")]
        Xtr,Xdv=Xtr[keep],Xdv[keep]
    coef=pd.Series(Ridge(alpha=1.0).fit(Xtr.fillna(Xtr.median()),y).coef_,index=Xtr.columns)
    allc=list(Xtr.columns); pa=favg(Xtr,y,allc)
    nmap={c:float(Xtr[c].mode().iloc[0]) for c in REMOVED}
    yb=y-(Xtr[REMOVED].fillna(0)*coef[REMOVED]).sum(1); pb=favg(Xtr,yb,[c for c in allc if c not in REMOVED])
    raw_m=C.dev_metrics(dv,pa(Xdv),dw); neu=C.dev_metrics(dv,pa(Xdv.assign(**nmap)),dw)
    bl=C.dev_metrics(dv,0.5*(pa(Xdv.assign(**nmap))+pb(Xdv)),dw)
    return raw_m,neu,bl,Xtr.shape[1]

print("%-38s %-7s %-12s %-8s %s"%("config","raw","neutralised","blend","feats"))
for name,a,r,f in [("baseline (main)",0,0,0),("+ aptitude % fix",1,0,0),
                   ("+ apt + last_rating text",1,1,0),("+ apt+rating + factors (FULL)",1,1,1)]:
    rm,ne,bl,nf=build(a,r,f)
    print("%-38s %-7.3f %-12.3f %-8.3f %d"%(name,rm["P@150"],ne["P@150"],bl["P@150"],nf))
