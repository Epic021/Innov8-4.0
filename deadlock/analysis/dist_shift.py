"""
Distribution-shift audit: train vs dev vs test.
  1. KS statistic + PSI (population stability index) per numeric feature.
  2. Categorical proportion shifts.
  3. Adversarial validation: train a classifier to tell train-vs-test apart; AUC
     >> 0.5 means the covariate distribution moved (SOTA drift diagnostic).
No labels used from dev/test -> no leakage.
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import numpy as np, pandas as pd, common as C
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

tr, dv, te = C.parse(C.load(".")[0]), C.parse(C.load(".")[1]), C.parse(C.load(".")[3])

def psi(a, b, bins=10):
    a, b = a.dropna(), b.dropna()
    if len(a) < 50 or len(b) < 50: return np.nan
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3: return np.nan
    pa = np.histogram(a, qs)[0] / len(a) + 1e-6
    pb = np.histogram(b, qs)[0] / len(b) + 1e-6
    return float(((pb - pa) * np.log(pb / pa)).sum())

NUMS = ["c_tech","c_apt","c_rating","c_kpi","c_exp","c_age","c_gy","c_nemp","c_psum","c_plen",
        "c_lvl","c_nskills","c_ncert","c_ctc","c_ectc","c_ctc_ratio","c_nd","c_trn","c_trh","c_ljc"]
print("=== NUMERIC SHIFT: KS stat & PSI (flag PSI>0.1 minor, >0.25 major) ===")
print("%-12s %8s %8s | %8s %8s  %s" % ("feature","KS_dv","KS_te","PSI_dv","PSI_te","note"))
for c in NUMS:
    ksd = ks_2samp(tr[c].dropna(), dv[c].dropna()).statistic
    kst = ks_2samp(tr[c].dropna(), te[c].dropna()).statistic
    pd_ = psi(tr[c], dv[c]); pt = psi(tr[c], te[c])
    note = ""
    if pt > 0.25: note = "*** MAJOR test shift"
    elif pt > 0.1: note = "* minor test shift"
    print("%-12s %8.3f %8.3f | %8.3f %8.3f  %s" % (c, ksd, kst, pd_, pt, note))

print("\n=== CATEGORICAL / STRUCTURE SHIFT (share of rows) ===")
def share(d, m): return 100*m.mean()
rows = [
 ("notice > 60d", lambda d:(d.c_nd>60)),
 ("title inflated", lambda d: C.flag_title_inflated(d)),
 ("fabricated (rule)", lambda d: C.flag_fake(d)),
 ("pcc present", lambda d: d.c_pcc.notna()),
 ("unseen institute", lambda d:~d.c_inst.isin(set(tr.c_inst))),
 ("aptitude>10 raw", lambda d: (d.aptitude_score.fillna("").astype(str).str.contains("%"))),
 ("referral channel", lambda d: d.recruitment_channel.fillna("").str.contains("Referral")),
]
print("%-20s %8s %8s %8s" % ("group","train%","dev%","test%"))
for nm, fn in rows:
    print("%-20s %8.1f %8.1f %8.1f" % (nm, share(tr,fn(tr)), share(dv,fn(dv)), share(te,fn(te))))

print("\n=== ADVERSARIAL VALIDATION (can a model tell the pools apart?) ===")
fb = C.FeatureBuilder().fit(tr, [dv, te])
def av(A, B, nA, nB):
    XA, XB = fb.transform(A), fb.transform(B)
    X = pd.concat([XA, XB]); yv = np.r_[np.zeros(len(XA)), np.ones(len(XB))]
    # drop pcc (test-only) so it doesn't trivially separate
    X = X[[c for c in X.columns if c not in ("pcc","c_pcc")]].fillna(-999)
    clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, random_state=42)
    p = cross_val_predict(clf, X, yv, cv=3, method="predict_proba")[:,1]
    return roc_auc_score(yv, p)
print("train vs dev  AUC = %.3f  (0.5 = identical; >0.6 = real shift)" % av(tr, dv, "tr","dv"))
print("train vs test AUC = %.3f" % av(tr, te, "tr","te"))
print("dev   vs test AUC = %.3f" % av(dv, te, "dv","te"))
