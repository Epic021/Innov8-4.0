"""
BIAS AUDIT of the OLD hiring panel (train.csv post_hire_score), guided by the
insider debrief (Problem_Statement 01:39): the six "pet preferences" the old panel
inflated -- big-name college, big metro, referral, big-brand employer, 'proper'
degree, gap-free CV -- plus the surviving old-boys' network.

For each marker we separate BIAS from MERIT three ways:
  1. raw gap          : mean post_hire_score(marker) - mean(no marker)
  2. merit-controlled : OLS coef with tech/rating/aptitude/kpi/skills/exp controlled
                        (the premium that is NOT explained by ability)
  3. merit-matched    : within tech x rating deciles, the residual marker gap
                        (non-parametric check that 2. isn't a functional-form artefact)
  4. representation   : P(marker | old top-5%) vs P(marker | pool)  -> selection lift
No dev/test labels used.
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import numpy as np, pandas as pd, common as C
from sklearn.linear_model import LinearRegression

tr = C.parse(C.load(".")[0])
fb = C.FeatureBuilder().fit(tr, [C.parse(C.load(".")[1]), C.parse(C.load(".")[3])])
X = fb.transform(tr)
y = tr.post_hire_score.map(C.num).astype(float)
top5 = y >= y.quantile(0.95)

MERIT = ["tech", "rating", "apt", "kpi", "nskills", "role_fit", "exp", "plen", "c_ctc"]
Xm = pd.DataFrame({
    "tech": tr.c_tech, "rating": tr.c_rating, "apt": tr.c_apt, "kpi": tr.c_kpi,
    "nskills": tr.c_nskills, "role_fit": X.role_fit, "exp": tr.c_exp, "plen": tr.c_plen,
    "c_ctc": np.log1p(tr.c_ctc)}).fillna(0)

MARKERS = [("big-name college","bigname"), ("big metro","metro"), ("referral","referral"),
           ("big-brand employer","big_emp"), ("'proper' degree","proper_deg"),
           ("gap-free CV","no_gap"), ("IIT","iit"), ("old-boys' network","old_boys")]

def merit_controlled(flag):
    d = X[flag].astype(float)
    Z = pd.concat([Xm, d.rename("marker")], axis=1)
    lr = LinearRegression().fit(Z, y)
    return lr.coef_[-1]

def merit_matched(flag):
    d = X[flag].astype(float)
    td = pd.qcut(tr.c_tech.rank(method="first"), 5, labels=False)
    rd = pd.qcut(tr.c_rating.fillna(tr.c_rating.median()).rank(method="first"), 5, labels=False)
    g = pd.DataFrame({"y": y, "m": d, "td": td, "rd": rd})
    diffs = []
    for _, cell in g.groupby(["td","rd"]):
        if cell.m.nunique() == 2 and len(cell) > 30:
            diffs.append(cell[cell.m==1].y.mean() - cell[cell.m==0].y.mean())
    return np.nanmean(diffs) if diffs else np.nan

print("OLD-PANEL BIAS (points of post_hire_score; mean score=%.1f, top-5%% line=%.1f)\n" % (y.mean(), y.quantile(0.95)))
print("%-20s %8s %8s %8s | %10s %10s" % ("pet preference","raw_gap","merit_ctrl","matched","P(top5%)","P(pool)"))
print("-"*80)
for name, flag in MARKERS:
    m = X[flag].astype(bool)
    raw = y[m].mean() - y[~m].mean()
    ctrl = merit_controlled(flag)
    match = merit_matched(flag)
    p_top = m[top5].mean()*100
    p_pool = m.mean()*100
    print("%-20s %8.1f %8.1f %8.1f | %9.1f%% %9.1f%%" % (name, raw, ctrl, match, p_top, p_pool))

print("\nINTERPRETATION")
print("  merit_ctrl > 0 with raw ~ merit_ctrl  => the gap is BIAS, not ability.")
print("  P(top5%) >> P(pool)                   => the marker was over-selected by the old panel.")
print("  old-boys': the one the debrief says SURVIVES the reorg -> re-added; the rest neutralised.")

# how much of the old model's ranking gain came from pedigree vs merit
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0).fit(X.fillna(X.median()), y)
coef = pd.Series(ridge.coef_, index=X.columns).abs()
ped = C.PEDIGREE + ["old_boys"]
print("\nShare of |Ridge coef| mass on pedigree block: %.1f%% (of all %d features)"
      % (100*coef[ped].sum()/coef.sum(), len(coef)))
