"""
The Corporate Heist -- Track A: "literal debrief" re-ranker.

    python code/main.py            (run from the folder that holds train/dev/dev_winners/test .csv)

1. Learn the OLD panel's rating function from train.csv with two de-biased scorers:
     A. LightGBM on all cleaned features, pedigree attributes as explicit columns, scored with every
        candidate given the same (train-mode) pedigree values -> pedigree cannot move anyone's rank;
     B. LightGBM trained on a de-biased target (post_hire_score minus the old panel's premium for
        each pedigree flag, measured by a linear fit) without the pedigree columns.
   The quality score is the average of A and B (each is a 3-seed average).
2. Apply the new-cycle rules from the insider's debrief on top: code-contribution fast-track,
   new-college stars, the old-boys' leg-up re-added at its measured size; hard-exclude fabricated
   profiles, duplicate people, notice periods over two months and inflated titles.
3. Write the top 500 to submission.csv, ordered by the adjusted score.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

T0 = time.time()

# --- tunables (documentation, sections 4 and 6) ------------------------------------------------
DEFAULTS = dict(
    pcc_low=10,          # below this a contribution count "means nothing"
    pcc_full=24,         # "dozens of merged contributions or more" -> full fast-track bonus
    pcc_q=0.65,          # full bonus lifts a 65th-percentile ("only good") profile to the top-5% line
    star_tech=85,        # "topped the technical assessment"
    star_q=0.50,         # star bonus lifts a median profile to the top-5% line
    star_role_fit=True,  # "genuinely fit the role": skill role-fit >= Vault median, or title in the role family
    old_boys_bonus=None, # None = the premium measured in the Archive (Ridge coefficient)
)
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=150,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=C.NUM_THREADS, deterministic=True, force_row_wise=True, verbose=-1)
N_ROUNDS = 1500
SEEDS = (42, 43, 44)
# Only the top of the distribution matters: scores below the Archive median (49) are clipped to 50 so the
# models spend no capacity separating bad hires from mediocre ones (Ledger NDCG 0.63 -> 0.65, rho 0.29 -> 0.42).
TARGET_CLIP = 50.0
REMOVED = C.PEDIGREE + ["old_boys"]     # every institute/pedigree column is taken out of the models


def log(msg):
    print("[%5.1fs] %s" % (time.time() - T0, msg), flush=True)


def fit_avg(X, y, cols):
    """Seed-averaged LightGBM; returns a predict(frame) callable."""
    models = [lgb.train(dict(LGB_PARAMS, seed=s), lgb.Dataset(X[cols], y), num_boost_round=N_ROUNDS) for s in SEEDS]
    return lambda F: np.mean([m.predict(F[cols]) for m in models], axis=0)


def prepare(data_dir="."):
    """Load, parse, build features, train the two de-biased scorers. Returns everything the ranking needs."""
    tr_raw, dv_raw, dw, te_raw = C.load(data_dir)
    tr, dv, te = C.parse(tr_raw), C.parse(dv_raw), C.parse(te_raw)
    y = tr.post_hire_score.map(C.num).astype(float)
    log("parsed train %d, dev %d, test %d" % (len(tr), len(dv), len(te)))

    fb = C.FeatureBuilder().fit(tr, [dv, te])
    Xtr, Xdv, Xte = fb.transform(tr), fb.transform(dv), fb.transform(te)
    log("features: %d columns (%d note sentences, %d skills)" % (Xtr.shape[1], len(fb.sentences), len(fb.top_skills)))

    # the old panel's premium per pedigree flag (points of post_hire_score), from a linear fit on train
    ridge = Ridge(alpha=1.0).fit(Xtr.fillna(Xtr.median()), y)
    coef = pd.Series(ridge.coef_, index=Xtr.columns)
    log("old panel premium (pts): " + ", ".join("%s=%.1f" % (c, coef[c]) for c in REMOVED))

    y_fit = np.clip(y, TARGET_CLIP, None)
    # scorer A: all columns; scored with pedigree neutralised
    all_cols = list(Xtr.columns)
    pred_a = fit_avg(Xtr, y_fit, all_cols)
    neutral = {c: float(Xtr[c].mode().iloc[0]) for c in REMOVED}
    neut = lambda F: F.assign(**neutral)
    # scorer B: de-biased target, pedigree columns dropped
    y_deb = y_fit - (Xtr[REMOVED].fillna(0) * coef[REMOVED]).sum(axis=1)
    cols_b = [c for c in all_cols if c not in REMOVED]
    pred_b = fit_avg(Xtr, y_deb, cols_b)

    # old-regime checks on the Ledger (dev): raw model should be ~0.5+, de-biased ones lower (expected)
    log("dev A raw pedigree:   " + C.fmt(C.dev_metrics(dv, pred_a(Xdv), dw)))
    pa_dv, pb_dv = pred_a(neut(Xdv)), pred_b(Xdv)
    log("dev A neutralised:    " + C.fmt(C.dev_metrics(dv, pa_dv, dw)) + "   <- lower is expected: dev is old-regime")
    log("dev B de-biased:      " + C.fmt(C.dev_metrics(dv, pb_dv, dw)))
    log("dev blend (A+B)/2:    " + C.fmt(C.dev_metrics(dv, 0.5 * (pa_dv + pb_dv), dw)))
    fdv = C.all_flags(dv)
    n_win_flagged = int(dv.candidate_id[fdv.excluded].isin(set(dw.candidate_id)).sum())
    log("exclusion rules on dev: %d rows flagged, %d of them winners (must be 0)" % (int(fdv.excluded.sum()), n_win_flagged))

    quality = pd.Series(0.5 * (pred_a(neut(Xte)) + pred_b(Xte)), index=te.index)
    return dict(tr=tr, te=te, Xte=Xte, quality=quality, flags=C.all_flags(te), coef=coef,
                unseen=~te.c_inst.isin(set(tr.c_inst)))


def shortlist(prep, **kw):
    """Apply the debrief rules to the quality score and return (ordered candidate ids, stats)."""
    P = dict(DEFAULTS, **kw)
    te, X, p, f = prep["te"], prep["Xte"], prep["quality"], prep["flags"]
    q = p.quantile
    top5_line = q(0.95)

    pcc = te.c_pcc
    ramp = ((pcc - P["pcc_low"]) / (P["pcc_full"] - P["pcc_low"])).clip(0, 1).fillna(0.0)
    pcc_bonus = top5_line - q(P["pcc_q"])
    b_pcc = ramp * pcc_bonus

    star = prep["unseen"] & (te.c_tech >= P["star_tech"])
    if P["star_role_fit"]:   # "genuinely fit the role": skills overlap the role's, or the current title is in the role family
        star &= (X.role_fit >= X.role_fit.median()) | (X.title_fit > 0)
    star_bonus = top5_line - q(P["star_q"])
    b_new = star.astype(float) * star_bonus

    ob_bonus = prep["coef"]["old_boys"] if P["old_boys_bonus"] is None else P["old_boys_bonus"]
    b_old = X.old_boys * ob_bonus

    score = p + b_pcc + b_new + b_old
    keep = ~f.excluded
    order = score[keep].sort_values(ascending=False)
    top = order.head(C.K_TEST).index
    sel = te.index.isin(top)
    base_top = set(p[keep].sort_values(ascending=False).head(C.K_TEST).index)
    stats = dict(
        excluded=dict(fake=int(f.fake.sum()), notice_gt60=int(f.notice_gt60.sum()),
                      title_inflated=int(f.title_inflated.sum()), dup_drop=int(f.dup_drop.sum()),
                      total=int(f.excluded.sum())),
        bonus_pts=dict(pcc=round(pcc_bonus, 1), star=round(star_bonus, 1), old_boys=round(float(ob_bonus), 1)),
        top5_line=round(top5_line, 1),
        composition=dict(fast_tracked=int((pcc[sel] >= P["pcc_full"]).sum()),
                         partial_ramp=int(((pcc[sel] >= P["pcc_low"]) & (pcc[sel] < P["pcc_full"])).sum()),
                         new_college_stars=int(star[sel].sum()), stars_total=int((star & keep).sum()),
                         old_boys=int((X.old_boys[sel] > 0).sum())),
        overlap_with_plain_quality_top500=len(base_top & set(top)),
        mean_tech=round(float(te.c_tech[sel].mean()), 1),
    )
    return te.candidate_id.loc[top].tolist(), stats


def main():
    prep = prepare(".")
    ids, stats = shortlist(prep)
    C.write_submission(ids, "submission.csv", test_ids=prep["te"].candidate_id)
    for k, v in stats.items():
        log("%s: %s" % (k, v))
    log("wrote submission.csv (%d rows)" % len(ids))


if __name__ == "__main__":
    main()
