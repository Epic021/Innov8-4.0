"""
The Corporate Heist -- Track C (Person C): a transparent consensus scorecard.

    python tracks/track_c.py        (run from the folder that holds train/dev/dev_winners/test .csv)

Thesis
------
A single gradient-boosted model can quietly overfit the Archive's quirks, and we
are shipping a shortlist we cannot inspect row by row. Track C hedges that risk
two ways and takes the **consensus**:

  1. an interpretable **signal scorecard** -- percentile-rank the handful of
     signals the analysis actually found to matter (technical assessment, last
     rating, aptitude, CTC, KPI met, skills count, role fit; experience and career
     length count *against*, because the old panel's best hires were early-career)
     and combine them by weight. It uses no pedigree at all and no training, so it
     cannot overfit;
  2. a **diversified model ensemble** on the de-biased target (post_hire_score
     minus the pedigree premiums, exactly as Track A's scorer B), but built from
     three *different* families -- Ridge, a random forest and a histogram
     gradient-booster (scikit-learn, no LightGBM) -- so no single learner's
     idiosyncrasies decide the list.

The two blocks are fused by percentile rank (50/50); that consensus is the
Track-C **quality**. The debrief bonuses and the four hard exclusions are the
shared rulebook in `rerank.py`, identical to Tracks A and B.

Deterministic: fixed seeds, 2 threads.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                  # rerank.py
sys.path.insert(0, os.path.join(_HERE, "..", "code"))      # common.py
import common as C   # noqa: E402
import rerank as R   # noqa: E402

T0 = time.time()
SEED = 42

# Interpretable signals and weights, read off documentation section 3 (model gain /
# correlations). Positive lifts the score; experience & career length are negative
# (r = -0.13 / -0.12: the old panel's best-rated hires were early-career).
SIGNAL_WEIGHTS = dict(tech=1.0, rating=1.0, apt=0.6, ctc=0.5, kpi=0.4, nskills=0.4,
                      role_fit=0.6, exp=-0.5, plen=-0.4)


def log(msg):
    print("[%5.1fs] %s" % (time.time() - T0, msg), flush=True)


def signal_score(te, X):
    """Weighted sum of percentile-ranked signals; higher is better. Missing values
    rank as 0.5 (neutral). Uses only cleaned features -- no pedigree, no training."""
    raw = dict(tech=te.c_tech, rating=te.c_rating, apt=te.c_apt, ctc=te.c_ctc,
               kpi=te.c_kpi, nskills=te.c_nskills, role_fit=X.role_fit, exp=te.c_exp, plen=te.c_plen)
    total = pd.Series(0.0, index=te.index)
    for name, w in SIGNAL_WEIGHTS.items():
        # a negative weight rewards a *low* value: low experience -> low rank -> w*low is least negative -> highest total.
        total = total + w * R.pctrank(raw[name], index=te.index)
    return total


def model_score(P, frames):
    """De-biased target (Track A's scorer B) on the clipped scale, fit by three different
    families; return a percentile-averaged prediction for each frame. No pedigree columns."""
    Xtr, y_fit, coef = P["Xtr"], P["y_fit"], P["coef"]
    cols_b = [c for c in Xtr.columns if c not in R.REMOVED]
    y_deb = y_fit - (Xtr[R.REMOVED].fillna(0) * coef[R.REMOVED]).sum(axis=1)
    med = Xtr[cols_b].median()
    mat = lambda F: F[cols_b].fillna(med)

    # shallow histogram trees echo Track A's regularisation; RF averages many deeper trees.
    models = [
        Ridge(alpha=1.0),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=SEED,
                              n_jobs=C.NUM_THREADS),
        HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=8,
                                      min_samples_leaf=150, random_state=SEED),
    ]
    for m in models:
        m.fit(mat(Xtr), y_deb)
    log("trained Ridge + RandomForest + HistGradientBoosting on the de-biased target")

    out = []
    for F in frames:
        preds = [R.pctrank(m.predict(mat(F)), index=F.index) for m in models]
        out.append(sum(preds) / len(preds))
    return out


def build_quality(P):
    """Consensus of the signal scorecard and the diversified model ensemble; returns
    (dev_quality, test_quality). The test score is calibrated onto the clipped
    post_hire_score scale, the dev score is the raw consensus (ranking-invariant)."""
    te, dv, dw = P["te"], P["dv"], P["dw"]
    Xte, Xdv = P["Xte"], P["Xdv"]

    model_dv, model_te = model_score(P, [Xdv, Xte])
    sig_dv, sig_te = signal_score(dv, Xdv), signal_score(te, Xte)

    # dev harness (old regime): report each block and the consensus
    log("dev signal block: " + C.fmt(C.dev_metrics(dv, sig_dv, dw)))
    log("dev model block:  " + C.fmt(C.dev_metrics(dv, model_dv, dw)) + "   <- lower is expected: de-biased on old-regime dev")
    cons_dv = 0.5 * R.pctrank(sig_dv, index=dv.index) + 0.5 * model_dv
    log("dev consensus (C):" + C.fmt(C.dev_metrics(dv, cons_dv, dw)))

    cons_te = 0.5 * R.pctrank(sig_te, index=te.index) + 0.5 * model_te
    return cons_dv, R.to_score_scale(cons_te, P["y_fit"])   # test onto the clipped scale, like Track A


def main(data_dir=".", out_path="submission_c.csv"):
    prep = R.prepare(data_dir)
    log("parsed train %d, dev %d, test %d" % (len(prep["tr"]), len(prep["dv"]), len(prep["te"])))
    log("features: %d columns" % prep["Xtr"].shape[1])
    n_flag, n_win = R.exclusion_check(prep["dv"], prep["dw"])
    log("exclusion rules on dev: %d rows flagged, %d of them winners (must be 0)" % (n_flag, n_win))

    _, quality = build_quality(prep)
    ids, stats = R.shortlist(prep["te"], prep["Xte"], quality, prep["unseen"], prep["ob_bonus"])
    C.write_submission(ids, out_path, test_ids=prep["te"].candidate_id)
    for k, v in stats.items():
        log("%s: %s" % (k, v))
    log("wrote %s (%d rows)" % (out_path, len(ids)))


if __name__ == "__main__":
    main()
