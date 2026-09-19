"""
The Corporate Heist -- Track B (Person B): direct ranking + top-k membership.

    python tracks/track_b.py        (run from the folder that holds train/dev/dev_winners/test .csv)

Thesis
------
Track A regresses the old panel's *continuous* rating and then de-biases it. But
the thing we are graded on is a **ranking**: land the people who become the top
5%, high in the list. So Track B optimises rank and membership directly instead
of a point estimate:

  1. a LightGBM **LambdaMART ranker** (`objective="lambdarank"`) trained to order
     the whole Archive by a graded relevance built from post_hire_score
     (top-1% = 3, top-5% = 2, top-15% = 1, else 0) -- it optimises NDCG, the
     ranking metric itself;
  2. two LightGBM **membership classifiers** -- P(top 5%) and P(top 15%) -- the
     "is this a great hire?" question as a probability (documentation section 7
     tried these; here they are ensembled with the ranker rather than used alone).

Each component is a 3-seed average, trained on every feature and scored with the
pedigree block neutralised to the train mode -- the same counterfactual as Track
A's scorer A, so pedigree cannot move a rank. The three components are fused by
averaging their percentile ranks; that fused score is the Track-B **quality**.

Everything above the quality score -- the code-contribution fast-track, the
new-college stars, the old-boys' leg-up and the four hard exclusions -- is the
shared debrief rulebook in `rerank.py`, identical to Track A.

Deterministic: seeds 42/43/44, LightGBM deterministic mode, 2 threads.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                  # rerank.py
sys.path.insert(0, os.path.join(_HERE, "..", "code"))      # common.py
import common as C   # noqa: E402
import rerank as R   # noqa: E402

T0 = time.time()

# Shallow trees + many rounds, the same regularisation Track A settled on (num_leaves 7,
# min_data_in_leaf 150): with pedigree neutralised, deep trees over-fit the Archive's quirks.
LGB_BASE = dict(learning_rate=0.03, num_leaves=7, min_data_in_leaf=150, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=C.NUM_THREADS,
                deterministic=True, force_row_wise=True, verbose=-1)
N_ROUNDS = 1500
SEEDS = (42, 43, 44)


def log(msg):
    print("[%5.1fs] %s" % (time.time() - T0, msg), flush=True)


def relevance(y):
    """Graded relevance for LambdaMART from the post_hire_score quantiles."""
    q85, q95, q99 = y.quantile(0.85), y.quantile(0.95), y.quantile(0.99)
    r = np.zeros(len(y), dtype=int)
    r[(y >= q85).to_numpy()] = 1
    r[(y >= q95).to_numpy()] = 2
    r[(y >= q99).to_numpy()] = 3
    return r


def fit_ranker(X, rel, cols, group_size=1000):
    """3-seed LambdaMART. LightGBM caps a query group at 10,000 rows, so the Archive is split into
    fixed random groups of `group_size` (a fixed permutation keeps it deterministic). Returns predict(frame)."""
    perm = np.random.RandomState(SEEDS[0]).permutation(len(X))
    Xs, rs = X[cols].iloc[perm], np.asarray(rel)[perm]
    n = len(X)
    groups = [group_size] * (n // group_size) + ([n % group_size] if n % group_size else [])
    models = []
    for s in SEEDS:
        dset = lgb.Dataset(Xs, label=rs, group=groups)
        params = dict(LGB_BASE, objective="lambdarank", metric="ndcg", ndcg_eval_at=[C.K_DEV], seed=s)
        models.append(lgb.train(params, dset, num_boost_round=N_ROUNDS))
    return lambda F: np.mean([m.predict(F[cols]) for m in models], axis=0)


def fit_clf(X, label, cols):
    """3-seed binary classifier; returns predict-proba(frame)."""
    models = [lgb.train(dict(LGB_BASE, objective="binary", seed=s),
                        lgb.Dataset(X[cols], label=label), num_boost_round=N_ROUNDS) for s in SEEDS]
    return lambda F: np.mean([m.predict(F[cols]) for m in models], axis=0)


def build_quality(P):
    """Train the three components and return (dev_quality, test_quality) plus per-component
    dev scores for the log. All components are scored with pedigree neutralised."""
    Xtr, Xdv, Xte, y = P["Xtr"], P["Xdv"], P["Xte"], P["y"]
    cols = list(Xtr.columns)                      # all features; pedigree neutralised at scoring
    nmap = R.neutral_map(Xtr)
    neut = lambda F: R.neutralize(F, nmap)

    rel = relevance(y)
    rank_pred = fit_ranker(Xtr, rel, cols)
    clf5 = fit_clf(Xtr, (y >= y.quantile(0.95)).astype(int), cols)
    clf15 = fit_clf(Xtr, (y >= y.quantile(0.85)).astype(int), cols)
    log("trained ranker + P(top5%) + P(top15%), 3 seeds each")

    def fuse(F):
        Fn = neut(F)
        return (R.pctrank(rank_pred(Fn)) + R.pctrank(clf5(Fn)) + R.pctrank(clf15(Fn))).to_numpy() / 3.0

    # dev harness (old regime): report each component and the fused score
    dv, dw = P["dv"], P["dw"]
    Xdv_n = neut(Xdv)
    log("dev ranker:      " + C.fmt(C.dev_metrics(dv, rank_pred(Xdv_n), dw)))
    log("dev P(top5%):    " + C.fmt(C.dev_metrics(dv, clf5(Xdv_n), dw)))
    log("dev P(top15%):   " + C.fmt(C.dev_metrics(dv, clf15(Xdv_n), dw)))
    dev_q = pd.Series(fuse(Xdv), index=dv.index)
    log("dev fused (B):   " + C.fmt(C.dev_metrics(dv, dev_q, dw)) + "   <- lower is expected: dev is old-regime")

    test_raw = pd.Series(fuse(Xte), index=P["te"].index)
    return R.to_score_scale(test_raw, P["y_fit"])   # onto the clipped scale, like Track A


def main(data_dir=".", out_path="submission_b.csv"):
    prep = R.prepare(data_dir)
    log("parsed train %d, dev %d, test %d" % (len(prep["tr"]), len(prep["dv"]), len(prep["te"])))
    log("features: %d columns" % prep["Xtr"].shape[1])
    n_flag, n_win = R.exclusion_check(prep["dv"], prep["dw"])
    log("exclusion rules on dev: %d rows flagged, %d of them winners (must be 0)" % (n_flag, n_win))

    quality = build_quality(prep)
    ids, stats = R.shortlist(prep["te"], prep["Xte"], quality, prep["unseen"], prep["ob_bonus"])
    C.write_submission(ids, out_path, test_ids=prep["te"].candidate_id)
    for k, v in stats.items():
        log("%s: %s" % (k, v))
    log("wrote %s (%d rows)" % (out_path, len(ids)))


if __name__ == "__main__":
    main()
