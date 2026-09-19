"""
The Corporate Heist -- Track D: the A + B + C ensemble (and a comparison report).

    python tracks/track_d.py        (run from the folder that holds train/dev/dev_winners/test .csv)

Thesis
------
Tracks A, B and C are deliberately different learners -- a de-biased regression, a
LambdaMART ranker + membership classifiers, and a consensus of interpretable
signals + a diversified ensemble. When several *good and different* rankers
disagree only at the margins, averaging their ranks is almost always at least as
good as the best single one and is more robust: no single model's blind spot
decides a borderline candidate. Track D is that average.

It rebuilds each track's quality on the same features (Track A's two de-biased
scorers are imported straight from `code/main.py`, so there is no second copy of
its hyper-parameters), fuses the three by **percentile rank**, and applies the
same shared debrief + exclusions. It also prints a side-by-side dev-harness table
for A / B / C / D and the overlap of each track's top-500 with the ensemble, so
the team can choose what to submit.

Deterministic; reuses `code/common.py`, `code/main.py` and `tracks/rerank.py`.
"""
import os
import sys
import time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                  # rerank / track_b / track_c
sys.path.insert(0, os.path.join(_HERE, "..", "code"))      # common / main
import common as C     # noqa: E402
import rerank as R     # noqa: E402
import main as A       # noqa: E402  (Track A building blocks: fit_avg + its LGB params, no side effects on import)
import track_b         # noqa: E402
import track_c         # noqa: E402

T0 = time.time()


def log(msg):
    print("[%5.1fs] %s" % (time.time() - T0, msg), flush=True)


def scorer_ab(P):
    """Track A's quality on dev and test: scorer A (all features, pedigree neutralised) and
    scorer B (de-biased target, pedigree dropped), averaged. Mirrors `code/main.py::prepare`
    using that module's own `fit_avg` and LightGBM parameters."""
    Xtr, Xdv, Xte, y_fit, coef = P["Xtr"], P["Xdv"], P["Xte"], P["y_fit"], P["coef"]
    all_cols = list(Xtr.columns)
    pred_a = A.fit_avg(Xtr, y_fit, all_cols)
    nmap = R.neutral_map(Xtr)
    neut = lambda F: R.neutralize(F, nmap)
    y_deb = y_fit - (Xtr[R.REMOVED].fillna(0) * coef[R.REMOVED]).sum(axis=1)
    cols_b = [c for c in all_cols if c not in R.REMOVED]
    pred_b = A.fit_avg(Xtr, y_deb, cols_b)
    dev_q = pd.Series(0.5 * (pred_a(neut(Xdv)) + pred_b(Xdv)), index=P["dv"].index)
    te_q = pd.Series(0.5 * (pred_a(neut(Xte)) + pred_b(Xte)), index=P["te"].index)
    log("Track A rebuilt (scorer A neutralised + scorer B de-biased)")
    return dev_q, te_q


def ens(parts, index):
    """Ensemble = mean of the component percentile ranks (rank fusion)."""
    return sum(R.pctrank(p, index=index) for p in parts) / len(parts)


def main(data_dir=".", out_path="submission_d.csv"):
    prep = R.prepare(data_dir)
    log("parsed train %d, dev %d, test %d | features: %d columns"
        % (len(prep["tr"]), len(prep["dv"]), len(prep["te"]), prep["Xtr"].shape[1]))
    n_flag, n_win = R.exclusion_check(prep["dv"], prep["dw"])
    log("exclusion rules on dev: %d rows flagged, %d of them winners (must be 0)" % (n_flag, n_win))

    a_dv, a_te = scorer_ab(prep)
    b_dv, b_te = track_b.build_quality(prep)
    c_dv, c_te = track_c.build_quality(prep)
    dv, dw, te = prep["dv"], prep["dw"], prep["te"]

    # --- side-by-side dev comparison (dev is the old-regime Ledger; de-biased tracks score
    #     lower here by design -- it is a sanity check, not the optimisation target) ---
    d_dv = ens([a_dv, b_dv, c_dv], dv.index)
    log("=== dev harness: P@150 / NDCG@150 / MAP@150 / rho ===")
    for name, q in [("A regression", a_dv), ("B ranking", b_dv), ("C consensus", c_dv), ("D ensemble", d_dv)]:
        log("  %-13s %s" % (name, C.fmt(C.dev_metrics(dv, q, dw))))

    # --- test ensemble -> submission ---
    quality = R.to_score_scale(ens([a_te, b_te, c_te], te.index), prep["y_fit"])
    ids, stats = R.shortlist(te, prep["Xte"], quality, prep["unseen"], prep["ob_bonus"])
    C.write_submission(ids, out_path, test_ids=te.candidate_id)

    # --- how much each single track would agree with the ensemble on the actual shortlist ---
    ens_ids = set(ids)

    def top500(qte):
        q = R.to_score_scale(qte, prep["y_fit"])
        tids, _ = R.shortlist(te, prep["Xte"], q, prep["unseen"], prep["ob_bonus"])
        return set(tids)

    log("=== top-500 overlap with the ensemble ===")
    for name, qte in [("A", a_te), ("B", b_te), ("C", c_te)]:
        log("  D vs %s: %d / %d" % (name, len(ens_ids & top500(qte)), C.K_TEST))

    for k, v in stats.items():
        log("%s: %s" % (k, v))
    log("wrote %s (%d rows)" % (out_path, len(ids)))


if __name__ == "__main__":
    main()
