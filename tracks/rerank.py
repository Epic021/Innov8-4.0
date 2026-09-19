"""
The Corporate Heist -- shared machinery for Tracks B and C.

Track A (`code/main.py`) is the "literal debrief" re-ranker. Tracks B and C keep
*exactly* the same problem framing -- the same parsing, the same features, the
same hard exclusions and the same debrief bonuses -- and differ only in how they
turn the cleaned candidate into a single **quality** score:

    Track A  pointwise de-biased LightGBM regression of the old panel's rating;
    Track B  direct learning-to-rank + top-k membership classifiers;
    Track C  a transparent consensus of interpretable signals and a
             diversified (non-LightGBM) model ensemble.

Holding the rulebook constant and varying only the quality model is deliberate:
it isolates each teammate's modelling contribution and keeps the three
submissions directly comparable on the dev harness.

This module provides that shared rulebook so Tracks B and C do not re-implement
(and drift from) it:

  * `prepare`             load / parse / build features / measure the old-panel
                          pedigree premiums -- everything a track needs before it
                          builds its own quality score;
  * `neutral_map`         the train-mode counterfactual for the pedigree block
                          (identical to `code/main.py`: pedigree cannot move a
                          rank);
  * `to_score_scale`      put any monotone quality on the post_hire_score scale so
                          the +points bonuses (old-boys, star, fast-track) mean the
                          same thing for every track;
  * `shortlist`           apply the debrief bonuses + hard exclusions to a quality
                          score and return the top 500. The maths mirrors
                          `code/main.py::shortlist` line for line.

Everything is deterministic (fixed seeds, 2 threads = judge box), reusing
`code/common.py` for the actual parsers, features, flags and dev harness.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# make the shared base (code/common.py) importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "code"))
import common as C  # noqa: E402

# Pedigree columns removed / neutralised for the new regime -- same set as Track A.
REMOVED = C.PEDIGREE + ["old_boys"]

# Only the top of the distribution matters: post_hire_score below 60 (~the Archive's 70th
# percentile) is clipped to 60 so models spend no capacity separating bad from mediocre hires.
# Same value and rationale as Track A (`code/main.py`); the pedigree premiums are measured, and
# the quality calibrated, on this clipped scale so every track shares one scale.
TARGET_CLIP = 60.0

# Debrief tunables (documentation sections 4 and 6). Identical defaults to Track A's
# `DEFAULTS`, minus `old_boys_bonus` (the old-boys premium is passed in explicitly,
# measured once in `prepare`).
DEFAULTS = dict(
    pcc_low=10,          # below this a contribution count "means nothing"
    pcc_full=24,         # "dozens of merged contributions or more" -> full fast-track bonus
    pcc_q=0.65,          # full bonus lifts a 65th-percentile ("only good") profile to the top-5% line
    star_tech=85,        # "topped the technical assessment"
    star_q=0.50,         # star bonus lifts a median profile to the top-5% line
    star_role_fit=True,  # "genuinely fit the role": skill role-fit >= median, or title in the role family
)


def pedigree_premiums(Xtr, y):
    """The old panel's premium (points of post_hire_score) for each pedigree flag,
    from a Ridge fit on train -- exactly as Track A measures it."""
    ridge = Ridge(alpha=1.0).fit(Xtr.fillna(Xtr.median()), y)
    return pd.Series(ridge.coef_, index=Xtr.columns)


def neutral_map(Xtr):
    """Train-mode value of every pedigree/old-boys column: the counterfactual signal
    every candidate is scored with so pedigree carries no information (Track A's
    trick -- the mode keeps predictions in the well-populated tree branches)."""
    return {c: float(Xtr[c].mode().iloc[0]) for c in REMOVED}


def neutralize(F, nmap):
    """Return a copy of feature frame F with the pedigree block set to `nmap`."""
    return F.assign(**nmap)


def to_score_scale(raw, y_train):
    """Map a monotone quality score onto the train post_hire_score distribution by
    percentile. Ranking is unchanged (so dev metrics are unaffected), but the score
    now lives in points -- so the old-boys +points premium and the quantile-gap
    bonuses in `shortlist` share one scale across all tracks."""
    raw = pd.Series(raw)
    pct = raw.rank(method="average", pct=True).to_numpy()
    ys = np.sort(np.asarray(y_train, dtype=float))
    idx = np.clip(np.round(pct * (len(ys) - 1)).astype(int), 0, len(ys) - 1)
    return pd.Series(ys[idx], index=raw.index)


def pctrank(a, index=None):
    """Percentile rank in [0, 1]; NaN -> 0.5 (neutral). Used to fuse heterogeneous
    component scores (a lambdarank score, a probability, a regression) on one axis."""
    s = pd.Series(a, index=index) if index is not None else pd.Series(a)
    return s.rank(pct=True).fillna(0.5)


def prepare(data_dir="."):
    """Load, parse, build features and measure the old-panel premiums.

    Returns a dict shared by every track: parsed frames (`tr`/`dv`/`te`), the winner
    ids (`dw`), the feature matrices (`Xtr`/`Xdv`/`Xte`), the raw target `y` and the
    clipped training target `y_fit`, the fitted `FeatureBuilder` `fb`, the pedigree
    premium vector `coef` (measured on the clipped target, like Track A), the old-boys
    bonus `ob_bonus`, and the `unseen` (new-college) mask.
    """
    tr_raw, dv_raw, dw, te_raw = C.load(data_dir)
    tr, dv, te = C.parse(tr_raw), C.parse(dv_raw), C.parse(te_raw)
    y = tr.post_hire_score.map(C.num).astype(float)
    y_fit = np.clip(y, TARGET_CLIP, None)

    fb = C.FeatureBuilder().fit(tr, [dv, te])
    Xtr, Xdv, Xte = fb.transform(tr), fb.transform(dv), fb.transform(te)

    # premiums on the clipped target -- the scale on which they are subtracted (Track C's
    # de-biasing) and the old-boys' bonus is added, matching Track A.
    coef = pedigree_premiums(Xtr, y_fit)
    ob_bonus = float(coef["old_boys"])
    unseen = ~te.c_inst.isin(set(tr.c_inst))
    return dict(tr=tr, dv=dv, te=te, dw=dw, fb=fb, Xtr=Xtr, Xdv=Xdv, Xte=Xte,
                y=y, y_fit=y_fit, coef=coef, ob_bonus=ob_bonus, unseen=unseen)


def exclusion_check(dv, dw):
    """Sanity line every track prints: the hard rules must flag 0 dev winners."""
    fdv = C.all_flags(dv)
    n_win_flagged = int(dv.candidate_id[fdv.excluded].isin(set(dw.candidate_id)).sum())
    return int(fdv.excluded.sum()), n_win_flagged


def shortlist(te, X, quality, unseen, ob_bonus, **kw):
    """Apply the debrief rules to a quality score and return (ordered ids, stats).

    `quality` is a per-candidate Series (any track's score, on the post_hire_score
    scale via `to_score_scale`); `X` is the *un-neutralised* feature frame (its
    old_boys / role_fit / title_fit columns drive the bonuses). The arithmetic is
    identical to `code/main.py::shortlist`, so all three tracks apply the same
    debrief; only `quality` differs.
    """
    P = dict(DEFAULTS, **kw)
    p = pd.Series(quality)
    f = C.all_flags(te)
    q = p.quantile
    top5_line = q(0.95)

    # code-contribution fast-track: a linear ramp from pcc_low..pcc_full contributions,
    # full bonus lifting a "pcc_q" profile to the top-5% line.
    pcc = te.c_pcc
    ramp = ((pcc - P["pcc_low"]) / (P["pcc_full"] - P["pcc_low"])).clip(0, 1).fillna(0.0)
    pcc_bonus = top5_line - q(P["pcc_q"])
    b_pcc = ramp * pcc_bonus

    # new-college stars: unseen institute, topped the assessment, genuinely fit the role.
    star = unseen & (te.c_tech >= P["star_tech"])
    if P["star_role_fit"]:
        star &= (X.role_fit >= X.role_fit.median()) | (X.title_fit > 0)
    star_bonus = top5_line - q(P["star_q"])
    b_new = star.astype(float) * star_bonus

    # surviving old-boys' leg-up, re-added at the size measured in the Archive.
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
        bonus_pts=dict(pcc=round(float(pcc_bonus), 1), star=round(float(star_bonus), 1),
                       old_boys=round(float(ob_bonus), 1)),
        top5_line=round(float(top5_line), 1),
        composition=dict(fast_tracked=int((pcc[sel] >= P["pcc_full"]).sum()),
                         partial_ramp=int(((pcc[sel] >= P["pcc_low"]) & (pcc[sel] < P["pcc_full"])).sum()),
                         new_college_stars=int(star[sel].sum()), stars_total=int((star & keep).sum()),
                         old_boys=int((X.old_boys[sel] > 0).sum())),
        overlap_with_plain_quality_top500=len(base_top & set(top)),
        mean_tech=round(float(te.c_tech[sel].mean()), 1),
    )
    return te.candidate_id.loc[top].tolist(), stats
