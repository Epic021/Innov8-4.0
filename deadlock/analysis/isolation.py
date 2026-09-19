"""
Layered anomaly ISOLATION for the Vault, SOTA-but-cheap (2 cores / <5 min / allowed libs only).

Core principle proven below: fabrications are OUTLIERS IN CONSISTENCY SPACE, not in magnitude space.
Running an outlier detector on raw features flags the BEST candidates (they are extreme but real);
running it on cross-field RESIDUALS flags fakes (their internal relationships are broken).

Layers:
  L0  messy-but-recoverable        -> parse, never flag (already handled by common.py)
  L1  logical impossibility        -> deterministic hard rules (high precision)  [fabricated]
  L2  statistical isolation        -> IsolationForest + LOF + ECOD on residuals  [soft fabricated]
  L3  duplicate isolation          -> exact keys + rapidfuzz blocked fuzzy match  [same person twice]
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
import numpy as np, pandas as pd
import common as C
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Consistency-residual features: each is ~0 for a coherent CV, large for a fake.
# These are RELATIONSHIPS between fields, so genuine excellence (high tech, high
# CTC, top institute) does NOT inflate them -- only internal contradictions do.
# ---------------------------------------------------------------------------
def residual_features(d):
    f = pd.DataFrame(index=d.index)
    yrs_since_grad = (C.NOW_YEAR - d.c_gy)
    f["age_grad_gap"]   = (d.c_gy - (C.NOW_YEAR - d.c_age)) - 22        # age at graduation vs 22
    f["exp_vs_age"]     = d.c_exp - (d.c_age - 18)                       # working before 18?
    f["path_vs_exp"]    = (d.c_psum - d.c_exp)                          # career-path years vs stated exp
    f["exp_vs_grad"]    = d.c_exp - (yrs_since_grad + 1)                # more exp than time since degree
    f["level_vs_exp"]   = d.c_lvl * 3.0 - d.c_exp                       # seniority faster than years
    f["ctc_vs_exp"]     = np.log1p(d.c_ctc) - (1.0 + 0.15 * d.c_exp)    # pay wildly ahead of experience
    f["ctc_ratio"]      = (d.c_ectc / d.c_ctc).clip(0, 10) - 1.2        # expected/current sanity
    f["emp_vs_exp"]     = d.c_nemp - (d.c_exp / 1.5 + 1)                # too many employers for the years
    f["tech_apt_gap"]   = (d.c_tech / 10.0) - d.c_apt                   # assessment vs aptitude coherence
    return f.replace([np.inf, -np.inf], np.nan)


def zfill(F):
    """Robust standardise (median/IQR), impute 0 = 'on-model'."""
    Z = pd.DataFrame(index=F.index)
    for c in F.columns:
        x = F[c]
        med = x.median()
        iqr = (x.quantile(0.75) - x.quantile(0.25)) or 1.0
        Z[c] = ((x - med) / iqr).fillna(0.0)
    return Z


def ecod_scores(Z):
    """ECOD (Li et al. 2022): parameter-free tail-probability outlier score.
    Sum of -log empirical-CDF tail across dims. Pure numpy, O(n log n), no lib needed."""
    n = len(Z)
    total = np.zeros(n)
    for c in Z.columns:
        x = Z[c].values
        # two-tailed: left tail via rank, right tail via reverse rank
        left = stats.rankdata(x, method="average") / (n + 1)
        right = stats.rankdata(-x, method="average") / (n + 1)
        tail = np.minimum(left, right)
        total += -np.log(np.clip(tail, 1e-12, None))
    return total


def statistical_isolation(d):
    """L2: ensemble outlier score on consistency residuals. Higher = more anomalous."""
    Z = zfill(residual_features(d))
    # IsolationForest
    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=SEED,
                          n_jobs=2, contamination="auto").fit(Z)
    s_if = -iso.score_samples(Z)                       # higher = more anomalous
    # LOF (density) -- novelty off, fit_predict style scoring
    lof = LocalOutlierFactor(n_neighbors=35, n_jobs=2)
    lof.fit_predict(Z)
    s_lof = -lof.negative_outlier_factor_
    # ECOD
    s_ec = ecod_scores(Z)
    # rank-average the three into one robust score in [0,1]
    def r(x): return stats.rankdata(x) / len(x)
    ens = (r(s_if) + r(s_lof) + r(s_ec)) / 3.0
    return pd.Series(ens, index=d.index), Z


# ---------------------------------------------------------------------------
def analyse(d, label, winners=None):
    print("\n" + "=" * 70 + "\n%s (n=%d)\n" % (label, len(d)) + "=" * 70)
    rule_fake = C.flag_fake(d)
    ens, Z = statistical_isolation(d)

    # Prove the core claim: raw-magnitude IF flags excellence, residual IF does not
    raw = pd.DataFrame({"tech": d.c_tech, "apt": d.c_apt, "rating": d.c_rating,
                        "ctc": np.log1p(d.c_ctc), "exp": d.c_exp}).fillna(d[[ ]].assign().median() if False else 0)
    rawZ = zfill(raw)
    iso_raw = IsolationForest(n_estimators=200, max_samples=256, random_state=SEED, n_jobs=2).fit(rawZ)
    s_raw = -iso_raw.score_samples(rawZ)
    top_raw = s_raw >= np.quantile(s_raw, 0.96)        # top ~4% "anomalies" by raw magnitude
    print("RAW-magnitude IF 'anomalies': mean_tech=%.1f  (these are the STRONG ones -> false alarms)"
          % d.c_tech[top_raw].mean())

    top_res = ens >= np.quantile(ens, 0.96)
    print("RESIDUAL-space ensemble 'anomalies': mean_tech=%.1f" % d.c_tech[top_res].mean())
    print("  overlap of residual-anomalies with logical-impossible rule: %d / %d"
          % ((top_res & rule_fake).sum(), rule_fake.sum()))

    # How much does L2 add beyond L1?
    extra = top_res & ~rule_fake
    print("L2 flags %d rows the rules miss; of those mean_tech=%.1f (vs pool %.1f)"
          % (extra.sum(), d.c_tech[extra].mean(), d.c_tech.mean()))

    if winners is not None:
        w = d.candidate_id.isin(winners)
        print("SAFETY: winners flagged by rule=%d  by L2-top4%%=%d  (both MUST be ~0)"
              % ((rule_fake & w).sum(), (top_res & w).sum()))
        print("        winners' mean residual-anomaly percentile = %.1f (low = good)"
              % (100 * ens[w].mean()))
    return ens, rule_fake


if __name__ == "__main__":
    tr_raw, dv_raw, dw, te_raw = C.load(".")
    tr, dv, te = C.parse(tr_raw), C.parse(dv_raw), C.parse(te_raw)
    win = set(dw.candidate_id)

    t0 = time.time()
    analyse(dv, "DEV / Ledger (validation: winners must survive)", winners=win)
    analyse(te, "TEST / Vault", winners=None)
    print("\n[%.1fs total]" % (time.time() - t0))
