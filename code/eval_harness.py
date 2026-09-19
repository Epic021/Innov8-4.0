"""
Leak-free measurement harness for the base quality model, plus the cross-cycle
identity ("backtracking to source") probe.

Two honest proxies for base-model quality (a change only counts if it holds on BOTH,
above the ~0.04 dev noise floor documented in STRATEGIES.md):
  * ARCHIVE CV : 5-fold out-of-fold ranking of the TRUE top-5% of post_hire_score
                 (train on the clipped target, score against the unclipped truth).
  * LEDGER raw : model trained on full train, scored on dev WITH pedigree -> P@150 etc.
                 (this reproduces main.py's "dev A raw pedigree" = 0.627 baseline).

The identity probe answers "can we recover a Vault candidate's known outcome by
matching them back to a train/dev record?" -- measured result: NO. The three files
are disjoint people (0 shared phones, 0 shared emails, 0 name+gradyear+institute
matches). See STRATEGIES.md "Backtracking to source".

Usage (from the folder holding the four CSVs, or from the repo root):
    python code/eval_harness.py            # baseline, 1 seed (fast)
    python code/eval_harness.py --full     # 5 seeds
    python code/eval_harness.py --probe    # identity / backtracking probe only
"""
import os, sys, pickle, time, tempfile
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C  # noqa: E402


def find_data():
    for d in (os.getcwd(), ROOT, os.path.join(ROOT, "corporate_heist_data")):
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    sys.exit("train.csv/dev.csv/test.csv not found in cwd, repo root, or corporate_heist_data/")


DATA = find_data()
CACHE = os.path.join(tempfile.gettempdir(), "heist_eval_cache.pkl")
TARGET_CLIP = 60.0
BASE_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=150,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   num_threads=C.NUM_THREADS, deterministic=True, force_row_wise=True, verbose=-1)
N_ROUNDS = 1500
SEEDS = (42, 43, 44, 45, 46)
_read = lambda n: pd.read_csv(os.path.join(DATA, n), dtype=str)


# ---- cross-cycle identity probe ("backtracking to source") -------------------
def identity_probe():
    tr, dv, te = _read("train.csv"), _read("dev.csv"), _read("test.csv")
    win = set(_read("dev_winners.csv").candidate_id)

    def keys(df):
        return pd.DataFrame({
            "cid": df.candidate_id,
            "pk": df.phone.map(C.phone_key), "ek": df.email.map(C.email_key),
            "nk": df.full_name.map(C.name_key),
            "gy": df.graduation_year.fillna("").astype(str).str.strip(),
            "inst": df.institute.map(C.canon_inst)})

    Ktr, Kdv, Kte = keys(tr), keys(dv), keys(te)

    def shared(a, b, col):
        sa = set(a[col][a[col] != ""]); sb = set(b[col][b[col] != ""])
        return len(sa & sb)

    print("rows: train %d  dev %d  test %d  winners %d" % (len(tr), len(dv), len(te), len(win)))
    print("shared phone  test<->train: %d   test<->dev: %d" % (shared(Kte, Ktr, "pk"), shared(Kte, Kdv, "pk")))
    print("shared email  test<->train: %d   test<->dev: %d" % (shared(Kte, Ktr, "ek"), shared(Kte, Kdv, "ek")))
    for cols in (["nk"], ["nk", "gy"], ["nk", "gy", "inst"]):
        a = Ktr.assign(k=Ktr[cols].agg("|".join, axis=1)); b = Kte.assign(k=Kte[cols].agg("|".join, axis=1))
        sa = set(a.k[(a[cols] != "").all(axis=1)]); sb = set(b.k[(b[cols] != "").all(axis=1)])
        print("shared %-16s test<->train: %d" % ("+".join(cols), len(sa & sb)))
    print("VERDICT: disjoint populations -- no source outcome to backtrack to.")


# ---- feature cache -----------------------------------------------------------
def build_cache():
    t = time.time()
    tr, dv, te = C.parse(_read("train.csv")), C.parse(_read("dev.csv")), C.parse(_read("test.csv"))
    dw = _read("dev_winners.csv")
    y = tr.post_hire_score.map(C.num).astype(float).values
    fb = C.FeatureBuilder().fit(tr, [dv, te])
    obj = dict(Xtr=fb.transform(tr), y=y, Xdv=fb.transform(dv), dv=dv, dw=dw,
               cols=list(fb.transform(tr.head(1)).columns))
    with open(CACHE, "wb") as f:
        pickle.dump(obj, f)
    print("built cache in %.1fs" % (time.time() - t))
    return obj


def load_cache(rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        return build_cache()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


# ---- metrics -----------------------------------------------------------------
def rank_metrics(score, relevant, ks=(150, 250, 1000)):
    rel = relevant[np.argsort(-score, kind="stable")].astype(float)
    nrel = int(relevant.sum())
    out = {}
    for k in ks:
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        out["P@%d" % k] = rel[:k].mean()
        out["NDCG@%d" % k] = (rel[:k] * disc).sum() / disc[:min(k, nrel)].sum()
    return out


def _predict(Xtr, ytr, Xev, params, n_seeds, rounds):
    ds = lgb.Dataset(Xtr, ytr)
    return np.mean([lgb.train(dict(params, seed=s), ds, num_boost_round=rounds).predict(Xev)
                    for s in list(SEEDS)[:n_seeds]], axis=0)


def cv_eval(cache, params=None, cols=None, n_seeds=1, rounds=None, folds=5, clip=TARGET_CLIP, seed=0):
    params = dict(BASE_PARAMS, **(params or {})); cols = cols or cache["cols"]; rounds = rounds or N_ROUNDS
    X, y = cache["Xtr"], cache["y"]
    yc = np.clip(y, clip, None) if clip is not None else y
    rel = y >= np.quantile(y, 0.95)
    fid = np.zeros(len(y), int)
    for i, f in enumerate(np.array_split(np.random.default_rng(seed).permutation(len(y)), folds)):
        fid[f] = i
    oof = np.zeros(len(y))
    for f in range(folds):
        m = fid != f
        oof[~m] = _predict(X[cols][m], yc[m], X[cols][~m], params, n_seeds, rounds)
    out = rank_metrics(oof, rel); out["rho"] = spearmanr(oof, y).correlation
    return out


def dev_eval(cache, params=None, cols=None, n_seeds=1, rounds=None, clip=TARGET_CLIP):
    params = dict(BASE_PARAMS, **(params or {})); cols = cols or cache["cols"]; rounds = rounds or N_ROUNDS
    y = cache["y"]; yc = np.clip(y, clip, None) if clip is not None else y
    pred = _predict(cache["Xtr"][cols], yc, cache["Xdv"][cols], params, n_seeds, rounds)
    return C.dev_metrics(cache["dv"], pred, cache["dw"])


def fmt(m):
    return "  ".join("%s=%.4f" % (k, v) for k, v in m.items())


if __name__ == "__main__":
    if "--probe" in sys.argv:
        identity_probe(); sys.exit()
    cache = load_cache(rebuild="--rebuild" in sys.argv)
    ns = 5 if "--full" in sys.argv else 1
    print("== BASELINE (n_seeds=%d) ==" % ns)
    print("Archive CV :", fmt(cv_eval(cache, n_seeds=ns)))
    print("Ledger raw :", fmt(dev_eval(cache, n_seeds=ns)))
