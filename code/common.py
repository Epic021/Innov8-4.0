"""
The Corporate Heist -- shared base used by every track.

Parsing of the recruiter-typed fields, feature construction, the hard exclusion
flags (fakes / duplicates / notice period / title inflation), the dev-set
scoring harness and the submission writer live here so that all tracks share
exactly the same cleaning and the same rules.

Everything is deterministic: fixed seeds, fixed thread count (2 = judge box).
"""
import os
import re
import sys
import numpy as np
import pandas as pd

SEED = 42
NUM_THREADS = 2
NOW_YEAR = 2026          # the Vault is this cycle (Sept 2026); train/dev/test ages line up with it
USD_INR = 95.72          # RBI reference rate, 12 Sep 2026 (see documentation, section 8)
K_TEST = 500
K_DEV = 150
HERE = os.path.dirname(os.path.abspath(__file__))

# Census million-plus metros + the NCR satellites that recruiters type as "big metros".
METRO_CITIES = {
    "mumbai", "bombay", "delhi", "new delhi", "bengaluru", "bangalore", "hyderabad",
    "chennai", "madras", "kolkata", "calcutta", "pune", "ahmedabad",
    "gurgaon", "gurugram", "noida",
}

# Institutes whose lift in train survives after controlling for everything else
# (residual +4.0..+4.6 vs +3.0 for a generic IIT) -- the "old-boys' network" handful.
OLD_BOYS = {"dtu", "nit warangal", "iit kharagpur", "bits pilani", "iisc", "nsut"}

SKILL_ALIASES = {
    "python 3": "python", "python3": "python", "py": "python",
    "t-sql": "sql", "structured query language": "sql", "mysql": "sql", "postgresql": "sql", "postgres": "sql",
    "dockerization": "docker", "k8s": "kubernetes", "js": "javascript", "reactjs": "react", "react.js": "react",
    "nodejs": "node", "node.js": "node", "ms excel": "excel", "amazon web services": "aws",
    "scikit learn": "scikit-learn", "sklearn": "scikit-learn", "tf": "tensorflow",
}


# ----------------------------------------------------------------------------
# Field parsers.  Every value is "as the recruiter typed it"; each parser
# documents the variants actually observed in the files.
# ----------------------------------------------------------------------------
def num(s):
    """First number in a string ('13+ yrs' -> 13, '₹15,70,000' -> 1570000)."""
    if pd.isna(s):
        return np.nan
    m = re.search(r"-?\d+(\.\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else np.nan


def parse_tech(s):
    """technical_assessment: '70', '0.45' (0-1 scale), '58.0 %', '48/100', 'not taken'."""
    if pd.isna(s):
        return np.nan
    t = str(s).strip().lower()
    if t in ("", "na", "n/a", "-", "none", "nan") or "not taken" in t or "skipped" in t:
        return np.nan
    m = re.match(r"(\d+(\.\d+)?)\s*/\s*100", t)
    if m:
        return float(m.group(1))
    v = num(t)
    if np.isnan(v):
        return np.nan
    return v * 100.0 if v <= 1.0 else v


def parse_exp_years(s):
    """total_experience: '9.4 years', '15.9 yrs', '8.3', '20+ years', '9 months', '197 months', 'Fresher'."""
    if pd.isna(s):
        return np.nan
    t = str(s).lower().strip()
    if "fresher" in t:
        return 0.0
    v = num(t)
    if np.isnan(v):
        return np.nan
    return v / 12.0 if "month" in t else v


def parse_notice_days(s):
    """notice_period: '30 days', '1 month', 'Immediate', 'Available now', 'Serving notice - 15 days'."""
    if pd.isna(s):
        return np.nan
    t = str(s).lower()
    if "immediate" in t or "available now" in t:
        return 0.0
    v = num(t)
    if np.isnan(v):
        return np.nan
    if "month" in t:
        return v * 30.0
    if "week" in t:
        return v * 7.0
    return v


def parse_ctc_inr(s):
    """current/expected CTC to absolute INR.
    '25.7 lpa' / 'INR 10.4 lakh' / '26.64L' -> lakh units; '₹15,70,000' -> absolute;
    '$58,650' -> USD * RBI rate; bare '41' -> lakh (all bare values are < 500)."""
    if pd.isna(s):
        return np.nan
    t = str(s).lower().strip()
    v = num(t)
    if np.isnan(v):
        return np.nan
    if "$" in t or "usd" in t:
        return v * USD_INR
    if "lpa" in t or "lakh" in t or re.search(r"\d\s*l\b", t):
        return v * 1e5
    if "₹" in t or "inr" in t or "rs" in t:
        return v * 1e5 if v < 500 else v
    return v * 1e5 if v < 500 else v


_PATH_SPLIT = re.compile(r"\s*(?:→|->|>|\|)\s*")
_PATH_ITEM = re.compile(r"(.*?)\s*[\(\[]\s*(\d+(\.\d+)?)\s*(yrs?|y|mo|months?)\s*[\)\]]\s*$")


def parse_path(s):
    """career_path -> [(title, years)].  Separators: → , -> , > , | ; durations: (1.2 yrs) (1.0y) [12 mo]."""
    if pd.isna(s):
        return []
    out = []
    for part in _PATH_SPLIT.split(str(s)):
        m = _PATH_ITEM.match(part.strip())
        if m:
            yrs = float(m.group(2))
            if m.group(4).startswith("mo"):
                yrs /= 12.0
            out.append((m.group(1).strip(), yrs))
    return out


def title_level(t):
    """Seniority ladder used for the title-inflation rule: 1 IC, 2 senior IC, 3 lead/manager, 4 head/director/VP/C-level."""
    t = str(t).lower()
    if re.search(r"\b(cto|ceo|coo|chief|vp|vice president|head|director)\b", t):
        return 4
    if re.search(r"\b(manager|lead|principal|staff|architect)\b", t):
        return 3
    if re.search(r"\bsenior\b|\bsr\b|\biii\b", t):
        return 2
    return 1


def yesno(series):
    m = {"yes": 1.0, "y": 1.0, "true": 1.0, "1": 1.0, "no": 0.0, "n": 0.0, "false": 0.0, "0": 0.0}
    return series.fillna("").astype(str).str.lower().str.strip().map(m)


def canon_inst(s):
    """Institute name as typed -> canonical key. Merges 'IIT Delhi' / 'I.I.T. Delhi' /
    'Indian Institute of Technology, Delhi' / 'IITD', DTU/DCE, NSIT/NSUT, BITS variants."""
    t = str(s).lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\bindian institute of technology\b", "iit", t)
    t = re.sub(r"\bnational institute of technology\b", "nit", t)
    t = re.sub(r"\bi i t\b", "iit", t)
    t = re.sub(r"\bn i t\b", "nit", t)
    short = {"b": "bombay", "d": "delhi", "m": "madras", "k": "kanpur", "kgp": "kharagpur",
             "r": "roorkee", "g": "guwahati", "h": "hyderabad"}
    t = re.sub(r"\biit ?(b|d|m|k|kgp|r|g|h)\b", lambda m: "iit " + short[m.group(1)], t)
    for a, b in (("birla institute of technology and science", "bits"),
                 ("delhi technological university", "dtu"),
                 ("delhi college of engineering", "dtu"),
                 ("netaji subhas university of technology", "nsut"),
                 ("netaji subhas institute of technology", "nsut"),
                 ("nsit delhi", "nsut"),
                 ("indian institute of science", "iisc"),
                 ("formerly dce", "")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def norm_skill(tok):
    t = tok.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return SKILL_ALIASES.get(t, t)


def skill_list(s):
    if pd.isna(s):
        return []
    return sorted({norm_skill(t) for t in str(s).split(",") if t.strip()})


def load_nirf():
    """NIRF India Rankings 2025 (Engineering), top 50 -- shipped as code/nirf_2025_engineering_top50.csv."""
    df = pd.read_csv(os.path.join(HERE, "nirf_2025_engineering_top50.csv"))
    return {canon_inst(x): int(r) for x, r in zip(df.institute, df["rank"])}


# ----------------------------------------------------------------------------
# Identity keys for duplicate detection.
# ----------------------------------------------------------------------------
def name_key(s):
    return " ".join(sorted(re.findall(r"[a-z]+", str(s).lower())))


def phone_key(s):
    d = re.sub(r"\D", "", str(s) if not pd.isna(s) else "")
    return d[-10:] if len(d) >= 10 else ""


def email_key(s):
    if pd.isna(s):
        return ""
    local = str(s).lower().split("@")[0]
    local = re.sub(r"\+.*$", "", local)
    return local.replace(".", "")


# ----------------------------------------------------------------------------
# Data loading / parsing
# ----------------------------------------------------------------------------
def load(data_dir="."):
    read = lambda n: pd.read_csv(os.path.join(data_dir, n), dtype=str)
    tr, dv, te = read("train.csv"), read("dev.csv"), read("test.csv")
    dw = read("dev_winners.csv")
    return tr, dv, dw, te


def parse(df):
    """Adds cleaned numeric columns (prefix c_) to a copy of the raw frame."""
    d = df.copy()
    d["c_tech"] = d.technical_assessment.map(parse_tech)
    d["c_apt"] = d.aptitude_score.map(num)
    d["c_rating"] = d.last_rating.map(num)
    d["c_kpi"] = yesno(d.kpi_met)
    d["c_exp"] = d.total_experience.map(parse_exp_years)
    d["c_age"] = d.age.map(num)
    d["c_gy"] = d.graduation_year.map(num)
    d["c_trn"] = d.trainings_last_year.map(num)
    d["c_trh"] = d.training_hours.map(num)
    d["c_awards"] = d.awards.fillna("").str.lower().str.strip().map(
        lambda v: 0.0 if v in ("", "-", "0", "none", "no") else 1.0)
    d["c_ot"] = yesno(d.overtime_history)
    d["c_enr"] = d.currently_enrolled.fillna("").str.lower().map(
        {"no_enrollment": 0.0, "part time course": 1.0, "full time course": 2.0})
    d["c_ljc"] = d.last_job_change.fillna("").str.lower().map(
        lambda v: 0.0 if v == "never" else (5.0 if ">" in v else num(v)))
    d["c_nemp"] = d.num_employers.map(num)
    P = d.career_path.map(parse_path)
    d["c_psum"] = P.map(lambda l: sum(y for _, y in l) if l else np.nan)
    d["c_plen"] = P.map(len)
    d["c_path_lvl"] = P.map(lambda l: max([title_level(t) for t, _ in l]) if l else 1)
    d["c_cur_lvl"] = d.current_title.fillna("").map(title_level)
    d["c_lvl"] = np.maximum(d["c_cur_lvl"], d["c_path_lvl"])
    d["c_skills"] = d.skills.map(skill_list)
    d["c_nskills"] = d["c_skills"].map(len)
    d["c_ncert"] = d.certifications.fillna("").map(lambda s: 0 if s.strip() in ("", "-") else len(s.split(",")))
    d["c_ctc"] = d.current_ctc.map(parse_ctc_inr)
    d["c_ectc"] = d.expected_ctc.map(parse_ctc_inr)
    d["c_ctc_ratio"] = d["c_ectc"] / d["c_ctc"]
    d["c_nd"] = d.notice_period.map(parse_notice_days)
    d["c_inst"] = d.institute.map(canon_inst)
    d["c_city"] = d.current_city.fillna("").str.lower().str.strip()
    if "public_code_contributions" in d.columns:
        d["c_pcc"] = pd.to_numeric(d.public_code_contributions.fillna("").str.replace("~", "", regex=False)
                                   .str.strip(), errors="coerce")   # 'not tracked' / blank -> NaN
    else:
        d["c_pcc"] = np.nan
    d["c_name_key"] = d.full_name.map(name_key)
    d["c_phone_key"] = d.phone.map(phone_key)
    d["c_email_key"] = d.email.map(email_key)
    return d


# ----------------------------------------------------------------------------
# Hard rules (the new panel's gates + data hygiene)
# ----------------------------------------------------------------------------
def flag_fake(d):
    """Profiles that 'look amazing and don't add up': internal inconsistencies between age,
    graduation year, total experience and the career path. Fires on 0 of 150 dev winners."""
    age_at_grad = d.c_gy - (NOW_YEAR - d.c_age)
    r1 = age_at_grad < 17
    r2 = d.c_exp > d.c_age - 18
    r3 = (d.c_psum - d.c_exp).abs() > np.maximum(2.0, 0.35 * d.c_exp)
    r4 = d.c_exp > (NOW_YEAR - d.c_gy) + 1
    return (r1 | r2 | r3 | r4).fillna(False)


def flag_notice(d):
    """'Anyone who can't join inside two months is dead to them this cycle.' train/dev max is 60 days."""
    return (d.c_nd > 60).fillna(False)


def flag_title_inflated(d):
    """'Titles climb faster than the years behind them.' In train (20k hires) no Head/Director/VP has
    < 10 years and no Lead/Manager has < 3 years; test contains ~120 such profiles."""
    return (((d.c_lvl >= 4) & (d.c_exp < 10)) | ((d.c_lvl == 3) & (d.c_exp < 4))).fillna(False)


def flag_dup_drop(d):
    """Same person entered twice under different IDs (case, phone format, email dots/+tags,
    'Last, First' ordering). Keep the FIRST occurrence in file order (matches both dev cases)."""
    k1 = d.c_name_key + "|" + np.where(d.c_phone_key != "", d.c_phone_key, d.c_email_key)
    ke = d.c_email_key
    return (pd.Series(k1, index=d.index).duplicated(keep="first") |
            ke.duplicated(keep="first") & (ke != ""))


def all_flags(d):
    f = pd.DataFrame(index=d.index)
    f["fake"] = flag_fake(d)
    f["notice_gt60"] = flag_notice(d)
    f["title_inflated"] = flag_title_inflated(d)
    f["dup_drop"] = flag_dup_drop(d)
    f["excluded"] = f.fake | f.notice_gt60 | f.title_inflated | f.dup_drop
    return f


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
PEDIGREE = ["iit", "nit", "bigname", "metro", "referral", "proper_deg", "big_emp", "no_gap"]
NUMERIC = ["c_tech", "c_apt", "c_rating", "c_kpi", "c_exp", "c_age", "c_gy", "c_trn", "c_trh", "c_awards",
           "c_ot", "c_enr", "c_ljc", "c_nemp", "c_psum", "c_plen", "c_lvl", "c_nskills", "c_ncert",
           "c_ctc", "c_ctc_ratio", "c_nd"]


class FeatureBuilder:
    """Fit on train (+ the other files for vocabularies), transform any parsed frame into a numeric matrix."""

    def __init__(self, n_skills=60):
        self.n_skills = n_skills

    def fit(self, tr, others):
        allf = pd.concat([tr] + list(others))
        notes = allf.recruiter_note.dropna().str.split(r"(?<=[.!?])\s+").explode().str.strip()
        self.sentences = [s for s in notes.value_counts().index if s]          # 24 templated sentences
        sk = allf.c_skills.explode().dropna()
        self.top_skills = list(sk.value_counts().head(self.n_skills).index)
        self.roles = sorted(allf.applied_role.dropna().unique())
        self.nirf = load_nirf()
        # role fit: the 20 most common skills among the top-quartile performers of each role in train
        y = tr.post_hire_score.map(num)
        self.role_skills = {}
        for r in self.roles:
            m = (tr.applied_role == r)
            top = tr[m & (y >= y[m].quantile(0.75))]
            self.role_skills[r] = set(top.c_skills.explode().dropna().value_counts().head(20).index)
        return self

    def transform(self, d):
        cols = {}
        for c in NUMERIC:
            cols[c[2:]] = d[c].astype(float)
        notes = d.recruiter_note.fillna("")
        for i, s in enumerate(self.sentences):
            cols["note_%02d" % i] = notes.str.contains(re.escape(s)).astype(float)
        for r in self.roles:
            cols["role_" + re.sub(r"[^a-z]", "_", r.lower())] = (d.applied_role == r).astype(float)
        for i, s in enumerate(self.top_skills):
            cols["skill_%02d" % i] = d.c_skills.map(lambda l, s=s: float(s in l))
        cols["role_fit"] = pd.Series([
            (len(set(sk) & self.role_skills.get(r, set())) / max(len(sk), 1)) if isinstance(sk, list) else 0.0
            for sk, r in zip(d.c_skills, d.applied_role)], index=d.index)
        # --- pedigree block: the six things the old panel inflated (+ IIT/NIT splits of "big name") ---
        inst = d.c_inst
        iit = inst.str.startswith("iit ")
        nit = inst.str.startswith("nit ")
        cols["iit"] = iit.astype(float)
        cols["nit"] = nit.astype(float)
        cols["bigname"] = (inst.isin(self.nirf) | iit | nit | inst.str.startswith("bits")
                           | inst.str.startswith("iiit") | (inst == "iisc")).astype(float)
        cols["metro"] = d.c_city.isin(METRO_CITIES).astype(float)
        cols["referral"] = d.recruitment_channel.fillna("").str.contains("Referral").astype(float)
        cols["proper_deg"] = d.degree.fillna("").str.lower().str.replace(".", "", regex=False).str.strip().isin(
            ["btech", "be", "bachelor of technology", "mtech", "master of technology", "ms", "phd", "me"]).astype(float)
        cols["big_emp"] = d.company_size.isin(["10000+", "5000-9999"]).astype(float)
        # "CVs without a single gap": years since graduation not covered by experience (<= 2 = gap-free).
        # 27% of train has a gap > 2 years and scores 5 points lower -- same size as the other pet preferences.
        idle = (NOW_YEAR - d.c_gy) - d.c_exp
        cols["no_gap"] = (idle <= 2).astype(float)
        # kept in the new regime
        cols["old_boys"] = inst.isin(OLD_BOYS).astype(float)
        return pd.DataFrame(cols, index=d.index)

    def neutralize(self, X, Xref):
        """Counterfactual for the new panel: every candidate gets the same pedigree signal.
        The shared value is the train mode of each flag, so predictions stay in the well-populated
        branches of the trees (zeroing everything pushed rows into 1%-branches and wrecked the ranking)."""
        Xn = X.copy()
        for c in PEDIGREE:
            Xn[c] = float(Xref[c].mode().iloc[0])
        return Xn


# ----------------------------------------------------------------------------
# Dev-set harness: dev.csv is an OLD-regime pool with 150 known winners.
# ----------------------------------------------------------------------------
def dev_metrics(dv, scores, dw, k=K_DEV, exclude=None):
    win = set(dw.candidate_id)
    o = pd.DataFrame({"id": dv.candidate_id.values, "s": np.asarray(scores)})
    if exclude is not None:
        o = o[~np.asarray(exclude)]
    o = o.sort_values("s", ascending=False).head(k)
    hits = o.id.isin(win).values.astype(float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg = (hits * disc).sum() / disc[: min(k, len(win))].sum()
    prec_at = np.cumsum(hits) / np.arange(1, k + 1)
    ap = (prec_at * hits).sum() / min(k, len(win))
    return {"P@%d" % k: hits.mean(), "NDCG@%d" % k: ndcg, "MAP@%d" % k: ap}


def fmt(m):
    return "  ".join("%s=%.3f" % (k, v) for k, v in m.items())


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_submission(ids, path="submission.csv", test_ids=None, k=K_TEST):
    ids = list(ids)
    assert len(ids) == k, "need exactly %d ids, got %d" % (k, len(ids))
    assert len(set(ids)) == k, "duplicate candidate_id in shortlist"
    if test_ids is not None:
        assert set(ids) <= set(test_ids), "id not in test.csv"
    pd.DataFrame({"rank": np.arange(1, k + 1), "candidate_id": ids}).to_csv(path, index=False)
    return path
