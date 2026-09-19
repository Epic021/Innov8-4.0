# deadlock/ — enhanced base, ensemble & analysis

A self-contained second-pass contribution on top of Tracks A/B/C. Everything trains
on `train.csv` only; `dev.csv` is scored once as a held-out check (no leakage, no dev
tuning). Run any script from a folder holding the four CSVs
(`train.csv`, `dev.csv`, `dev_winners.csv`, `test.csv`).

## TL;DR — the one thing that matters

**The quality model is at a noise ceiling.** `post_hire_score = f(features) + Gaussian
noise` with **noise std 15.5 > signal std 13.1** (SNR 0.71). The best achievable
top-5% precision is **≈0.32** (`analysis/noise_ceiling.py`), and *every* model variant
the team tried lands at CV 0.326–0.343 — i.e. on the ceiling. So model tuning is spent;
the real edges are **(1) cleaning correctness, (2) the deterministic debrief rules, and
(3) the de-biasing strength λ, which only the live leaderboard can settle.**

## Files

| File | What it is |
|---|---|
| `common.py` | Enhanced shared base. Adds three things to `code/common.py`: the **aptitude `%`/0–100 scale fix** (`parse_apt`), **`last_rating` word→number recovery** (`parse_rating`: Outstanding=5…Unsatisfactory=1), and the previously-unused **`major` / `company_type` / expected-CTC** features. Drop-in compatible with every track. |
| `ensemble.py` | Leakage-free 5-model rank fusion (2× LGB regression + LambdaMART + grid-searched HistGBM + ExtraTrees; HGB params picked by 3-fold CV **on train**). |
| `analysis/noise_ceiling.py` | Measures the achievable top-5% ceiling directly on train (≈0.32) — the headline. |
| `analysis/reverse_eng.py` | Reverse-engineers the generator: R²=0.42, clean homoscedastic Gaussian residual. |
| `analysis/creative_feats.py` | Career-trajectory + role-relative features (only +0.013 R² — confirms the ceiling). |
| `analysis/bias_audit.py` | Old-panel pet-preference premiums are **bias not merit** (merit-controlled ≈ raw); pedigree = 14.1% of the ranking signal. |
| `analysis/dist_shift.py` | KS+PSI+adversarial validation: train≈dev (AUC 0.508), test differs (0.629) **only** via the debrief columns. |
| `analysis/isolation.py` | Layered anomaly isolation (logical L1 + IsolationForest/LOF/ECOD on consistency residuals). Surfaced the aptitude bug + a missed duplicate. |
| `analysis/lambda_knob.py` | Raw↔de-biased interpolation; dev slides **0.62→0.49** as pedigree is removed — that slide *is* the bias premium. |
| `analysis/ablation.py` | Incremental dev impact of each cleaning fix. |

## Cleaning fixes (measured, `analysis/ablation.py`)

| Fix | Rows affected | Effect on de-biased dev | Verdict |
|---|---|---|---|
| **aptitude `%`/0–100 → ÷10** | 463 test, 943 train, **7 dev winners** | neutralised P@150 0.453→**0.473** | ship (correct + helps) |
| **`last_rating` word recovery** | 16% of rows (a top-3 signal) | neutral within ±0.04 dev noise | ship for correctness, not score |
| `major`/`company_type`/expected-CTC | all rows | neutral | ship (completeness) |

Both scale bugs are independently confirmed by the teammate audit workbook
(`updated/APPROACH.md`: `69%→6.9`, `Meets Expectations→3`).

## Final metrics (de-biased dev — the fair comparison)

| Track | P@150 | NDCG@150 | MAP@150 |
|---|---|---|---|
| A regression (shipped) | 0.480 | 0.528 | 0.293 |
| B ranking | 0.473 | 0.525 | 0.302 |
| C consensus | 0.407 | 0.436 | 0.217 |
| D (A+B+C fusion) | 0.433 | 0.494 | 0.276 |
| A raw *(pedigree sanity, unattainable on Vault)* | 0.613 | 0.669 | 0.487 |

All de-biased tracks cluster inside dev's ±0.04 noise — consistent with the ceiling.

## Recommendations

1. **Ship the aptitude fix** — a genuine correctness win that touches 7 dev winners.
2. **Stop tuning the model** — it is saturated; spend effort on the rules.
3. **Bet on `public_code_contributions`** — the one Vault-only, noise-free, debrief-endorsed signal.
4. **A/B the de-biasing λ on the live leaderboard** (re-uploads are free): ship λ=1 (principled), test a λ=0.5 hedge.

Deterministic (seeds 42/43/44, 2 threads). Libraries: pandas, numpy, scikit-learn, LightGBM, scipy.
