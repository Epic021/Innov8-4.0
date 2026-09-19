# Strategies and metrics — Syntax Samurai, The Corporate Heist (19 Sep 2026)

Every number below was measured in-session on the real files. Harnesses:
- **dev** = the Ledger (`dev.csv`, 2,999 rows, 150 known winners): P@150 / NDCG@150 / MAP@150 / rho (Spearman between our score and the winners' official rank). Random P@150 = 0.05. Noise about ±0.04 on P@150.
- **CV** = 5-fold cross-validation on the Archive (`train.csv`, 20k rows, 1,000 top-5% hires): P@5% / NDCG@5%. Steadier than dev.
- "raw" = the old-regime model with pedigree; "de-biased" = the quality score actually applied to the Vault (expected to score lower on dev, because the Ledger's winners were picked by the old, biased panel).

Shipped: **Track A** (`code/main.py`). Final dev (after the aptitude/rating parser fixes of section 7.2): raw 0.620 / 0.673 / 0.484 / rho 0.44; de-biased blend 0.460 / 0.513 / 0.283 / rho 0.36; Archive CV P@5% 0.344 (0.329 before the fixes).

## 1. Rules layer (no ground truth on the Vault — validated indirectly)

| Strategy | Brief | Vault rows removed / affected | Evidence | Verdict |
|---|---|---|---|---|
| Fabricated-profile exclusion | 4 consistency rules: graduated before 17; experience > age-18; career-path years vs total experience off by >35%; experience > years since graduation + 1 | 383 | flagged rows average tech 92 vs 56; on dev 84 flagged, **0/150 winners**; after removal the Vault's 90+ share (2.1%) matches the dev pool (1.8%) | shipped |
| Duplicate-person exclusion | key = sorted name tokens + last-10 phone digits, or normalised email local part; keep first occurrence | 244 | 0 dev groups with 2 winners; winner = first occurrence in 2/2 dev cases; fuzzy matching found 0 extra pairs | shipped |
| Notice > 60 days | "can't join inside two months" | 208 | train/dev max = 60 days; Vault has 90-180 | shipped |
| Inflated titles | Head/Director/VP with <10 yrs or Lead/Manager with <4 yrs | 107 | 0 such rows in train/dev; these rows sit at the 72nd pct of quality (24% would enter a naive top-5%) | shipped |
| Contribution parser fix | `10+`, `3 PRs`, `40 merged PRs` parsed as numbers | 1,419 values | eligible >=24 went 436 -> 602; about 35 seats of the 500 | shipped |
| Pedigree neutralisation | six "pet preferences" set to their train mode at scoring; pedigree cannot move a rank | all rows | zero-all: dev 0.267; **mode: 0.440**; marginalisation: 0.440 (458/500 same list); retrain without: 0.420 | shipped (mode) |
| De-biased target (scorer B) | subtract the Ridge-measured premium per flag from the clipped target, retrain without pedigree | all rows | CV vs de-biased target 0.323 (best of the neutralisation variants); dev 0.493 | shipped (blended with A) |
| Code-contribution fast-track | linear ramp 10->24, full bonus = gap 65th pct -> top-5% line (8.0 pts) | 158 fast-tracked + 86 ramp in the 500 | sensitivity: threshold 12/18/36 -> 407/467/454 overlap; anchor 50/75/85th -> 464/470/416 | shipped |
| New-college stars | unseen institute and tech >= 85 and (skill role-fit >= median or title in role family); bonus = gap median -> top-5% line (9.2 pts) | 44 of 49 eligible in the 500 | cut 80/90 -> 486/478 overlap; role-fit off -> 497 | shipped |
| Old-boys' leg-up | DTU, NIT Warangal, IIT KGP, BITS Pilani, IISc, NSUT; +1.6 (premium on the clipped scale) | 76 in the 500 | residual +4.0-4.6 vs +3.0 generic IIT on the raw scale; bonus 0/2.5 -> 477/488 overlap | shipped |

## 2. Base model (dev, raw)

| Step | P@150 | NDCG | MAP | rho |
|---|---|---|---|---|
| technical assessment alone | 0.173 | - | - | - |
| aptitude / last rating alone | 0.073 / 0.107 | - | - | - |
| first LightGBM, ~50 features | 0.507 | 0.556 | - | - |
| + 24 recruiter-note sentence flags, top-60 skills, role one-hots (126 feats) | 0.547 | 0.587 | 0.372 | - |
| + 3 seeds, 15 leaves | 0.547 | 0.595 | 0.398 | - |
| + title-fit feature | 0.573 | 0.620 | 0.417 | 0.29 |
| + target clipped at 50, 7 leaves / 1500 rounds, assessment z-scored within role | 0.607 | 0.662 | 0.470 | 0.43 |
| + clip 60 | 0.633 | 0.682 | 0.493 | 0.44 |
| + 5 seeds, institute aliases (first upload) | 0.627 | 0.678 | 0.490 | 0.43 |
| **+ aptitude `%` scale fix and word-rating mapping (final upload; Archive CV 0.329 -> 0.344)** | **0.620** | **0.673** | **0.484** | **0.44** |

## 3. Model candidates tested (2-seed; CV P@5% / NDCG; dev raw P@150 / NDCG / rho)

| Candidate | Brief | CV | dev | Verdict |
|---|---|---|---|---|
| clip 60 | train on max(score, 60) | 0.335 / 0.376 | 0.633 / 0.683 / 0.43 | adopted (up on both) |
| clip 40 / 45 / 50 / 55 | lower clips | 0.335 / 0.373 (50) | 0.573 / 0.632; 0.593 / 0.646; 0.607 / 0.662; 0.607 / 0.662 | no |
| no clip | raw target | - | 0.580 / 0.627 / 0.29 | no |
| 7 leaves vs 15 vs 31 | tree depth | - | 0.607 / 0.655 vs 0.573 / 0.632 vs 0.567 / 0.622 | 7 adopted |
| 1000 / 2200 rounds | | 0.330 / 0.368; 0.339 / 0.378 | 0.607 / 0.660; 0.587 / 0.648 | split -> no |
| monotone constraints on tech/apt/rating/kpi | | 0.326 / 0.376 | 0.587 / 0.643 | no |
| feature_fraction 0.5 | | 0.338 / 0.376 | 0.587 / 0.648 | split -> no |
| LambdaRank, graded labels | rank objective | 0.342 / 0.379 | 0.587 / 0.643 / 0.40 | split -> no |
| CatBoost depth 4 | different learner | 0.335 / 0.378 | 0.580 / 0.644 / 0.45 | split -> no |
| rank-blend LightGBM + CatBoost | | 0.343 / 0.385 | 0.587 / 0.652 / 0.44 | split -> no |
| 3-way rank blend regression + LambdaRank + CatBoost (1 seed) | | 0.337 / 0.382 vs 0.329 / 0.372 | 0.600 / 0.656 vs 0.613 / 0.667 | split -> no |
| Ledger labels as extra training signal (classifier on train + dev winners, rank-blended) | | 0.335 / 0.379 vs 0.329 / 0.372 (+1 hit in 200) | not measurable (trains on dev) | no: inside CV noise, and it spends the validation set |
| binary top-5% / top-15% classifier | | - | 0.507 / 0.551; 0.540 / 0.590 | no |
| Huber / (y/100)^2 / rank-Gauss target | | - | 0.560; 0.567; 0.573 | no |
| 5 vs 3 seeds | variance only | - | 0.627 vs 0.633 (noise) | adopted for stability |

## 4. Feature candidates (dev raw, on the clip-40 base 0.573 / 0.632 / rho 0.38)

| Feature | Brief | dev | Verdict |
|---|---|---|---|
| assessment z-scored within applied role | the assessment is role-specific; its link to performance is 2x stronger for DS/ML than Product/Full-Stack | 0.587 / 0.645 / 0.40 | adopted |
| title-fit | current title in the applied role's family | +0.026 P@150 | adopted |
| 24 recruiter-note sentence flags | notes are 24 templated sentences; the "hype" sentences carry -7 to -8 pts | part of the +0.04 step | adopted |
| career velocity + average tenure | promotions per year, years per title | 0.567 / 0.626 | no |
| certification-to-role fit | | 0.567 / 0.628 | no |
| TF-IDF + SVD(24) of skills / certs / titles | | CV 0.328 vs 0.335; dev 0.580 vs 0.633 | no |
| sentence embeddings (all-MiniLM-L6-v2) | | unnecessary: notes are templated | no |

## 5. Tracks (identical rulebook, different quality model; de-biased dev = the fair comparison)

| Track | Brief | P@150 | NDCG | MAP | rho | overlap with A's 500 | runtime |
|---|---|---|---|---|---|---|---|
| **A** (shipped) | de-biased LightGBM regression: mode-neutralised scorer + de-biased-target scorer, averaged; 5 seeds | 0.480 | 0.521 | 0.287 | 0.36 | - | 20 s |
| B | LambdaMART + P(top-5%) + P(top-15%), percentile-rank fused (components 0.453 / 0.447 / 0.447) | 0.480 | 0.526 | 0.300 | 0.33 | 430 / 500; A's top-100 all inside | 31 s |
| C | untrained signal scorecard (0.280 alone) fused with Ridge / RandomForest / HistGB on the de-biased target (0.447 alone) | 0.400 | 0.453 | 0.221 | 0.21 | 344 / 500 | 29 s |
| D | rank fusion of A + B + C (teammate's exploratory harness) | see `tracks/README.md` | | | | | |

Merge rule agreed before the results: >80% overlap between the leading tracks -> ship the simpler one. A and B are tied and share 86% of the list, so A is shipped.

## 6. Hedges and sanity checks (shortlist level)

| Check | Result |
|---|---|
| Rank-ensemble over every reading of "dozens" / "only good" / star cut | 488 / 500 identical to the shipped list |
| Neutralising the six "hype" sentences as if they were "glowing referrals" | 487 / 500 |
| Duplicate-record merge (use the more complete record) | 0 pairs differ |
| Rows with missing assessment in the 500 | 2 |
| Rows with blank age in the 500 (two fake rules cannot fire) | 15, all internally consistent; blank-age rate identical across train / dev winners / test |
| Fuzzy institute variants counted as "new colleges" | only NIT Surathkal and CCS University (fixed by alias); about 6 rows, no membership change |
| Clean-folder reproduction (`python code/main.py` with only the 4 CSVs) | byte-identical `submission.csv`, 20 s (limit 300 s) |

## 7. Additional analysis (second exploration pass)

Sections 1–6 above are taken as ground truth. The items below were measured
separately on the same real files (train-only training; dev scored once). They are
mostly *diagnostic* — the headline is that **the model is at the noise ceiling**, so
the remaining edge is in cleaning correctness and the deterministic rule layer, not
in the quality model.

### 7.1 The noise ceiling — why every model candidate clusters at CV ≈ 0.33

`post_hire_score = f(features) + Gaussian noise`: on train the linear residual is
near-perfectly Gaussian (skew −0.15, kurtosis 0.16) and **homoscedastic** (std ≈17.5
flat across all fitted deciles) — the signature of an additive-noise generator.
Decomposition: total std 20.3, **noise std 15.5, signal std 13.1 → signal-to-noise
variance ratio 0.71 (noise dominates).**

Ranking train by an out-of-fold GBM against the *true* top-5% gives **P@5% = 0.323**;
a Monte-Carlo (rank by perfect signal vs a noisy realisation) gives **0.314 ± 0.014**.
That is the achievable ceiling. Section 3's model search (CatBoost, LambdaRank,
blends, Huber/rank-Gauss targets, monotone constraints, TF-IDF, career velocity) all
land at **CV P@5% 0.326–0.343** — i.e. the whole search sits *on the ceiling*. The
model is saturated; further model tuning is inside the noise. The high dev *raw*
number (0.63) is an old-regime artefact: the Ledger's winners were pedigree-selected,
so a pedigree-heavy model matches them; the Vault is de-biased, so expect Vault
precision far below 0.63, nearer the 0.3–0.45 band.

### 7.2 Cleaning fix the base is still missing: `aptitude_score` scale

`aptitude_score` is on 0–10 but recruiters also typed it as a percentage (`"83%"`)
or bare 0–100; the current `num()` parser leaves those as `83` instead of `8.3`.
This mis-scales **463 test rows + 943 train rows, and 7 of the 150 dev winners** — a
spurious 50–100 outlier fed straight to the model (same class of bug as the
`technical_assessment` 0–1 scale, but not yet fixed). A `parse_apt()` that divides
the `%`/0–100 values by 10 lifts the de-biased Track A neutralised P@150 0.453→0.467
and the blend 0.480→0.487, still 0 winners excluded. **Recommend shipping this fix.**

### 7.3 Two columns never fed to any model: `major`, `company_type`

Both are unused. They are job-relevant and **not** in the debrief's pet-preference
list, so they can be added as ordinary (non-neutralised) one-hots. `company_type`
carries a small real signal (Early-Stage/Funded Startup +2–3 pts vs Pvt Ltd); `major`
is near-flat. Adding them + expected-CTC level moved the numbers in 7.2.

### 7.4 Bias audit of the old panel (merit-controlled)

For each pet preference: raw gap ≈ merit-controlled OLS premium ≈ merit-matched gap,
so the premium is **bias, not ability**: big-name +5.7, metro **+7.2** (grows after
controlling for merit), gap-free +4.4, IIT +5.2, referral +4.3, employer +4.6,
degree +3.1, old-boys +6.2. Each marker is 1.3–1.6× over-represented in the old
top-5% vs the pool. Pedigree carries **14.1%** of the Ridge coefficient mass — the
share de-biasing removes.

### 7.5 Distribution shift, train vs dev vs test

KS < 0.05 and PSI < 0.02 for every numeric feature; **adversarial validation** AUC:
train~dev **0.508** (identical — a valid held-out old-regime sample), train-vs-test
**0.629**, dev-vs-test 0.607. The test shift is driven **only** by the debrief columns
(pcc present 0→55%, notice>60 0→2.1%, inflated titles 0→1.1%, unseen institutes
0→3.8%, fabrications 2.7→3.8%); the continuous candidate signal is stable, so the
train→test transfer is sound and the shift is exactly the new-cycle rules.

### 7.6 Anomaly isolation (diagnostic, not shipped)

Layered: L1 logical-impossibility (the 4 shipped rules; 0 winners, 92.5% recall on
injected synthetic fakes) + L2 statistical isolation (IsolationForest + LOF + ECOD)
on **consistency residuals** — outliers in *relationship* space, not magnitude, so
genuine excellence isn't flagged. A blanket L2 cut clips real winners (4/150), so L2
belongs as a graded risk **discount**, never a hard delete. L2 surfaced the aptitude
bug (7.2) and one duplicate the exact-key dedup missed (CH-2TJNBK ≡ CH-Y53B1D).

### 7.7 De-biased ensemble + the λ knob

A 5-model de-biased ensemble (2× LGB regression + LambdaMART + grid-searched HistGBM
+ ExtraTrees, params chosen by 3-fold CV **on train**) on the 7.2/7.3 features gives
de-biased dev **0.507 / 0.541 / 0.300** — above A (0.480), B (0.480), C (0.400).
A raw↔de-biased interpolation knob shows dev sliding **0.62→0.49** as pedigree is
removed (λ 0→1): that slide *is* the bias premium. Since re-uploads are free, ship
λ=1 (principled per the debrief) and A/B a λ=0.5 hedge on the live leaderboard —
the only way to learn whether the Vault rewards any residual pedigree.
## 8. Backtracking to source (cross-cycle identity) — tested, empty

Idea: recover a Vault candidate's known outcome by matching them back to their record in the
Archive/Ledger via the identity keys (`code/common.py`: name / phone / email), i.e. if the same
person recurs across cycles we already know their `post_hire_score` or winner status. Measured
with `code/eval_harness.py --probe` over all 10,000 Vault rows:

| Match test → train / dev | Shared |
|---|---|
| phone (last-10 digits) | **0 / 0** |
| email (normalised local part) | **0 / 0** |
| name + graduation-year + institute | **0** |
| name only | 473 — but 0 of them share a phone, email, or institute |

Verdict: **the three files are disjoint populations by design** — there is no source outcome to
backtrack to. The 473 shared bare names are common-name collisions, not recurring people. This is
consistent with the debrief ("the Vault is a new cycle"). Not usable; kept here so the idea is not
re-tried. (`code/eval_harness.py` also holds the leak-free base-model harness: Archive 5-fold CV +
Ledger raw-pedigree P@150, the two proxies any base-model change must improve together.)
