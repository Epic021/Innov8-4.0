# Alternative strategies — Tracks B and C

Three of us each built a ranking strategy for the same shortlist. They share
**everything except the quality model**: the same parsing, features, hard
exclusions and debrief bonuses (all in `code/common.py` and, for B/C, the shared
re-ranker `tracks/rerank.py`). Holding the rulebook constant and varying only how
a candidate is turned into a single *quality* score isolates each approach and
keeps the three submissions directly comparable on the dev harness.

| Track | Where | Quality model | One-line thesis |
|---|---|---|---|
| **A** (Person A) | `code/main.py` | De-biased LightGBM **regression** of the old panel's rating (scorer A neutralised + scorer B de-biased, averaged) | Learn the old rating, strip pedigree, apply the debrief literally. |
| **B** (Person B) | `tracks/track_b.py` | LightGBM **LambdaMART ranker** + P(top-5%) + P(top-15%) classifiers, fused by percentile rank | We are graded on a ranking — so optimise rank and membership directly, not a point estimate. |
| **C** (Person C) | `tracks/track_c.py` | **Consensus** of an interpretable signal scorecard and a diversified Ridge + RandomForest + HistGradientBoosting ensemble (no LightGBM) | Don't bet the list on one GBM: fuse auditable signals with a multi-family ensemble. |

All three then apply the identical debrief layer — code-contribution fast-track,
new-college stars, the old-boys' leg-up — and the four hard exclusions
(fabricated profiles, duplicates, notice > 2 months, inflated titles).

## Files
- `rerank.py` — shared machinery for B and C: `prepare` (load/parse/features/premiums), pedigree neutralisation, `to_score_scale` (puts any quality on the post_hire_score scale so the +points bonuses mean the same thing), and `shortlist` (the debrief bonuses + exclusions, arithmetic identical to `code/main.py`).
- `track_b.py` — Person B's ranking + membership ensemble → `submission_b.csv`.
- `track_c.py` — Person C's consensus scorecard → `submission_c.csv`.

## Run
Exactly like Track A — from a folder holding `train.csv`, `dev.csv`, `dev_winners.csv`, `test.csv`:

```
python tracks/track_b.py     # writes submission_b.csv
python tracks/track_c.py     # writes submission_c.csv
```

Each prints the dev-harness metrics (P@150 / NDCG@150 / MAP@150), the exclusion
sanity check (0 winners may be flagged), the bonus sizes and the shortlist
composition, then writes its 500-row submission. Both are deterministic (fixed
seeds 42/43/44, 2 threads) and reuse `code/common.py` and `code/nirf_2025_engineering_top50.csv`.

Dependencies are the same as Track A: pandas, numpy, scikit-learn, LightGBM.
