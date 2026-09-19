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

## Results (run 19 Sep, dev harness = Ledger, de-biased quality; random = 0.05)

| Track | P@150 | NDCG@150 | MAP@150 | overlap with A's 500 | runtime |
|---|---|---|---|---|---|
| A (shipped) | 0.480 | 0.521 | 0.287 | — | 20 s |
| B | 0.480 | 0.526 | 0.300 | 430 / 500 (A's top 100 all included) | 31 s |
| C | 0.400 | 0.453 | 0.221 | 344 / 500 | 29 s |

Components: B's ranker alone 0.453, P(top-5%) 0.447, P(top-15%) 0.447; C's scorecard alone 0.280, its model
block alone 0.447. A and B are tied; by the pre-agreed rule (>80% overlap -> ship the simpler one) Track A is the
submission. Notes: LightGBM caps a query group at 10,000 rows, so B's ranker uses fixed random groups of 1,000;
`to_score_scale` maps quality onto the clipped target distribution, which makes B/C's quantile-gap bonuses larger
in points (21.9) than A's (7.9) -- the rank semantics are the same but the two are not point-for-point identical.
