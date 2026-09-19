# Innov8 4.0 — The Corporate Heist (preliminary round)

Ranked shortlist of the 500 Vault applicants most likely to become Nightingale's top 5% this cycle.

## Layout
- `code/main.py` — entry point; regenerates `submission.csv` (about 10 s, seeds fixed, 2 threads)
- `code/common.py` — parsers for the recruiter-typed fields, features, exclusion rules (fakes, duplicates, notice period, inflated titles), dev-set harness
- `code/sweep.py` — sensitivity sweep over the debrief-rule parameters (helper, not needed to reproduce)
- `code/nirf_2025_engineering_top50.csv` — external data (NIRF 2025 Engineering rankings)
- `submission.csv` — the shortlist (`rank,candidate_id`, 500 rows)
- `documentation.md` / `documentation.pdf` — solution documentation (template sections 1–9)
- `make_docs.py` — renders `documentation.md` to PDF (helper)

## Run
Put `train.csv`, `dev.csv`, `dev_winners.csv`, `test.csv` (from the Unstop dataset zip; not committed) in the repo root, then:

```
python code/main.py
python check_format.py submission.csv test.csv
```
