# The Corporate Heist - Solution Documentation
Team name: TEAM_NAME &nbsp;&nbsp;&nbsp; Unstop team ID: TEAM_ID

Members: MEMBER_1, MEMBER_2, MEMBER_3

## 1. Summary
The Archive and the Ledger were produced under Nightingale's **old** hiring standard; the Vault is the **new** one described in the insider's debrief, and every change the debrief describes is measurable in the files (section 4). We therefore (a) learn the old panel's rating function from `train.csv` with gradient-boosted models whose inputs include the six "pet preferences" as explicit columns, (b) score the Vault with those columns neutralised so pedigree can no longer move anyone's rank, and (c) apply the debrief literally on top: a code-contribution fast-track, a bonus for new-college candidates who topped the assessment and fit the role, the surviving old-boys' leg-up at the size measured in the Archive, and hard exclusions for fabricated profiles, duplicate people, notice periods over two months and inflated titles. Every rule was checked on the Ledger: the exclusions remove 0 of its 150 winners, and the base model ranks them with P@150 = 0.61 (random = 0.05). The 500 are ordered by the adjusted score.

## 2. Data cleaning and parsing
All fields are parsed by small documented functions in `code/common.py`; values actually found:

| Field | Variants observed | Rule |
|---|---|---|
| `technical_assessment` | `70`, `0.45`, `58.0 %`, `48/100`, `not taken` | `x/100` -> x; values <= 1 are a 0-1 scale (x100; their distribution matches the 0-100 rows); "not taken" -> missing |
| `total_experience` | `9.4 years`, `15.9 yrs`, `8.3`, `20+ years`, `9 months`, `197 months`, `Fresher` | months / 12, `Fresher` = 0, `20+` = 20 |
| `notice_period` | `30 days`, `1 month`, `Immediate`, `Available now`, `Serving notice - 15 days`, `4 months` | days; months x 30; immediate = 0 |
| `current_ctc`, `expected_ctc` | `25.7 lpa`, `INR 10.4 lakh`, `26.64L`, `Rs 15,70,000`, `$58,650`, `41` | lakh-units x 1e5; `$` x 95.72 (RBI reference rate); bare numbers are lakh |
| `career_path` | separators `->`, `>`, `\|`, arrow; durations `(1.2 yrs)`, `(1.0y)`, `[12 mo]` | list of (title, years): sum, length, highest seniority level (IC / senior / lead-manager / head-director-VP) |
| `kpi_met`, `overtime_history` | `Yes/No/Y/N/TRUE/FALSE/1/0/yes/no` | boolean |
| `last_rating` | `4/5`, `3.0`, `Unsatisfactory` | first number; text -> missing |
| `institute` | `IIT Delhi`, `I.I.T. Delhi`, `Indian Institute of Technology, Delhi`, `IITD`, `DTU (formerly DCE)`, `NSIT Delhi` | canonical key: punctuation, IIT/NIT expansions, alias table (710 spellings -> 410 institutes) |
| `skills` | `Python`/`python3`/`Python 3`; `SQL`/`T-SQL`/`Structured Query Language` | lower-case + alias table; top-60 skills as flags; role-fit = share of skills among the 20 most common skills of the role's top-quartile hires |
| `recruiter_note` | free text | only **24 distinct sentences** exist in all 33k notes: each becomes a flag |
| `public_code_contributions` | `22`, `~3`, `10+`, `3 PRs`, `40 merged PRs`, `not tracked`, blank | the number in the text (`~3` -> 3, `10+` -> 10, `40 merged PRs` -> 40); not tracked / blank -> unknown (no bonus) |
| `full_name`, `phone`, `email` | `Viral Hajariya` / `Hajariya, Viral`; `+91-61491-06394` / `6149106394`; `x.y971@` / `xy971+jobs@` | identity keys: sorted name tokens + last 10 phone digits; email local part without dots, `+tags`, case |

## 3. What drives a great hire - your findings
On the 20,000 Archive hires (`post_hire_score`: mean 48.8, top-5% line 81.9):

- Strongest signals: technical assessment (r = +0.21, 26% of model gain), last rating (+0.21), aptitude, current CTC, KPI met, skills count and role fit. Ranking the Ledger by technical assessment alone gives P@150 = 0.17; the full model gives 0.61 (NDCG@150 0.66, MAP@150 0.47, Spearman 0.43 between our score and the winners' official rank).
- More experience and longer career paths score *lower* (r = -0.13 / -0.12): the old panel's best-rated hires were early-career.
- The six "pet preferences" carry a clear premium in the old ratings (linear fit, points of post-hire score, everything else controlled): big-metro city +6.5, referral channel +4.4, employer with 5,000+ staff +4.1, gap-free CV (<= 2 idle years since graduation) +4.1, IIT +3.6 on top of +2.4 for any NIRF-2025 top-50 / IIT / NIT / BITS / IIIT institute, "proper" engineering degree +3.2. Ledger winners are 48% IIT vs 31% of the pool and 35% referrals vs 17%.
- A handful of institutes keep a further +5.0 after all of that: DTU, NIT Warangal, IIT Kharagpur, BITS Pilani, IISc, NSUT (residual +4.0 to +4.6 vs +3.0 for a generic IIT). We read these as the "old-boys' network" the debrief says survived.
- Recruiter "hype" sentences (mentored juniors / shipped 1M+ users / led migration / excellent system-design round ...) go with **lower** post-hire scores (-7 to -8 points); the model keeps them as learned features.
- Validation: 5-fold CV on the Archive plus the Ledger harness (P@150, NDCG@150, MAP@150 and the rank correlation among winners, against `dev_winners.csv`).

## 4. The current cycle
Everything below exists in the Vault and in **neither** train nor dev, so the debrief is a literal description of this cycle:

- `public_code_contributions` (new column): 602 rows with >= 24 ("dozens"), 899 with 12-23, 1,593 with 6-11, 2,420 with 1-5; 4,448 are blank or "not tracked".
- Notice periods above 60 days: train/dev maximum is 60 days; the Vault has 208 rows at 90-180 days.
- Inflated titles: the Archive contains no Head/Director/VP with under 10 years and no Lead/Manager with under 3; the Vault has 107 (e.g. "Head of AI, 3.4 years", "VP Engineering after an internship and an apprenticeship").
- 341 Vault rows come from 92 institutes Nightingale never hired from; 57 of them scored >= 85 on the technical assessment and pass every fake check, and their base quality is already high (median at the 93rd percentile).
- The excluded groups are traps for a naive model: inflated-title rows sit at the 72nd percentile of our quality score (24% would land in a plain top 5%), long-notice rows at the 59th (12%), duplicates at the 58th (12%); fabricated rows average 92 on the technical assessment.

How the shortlist accounts for it (`code/main.py`, `shortlist()`): the quality score is the neutralised old-panel model (section 6). A contribution bonus ramps linearly from 10 to 24 contributions and at 24+ equals the gap between the 65th percentile and the top-5% line of the quality score (11.1 points: "fast-tracked even if the rest is only good"). New-college candidates with technical assessment >= 85 whose skills or current title fit the applied role receive the gap between the median and the top-5% line (13.4 points). The old-boys' handful receives the +5.0 measured in the Archive. Notice > 60 days and inflated titles are removed outright.

## 5. Profiles you excluded and why
- **Fabricated (383 rows)**: any of four impossible combinations - graduated before age 17; experience exceeds age - 18; career-path years disagree with total experience by more than 35% (minimum 2 years); experience exceeds years since graduation + 1. Flagged rows average 92 on the technical assessment vs 56 for the rest ("look amazing and don't add up"). On the Ledger the same rules flag 84 rows and **0 winners**; after removing them the Vault's share of 90+ assessments (2.1%) matches the Ledger pool (1.8%), so we found no second population of fakes.
- **Duplicate people (244 rows dropped)**: same identity key (section 2). We keep the first occurrence in file order: in both Ledger duplicate groups that contain a winner, the winner is the first occurrence. Fuzzy name matching within institute and graduation year found no further pairs.
- **Cannot join inside two months (208 rows)**: notice period > 60 days.
- **Inflated titles (107 rows)**: head/director/VP level with < 10 years or lead/manager level with < 4 years of total experience (the Archive's envelope).
Overlap between rules is small; 918 rows are excluded in total.

## 6. Model and selection procedure
- Features (128): cleaned numerics, 24 note-sentence flags, applied role, top-60 skill flags, skill role-fit, title-fit, the assessment z-scored within the applied role (the assessment is role-specific and its link to performance is twice as strong for DS/ML as for Product/Full-Stack), CTC in INR and expected/current ratio, notice days, and the pedigree block (IIT, NIT, big-name, metro, referral, proper degree, big employer, gap-free) plus the old-boys flag.
- Two de-biased scorers, each a 3-seed average of LightGBM (1,500 rounds, learning rate 0.03, 7 leaves, >= 150 rows per leaf, 2 threads, deterministic), trained on the post-hire score clipped from below at 50 (the Archive median) so that model capacity goes to separating the top of the distribution rather than the bottom: **A** is trained on all features and scored with every candidate given the train-mode value of each pedigree flag (a counterfactual: pedigree carries no information); **B** is trained without the pedigree columns on a de-biased target (post-hire score minus the premiums of section 3). Quality = (A + B) / 2. On the Ledger A scores 0.61 with pedigree and 0.45 without; B 0.45 - lower is *expected* there because the Ledger's winners were chosen by the old panel.
- Selection: quality + bonuses (section 4), exclusions (section 5), the 500 highest scores in score order. Composition: 159 fast-tracked (>= 24 contributions), 86 on the partial ramp, 47 new-college stars (of 51 eligible), 101 from the old-boys' institutes; 306 of the 500 coincide with the plain quality top 500.
- Sensitivity (`code/sweep.py`, overlap with the final 500): contribution threshold 12 / 18 / 36 instead of 24 -> 396 / 457 / 461; "handful" cut 5 / 15 instead of 10 -> 459 / 482; bonus anchored at the 50th / 75th / 85th percentile instead of the 65th -> 449 / 468 / 414; star assessment cut 80 / 90 -> 487 / 477; old-boys bonus 0 / 2.5 -> 447 / 477. The list is stable to the star settings and moves mostly with how "dozens" and "only good" are read.

## 7. What did not work
- Zeroing every pedigree flag instead of using the train mode pushed candidates into 1%-populated tree branches and collapsed the Ledger score to 0.27; the mode counterfactual (0.44) agrees with a full marginalisation over sampled pedigree vectors (458 / 500 identical).
- A binary top-5% classifier (0.51 on the Ledger) and a top-15% classifier (0.54) did not beat seed-averaged regression (0.57 at the time); the regression is kept.
- Training on the raw score: clipping it at the median raised Ledger NDCG@150 from 0.63 to 0.65 and the winner-rank correlation from 0.29 to 0.42; squared, Huber and rank-Gaussian targets did not. Career-velocity, average-tenure and certification-to-role-fit features did not help either.
- Sentence embeddings (`all-MiniLM-L6-v2`) for recruiter notes were unnecessary: the notes are 24 templated sentences.
- A numeric-only parser for `public_code_contributions` silently treated 1,419 values such as `10+` and `3 merged PRs` as unknown and kept those candidates out of the fast-track; listing every non-numeric value caught it.
- Email addresses alone do not identify duplicates (all 10,000 Vault emails are unique); name + phone and the normalised email local part are needed.
- Optimising the Ledger score after neutralisation is a trap (it must fall); the Ledger was used only to validate parsing, the base model and the exclusion rules.

## 8. External resources and AI tools (mandatory)
- NIRF India Rankings 2025, Engineering (nirfindia.org/Rankings/2025/EngineeringRanking.html): top-50 list shipped as `code/nirf_2025_engineering_top50.csv`, used only for the "big-name institute" flag that is then neutralised.
- RBI USD/INR reference rate of 12 Sep 2026 (95.72): converts dollar CTC values (436 rows).
- Libraries: pandas, numpy, scikit-learn (Ridge), LightGBM; Python 3.11.
- AI assistant: Claude Code (Anthropic) was used to write and review the code, run the data exploration and draft this document. Every rule and threshold was checked by the team on the files.

## 9. How to run
`python code/main.py` from a folder containing `train.csv`, `dev.csv`, `dev_winners.csv` and `test.csv`; it writes `submission.csv` there. Runtime about 15 seconds on 2 CPU cores (< 1 GB RAM); seeds 42/43/44, LightGBM deterministic mode, 2 threads. A clean-folder rerun reproduces the shipped file byte-for-byte. Then `python check_format.py submission.csv test.csv`.
