# The Corporate Heist - Solution Documentation
Team name: TEAM_NAME &nbsp;&nbsp;&nbsp; Unstop team ID: TEAM_ID
Members: MEMBER_1, MEMBER_2, MEMBER_3

## 1. Summary
The Archive and the Ledger were produced under Nightingale's **old** hiring standard; the Vault is the **new** one described in the insider's debrief. We therefore (a) learn the old panel's rating function from `train.csv` with a gradient-boosted model whose inputs include the six "pet preferences" as explicit columns, (b) score the Vault with those six columns neutralised, so pedigree can no longer move anyone's rank, and (c) apply the debrief literally as rules on top: code-contribution fast-track, new-college stars, the surviving old-boys' leg-up, and hard exclusions for fabricated profiles, duplicate people, notice periods over two months and inflated titles. Every rule was checked against the Ledger: our exclusions remove 0 of the 150 known winners, and the base model ranks the Ledger's winners with P@150 = 0.55 (random = 0.05). The final 500 are ordered by the adjusted score.

## 2. Data cleaning and parsing
All fields are parsed by small, documented functions in `code/common.py`; examples of values actually found:

| Field | Variants observed | Rule |
|---|---|---|
| `technical_assessment` | `70`, `0.45`, `58.0 %`, `48/100`, `not taken` | `x/100` -> x; values <= 1 are a 0-1 scale (x100; verified: their distribution matches the 0-100 rows); "not taken" -> missing |
| `total_experience` | `9.4 years`, `15.9 yrs`, `8.3`, `20+ years`, `9 months`, `197 months`, `Fresher` | months / 12, `Fresher` = 0, `20+` = 20 |
| `notice_period` | `30 days`, `1 month`, `Immediate`, `Available now`, `Serving notice - 15 days`, `4 months` | days; months x 30; immediate = 0 |
| `current_ctc` / `expected_ctc` | `25.7 lpa`, `INR 10.4 lakh`, `26.64L`, `Rs 15,70,000`, `$58,650`, `41` | lakh-units x 1e5; `$` x 95.72 (RBI reference rate); bare numbers are lakh |
| `career_path` | separators `->`, `>`, `\|`, arrow; durations `(1.2 yrs)`, `(1.0y)`, `[12 mo]` | list of (title, years); sum, length, max seniority level |
| `kpi_met`, `overtime_history` | `Yes/No/Y/N/TRUE/FALSE/1/0/yes/no` | boolean |
| `last_rating` | `4/5`, `3.0`, `Unsatisfactory` | first number; text -> missing |
| `institute` | `IIT Delhi`, `I.I.T. Delhi`, `Indian Institute of Technology, Delhi`, `IITD`, `DTU (formerly DCE)`, `NSIT Delhi` | canonical key (punctuation, IIT/NIT expansions, alias table) |
| `skills` | `Python`, `python3`, `Python 3`; `SQL`, `T-SQL`, `Structured Query Language` | lower-case + alias table; top-60 skills as flags |
| `recruiter_note` | free text | only **24 distinct sentences** exist in all 33k notes: each becomes a binary flag |
| `public_code_contributions` | `22`, `~3`, `not tracked`, blank | number; `~3` -> 3; not tracked / blank -> unknown (no bonus) |

## 3. What drives a great hire - your findings
On the 20,000 Archive hires (`post_hire_score`, mean 48.8, top-5% line 81.9):
- Strongest single signals: technical assessment (r = +0.21), last rating (+0.21), aptitude, KPI met; ranking the Ledger by technical assessment alone already gives P@150 = 0.17.
- More experience and longer career paths score *lower* (r = -0.13 / -0.12): the old panel's best-rated hires were early-career.
- The six "pet preferences" carry a clear premium in the old ratings: IIT +5.0 points, big-metro city +5.0, referral channel +4.3, employer with 5,000+ staff +3.9, "proper" engineering degree +2.9, gap-free CV (7th most important feature). Ledger winners are 48% IIT vs 31% of the pool, and 35% referrals vs 17%.
- A handful of institutes keep a premium even after controlling for everything else (residual +4.0 to +4.6 vs +3.0 for a generic IIT): DTU, NIT Warangal, IIT Kharagpur, BITS Pilani, IISc, NSUT. We read these as the "old-boys' network" that the debrief says survived.
- Recruiter "hype" sentences (mentored juniors / shipped 1M+ users / led migration / excellent system-design round ...) are associated with **lower** post-hire scores (-7 to -8 points); the model keeps them as learned features.
- Validation: 5-fold CV on the Archive and the Ledger harness (P@150, NDCG@150, MAP@150 against `dev_winners.csv`). Base model on the Ledger: P@150 = 0.55, NDCG@150 = 0.59.

## 4. The current cycle
What is different in the Vault, verified on the files (none of these exist in train or dev):
- `public_code_contributions` (new column): 4,448 rows not tracked; 436 rows with >= 24 ("dozens"), 1,097 with >= 12.
- Notice periods above 60 days: 0 in train/dev, 208 in the Vault (90-180 days).
- Inflated titles (Head/Director/VP with < 10 years, Lead/Manager with < 4): 0 in train, 107 in the Vault.
- 383 Vault rows come from 92 institutes Nightingale never hired from; 64 of them scored >= 85 on the technical assessment and pass all fake checks.
How the shortlist accounts for it: pedigree columns are set to the same value for every candidate before scoring (neutralised); a code-contribution bonus ramps from 10 to 24 contributions and at 24+ equals the gap between the 65th percentile and the top-5% line of the neutralised score ("fast-tracked even if the rest is only good"); new-college candidates with technical assessment >= 85 and above-median role fit receive the gap between the 70th percentile and the top-5% line; the old-boys' handful keeps +1.5 points. Notice > 60 days and inflated titles are excluded outright.

## 5. Profiles you excluded and why
- **Fabricated (383 rows)**: any of four impossible combinations - graduated before age 17; experience exceeds age - 18; career-path years disagree with total experience by more than 35% (min 2 years); experience exceeds years since graduation + 1. These profiles average 92 on the technical assessment vs 56 for the rest ("look amazing and don't add up"). On the Ledger the same rules flag 84 rows and **0 winners**.
- **Duplicate people (244 rows dropped)**: same normalised name + phone digits, or same email local part after removing dots, `+tags` and case (`Viral Hajariya` / `Hajariya, Viral`, `abhishek.gehlot971` / `abhishekgehlot971`). We keep the first occurrence in file order; in both Ledger duplicate groups that contain a winner, the winner is the first occurrence.
- **Cannot join inside two months (208 rows)**: notice period > 60 days.
- **Inflated titles (107 rows)**: see section 4; the Archive contains no Head/Director/VP with under 10 years.

## 6. Model and selection procedure
- Features (126): cleaned numerics, 24 recruiter-note sentence flags, applied role, top-60 skill flags, a role-fit score (share of the candidate's skills among the 20 most common skills of the role's top-quartile hires), CTC in INR and expected/current ratio, and the pedigree block (IIT, NIT, NIRF-2025 big-name, metro, referral, proper degree, big employer, gap-free) plus the old-boys flag.
- Model: LightGBM regression on `post_hire_score` (600 rounds, lr 0.03, 31 leaves, min 40 rows per leaf, seed 42, 2 threads). Trains in about one second.
- Selection: neutralised score + bonuses (section 4); exclusions (section 5); the 500 highest scores, ranked by score. Composition of the final 500: 116 fast-tracked (>= 24 contributions), 79 on the partial ramp, 22 new-college stars, 140 from the old-boys' institutes; 352 of the 500 coincide with the plain neutralised model's top 500.

## 7. What did not work
- Sentence embeddings (`all-MiniLM-L6-v2`) for recruiter notes were unnecessary: the notes are built from 24 templated sentences, so exact flags are lossless and free.
- Optimising the Ledger score after neutralisation is a trap: the Ledger is an old-regime pool, so removing pedigree *must* lower its score (0.55 -> lower); we used the Ledger only to validate parsing, the base model and the exclusion rules.
- Email addresses alone do not identify duplicates (all 10,000 Vault emails are unique); name + phone and normalised email local part are needed.

## 8. External resources and AI tools (mandatory)
- NIRF India Rankings 2025, Engineering (nirfindia.org/Rankings/2025/EngineeringRanking.html) - top-50 list shipped as `code/nirf_2025_engineering_top50.csv`, used for the "big-name institute" flag.
- RBI USD/INR reference rate, 12 Sep 2026 (95.72) - used to convert dollar CTC values.
- Libraries: pandas, numpy, scikit-learn, LightGBM (Python 3.11).
- AI assistant: Claude Code (Anthropic) was used to write and review the code, run the data exploration and draft this document. All rules and thresholds were checked by the team on the files.

## 9. How to run
`python code/main.py` from a folder containing `train.csv`, `dev.csv`, `dev_winners.csv`, `test.csv`. Writes `submission.csv` to that folder. Runtime about 5 seconds on 2 CPU cores; all seeds fixed (42), LightGBM in deterministic mode. Then `python check_format.py submission.csv test.csv`.
