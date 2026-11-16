# Yağız Efe Gökçe — presentation and repository organization

## What I built

- The GitHub repository itself: I created it and set up the layout the whole team then worked in — `src/` for pipeline code, `scripts/` for data acquisition, `tests/` with fixtures, `results/` for run files and predictions, `docs/`, and `contributions/`.
- The working rules the others followed: feature branches with pull requests and a teammate review before every merge, `results/` written only by the scripts (never by hand), and test splits read only through `--final`.
- [`contributions/`](.) itself: one file per member with one fixed section structure (the seven sections used across the four records above), so each member's share is findable for grading.

## Key decisions and why

- One module per pipeline stage in a flat `src/` (data, models, metrics, benchmark) instead of sub-packages: four people could own disjoint files and review each other's diffs without path churn.
- Every merge through a reviewed pull request, even small ones: the review caught a real bug before merge (the missing Viterbi start mask found on [PR #1](https://github.com/gorkemgoksu/API_Lens/pull/1)).
- A shared contribution-record format fixed by me at the start, rather than free-form notes: comparable sections make individual shares auditable from one folder.
- Presentation delivery under 2:00 with a prepared cut list, because pace is graded (<2 min = full marks); I rehearsed the checkpoints rather than reading the notes.

## Numbers I present

Every number I say aloud, traced to its file before delivery:

- ModernBERT-CRF test macro-F1 **0.851** (mean of 3 seeds, 95% CI [0.814, 0.878]) and heldout (PyTorch) **0.5003** — `results/benchmark.csv`.
- Hand audit of 150 test sentences: **3.6%** wrong spans, second annotator agreement 50/50 — `results/label_audit.csv`.
- **19,504** mentions over 16,796 sentences — `data/processed/stats.json`.
- Rules keep **0.4407** on PyTorch while the learned classical models fall to 0.1708 (CRF) — `results/benchmark.csv`.

## One-sentence presentation finding

Pre-training plus structure wins in-domain (ModernBERT-CRF 0.851) but transfers only halfway to an unseen library (0.5003, barely above the rules' 0.4407), so cross-library generalization stays the open problem (`results/benchmark.csv`).

## How I verified the work

- Each number in the delivery was checked against `results/benchmark.csv`, `results/label_audit.csv` and `data/processed/stats.json`, not against the slides' text.
- Timed rehearsals against the script's checkpoints (target < 2:00) with the cut list ready.
- The Q&A answers I rehearsed all trace to result files or module docstrings, so no answer rests on memory alone.

## How I used AI tools

- The repository conventions and this record were drafted with a coding assistant; the structure, rules and every spoken number were checked against the actual repository and result files before use.
- No assistant output was accepted on its own description.

## Limitations and next steps

- My organization work shows in the repository structure, the PR history and this folder, not in code authorship; I have no commits under my name.
- The deck and speaker script are the team's work; my claim is the delivery, which is graded once, live.
- The Q&A depth score depends on the questions asked; the prepared table covers the eleven most likely ones.
