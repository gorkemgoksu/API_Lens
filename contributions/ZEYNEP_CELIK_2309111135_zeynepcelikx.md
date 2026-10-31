# Zeynep Çelik — evaluation and analysis

## What I built

- [`src/metrics.py`](../src/metrics.py): the single source of every number in the project.
  - `evaluate`: strict entity-level metrics (macro and micro P/R/F1, per-type scores, span-only F1, 6 x 7 type confusion, token accuracy, invalid-IOB2 rate).
  - `slices`: results per library and per source type, plus recall on gold spans whose target is seen or unseen in train.
  - `bootstrap_mean_ci` and `bootstrap_ci`: 95% bootstrap intervals over sentences.
  - `validate_preds`, `write_preds`, `write_result`: the prediction and run-file contract the other scripts call.
  - `cohen_kappa`: agreement between the two label auditors.
- [`src/build_benchmark.py`](../src/build_benchmark.py): checks every run in `results/`, recomputes each stored metric from `results/preds/`, aggregates seeds and writes `results/benchmark.csv`, `results/ablations.csv` and four SVG figures ([PR #18](https://github.com/gorkemgoksu/API_Lens/pull/18)). It also has the label-audit flags `--sample-audit` and `--audit-report` ([PR #19](https://github.com/gorkemgoksu/API_Lens/pull/19)).
- [`tests/test_metrics.py`](../tests/test_metrics.py): 18 tests with hand-computed expectations.
- `results/label_audit.csv`: the 150-sentence audit sample (PR #19).

## Key decisions and why

- Entities are parsed with seqeval's strict IOB2 scheme, so a prediction counts only when boundaries and type both match. A test with gold `B-CLASS I-CLASS` and prediction `B-CLASS O` shows strict F1 0 while token accuracy is 2/3.
- Macro-F1 averages only over the types that occur in the gold labels of the evaluated split. A type with no gold example is left out instead of scored 0. This matters for heldout (PyTorch), which has no EXCEPTION and only 6 ATTRIBUTE mentions (`support` in the run files).
- Bootstrap intervals resample sentences, recompute F1 from summed entity counts (not the mean of per-sentence F1) and use the same draw for every seed before averaging the seeds.
- Seen versus unseen target is reported as recall only: a predicted span has no target string, so a false positive cannot be assigned to either group.
- `write_result` accepts only `config`, `train_time_s`, `hardware`, `n_params` and `best_epoch` in `extra`; missing keys become null and unknown keys raise an error, so a typo cannot silently change the run-file contract.
- `build_benchmark.py` needs `--final` to read the test splits (hard rule 1). Without it only dev is verified and nothing is written.
- Figure colours are the first six slots of a categorical palette checked with a validator script (colour-blind separation passed). Three slots have low contrast on white, so every bar carries its value label and the numbers are also in the CSV.
- The test split has only 14 EXCEPTION mentions, so "at least 15 per type" in the audit sample cannot hold for that type. The sample takes all 14 and the script prints a note.

## Numbers I produced

All from `results/benchmark.csv` (columns `macro_f1`, `macro_f1_ci_low`, `macro_f1_ci_high`); neural models are means over seeds 13, 42 and 2026.

| Model | Test macro-F1 [95% CI] | Heldout (PyTorch) macro-F1 |
| --- | --- | --- |
| Rules | 0.6176 [0.575, 0.649] | 0.4407 |
| HMM | 0.5194 [0.4693, 0.5665] | 0.0858 |
| CRF | 0.8060 [0.7555, 0.8439] | 0.1708 |
| BiLSTM-CRF | 0.7960 [0.7556, 0.8267] | 0.3091 |
| ModernBERT | 0.8332 [0.7991, 0.8608] | 0.4631 |
| ModernBERT-CRF | 0.8507 [0.814, 0.8778] | 0.5003 |

- `invalid_iob2_rate` for ModernBERT with the softmax head: 0.0239 on test and 0.16 on heldout; 0.0 for every other model in the main table.
- Audit sample (`results/label_audit.csv`, 150 sentences): FUNCTION 40, METHOD 25, CLASS 60, MODULE 15, ATTRIBUTE 15, EXCEPTION 14 mentions, printed by `--sample-audit`. 50 sentences are marked `second=1` for the second auditor.
- `hmm-nomask` and `hmm` give byte-identical predictions on test (`results/ablations.csv`, same macro-F1 0.5194).

## How I verified the work

- Every test expectation was computed by hand: a three-sentence example with one boundary error and one type error gives micro-F1 0.5, span-F1 0.75, token accuracy 10/12 and macro-F1 (2/3 + 1 + 0 + 0) / 4. The micro-F1 also matches seqeval's own `f1_score`. The kappa test is a hand-computed 0.32 / 0.52.
- `build_benchmark.py` recomputed all 16 runs from `results/preds/` and none of the stored numbers differed. I also changed one stored dev macro-F1 to 0.9999 on purpose and the script stopped with the exact key path; I then restored the file.
- I tested the figure code on dev results only and looked at the rendered images before running `--final`; I fixed a colour scale that hid rare confusions.
- `--sample-audit` refuses to overwrite an existing file; I tried a second run to confirm.
- The test splits were read only through `--final`.
- Pull requests [#2](https://github.com/gorkemgoksu/API_Lens/pull/2), [#3](https://github.com/gorkemgoksu/API_Lens/pull/3) and [#15](https://github.com/gorkemgoksu/API_Lens/pull/15) were merged with "Rebase and merge", so the commits keep my name.

## How I used AI tools

- I used a coding assistant to draft and refine the metrics, benchmark aggregation, audit tooling, tests, figures, and this contribution record.
- I checked generated work with hand-computed metric examples, seqeval comparisons, recomputation of stored results from prediction files, deliberate contract-failure tests, and visual inspection of the rendered figures.
- No assistant output was accepted on its description alone; every published number was traced to versioned run artifacts and reproducible checks.

## Limitations and next steps

- The audit is not finished at the time of writing, so wrong-span rate, missing mentions per 100 sentences and Cohen's kappa are not in this record yet. They will be added from `--audit-report`.
- Bootstrap intervals resample sentences as if independent, but sentences from one page are correlated, so the intervals are probably too narrow. Rules, HMM and CRF have one run each, so their intervals show sentence sampling only, not seed variance.
- Heldout macro-F1 averages five types (no EXCEPTION) and ATTRIBUTE has 6 mentions, so it is noisy.
- The CRF and BiLSTM-CRF test intervals overlap, so the data do not show which of the two is better.
- The `hmm-nomask` and `hmm` predictions are identical because the smoothed transitions never select an illegal step; `invalid_iob2_rate` is 0.0 for `hmm-nomask` on all four splits.
- Next: finish the audit and the error analysis (`results/error_analysis.md`), then give the numbers to the README.
