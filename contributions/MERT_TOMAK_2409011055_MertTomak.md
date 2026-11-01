# Mert Tomak — coordination and neural models

## What I built

- [`src/train_bilstm_crf.py`](../src/train_bilstm_crf.py): BiLSTM-CRF tagger with a from-scratch linear-chain CRF layer (batched forward algorithm for the log-partition, gold path score, negative log-likelihood loss) and a character CNN. Decoding calls the shared `viterbi` with the transition and start masks from `bio_allowed`.
- The same script provides the ablations `bilstm_crf-nochar` (`--no-char`) and `bilstm-softmax` (`--softmax`).
- [`src/train_transformer.py`](../src/train_transformer.py): ModernBERT-base fine-tuning with a softmax head (`modernbert`) or the same CRF layer (`modernbert_crf`), both reading scores at the first sub-token of each word. It also contains the learning-rate sweep (`--lr-sweep`).
- [`src/hallucination_check.py`](../src/hallucination_check.py): demo that tags API mentions in a text file with a saved BiLSTM-CRF checkpoint and looks each one up in the local Sphinx inventories (FOUND, NOT FOUND or AMBIGUOUS, plus whether the inventory type agrees).
- The neural final runs: `bilstm_crf`, `modernbert` and `modernbert_crf` with seeds 13, 42 and 2026, and the two BiLSTM ablations with seed 42.
- Work taken over from Zeynep Çelik when she had to travel: the first-annotator pass of the label audit ([`results/label_audit.csv`](../results/label_audit.csv), all 150 sentences), the `--errors` option of [`src/build_benchmark.py`](../src/build_benchmark.py) with one test, and [`results/error_analysis.md`](../results/error_analysis.md) with the 60 classified examples ([`results/error_labels.csv`](../results/error_labels.csv)).
- [`README.md`](../README.md) and [`LICENSE`](../LICENSE).

## Key decisions and why

- `max_length` is 256 for ModernBERT. The longest train sentence needs 140 sub-tokens on data-v2 (`longest_subtokens` in the run files), and `encode()` exits with an error if any split would be truncated.
- The learning rate was chosen once, on dev, with the softmax head and seed 42, from the grid 3e-5, 5e-5 and 8e-5. Dev macro-F1 was 0.8256 at 3e-5, 0.8512 at 5e-5 and 0.8691 at 8e-5 ([PR #12](https://github.com/gorkemgoksu/API_Lens/pull/12), description). 8e-5 won and is the `--lr` default for both heads. The CRF parameters keep their own rate of 1e-3.
- Precision is an explicit `--precision` flag. The T4 has no native bf16, so the script refuses bf16 there and the final runs use fp16 with a gradient scaler. The CRF loss is computed in float32.
- All seeds of a model run on one device: BiLSTM-CRF on a Mac CPU, ModernBERT on a Tesla T4 (`hardware` in the run files). MPS is skipped so the same seed reproduces the same score.
- The CRF layer fixes illegal IOB2 transitions and illegal starts to a large negative score, so the BiLSTM-CRF and ModernBERT-CRF heads never emit invalid IOB2 sequences.
- Data-v2: I measured the drop table, traced the pandas `{{ header }}` template and the scikit-learn badge substitutions, and reported them to the data owner. The fix landed in [PR #5](https://github.com/gorkemgoksu/API_Lens/pull/5), which I reviewed and merged, and I tagged `data-v2`.
- ModernBERT words after the first get a leading space before tokenization, so they look like words in running text.

## Numbers I produced

Dev macro-F1 only, read from `results/<run>.json` under `splits.dev.macro_f1`. Test, heldout and test_noparen numbers are in `results/benchmark.csv`.

| Run | Seed 13 | Seed 42 | Seed 2026 |
| --- | --- | --- | --- |
| `bilstm_crf` | 0.8381 | 0.8300 | 0.8269 |
| `modernbert` (softmax) | 0.8605 | 0.8691 | 0.8736 |
| `modernbert_crf` | 0.8707 | 0.8618 | 0.8774 |

- Ablations, seed 42: `bilstm_crf-nochar` 0.8111 and `bilstm-softmax` 0.8009 (`results/bilstm_crf-nochar_seed42.json`, `results/bilstm-softmax_seed42.json`).
- Invalid IOB2 rate on dev (`splits.dev.invalid_iob2_rate`): 0.0 for every CRF run, 0.038 for `bilstm-softmax`, and 0.0216 to 0.0226 for the three `modernbert` runs.
- Lr sweep (softmax head, seed 42, dev macro-F1): 3e-5 0.8256, 5e-5 0.8512, 8e-5 0.8691, as stated in [PR #12](https://github.com/gorkemgoksu/API_Lens/pull/12). The 8e-5 value equals `modernbert_seed42.json`, `splits.dev.macro_f1`; the other two were only printed to the console and are not stored in a result file.

- Label audit (first annotator, 150 test sentences; `python src/build_benchmark.py --audit-report --second results/label_audit_second.csv`): 6 of 169 marked spans wrong (0.0355), 6 of 150 sentences with a wrong span, 15.33 unmarked API mentions per 100 sentences. On the 50 sentences shared with the second annotator we agreed on all 50 spans_ok answers, so Cohen's kappa is undefined; on the same 50 sentences I counted 6 unmarked mentions and the second annotator 15.
- Error analysis classification (`results/error_labels.csv`): model error / label noise / ambiguous = 10/5/5 (type), 14/0/6 (missed), 5/12/3 (spurious).

## How I verified the work

- `tests/test_viterbi.py` checks that the CRF log-partition equals a brute-force log-sum-exp over all tag paths, with and without the IOB2 masks.
- It also checks that padding is ignored and that CRF decoding never starts with an inside label (regression test for the start mask).
- Same-seed determinism: I trained the same seed twice on a T4 and got identical results except wall-clock time, for ModernBERT softmax (comment on [PR #1](https://github.com/gorkemgoksu/API_Lens/pull/1)) and for BiLSTM-CRF (comment on [PR #12](https://github.com/gorkemgoksu/API_Lens/pull/12)). On the Mac CPU the same seed also gave identical runs (same comment on PR #12), and the final `bilstm_crf` seed 42 run reproduces the dev macro-F1 of 0.8300 at best epoch 10 of an earlier, separate dev-only run.
- I ran the tests locally and approved [PR #2](https://github.com/gorkemgoksu/API_Lens/pull/2), [PR #3](https://github.com/gorkemgoksu/API_Lens/pull/3), [PR #5](https://github.com/gorkemgoksu/API_Lens/pull/5) and [PR #6](https://github.com/gorkemgoksu/API_Lens/pull/6), the Viterbi PR authored by Cemal Batuhan Bas; for #6 I tested on current main. The approvals are on the PR pages.
- The scripts follow the repository rules: without `--final` they train on train and report dev only, and the seed is set for Python, NumPy and PyTorch with deterministic kernels enabled. ModernBERT has a few kernels with no deterministic variant, so the repeat runs above are the reproducibility check there.

## How I used AI tools

- I used a coding assistant to draft the training scripts, tests and this record.
- I read every diff, ran the test suite, and checked the numbers against the run files before anything went to a pull request.
- Reviewer feedback from teammates was applied before merge; no assistant output was accepted on its own description.
- The first CRF draft passed only the transition mask to `viterbi()`. A teammate's review on [PR #1](https://github.com/gorkemgoksu/API_Lens/pull/1) caught the missing start mask, and I fixed it with a regression test that forces a 1e6 `I-FUNCTION` score on the first token.
- Checks I relied on instead of the assistant's description: the brute-force log-partition test, two identical same-seed runs per model, and comparing prediction counts with split sizes before committing results.
- Label audit: the assistant wrote a temporary input helper (outside the repository), explained the API objects I asked about and pointed out names that might be unmarked mentions; I made the final y/n and count decision for each of the 150 sentences.
- Error analysis: the 60 sampled errors were classified with the assistant proposing a class for each example under written rules; I read every example and accepted or changed the proposal. The rules are listed in `results/error_analysis.md`.
- The README was drafted with the assistant from `results/benchmark.csv`, `results/ablations.csv`, `data/processed/stats.json` and the audit files; I checked every number against those files.

## Limitations and next steps

- Ablations ran with a single seed (42), so differences between ablations are not backed by seed variance.
- The learning rate was tuned with the softmax head only and reused for the CRF head.
- BiLSTM-CRF and ModernBERT ran on different hardware, so the reported training times are not comparable.
- The hallucination demo checks names against the local inventories only; it cannot judge whether a real name is used correctly, and it is not an evaluated result.
- The error-analysis classification has one annotator and 20 examples per group.
- Next: store the lr sweep table in a result file, and run the ablations with all three seeds if time allows.
