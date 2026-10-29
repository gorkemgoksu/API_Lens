# Cemal Batuhan Baş — classical sequence models

## What I built

- [`src/viterbi.py`](../src/viterbi.py): the shared log-space Viterbi decoder, `viterbi(emissions, transitions, start, end, allowed)`, in O(T·K²) time with backpointers, and `bio_allowed(labels)`, which builds the IOB2 masks (`I-X` may only follow `B-X` or `I-X`; no sentence starts with `I-X`). The HMM, the BiLSTM-CRF and the ModernBERT-CRF all decode with it.
- [`tests/test_viterbi.py`](../tests/test_viterbi.py): the decoder tests (the CRF-layer tests further down the file are Mert's).
- [`src/rule_baseline.py`](../src/rule_baseline.py): a transparent surface-form reference with no learning; every rule is listed in the module docstring.
- [`src/train_hmm.py`](../src/train_hmm.py): a first-order HMM written from scratch with NumPy (counting, add-k smoothing, word-shape classes for rare and unseen words) and decoded with the shared Viterbi; `--no-mask` is the mask ablation.
- [`src/train_crf.py`](../src/train_crf.py): a feature-based CRF (sklearn-crfsuite, L-BFGS) whose features use only token form and a ±2 window; `--no-context` is the window ablation.
- Final runs: `rules`, `hmm`, `crf` and the ablations `hmm-nomask` and `crf-nocontext` (`results/*_seed0.json`, `results/preds/`).

## Key decisions and why

- One decoder for every structured model, so the HMM, the CRF layers and the masks are compared on the same search procedure, not on different decoding code.
- Masks are applied as −∞ scores inside the recurrence, not by post-processing the output, so the decoder returns the best *valid* path.
- The rule baseline never tags a single lowercase word without `()`: the word alone is too ambiguous, and the reference must stay precise rather than recall-hungry.
- The HMM replaces words seen at most once in train with a shape class (`min_count` 2 in `hmm_seed0.json`), so the model learns emissions for unseen words instead of giving them zero probability.
- The CRF uses no inventory or name list: it must learn from form and context, otherwise the test would measure lookup, not tagging.
- All hyperparameters were chosen on dev only: HMM k = 0.01 from {0.01, 0.1, 1.0}; CRF c1 = 0.01, c2 = 0.5 from {0.01, 0.1, 0.5}²; CRF without context c1 = 0.1, c2 = 0.01 (`config` in each run file). The test, held-out and no-parenthesis splits were read once, by the `--final` runs.

## Numbers I produced

Macro-F1, from `results/benchmark.csv` (one deterministic run, seed 0):

| Model | dev | test | heldout | test_noparen |
|---|---|---|---|---|
| Rules | 0.5979 | 0.6176 | 0.4407 | 0.3411 |
| HMM | 0.5711 | 0.5194 | 0.0858 | 0.4765 |
| CRF | 0.8192 | 0.8060 | 0.1708 | 0.7057 |

Ablations, from `results/ablations.csv`:

- `hmm-nomask` gives the same scores as `hmm` on all four splits, and the predictions are identical token for token. With k = 0.01 the smoothed transition into an illegal `I-X` is so small that the unmasked Viterbi never chooses it; `invalid_iob2_rate` is 0.0 with and without the mask.
- `crf-nocontext` drops test macro-F1 from 0.806 to 0.7705 and test unseen-target recall from 0.5524 to 0.4113. Without the window, the strongest state features become memorized lowercase names (e.g. `lower:fromfunction`), while the full model's strongest features are context words such as "method", "module" and "attribute" next to the mention.

## One-sentence presentation finding

The surface-form rules keep a held-out (PyTorch) macro-F1 of 0.4407, while the learned classical models fall to 0.1708 (CRF) and 0.0858 (HMM): learning from one set of libraries buys in-domain accuracy (CRF test 0.806) at the price of memorizing that domain's vocabulary (`benchmark.csv`).

## How I verified the work

- Viterbi equals brute-force enumeration of all K^T paths (path and score) on 200 random problems, also with random masks; every masked output is valid IOB2; T = 1, ties, empty input and malformed shapes are tested. The full suite passes (108 tests).
- Ran rules and HMM twice on dev: the scores were identical, and the final-run dev scores equal the scores from model selection.
- `python src/build_benchmark.py --final` recomputed every stored number from `results/preds/` and found no mismatch.
- Checked that the HMM's most probable transitions are the expected ones (`B-X → I-X`, `O → O`) and that the CRF's strongest transitions forbid `O → I-X` with large negative weights.

## How I used AI tools

- I used a coding assistant to draft the decoder, the three models and their tests from the prompts in the project plan.
- I accepted nothing on its description: the decoder is checked against brute force, every model is checked against `src/metrics.py` on dev, the final numbers are recomputed from the prediction files, and the printed transitions and features were read for plausibility.

## Limitations and next steps

- HMM k (0.01) and CRF c2 (0.5) were chosen at the edge of their grids; a wider grid could move both.
- The CRF decodes with crfsuite's own unconstrained Viterbi, so it can output invalid IOB2 (`invalid_iob2_rate` 0.0005 on dev); decoding its scores with `src/viterbi.py` and the masks would remove this.
- The classical models are single deterministic runs, so their uncertainty is the bootstrap CI over sentences only, not variation over seeds.
- Both learned classical models collapse on the held-out library; features that do not depend on the library's own vocabulary are the obvious next step.
