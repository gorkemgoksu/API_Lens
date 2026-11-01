# Error analysis

## 1. Setup

- **Model:** ModernBERT-CRF, the best model by mean test macro-F1 in `results/benchmark.csv` (0.8507 over 3 seeds). We analyse seed 42.
- **Split:** test (1630 sentences, 1695 gold spans), produced by `python src/build_benchmark.py --final --errors`.
- **Groups.** Each error is assigned to one group:

  | Group | Meaning |
  |---|---|
  | type | same boundaries as a gold span, different type |
  | missed | gold span with no overlapping predicted span |
  | spurious | predicted span with no overlapping gold span |
  | too_short, too_long, split, merged, shifted | boundary errors against overlapping spans |

  `type`, `merged` and `spurious` count predicted spans; the other groups count gold spans.
- **Sample.** For the groups `type`, `missed` and `spurious`, 20 examples each were drawn with seed 0 and classified by hand into three classes:

  | Class | Rule |
  |---|---|
  | model | The gold span is right and the model gave a wrong type or boundary, or missed a real API; or the model tagged something that is not an API (a common word, a parameter or variable name, a value, a library name). |
  | noise | The model found a real API name the authors did not mark; or the text calls it a method/function and the gold type says the opposite, and the model followed the text; or the inventory type contradicts the text (ufuncs or class aliases typed ATTRIBUTE) and the model's type matches the text; or the gold span covers a link title instead of an API name. |
  | ambiguous | The name exists as both a function and a method (or a module and a class) and the text does not say which; or the gold span is an attribute whose name reads as a common word (`data`, `mask`); or the reference is internal or an abbreviation; or the example mixes a label-noise span and a model-error span; or the gold type contradicts the text and the model's type does not match the text either; or the gold span is a built-in constant that reads as a value (`True`). |

  Distribution objects typed ATTRIBUTE were treated as correct gold labels, consistent with the label audit.

## 2. Counts

Top 5 type confusions (exact boundaries, gold -> predicted):

| Gold | Predicted | Count |
|---|---|---|
| METHOD | FUNCTION | 12 |
| ATTRIBUTE | FUNCTION | 12 |
| FUNCTION | METHOD | 11 |
| ATTRIBUTE | CLASS | 8 |
| CLASS | FUNCTION | 8 |

Error groups:

| Group | Count |
|---|---|
| type | 72 |
| too_short | 1 |
| too_long | 0 |
| split | 0 |
| merged | 0 |
| missed | 109 |
| spurious | 153 |
| shifted | 0 |

Boundary errors are almost absent. The errors are wrong types (72), missed mentions (109) and extra predictions (153).

## 3. Hand classification

| Group | n | model | noise | ambiguous |
|---|---|---|---|---|
| type | 20 | 10 | 5 | 5 |
| missed | 20 | 14 | 0 | 6 |
| spurious | 20 | 5 | 12 | 3 |

## 4. Findings

**The `()` cue.** In the `type` sample, 3 of 20 examples are classes written with `()` that the model typed FUNCTION (e.g. "scipy. sparse. csr_array(), ... coo_array()"; note: "class written with (), model said FUNCTION"). CLASS -> FUNCTION is one of the top 5 confusions (8 cases). The parentheses appear to push the model toward FUNCTION. Removing them in `test_noparen` lowers ModernBERT-CRF test macro-F1 from 0.8507 to 0.7865 (0.0642 absolute).

**Function vs method.** METHOD -> FUNCTION (12) and FUNCTION -> METHOD (11) together make 23 of the 72 type errors. The sample shows two causes. Names that exist as both: `hexbin`, `pivot`, `outer` (3 ambiguous; "exists as function and method, text unclear"). Pandas methods taken for functions: `mask()` and `where()` (2 model; "mask is a DataFrame/Series method"). In 3 noise examples the gold type contradicts the text: "the filled() method" and "the compressed() method" are gold FUNCTION, and "The ndarray creation functions ... random()" is gold METHOD.

**Inventory types that contradict the text.** `objects.inv` types some objects differently from how the text uses them. In the sample, an alias of a class is typed ATTRIBUTE (noise; "alias of a class, typed ATTRIBUTE by inventory") and a ufunc is typed ATTRIBUTE while the text reads as a function (ambiguous; "ufunc typed ATTRIBUTE, model said METHOD, reads as function"). A third case, a scipy distribution object typed ATTRIBUTE, was treated as a correct gold label and counted as a model error because the model predicted FUNCTION. Gold ATTRIBUTE spans predicted FUNCTION (12) or CLASS (8) are 20 of the 72 type errors; this sample covers only some of them.

**Unlabeled API mentions behind spurious predictions.** 12 of 20 sampled spurious predictions are noise: the predicted span is a real API mention that the doc authors did not mark with a role (e.g. "unlabeled API Series class found", "unlabeled API SVC class found"). Only 5 of 20 are model errors, such as the math variable `s_k` and the adjective in "Datetime-like". Another 3 are ambiguous (e.g. `SVC` as an abbreviation for `LinearSVC`). So the measured precision understates the model. This is a property of the labeling scheme, which marks only role-annotated mentions.

**Spans merged across a sentence-final dot.** One of the 20 spurious examples is a single span "pandas.compat. pandas.compat" that joins the end of one sentence and the start of the next and is typed CLASS (note: "span merged across a sentence boundary"). The `merged` group is 0 because this event has no overlapping gold spans and falls into `spurious`.

**Missed mentions.** 14 of 20 sampled misses are model errors, mostly real classes, methods and modules written without a role cue (e.g. "missed make_norm_from_scale function", "missed bytes builtin class"). Five of the 6 ambiguous ones are attribute names or constants that read as common words or values (`data` twice, `mask`, `operands`, `True`); the sixth mixes a link-title span with a missed `memoryview`.

## 5. Limitations

- Only 20 examples per group were classified, drawn from groups of 72, 109 and 153 errors; the shares above are rough and carry a wide sampling error.
- One seed (42) and one split (test) were analysed.
- One annotator classified all 60 examples, so there is no agreement measure. The ambiguous class depends on that person's judgment. Two missed examples come from the same sentence.

## Sources of numbers

| Number | Source |
|---|---|
| 0.8507, 3 seeds | `results/benchmark.csv`, row modernbert_crf / test (`macro_f1`, `n_seeds`) |
| 0.7865 | `results/benchmark.csv`, row modernbert_crf / test_noparen (`macro_f1`) |
| 0.0642 | 0.8507 - 0.7865 |
| 1630 sentences, 1695 gold spans | `--errors` output (header) |
| Confusion counts 12, 12, 11, 8, 8 | `--errors` output (type confusions) |
| Group counts 72, 1, 0, 0, 0, 109, 153, 0 | `--errors` output (error groups) |
| 23 | 12 + 11 |
| 20 of the 72 | 12 + 8 |
| 20 examples per group, seed 0 | `src/build_benchmark.py --errors` |
| Classification counts (10/5/5, 14/0/6, 5/12/3) | `results/error_labels.csv`, `label` column counted per group |
| 3 of 20 (classes with `()`) | `results/error_labels.csv`, type 18, 19, 20 |
| 3 ambiguous (hexbin, pivot, outer) | `results/error_labels.csv`, type 3, 14, 7 |
| 2 model (mask, where) | `results/error_labels.csv`, type 15, 16 |
| 3 noise (filled, compressed, random) | `results/error_labels.csv`, type 8, 9, 12 |
| Alias and ufunc typed ATTRIBUTE; distribution object | `results/error_labels.csv`, type 4, 5, 17 |
| 12 of 20, 5 of 20, 3 of 20 (spurious) | `results/error_labels.csv`, spurious rows by label |
| 1 of 20 (merged across dot) | `results/error_labels.csv`, spurious 11 |
| 14 of 20, 6 ambiguous (missed) | `results/error_labels.csv`, missed rows by label |
