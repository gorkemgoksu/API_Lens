# Dataset and preprocessing

APILens is a sentence-level IOB2 dataset of Python API mentions in official
library documentation. Annotation is derived from upstream documentation
metadata rather than manual labels: author-written Sphinx or MyST roles locate
the mention, and the matching library `objects.inv` entry supplies its entity
type. Candidates that cannot be resolved uniquely are discarded.

## Sources and pins

`data/raw/SOURCES.json` is the authoritative machine-readable source record.
The corpus uses these documentation revisions:

- NumPy `v2.5.4`, commit `a3d0ef1df045b858e68115001fdabc930bb3b68c`.
- SciPy `v1.18.0`, commit `54ef5423f2e4376230ec3bfda6912a07a50958e3`.
- pandas `v3.1.0`, commit `c6f46769b808eb60218189d089dd67841b44f1b9`.
- scikit-learn `1.9.1`, commit `866c0f51e7560ef0303cbcc5f159df5382ea9e3f`.
- Matplotlib `v3.11.2`, commit `d3ca9172e392048ec242b90e8423e88d1a3000b2`.
- PyTorch `v2.14.1`, commit `5c4886908584029761b579af026dcfb627c84070`.
- Python `3.15` contributes an inventory only; it does not contribute prose.

Only upstream documentation text, the documentation configuration, and the
upstream license are retained. Versioned inventories are downloaded from the
URLs recorded in `SOURCES.json`; `data/raw/checksums.txt` covers every raw file.

## Pipeline

1. `scripts/download_data.py` fetches the pinned sources and inventories,
   records resolved commits and inventory metadata, and writes checksums.
2. `scripts/build_dataset.py` reads reStructuredText and MyST sources, skips
   code, prompts, directives, tables, comments, and navigation structures, and
   classifies each document as guide, tutorial, reference, or changelog.
3. Inline roles are rendered as a reader sees them while their exact character
   offsets, role, and target are retained. Function and method display text may
   include terminal parentheses.
4. Each candidate is resolved against the combined Python object inventories.
   Exact names are preferred, then the active module, then a unique qualified
   suffix. The inventory's Python object role determines FUNCTION, METHOD,
   CLASS, MODULE, ATTRIBUTE, or EXCEPTION.
5. Paragraphs are split into sentences while protecting entity-qualified dots
   and common abbreviations. `src/data_io.tokenize()` produces tokens and
   offsets; resolved spans are aligned and converted to strict IOB2 tags.
6. Cleaning rules R1--R7 are applied in order, so each rejected sentence is
   counted only under its first matching rule.
7. Every PyTorch sentence is assigned to `heldout`. Other complete documents
   are assigned by a seeded hash to train, dev, or test with an approximately
   80/10/10 policy. The recorded seed is 42, and no document path is shared by
   two splits.
8. R8 removes normalized-whitespace duplicates across the corpus, retaining
   one record by `heldout > test > dev > train` priority. All entity-bearing
   sentences are retained; zero-entity sentences are sampled deterministically
   so they form approximately 20% of each library's final records.
9. Every JSONL record is checked by `src/data_io.validate()` before writing.
   `stats.json` is derived from the final records.

## Drop rules

The totals below come from `data/processed/stats.json`:

- R1, residual reStructuredText or MyST markup: 5,570.
- R2, at least one unresolved role target: 8,698.
- R3, at least one ambiguously resolved role target: 76.
- R4, token alignment failure or overlapping entity spans: 14.
- R5, an inventory API appears in literal markup without an API role: 6,161.
- R6, fewer than 5 or more than 60 tokens: 14,349.
- R7, alphabetic-character share below 0.5: 1,735.
- R8, duplicate normalized sentence text: 3,818.

## Splits and `test_noparen`

Train and dev are available during development. Test, heldout, and
`test_noparen` are protected by `src/data_io.load_split()` and require
`final=True`. PyTorch is the held-out library, providing a cross-library
generalization evaluation.

`test_noparen` is derived from test. When a FUNCTION or METHOD entity ends in
the separate `(` and `)` tokens, those tokens are removed and every span and
IOB2 tag is realigned. Other tokens, records, and entity targets are unchanged.

## JSONL schema

Each line has this shape. Span `end` is exclusive.

```json
{"id":"string","library":"string","doc_path":"string","source_type":"guide | tutorial | reference | changelog","tokens":["string"],"tags":["IOB2 label"],"spans":[{"start":0,"end":1,"type":"entity type","target":"inventory name","role":"source role"}]}
```

The following are unmodified records selected from `train.jsonl`, one for each
entity type.

FUNCTION:

```json
{"doc_path":"doc/api/animation_api.rst","id":"matplotlib/doc/api/animation_api.rst#s00020","library":"matplotlib","source_type":"reference","spans":[{"end":9,"role":"obj","start":6,"target":"functools.partial","type":"FUNCTION"}],"tags":["O","O","O","O","O","O","B-FUNCTION","I-FUNCTION","I-FUNCTION","O","O","O","O","O","O","O"],"tokens":["The","second","method","is","to","use","functools",".","partial","to","pass","arguments","to","the","function",":"]}
```

METHOD:

```json
{"doc_path":"doc/api/animation_api.rst","id":"matplotlib/doc/api/animation_api.rst#s00042","library":"matplotlib","source_type":"reference","spans":[{"end":1,"role":"obj","start":0,"target":"matplotlib.animation.AbstractMovieWriter.setup","type":"METHOD"}],"tags":["B-METHOD","O","O","O","O","O","O","O","O","O","O","O","O","O"],"tokens":["setup","prepares","the","writer","(","e",".","g",".","opening","a","pipe",")","."]}
```

CLASS:

```json
{"doc_path":"doc/api/animation_api.rst","id":"matplotlib/doc/api/animation_api.rst#s00040","library":"matplotlib","source_type":"reference","spans":[{"end":15,"role":"obj","start":14,"target":"matplotlib.figure.Figure","type":"CLASS"}],"tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-CLASS","O"],"tokens":["The","writer","classes","provide","a","way","to","grab","sequential","frames","from","the","same","underlying","Figure","."]}
```

MODULE:

```json
{"doc_path":"doc/api/index.rst","id":"matplotlib/doc/api/index.rst#s00006","library":"matplotlib","source_type":"reference","spans":[{"end":1,"role":"mod","start":0,"target":"matplotlib.axes","type":"MODULE"}],"tags":["B-MODULE","O","O","O","O","O","O","O","O","O"],"tokens":["axes",":","add","data",",","limits",",","labels","etc","."]}
```

ATTRIBUTE:

```json
{"doc_path":"doc/api/prev_api_changes/api_changes_1.3.x.rst","id":"matplotlib/doc/api/prev_api_changes/api_changes_1.3.x.rst#s00046","library":"matplotlib","source_type":"reference","spans":[{"end":10,"role":"attr","start":3,"target":"matplotlib.cm.ScalarMappable.colorbar","type":"ATTRIBUTE"}],"tags":["O","O","O","B-ATTRIBUTE","I-ATTRIBUTE","I-ATTRIBUTE","I-ATTRIBUTE","I-ATTRIBUTE","I-ATTRIBUTE","I-ATTRIBUTE","O","O","O","O","O"],"tokens":["Instead",",","the","matplotlib",".","cm",".","ScalarMappable",".","colorbar","attribute","should","be","used","."]}
```

EXCEPTION:

```json
{"doc_path":"doc/api/prev_api_changes/api_changes_1.5.0.rst","id":"matplotlib/doc/api/prev_api_changes/api_changes_1.5.0.rst#s00035","library":"matplotlib","source_type":"reference","spans":[{"end":6,"role":"obj","start":5,"target":"ValueError","type":"EXCEPTION"}],"tags":["O","O","O","O","O","B-EXCEPTION","O"],"tokens":["Previously",",","they","raised","a","ValueError","."]}
```

## Known limitations

- Plain-text API mentions that upstream authors did not mark with an API role
  are generally zero-entity examples; the pipeline does not infer new labels.
- Source composition follows upstream writing practices. Libraries with large
  release-note histories can therefore be changelog-heavy.
- An inventory describes a published documentation version while prose is
  pinned to a repository revision. A name changed between those artifacts may
  be unresolved and removed.
- Role text can be intentionally abbreviated. Labels remain objective with
  respect to the source metadata, but the visible mention may not contain the
  fully qualified inventory target.

## Rebuild and verify

Review the pins, download the raw sources, build the processed data, and then
perform a byte-for-byte deterministic rebuild:

```powershell
python scripts/download_data.py
python scripts/download_data.py --download
python scripts/build_dataset.py
python scripts/build_dataset.py --check
python src/data_io.py train
python src/data_io.py test --final
```

The check rebuilds into a temporary project directory and compares SHA-256
digests for every generated file against `data/processed/`.
