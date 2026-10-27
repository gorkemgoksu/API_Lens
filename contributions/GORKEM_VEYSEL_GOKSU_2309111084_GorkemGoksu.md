# Görkem Veysel Göksu — data pipeline

## What I built

- [`src/data_io.py`](../src/data_io.py): fixed label vocabulary, tokenizer, IOB2 conversion, schema validation, and protected split loading.
- [`scripts/download_data.py`](../scripts/download_data.py): version-pinned documentation and Sphinx inventory acquisition with checksums.
- [`scripts/build_dataset.py`](../scripts/build_dataset.py): extraction, Sphinx-style rendering, inventory resolution, ordered cleaning, document-level splitting, deduplication, statistics, and deterministic rebuild checks.
- [`docs/preprocessing.md`](../docs/preprocessing.md): sources, pipeline, filtering rules, split policy, schema examples, rebuild instructions, and limitations.
- [`tests/test_data_io.py`](../tests/test_data_io.py), [`tests/test_build.py`](../tests/test_build.py), and [`tests/fixtures/mini.jsonl`](../tests/fixtures/mini.jsonl): contract and pipeline tests.

## Key decisions and why

- Entity positions come from author-written Sphinx/MyST roles, while entity types come from version-matched `objects.inv` inventories. This avoids subjective manual type assignment.
- Documents, rather than individual sentences, are assigned to splits to prevent page-level leakage.
- PyTorch is reserved for held-out evaluation so cross-library generalization can be measured.
- Cleaning rules are ordered and counted; unresolved, ambiguous, malformed, or likely unlabeled API mentions are removed instead of guessed.
- The original dataset remains frozen at Git tag `data-v1`; approved corrections are versioned separately instead of moving an existing tag.
- The `data-v2` rendering update applies each library's configured `currentmodule` at `{{ header }}` lines and renders raw-HTML badge substitutions as visible text, recovering valid sentences without changing any cleaning rule, threshold, seed, split ratio, or sampling policy.

## Numbers I produced

- Split sizes and all corpus statistics are recorded in [`data/processed/stats.json`](../data/processed/stats.json).
- `train`: 12,486 sentences; `dev`: 2,080; `test`: 1,630; `heldout`: 600; `test_noparen`: 1,630 (`stats.json -> splits`).
- Entity totals are ATTRIBUTE 990, CLASS 7,904, EXCEPTION 131, FUNCTION 4,929, METHOD 4,886, and MODULE 664 (`stats.json -> entity_types`).
- Shared document paths across splits: 0 (`stats.json -> shared_doc_paths`).

## One-sentence presentation finding

APILens deterministically turns version-pinned documentation into 16,796 leak-free sentences with 19,504 objectively typed API mentions, while reserving PyTorch as an unseen-library evaluation set (`stats.json -> splits`, `entity_types`, and `shared_doc_paths`).

## How I verified the work

- Ran the current test suite: 88 tests passed.
- Rebuilt all six processed artifacts and confirmed byte-for-byte SHA-256 equality.
- Verified all 1,766 recorded raw-source checksums.
- Reviewed a seeded sample of 120 final sentences, with 20 examples from each source library.
- Reviewed 10 rejected examples for each cleaning rule R1–R7; the first-matching rule was appropriate in the sampled cases.
- Checked five examples for each of the six entity types against their versioned official documentation. All 30 URLs returned HTTP 200 and their Sphinx inventory roles agreed with the assigned entity types.
- Confirmed no raw source file exceeds the 50 MB repository limit.
- Verified the pinned source licenses: NumPy, SciPy, pandas, scikit-learn, and PyTorch use BSD-style terms; Matplotlib uses its PSF-derived license agreement; the Python documentation and inventory use the PSF License Version 2, with documentation code examples also available under Zero-Clause BSD.

## How I used AI tools

- I used a coding assistant to draft and refine the data pipeline, tests, documentation, and reproducibility checks.
- I reviewed every generated diff through automated tests, deterministic rebuilds, checksum verification, balanced sentence samples, cleaning-rule samples, inventory-role checks, and official documentation links.
- No assistant output was accepted on its own description; each published dataset artifact was checked against executable or source-backed evidence.

## Limitations and next steps

- Only mentions explicitly marked by upstream documentation authors can receive entity positions; unmarked mentions can remain unlabeled.
- Changelog-heavy libraries contain short release-note and issue-title language even after cleaning.
- Inventory and documentation versions must remain aligned; changing either requires a new dataset version and rerunning final experiments.
- I will independently audit the 50 shared sentences marked `second=1` after the label-audit sample is generated.
