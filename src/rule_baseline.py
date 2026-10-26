"""Run the deterministic APILens rule-based baseline.

A transparent surface-form reference with no learning: it measures how much of
the task the shape of the tokens alone can solve. Tokens are scanned left to
right and each candidate is labeled by the first rule that applies.

1. A candidate is a maximal run IDENT ("." IDENT)*, IDENT = [A-Za-z_]\\w*,
   optionally followed by the tokens "(" ")"; the parentheses join the span.
2. With "()": METHOD if a component before the last starts with an uppercase
   letter (DataFrame.groupby()), otherwise FUNCTION (numpy.mean(), fit()).
3. Without "()", by the last component:
   a. ends with Error, Warning or Exception -> EXCEPTION (ValueError);
   b. CamelCase -> CLASS (pandas.Series, KMeans). CamelCase means a leading
      uppercase letter and at least one lowercase letter; a lone token also
      needs a second uppercase letter, otherwise every capitalized
      sentence-initial word would be a class;
   c. exactly two lowercase components whose first is a library root
      (numpy, np, scipy, pandas, pd, sklearn, matplotlib, plt, torch) -> MODULE;
   d. any other dotted run -> ATTRIBUTE (ndarray.shape);
   e. anything else, e.g. a single lowercase word, is not an entity.
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import metrics
from src.data_io import load_split


MODEL = "rules"
SEED = 0
FINAL_SPLITS = ("dev", "test", "heldout", "test_noparen")
IDENT = re.compile(r"[A-Za-z_]\w*")
MODULE_ROOTS = {"numpy", "np", "scipy", "pandas", "pd", "sklearn", "matplotlib", "plt", "torch"}
EXCEPTION_SUFFIXES = ("Error", "Warning", "Exception")


def span_type(parts: list[str], called: bool) -> str | None:
    """Return the entity type of a dotted run, or None when no rule labels it."""
    last = parts[-1]
    if called:
        return "METHOD" if any(part[0].isupper() for part in parts[:-1]) else "FUNCTION"
    if last.endswith(EXCEPTION_SUFFIXES):
        return "EXCEPTION"
    if last[0].isupper() and any(char.islower() for char in last):
        if len(parts) > 1 or any(char.isupper() for char in last[1:]):
            return "CLASS"
    if len(parts) == 2 and parts[0] in MODULE_ROOTS and all(part.islower() for part in parts):
        return "MODULE"
    return "ATTRIBUTE" if len(parts) > 1 else None


def predict(tokens: list[str]) -> list[str]:
    """Tag one tokenized sentence with the surface-form rules."""
    tags = ["O"] * len(tokens)
    start = 0
    while start < len(tokens):
        if not IDENT.fullmatch(tokens[start]):
            start += 1
            continue
        end = start + 1
        while end + 1 < len(tokens) and tokens[end] == "." and IDENT.fullmatch(tokens[end + 1]):
            end += 2
        called = tokens[end : end + 2] == ["(", ")"]
        entity_type = span_type(tokens[start:end:2], called)
        if entity_type is not None:
            end += 2 if called else 0
            tags[start:end] = [f"B-{entity_type}"] + [f"I-{entity_type}"] * (end - start - 1)
        start = end
    return tags


def main() -> None:
    """Report dev scores, or write final predictions and the run file with --final."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", action="store_true", help="predict all final splits and write results")
    args = parser.parse_args()

    if not args.final:
        records = load_split("dev")
        scores = metrics.evaluate([record["tags"] for record in records], [predict(r["tokens"]) for r in records])
        print(f"dev macro-F1 {scores['macro_f1']:.4f}")
        return

    splits: dict[str, Any] = {}
    for split in FINAL_SPLITS:
        records = load_split(split, final=True)
        predictions = [predict(record["tokens"]) for record in records]
        metrics.write_preds(MODEL, SEED, split, [record["id"] for record in records], predictions)
        splits[split] = metrics.evaluate([record["tags"] for record in records], predictions)
        print(f"{split}: macro-F1 {splits[split]['macro_f1']:.4f}")
    run_info = {
        "config": {"module_roots": sorted(MODULE_ROOTS), "exception_suffixes": list(EXCEPTION_SUFFIXES)},
        "train_time_s": 0.0,
        "hardware": f"cpu ({platform.system()} {platform.machine()})",
        "n_params": 0,
        "best_epoch": None,
    }
    metrics.write_result(MODEL, SEED, splits, run_info)


if __name__ == "__main__":
    main()
