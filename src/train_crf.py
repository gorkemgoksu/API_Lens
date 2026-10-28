"""Train and evaluate the APILens feature-based CRF (sklearn-crfsuite, L-BFGS).

Every feature comes from the form of the tokens and their neighbors; no API
name is looked up in any inventory or list. c1 and c2 are chosen on dev from a
3 x 3 grid. --no-context is the ablation that drops the +-2 token window.
"""

from __future__ import annotations

import argparse
import itertools
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import sklearn_crfsuite

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import metrics
from src.data_io import ENTITY_TYPES, load_split


SEED = 0
GRID = (0.01, 0.1, 0.5)
MAX_ITERATIONS = 200
FINAL_SPLITS = ("dev", "test", "heldout", "test_noparen")
CAMEL = re.compile(r"[A-Z][a-z0-9]+[A-Z]")


def shape(word: str) -> str:
    """Map letters and digits to X / x / d and collapse repeats: DataFrame -> XxXx."""
    mapped = re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"[0-9]", "d", word)))
    return re.sub(r"(.)\1+", r"\1", mapped)


def length_bucket(word: str) -> str:
    """Bucket the token length so that the CRF does not learn one weight per length."""
    return "1" if len(word) == 1 else "2-3" if len(word) <= 3 else "4-6" if len(word) <= 6 else "7+"


def token_features(tokens: list[str], i: int, context: bool) -> dict[str, Any]:
    """Return the features of token i; with context=False the +-2 window is left out."""
    word = tokens[i]
    features: dict[str, Any] = {
        "bias": 1.0,
        # the token itself
        "lower": word.lower(),
        "shape": shape(word),
        **{f"prefix{n}": word[:n] for n in (1, 2, 3)},
        **{f"suffix{n}": word[-n:] for n in (1, 2, 3)},
        "is_title": word.istitle(),
        "is_upper": word.isupper(),
        "is_camel": CAMEL.match(word) is not None,
        "has_underscore": "_" in word,
        "has_digit": any(char.isdigit() for char in word),
        "is_punct": not any(char.isalnum() for char in word),
        "length": length_bucket(word),
        # direct neighbors that mark dotted names and calls
        "prev_is_dot": i > 0 and tokens[i - 1] == ".",
        "next_is_dot": i + 1 < len(tokens) and tokens[i + 1] == ".",
        "next_is_paren": i + 1 < len(tokens) and tokens[i + 1] == "(",
        # sentence boundaries
        "BOS": i == 0,
        "EOS": i == len(tokens) - 1,
    }
    if context:
        # lowercase word and shape of the tokens at offsets -2, -1, +1, +2
        for offset in (-2, -1, 1, 2):
            j = i + offset
            if 0 <= j < len(tokens):
                features[f"{offset:+d}:lower"] = tokens[j].lower()
                features[f"{offset:+d}:shape"] = shape(tokens[j])
    return features


def featurize(records: list[dict[str, Any]], context: bool) -> list[list[dict[str, Any]]]:
    """Return the feature dicts of every token of every sentence."""
    return [[token_features(r["tokens"], i, context) for i in range(len(r["tokens"]))] for r in records]


def train(features: list[list[dict[str, Any]]], tags: list[list[str]], c1: float, c2: float) -> sklearn_crfsuite.CRF:
    """Fit one L-BFGS CRF with L1 weight c1 and L2 weight c2."""
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=c1, c2=c2, max_iterations=MAX_ITERATIONS, all_possible_transitions=True
    )
    crf.fit(features, tags)
    return crf


def predict(crf: sklearn_crfsuite.CRF, features: list[list[dict[str, Any]]]) -> list[list[str]]:
    """Return plain lists of tags; sklearn-crfsuite returns a NumPy object array."""
    return [[str(tag) for tag in sentence] for sentence in crf.predict(features)]


def select(train_records: list[dict[str, Any]], dev: list[dict[str, Any]], context: bool) -> tuple[sklearn_crfsuite.CRF, float, float, float]:
    """Fit the 3 x 3 grid, print every dev macro-F1 and return the best (crf, c1, c2, fit seconds)."""
    train_features, train_tags = featurize(train_records, context), [r["tags"] for r in train_records]
    dev_features, dev_tags = featurize(dev, context), [r["tags"] for r in dev]
    best: tuple[sklearn_crfsuite.CRF, float, float, float] | None = None
    best_f1 = -1.0
    for c1, c2 in itertools.product(GRID, GRID):
        started = time.perf_counter()
        crf = train(train_features, train_tags, c1, c2)
        fit_time = time.perf_counter() - started
        f1 = metrics.evaluate(dev_tags, predict(crf, dev_features))["macro_f1"]
        print(f"c1={c1:<4} c2={c2:<4} dev macro-F1 {f1:.4f}  {fit_time:.0f}s")
        if f1 > best_f1:
            best, best_f1 = (crf, c1, c2, fit_time), f1
    assert best is not None
    return best


def describe(crf: sklearn_crfsuite.CRF) -> None:
    """Print the 15 strongest transition weights and the 10 strongest positive state features per type."""
    print("15 strongest transitions:")
    transitions = sorted(crf.transition_features_.items(), key=lambda item: -abs(item[1]))[:15]
    for (previous, current), weight in transitions:
        print(f"  {previous:>12} -> {current:<12} {weight:+.3f}")
    for entity_type in ENTITY_TYPES:
        state = [(attr, label, w) for (attr, label), w in crf.state_features_.items() if label.endswith(entity_type) and w > 0]
        top = sorted(state, key=lambda item: -item[2])[:10]
        print(f"  {entity_type}: " + ", ".join(f"{attr} ({label[0]}) {w:+.2f}" for attr, label, w in top))


def main() -> None:
    """Choose c1 and c2 on dev; with --final also write predictions and the run file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="predict all final splits and write results")
    parser.add_argument("--no-context", action="store_true", help="ablation: drop the +-2 token window features")
    args = parser.parse_args()
    context = not args.no_context
    name = "crf" if context else "crf-nocontext"

    crf, c1, c2, train_time = select(load_split("train"), load_split("dev"), context)
    print(f"chosen c1={c1} c2={c2} (fit {train_time:.0f}s)")
    describe(crf)
    if not args.final:
        return

    splits: dict[str, Any] = {}
    for split in FINAL_SPLITS:
        records = load_split(split, final=True)
        predictions = predict(crf, featurize(records, context))
        metrics.write_preds(name, SEED, split, [record["id"] for record in records], predictions)
        splits[split] = metrics.evaluate([record["tags"] for record in records], predictions)
        print(f"{split}: macro-F1 {splits[split]['macro_f1']:.4f}")
    run_info = {
        "config": {"c1": c1, "c2": c2, "grid": list(GRID), "max_iterations": MAX_ITERATIONS, "context": context},
        "train_time_s": round(train_time, 1),
        "hardware": f"cpu ({platform.system()} {platform.machine()})",
        "n_params": len(crf.state_features_) + len(crf.transition_features_),
        "best_epoch": None,
    }
    metrics.write_result(name, SEED, splits, run_info)


if __name__ == "__main__":
    main()
