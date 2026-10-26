"""Train and evaluate the APILens first-order HMM tagger, decoded with src/viterbi.py.

Start, transition, end-of-sentence and emission probabilities are counted on
train and smoothed with add-k; k is chosen on dev. Words seen at most once in
train are replaced by a word-shape class, so an unseen word at test time uses
the emissions of its shape. Decoding applies the IOB2 masks from bio_allowed();
--no-mask is the ablation that drops them.
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import metrics
from src.data_io import LABEL2ID, LABELS, load_split
from src.viterbi import bio_allowed, viterbi


SEED = 0
K_GRID = (0.01, 0.1, 1.0)
MIN_COUNT = 2
FINAL_SPLITS = ("dev", "test", "heldout", "test_noparen")
SHAPES = (
    "<number>",
    "<punct>",
    "<has_digit>",
    "<snake_case>",
    "<ALLCAPS>",
    "<CamelCase>",
    "<Capitalized>",
    "<lower>",
    "<other>",
)
NUMBER = re.compile(r"\d+(?:\.\d+)*")


def shape(word: str) -> str:
    """Return the word-shape class that stands in for a rare or unseen word."""
    if NUMBER.fullmatch(word):
        return "<number>"
    if len(word) == 1 and not word.isalnum():
        return "<punct>"
    if any(char.isdigit() for char in word):
        return "<has_digit>"
    if "_" in word:
        return "<snake_case>"
    if word.isupper():
        return "<ALLCAPS>"
    if word[0].isupper():
        return "<CamelCase>" if any(char.isupper() for char in word[1:]) else "<Capitalized>"
    if word.islower():
        return "<lower>"
    return "<other>"


@dataclass
class Counts:
    """Raw label and emission counts from train, shared by every k."""

    vocab: dict[str, int]
    start: np.ndarray
    transitions: np.ndarray
    end: np.ndarray
    emissions: np.ndarray
    shapes: Counter[str]


@dataclass
class HMM:
    """Smoothed log-probabilities; each transition row plus its end score sums to one."""

    vocab: dict[str, int]
    start: np.ndarray
    transitions: np.ndarray
    end: np.ndarray
    emissions: np.ndarray


def word_id(vocab: dict[str, int], word: str) -> int:
    """Map a word to its own id, or to its shape class when it is rare or unseen."""
    return vocab[word] if word in vocab else vocab[shape(word)]


def count(records: list[dict[str, Any]]) -> Counts:
    """Build the vocabulary (shape classes + words seen at least MIN_COUNT times) and count."""
    frequency = Counter(token for record in records for token in record["tokens"])
    words = sorted(word for word, n in frequency.items() if n >= MIN_COUNT)
    vocab = {word: index for index, word in enumerate([*SHAPES, *words])}

    n_labels = len(LABELS)
    counts = Counts(
        vocab,
        np.zeros(n_labels),
        np.zeros((n_labels, n_labels)),
        np.zeros(n_labels),
        np.zeros((n_labels, len(vocab))),
        Counter(),
    )
    for record in records:
        tags = [LABEL2ID[tag] for tag in record["tags"]]
        if not tags:
            continue
        counts.start[tags[0]] += 1
        counts.end[tags[-1]] += 1
        for previous, current in zip(tags, tags[1:]):
            counts.transitions[previous, current] += 1
        for token, tag in zip(record["tokens"], tags):
            if token not in vocab:
                counts.shapes[shape(token)] += 1
            counts.emissions[tag, word_id(vocab, token)] += 1
    return counts


def smooth(counts: Counts, k: float) -> HMM:
    """Turn counts into add-k smoothed log-probabilities.

    Every label is followed either by another label or by the end of the
    sentence, so a transition row has K + 1 outcomes.
    """
    n_labels, n_words = counts.emissions.shape
    outgoing = counts.transitions.sum(axis=1) + counts.end + k * (n_labels + 1)
    return HMM(
        counts.vocab,
        np.log((counts.start + k) / (counts.start.sum() + k * n_labels)),
        np.log((counts.transitions + k) / outgoing[:, None]),
        np.log((counts.end + k) / outgoing),
        np.log((counts.emissions + k) / (counts.emissions.sum(axis=1, keepdims=True) + k * n_words)),
    )


def predict(model: HMM, records: list[dict[str, Any]], use_mask: bool) -> list[list[str]]:
    """Decode every sentence with the shared Viterbi."""
    allowed, start_allowed = bio_allowed(LABELS) if use_mask else (None, None)
    predictions = []
    for record in records:
        ids = [word_id(model.vocab, token) for token in record["tokens"]]
        path = viterbi(model.emissions[:, ids].T, model.transitions, model.start, model.end, allowed, start_allowed)
        predictions.append([LABELS[index] for index in path])
    return predictions


def select_k(counts: Counts, dev: list[dict[str, Any]], use_mask: bool) -> float:
    """Return the k from K_GRID with the best dev macro-F1, printing every score."""
    gold = [record["tags"] for record in dev]
    best_k, best_f1 = K_GRID[0], -1.0
    for k in K_GRID:
        f1 = metrics.evaluate(gold, predict(smooth(counts, k), dev, use_mask))["macro_f1"]
        print(f"k={k:<5} dev macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_k, best_f1 = k, f1
    return best_k


def describe(model: HMM, counts: Counts) -> None:
    """Print the shape classes, the most probable transitions and emitted words per type."""
    print("shape classes (train tokens):", ", ".join(f"{name} {counts.shapes[name]}" for name in SHAPES))
    print("10 most probable transitions:")
    for flat in np.argsort(-model.transitions, axis=None, kind="stable")[:10]:
        previous, current = np.unravel_index(flat, model.transitions.shape)
        print(f"  {LABELS[previous]:>12} -> {LABELS[current]:<12} {np.exp(model.transitions[previous, current]):.3f}")
    words = list(model.vocab)
    for label in LABELS:
        if label.startswith("B-"):
            top = np.argsort(-model.emissions[LABEL2ID[label]], kind="stable")[:10]
            print(f"  {label}: {' '.join(words[index] for index in top)}")


def main() -> None:
    """Choose k on dev; with --final also write predictions and the run file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="predict all final splits and write results")
    parser.add_argument("--no-mask", action="store_true", help="ablation: decode without the IOB2 masks")
    args = parser.parse_args()
    use_mask = not args.no_mask
    name = "hmm" if use_mask else "hmm-nomask"

    started = time.perf_counter()
    counts = count(load_split("train"))
    k = select_k(counts, load_split("dev"), use_mask)
    model = smooth(counts, k)
    train_time = time.perf_counter() - started
    print(f"chosen k={k}")
    describe(model, counts)
    if not args.final:
        return

    splits: dict[str, Any] = {}
    for split in FINAL_SPLITS:
        records = load_split(split, final=True)
        predictions = predict(model, records, use_mask)
        metrics.write_preds(name, SEED, split, [record["id"] for record in records], predictions)
        splits[split] = metrics.evaluate([record["tags"] for record in records], predictions)
        print(f"{split}: macro-F1 {splits[split]['macro_f1']:.4f}")
    run_info = {
        "config": {"k": k, "k_grid": list(K_GRID), "min_count": MIN_COUNT, "mask": use_mask},
        "train_time_s": round(train_time, 1),
        "hardware": f"cpu ({platform.system()} {platform.machine()})",
        "n_params": int(sum(array.size for array in (model.start, model.transitions, model.end, model.emissions))),
        "best_epoch": None,
    }
    metrics.write_result(name, SEED, splits, run_info)


if __name__ == "__main__":
    main()
