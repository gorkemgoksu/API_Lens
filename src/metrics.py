"""Compute every APILens evaluation number: strict entity-level metrics and result writers.

Entities are parsed with seqeval's strict IOB2 scheme: a span exists only if it starts
with B-X and continues with I-X. A stray I-X (after O or after another type) is not an
entity, so a prediction counts only when boundaries and type both match the gold span.
Macro-F1 averages only over the entity types that occur in the gold labels of the evaluated
split; a type with no gold example is excluded, not scored 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from seqeval.scheme import IOB2, Entities

try:  # works both as ``python src/metrics.py`` and as ``from src import metrics``
    from data_io import ENTITY_TYPES, LABELS
except ImportError:  # pragma: no cover
    from src.data_io import ENTITY_TYPES, LABELS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RUN_FILE_KEYS = ("config", "train_time_s", "hardware", "n_params", "best_epoch")
MISSED = "missed"

Tags = Sequence[Sequence[str]]


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float | int]:
    """Precision, recall and F1 from counts; 0.0 when a denominator is 0."""
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": n_gold}


def _check_inputs(gold: Tags, pred: Tags) -> None:
    """Fail early if gold and pred do not line up sentence by sentence."""
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} sentences but pred has {len(pred)}")
    for i, (g, p) in enumerate(zip(gold, pred)):
        if len(g) != len(p):
            raise ValueError(f"sentence {i}: gold has {len(g)} tags but pred has {len(p)}")
        unknown = {t for t in p if t not in LABELS}
        if unknown:
            raise ValueError(f"sentence {i}: unknown labels in pred: {sorted(unknown)}")


def entity_set(tags: Tags) -> set[tuple[int, str, int, int]]:
    """Strict IOB2 entities as (sentence, type, start, end_exclusive) tuples."""
    entities = Entities([list(t) for t in tags], IOB2, False)
    return {e.to_tuple() for sentence in entities.entities for e in sentence}


def is_valid_iob2(tags: Sequence[str]) -> bool:
    """True if every I-X directly follows B-X or I-X of the same type."""
    previous = "O"
    for tag in tags:
        if tag.startswith("I-") and previous[2:] != tag[2:]:
            return False
        previous = tag
    return True


def _span_only(entities: set[tuple[int, str, int, int]]) -> set[tuple[int, int, int]]:
    """Drop the type from entity tuples (boundaries only)."""
    return {(sent, start, end) for sent, _, start, end in entities}


def _confusion(gold_ents: set, pred_ents: set) -> dict[str, Any]:
    """Type confusion over gold spans: boundary match -> predicted type, else 'missed'."""
    pred_type_at = {(s, a, b): t for s, t, a, b in pred_ents}
    columns = list(ENTITY_TYPES) + [MISSED]
    matrix = [[0] * len(columns) for _ in ENTITY_TYPES]
    for sent, gold_type, start, end in gold_ents:
        predicted = pred_type_at.get((sent, start, end), MISSED)
        matrix[ENTITY_TYPES.index(gold_type)][columns.index(predicted)] += 1
    return {"rows": list(ENTITY_TYPES), "columns": columns, "matrix": matrix}


def evaluate(gold: Tags, pred: Tags) -> dict[str, Any]:
    """Score predicted IOB2 tag sequences against gold; every project number comes from here.

    Args:
        gold, pred: one list of IOB2 label strings per sentence, same shape.

    Returns a JSON-serializable dict with:
        macro_f1 / macro_precision / macro_recall: unweighted mean over the entity types that
            occur in gold (support > 0). A type absent from gold is left out rather than
            scored 0, so it cannot lower the average for something that cannot be measured.
        micro_f1 / micro_precision / micro_recall: all entities pooled.
        per_type: {type: precision, recall, f1, support}.
        span_f1: boundaries only, types ignored.
        confusion: 6 x 7 type confusion (see _confusion).
        token_accuracy: secondary metric (most tokens are O).
        invalid_iob2_rate: share of predicted sequences containing an illegal I-X.
        n_sentences, n_gold_entities, n_pred_entities.
    """
    _check_inputs(gold, pred)
    gold_ents, pred_ents = entity_set(gold), entity_set(pred)
    correct = gold_ents & pred_ents

    per_type = {}
    for etype in ENTITY_TYPES:
        per_type[etype] = _prf(
            sum(1 for e in correct if e[1] == etype),
            sum(1 for e in pred_ents if e[1] == etype),
            sum(1 for e in gold_ents if e[1] == etype),
        )
    present = [m for m in per_type.values() if m["support"] > 0]
    macro = {k: sum(m[k] for m in present) / len(present) if present else 0.0
             for k in ("precision", "recall", "f1")}
    micro = _prf(len(correct), len(pred_ents), len(gold_ents))
    span = _prf(len(_span_only(gold_ents) & _span_only(pred_ents)),
                len(_span_only(pred_ents)), len(_span_only(gold_ents)))

    n_tokens = sum(len(g) for g in gold)
    n_right = sum(a == b for g, p in zip(gold, pred) for a, b in zip(g, p))
    invalid = sum(not is_valid_iob2(p) for p in pred)
    return {
        "macro_f1": macro["f1"], "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "micro_f1": micro["f1"], "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "per_type": per_type,
        "span_f1": span["f1"],
        "confusion": _confusion(gold_ents, pred_ents),
        "token_accuracy": n_right / n_tokens if n_tokens else 0.0,
        "invalid_iob2_rate": invalid / len(pred) if len(pred) else 0.0,
        "n_sentences": len(gold),
        "n_gold_entities": len(gold_ents),
        "n_pred_entities": len(pred_ents),
    }


def slices(records: Sequence[dict[str, Any]], gold: Tags, pred: Tags,
           train_targets: set[str]) -> dict[str, Any]:
    """Break results down by library, by source_type and by seen/unseen entity target.

    ``records`` are the split records (same order as gold/pred); ``train_targets`` is the set of
    span targets that occur in train. Returns {"by_library", "by_source_type"} (full evaluate()
    output on each sentence subset) and "target_recall": recall on gold spans whose target is
    unseen in train vs. seen.

    Only recall is reported for seen/unseen: a predicted span carries no target string, so a
    false positive cannot be assigned to "seen" or "unseen"; precision would be undefined.
    """
    if not len(records) == len(gold) == len(pred):
        raise ValueError("records, gold and pred must have the same number of sentences")
    out: dict[str, Any] = {}
    for name, field in (("by_library", "library"), ("by_source_type", "source_type")):
        groups: dict[str, list[int]] = {}
        for i, record in enumerate(records):
            groups.setdefault(record[field], []).append(i)
        out[name] = {key: evaluate([gold[i] for i in idx], [pred[i] for i in idx])
                     for key, idx in sorted(groups.items())}
    found = entity_set(pred)
    hits = {"seen": [0, 0], "unseen": [0, 0]}  # [recalled, total]
    for i, record in enumerate(records):
        for span in record["spans"]:
            group = "seen" if span["target"] in train_targets else "unseen"
            hits[group][1] += 1
            hits[group][0] += (i, span["type"], span["start"], span["end"]) in found
    out["target_recall"] = {k: {"recall": h / n if n else 0.0, "support": n}
                            for k, (h, n) in hits.items()}
    return out


def _sentence_counts(gold: Tags, pred: Tags) -> np.ndarray:
    """Per-sentence [tp, n_pred, n_gold] for every entity type, shape (N, 3, 6)."""
    counts = np.zeros((len(gold), 3, len(ENTITY_TYPES)), dtype=np.int64)
    gold_ents, pred_ents = entity_set(gold), entity_set(pred)
    for column, ents in ((0, gold_ents & pred_ents), (1, pred_ents), (2, gold_ents)):
        for sent, etype, _, _ in ents:
            counts[sent, column, ENTITY_TYPES.index(etype)] += 1
    return counts


def _f1_from_counts(tp: np.ndarray, n_pred: np.ndarray, n_gold: np.ndarray) -> tuple[float, float]:
    """(micro_f1, macro_f1) from per-type summed counts; macro skips types absent from gold."""
    def f1(t: float, p: float, g: float) -> float:
        return 2 * t / (p + g) if p + g else 0.0  # equals 2PR/(P+R)
    micro = f1(tp.sum(), n_pred.sum(), n_gold.sum())
    present = n_gold > 0
    macro = float(np.mean([f1(t, p, g) for t, p, g in
                           zip(tp[present], n_pred[present], n_gold[present])])) if present.any() else 0.0
    return micro, macro


def bootstrap_ci(gold: Tags, pred: Tags, n: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    """95% bootstrap confidence intervals for micro and macro F1, resampling sentences.

    Each resample draws N sentences with replacement, sums their entity counts and recomputes
    F1 from the summed counts (not the mean of per-sentence F1, which would weight a one-entity
    sentence like a thirty-entity one). Returns {"micro_f1": [low, high], "macro_f1": [...]}.
    """
    _check_inputs(gold, pred)
    counts = _sentence_counts(gold, pred)
    rng = np.random.default_rng(seed)
    scores = np.zeros((n, 2))
    for b in range(n):
        weights = np.bincount(rng.integers(0, len(gold), len(gold)), minlength=len(gold))
        tp, n_pred, n_gold = np.tensordot(weights, counts, axes=(0, 0))
        scores[b] = _f1_from_counts(tp, n_pred, n_gold)
    low, high = np.percentile(scores, [2.5, 97.5], axis=0)
    return {"micro_f1": [float(low[0]), float(high[0])],
            "macro_f1": [float(low[1]), float(high[1])]}


def validate_preds(split: Sequence[dict[str, Any]],
                   preds: Sequence[dict[str, Any]]) -> list[list[str]]:
    """Check prediction rows ({"id", "pred"}) against the split records; return tags in split order.

    Raises ValueError for duplicate, missing or extra ids, length mismatches and unknown labels.
    """
    by_id: dict[str, list[str]] = {}
    for row in preds:
        if row["id"] in by_id:
            raise ValueError(f"duplicate prediction id: {row['id']}")
        by_id[row["id"]] = row["pred"]
    split_ids = {record["id"] for record in split}
    if missing := split_ids - by_id.keys():
        raise ValueError(f"{len(missing)} split ids have no prediction, e.g. {sorted(missing)[:3]}")
    if extra := by_id.keys() - split_ids:
        raise ValueError(f"{len(extra)} predicted ids are not in the split, e.g. {sorted(extra)[:3]}")
    ordered = [by_id[record["id"]] for record in split]
    for record, tags in zip(split, ordered):
        if len(tags) != len(record["tokens"]):
            raise ValueError(f"{record['id']}: {len(tags)} tags for {len(record['tokens'])} tokens")
        if unknown := {t for t in tags if t not in LABELS}:
            raise ValueError(f"{record['id']}: unknown labels {sorted(unknown)}")
    return ordered


def write_preds(model: str, seed: int, split: str, ids: Sequence[str],
                preds: Tags) -> Path:
    """Write results/preds/<model>_seed<seed>_<split>.jsonl, one {"id", "pred"} per line."""
    if len(ids) != len(preds):
        raise ValueError(f"{len(ids)} ids but {len(preds)} predictions")
    path = RESULTS_DIR / "preds" / f"{model}_seed{seed}_{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sent_id, tags in zip(ids, preds):
            handle.write(json.dumps({"id": sent_id, "pred": list(tags)}) + "\n")
    return path


def write_result(model: str, seed: int, splits: dict[str, dict[str, Any]],
                 extra: dict[str, Any]) -> Path:
    """Write results/<model>_seed<seed>.json following the run-file contract.

    ``splits`` maps split name -> evaluate() output. ``extra`` may hold config, train_time_s,
    hardware, n_params, best_epoch; missing keys are stored as null, unknown keys are an error.
    """
    unknown = set(extra) - set(RUN_FILE_KEYS)
    if unknown:
        raise ValueError(f"unknown keys in extra: {sorted(unknown)}; allowed: {RUN_FILE_KEYS}")
    run = {"model": model, "seed": seed}
    run.update({key: extra.get(key) for key in RUN_FILE_KEYS})
    run["splits"] = splits
    path = RESULTS_DIR / f"{model}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (metrics is a library; this only provides --help)."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
