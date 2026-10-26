"""Compute every APILens evaluation number: strict entity-level metrics and result writers.

Entities are parsed with seqeval's strict IOB2 scheme: a span exists only if it starts
with B-X and continues with I-X. A stray I-X (after O or after another type) is not an
entity, so a prediction counts only when boundaries and type both match the gold span.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

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
