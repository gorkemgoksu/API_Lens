"""Constrained log-space Viterbi decoding shared by the HMM and CRF taggers.

The HMM, the BiLSTM-CRF and the ModernBERT-CRF all decode with ``viterbi``;
they differ only in where the emission and transition scores come from.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np


def bio_allowed(labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the IOB2 masks ``(allowed[K, K], start_allowed[K])`` for ``labels``.

    ``allowed[i, j]`` is True when label ``j`` may directly follow label ``i``:
    ``I-X`` may only follow ``B-X`` or ``I-X``, and every other label may follow
    anything. ``start_allowed[j]`` is False exactly for the ``I-X`` labels,
    because a sentence cannot begin inside an entity.
    """
    label_list = list(labels)
    if len(set(label_list)) != len(label_list):
        raise ValueError("labels must be unique")
    for label in label_list:
        if label != "O" and not (label[:2] in ("B-", "I-") and len(label) > 2):
            raise ValueError(f"label {label!r} is not 'O', 'B-<TYPE>' or 'I-<TYPE>'")

    n_labels = len(label_list)
    allowed = np.ones((n_labels, n_labels), dtype=bool)
    start_allowed = np.ones(n_labels, dtype=bool)
    for j, label in enumerate(label_list):
        if not label.startswith("I-"):
            continue
        entity_type = label[2:]
        start_allowed[j] = False
        for i, previous in enumerate(label_list):
            allowed[i, j] = previous in (f"B-{entity_type}", f"I-{entity_type}")
    return allowed, start_allowed


def _as_scores(name: str, values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.isnan(array).any():
        raise ValueError(f"{name} contains NaN")
    return array


def _as_mask(name: str, values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def viterbi(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    allowed: np.ndarray | None = None,
    start_allowed: np.ndarray | None = None,
) -> list[int]:
    r"""Return the label indices of the highest-scoring path.

    All scores are in log space and add up. With emissions ``E[T, K]``,
    transitions ``A[K, K]`` (row = previous label, column = next label) and
    ``start[K]``, ``end[K]``, a path ``y_1 .. y_T`` scores

        score(y) = start[y_1] + E_1(y_1) + sum_{t=2..T} (A[y_{t-1}, y_t] + E_t(y_t)) + end[y_T]

    The dynamic program keeps, for every label ``j`` at step ``t``, the best
    score of any prefix ending in ``j`` and the label it came from:

        delta_1(j) = start[j] + E_1(j)
        delta_t(j) = max_i [delta_{t-1}(i) + A[i, j]] + E_t(j)
        psi_t(j)   = argmax_i [delta_{t-1}(i) + A[i, j]]
        y_T        = argmax_j [delta_T(j) + end[j]]
        y_{t-1}    = psi_t(y_t)

    Time is O(T * K^2) and memory O(T * K), instead of scoring all K^T paths.
    Transitions where ``allowed`` is False and start labels where
    ``start_allowed`` is False score -inf, so they are never chosen; pass both
    masks from ``bio_allowed`` to guarantee valid IOB2. Ties go to the lowest
    label index, so the output is deterministic.
    """
    emission_array = np.asarray(emissions, dtype=np.float64)
    if emission_array.ndim != 2:
        raise ValueError(f"emissions must have shape [T, K], got {emission_array.shape}")
    n_steps, n_labels = emission_array.shape
    if n_labels == 0:
        raise ValueError("emissions must have at least one label column")
    emission_array = _as_scores("emissions", emission_array, (n_steps, n_labels))
    transition_array = _as_scores("transitions", transitions, (n_labels, n_labels))
    start_array = _as_scores("start", start, (n_labels,))
    end_array = _as_scores("end", end, (n_labels,))
    if allowed is not None:
        transition_array = np.where(_as_mask("allowed", allowed, (n_labels, n_labels)), transition_array, -np.inf)
    if start_allowed is not None:
        start_array = np.where(_as_mask("start_allowed", start_allowed, (n_labels,)), start_array, -np.inf)
    if n_steps == 0:
        return []

    delta = start_array + emission_array[0]
    backpointers = np.zeros((n_steps, n_labels), dtype=np.int64)
    for t in range(1, n_steps):
        candidates = delta[:, None] + transition_array
        backpointers[t] = candidates.argmax(axis=0)
        delta = candidates.max(axis=0) + emission_array[t]

    final = delta + end_array
    best = int(final.argmax())
    if np.isneginf(final[best]):
        raise ValueError("no label path has a finite score; check the allowed masks")
    path = [best]
    for t in range(n_steps - 1, 0, -1):
        path.append(int(backpointers[t, path[-1]]))
    path.reverse()
    return path


def main() -> None:
    """Decode the three-label worked example from the project plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    labels = ["O", "B-FUNCTION", "I-FUNCTION"]
    tokens = ["call", "numpy", ".", "mean"]
    transitions = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 2.0], [0.0, -1.0, 1.0]])
    emissions = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 1.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0]])
    allowed, start_allowed = bio_allowed(labels)
    path = viterbi(emissions, transitions, np.zeros(3), np.zeros(3), allowed, start_allowed)
    for token, index in zip(tokens, path):
        print(f"{token:>6}  {labels[index]}")


if __name__ == "__main__":
    main()
