"""Tests for constrained Viterbi decoding and the IOB2 masks."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.data_io import LABEL2ID, LABELS, tags_to_spans
from src.viterbi import bio_allowed, viterbi


def path_score(
    path: tuple[int, ...] | list[int],
    emissions: np.ndarray,
    transitions: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    score = start[path[0]] + emissions[0, path[0]] + end[path[-1]]
    for t in range(1, len(path)):
        score += transitions[path[t - 1], path[t]] + emissions[t, path[t]]
    return float(score)


def brute_force(
    emissions: np.ndarray, transitions: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[list[int], float]:
    n_steps, n_labels = emissions.shape
    best = max(
        itertools.product(range(n_labels), repeat=n_steps),
        key=lambda path: path_score(path, emissions, transitions, start, end),
    )
    return list(best), path_score(best, emissions, transitions, start, end)


def random_problem(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    n_steps = int(rng.integers(1, 7))
    n_labels = int(rng.integers(1, 5))
    return (
        rng.normal(size=(n_steps, n_labels)),
        rng.normal(size=(n_labels, n_labels)),
        rng.normal(size=n_labels),
        rng.normal(size=n_labels),
    )


def test_matches_brute_force_on_200_random_problems() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        emissions, transitions, start, end = random_problem(rng)
        path = viterbi(emissions, transitions, start, end)
        expected_path, expected_score = brute_force(emissions, transitions, start, end)
        assert path == expected_path
        assert path_score(path, emissions, transitions, start, end) == pytest.approx(expected_score)


def test_matches_brute_force_with_random_masks() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        emissions, transitions, start, end = random_problem(rng)
        n_labels = emissions.shape[1]
        allowed = rng.random((n_labels, n_labels)) < 0.6
        np.fill_diagonal(allowed, True)
        path = viterbi(emissions, transitions, start, end, allowed)
        masked = np.where(allowed, transitions, -np.inf)
        assert path == brute_force(emissions, masked, start, end)[0]
        assert all(allowed[a, b] for a, b in zip(path, path[1:]))


def test_worked_example_from_the_plan() -> None:
    labels = ["O", "B-FUNCTION", "I-FUNCTION"]
    transitions = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 2.0], [0.0, -1.0, 1.0]])
    emissions = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 1.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0]])
    start, end = np.zeros(3), np.zeros(3)
    allowed, start_allowed = bio_allowed(labels)
    path = viterbi(emissions, transitions, start, end, allowed, start_allowed)
    assert path == [0, 1, 2, 2]
    assert path_score(path, emissions, transitions, start, end) == 12.0


def test_bio_allowed_masks() -> None:
    allowed, start_allowed = bio_allowed(LABELS)
    o, b_func, i_func = LABEL2ID["O"], LABEL2ID["B-FUNCTION"], LABEL2ID["I-FUNCTION"]
    assert allowed.shape == (13, 13) and start_allowed.shape == (13,)
    assert allowed[b_func, i_func] and allowed[i_func, i_func] and allowed[i_func, o]
    assert not allowed[o, i_func]
    assert not allowed[LABEL2ID["B-METHOD"], i_func]
    assert not allowed[LABEL2ID["I-METHOD"], i_func]
    assert (~allowed).sum() == 6 * 11
    assert [LABELS[j] for j in np.flatnonzero(~start_allowed)] == [l for l in LABELS if l.startswith("I-")]


@pytest.mark.parametrize("labels", [["O", "O"], ["O", "X-FUNCTION"], ["O", "B-"]])
def test_bio_allowed_rejects_bad_labels(labels: list[str]) -> None:
    with pytest.raises(ValueError):
        bio_allowed(labels)


def test_iob2_mask_always_gives_valid_sequences() -> None:
    rng = np.random.default_rng(2)
    allowed, start_allowed = bio_allowed(LABELS)
    n_labels = len(LABELS)
    for _ in range(200):
        n_steps = int(rng.integers(1, 9))
        path = viterbi(
            rng.normal(scale=3.0, size=(n_steps, n_labels)),
            rng.normal(size=(n_labels, n_labels)),
            rng.normal(size=n_labels),
            rng.normal(size=n_labels),
            allowed,
            start_allowed,
        )
        tags_to_spans([LABELS[index] for index in path])


def test_mask_fixes_an_invalid_unmasked_decode() -> None:
    allowed, start_allowed = bio_allowed(LABELS)
    n_labels = len(LABELS)
    emissions = np.zeros((2, n_labels))
    emissions[:, LABEL2ID["I-FUNCTION"]] = 5.0
    zeros = np.zeros(n_labels)
    transitions = np.zeros((n_labels, n_labels))

    unmasked = [LABELS[i] for i in viterbi(emissions, transitions, zeros, zeros)]
    masked = [LABELS[i] for i in viterbi(emissions, transitions, zeros, zeros, allowed, start_allowed)]
    assert unmasked == ["I-FUNCTION", "I-FUNCTION"]
    with pytest.raises(ValueError, match="invalid IOB2"):
        tags_to_spans(unmasked)
    assert masked == ["B-FUNCTION", "I-FUNCTION"]


def test_single_token_takes_best_start_plus_emission_plus_end() -> None:
    emissions = np.array([[0.5, 2.0, 1.0]])
    start = np.array([0.0, -1.5, 0.0])
    end = np.array([0.0, 0.0, 0.2])
    assert viterbi(emissions, np.zeros((3, 3)), start, end) == [2]


def test_ties_are_deterministic_and_go_to_the_lowest_index() -> None:
    emissions = np.zeros((5, 4))
    transitions = np.zeros((4, 4))
    paths = [viterbi(emissions, transitions, np.zeros(4), np.zeros(4)) for _ in range(3)]
    assert paths == [[0, 0, 0, 0, 0]] * 3


def test_empty_sentence_returns_empty_path() -> None:
    assert viterbi(np.zeros((0, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3)) == []


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="transitions must have shape"):
        viterbi(np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(3), np.zeros(3))


def test_no_finite_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="no label path"):
        viterbi(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), np.zeros(2), np.zeros((2, 2), dtype=bool))
