"""Tests for constrained Viterbi decoding and the IOB2 masks."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from src.data_io import LABEL2ID, LABELS, tags_to_spans
from src.metrics import is_valid_iob2
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


# CRF layer tests.
def crf_modules() -> tuple[Any, Any]:
    """Return torch and the CRF class, skipping the test when torch is missing."""
    torch = pytest.importorskip("torch")
    from src.train_bilstm_crf import CRF

    return torch, CRF


def random_crf(allowed: np.ndarray, start_allowed: np.ndarray, generator: Any) -> Any:
    torch, CRF = crf_modules()
    crf = CRF(allowed, start_allowed).double()
    with torch.no_grad():
        for parameter in crf.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64))
    return crf


def brute_force_log_partition(crf: Any, emissions: Any) -> Any:
    torch, _ = crf_modules()
    transitions, start, end = crf.masked_scores()
    length, num_tags = emissions.shape
    paths = torch.tensor(list(itertools.product(range(num_tags), repeat=length)))
    scores = start[paths[:, 0]] + end[paths[:, -1]] + emissions[torch.arange(length), paths].sum(dim=1)
    scores = scores + transitions[paths[:, :-1], paths[:, 1:]].sum(dim=1)
    return torch.logsumexp(scores, dim=0)


def test_crf_log_partition_matches_brute_force() -> None:
    torch, CRF = crf_modules()
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        num_tags = int(torch.randint(2, 5, (1,), generator=generator))
        length = int(torch.randint(1, 6, (1,), generator=generator))
        crf = random_crf(np.ones((num_tags, num_tags), bool), np.ones(num_tags, bool), generator)
        emissions = torch.randn(length, num_tags, generator=generator, dtype=torch.float64)
        mask = torch.ones(1, length, dtype=torch.bool)
        expected = brute_force_log_partition(crf, emissions)
        torch.testing.assert_close(crf.log_partition(emissions.unsqueeze(0), mask)[0], expected)


def test_crf_log_partition_with_iob2_mask_matches_brute_force() -> None:
    torch, CRF = crf_modules()
    generator = torch.Generator().manual_seed(1)
    crf = random_crf(*bio_allowed(LABELS), generator)
    for length in (1, 2, 3):
        emissions = torch.randn(length, len(LABELS), generator=generator, dtype=torch.float64)
        mask = torch.ones(1, length, dtype=torch.bool)
        expected = brute_force_log_partition(crf, emissions)
        torch.testing.assert_close(crf.log_partition(emissions.unsqueeze(0), mask)[0], expected)


def test_crf_ignores_padding() -> None:
    torch, CRF = crf_modules()
    generator = torch.Generator().manual_seed(2)
    crf = random_crf(*bio_allowed(LABELS), generator)
    emissions = torch.randn(2, 6, len(LABELS), generator=generator, dtype=torch.float64)
    tags = torch.tensor([[0, 1, 2, 0, 5, 0], [3, 4, 4, 0, 0, 0]])
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    log_z = crf.log_partition(emissions, mask)
    score = crf.path_score(emissions, tags, mask)
    torch.testing.assert_close(log_z[1], crf.log_partition(emissions[1:, :4], mask[1:, :4])[0])
    torch.testing.assert_close(score[1], crf.path_score(emissions[1:, :4], tags[1:, :4], mask[1:, :4])[0])
    assert bool((log_z >= score).all())


def test_crf_decode_never_starts_with_inside_label() -> None:
    torch, CRF = crf_modules()
    crf = CRF(*bio_allowed(LABELS)).double()
    emissions = torch.zeros(1, 4, len(LABELS), dtype=torch.float64)
    emissions[0, 0, LABEL2ID["I-FUNCTION"]] = 1e6
    path = crf.decode(emissions, torch.ones(1, 4, dtype=torch.bool))[0]
    tags = [LABELS[i] for i in path]
    assert not tags[0].startswith("I-")
    assert is_valid_iob2(tags)
