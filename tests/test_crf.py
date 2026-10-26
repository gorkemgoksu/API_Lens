"""Test the from-scratch CRF layer against brute-force enumeration."""

from __future__ import annotations

import itertools

import numpy as np
import torch

from src.data_io import LABELS
from src.train_bilstm_crf import CRF
from src.viterbi import bio_allowed


def random_crf(allowed: np.ndarray, start_allowed: np.ndarray, generator: torch.Generator) -> CRF:
    crf = CRF(allowed, start_allowed).double()
    with torch.no_grad():
        for parameter in crf.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64))
    return crf


def brute_force_log_partition(crf: CRF, emissions: torch.Tensor) -> torch.Tensor:
    transitions, start, end = crf.masked_scores()
    length, num_tags = emissions.shape
    paths = torch.tensor(list(itertools.product(range(num_tags), repeat=length)))
    scores = start[paths[:, 0]] + end[paths[:, -1]] + emissions[torch.arange(length), paths].sum(dim=1)
    scores = scores + transitions[paths[:, :-1], paths[:, 1:]].sum(dim=1)
    return torch.logsumexp(scores, dim=0)


def test_crf_log_partition_matches_brute_force() -> None:
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
    generator = torch.Generator().manual_seed(1)
    crf = random_crf(*bio_allowed(LABELS), generator)
    for length in (1, 2, 3):
        emissions = torch.randn(length, len(LABELS), generator=generator, dtype=torch.float64)
        mask = torch.ones(1, length, dtype=torch.bool)
        expected = brute_force_log_partition(crf, emissions)
        torch.testing.assert_close(crf.log_partition(emissions.unsqueeze(0), mask)[0], expected)


def test_crf_ignores_padding() -> None:
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
