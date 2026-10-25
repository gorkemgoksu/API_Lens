"""Tests for the shared APILens data contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_io import LABELS, load_split, spans_to_tags, tags_to_spans, tokenize, validate


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mini.jsonl"


def test_spans_tags_round_trip() -> None:
    spans = [
        {"start": 1, "end": 4, "type": "METHOD"},
        {"start": 6, "end": 7, "type": "EXCEPTION"},
    ]
    tags = spans_to_tags(8, spans)
    assert tags == ["O", "B-METHOD", "I-METHOD", "I-METHOD", "O", "O", "B-EXCEPTION", "O"]
    assert tags_to_spans(tags) == spans


@pytest.mark.parametrize(
    "tags",
    [
        ["I-CLASS"],
        ["B-CLASS", "I-METHOD"],
        ["O", "I-FUNCTION"],
    ],
)
def test_invalid_iob2_is_rejected(tags: list[str]) -> None:
    with pytest.raises(ValueError, match="invalid IOB2"):
        tags_to_spans(tags)


def test_tokenizer_offsets_reconstruct_every_token() -> None:
    text = "Use DataFrame.groupby() with NumPy 1.26.4."
    output = tokenize(text)
    assert [token for token, _, _ in output] == [
        "Use",
        "DataFrame",
        ".",
        "groupby",
        "(",
        ")",
        "with",
        "NumPy",
        "1.26.4",
        ".",
    ]
    assert all(text[start:end] == token for token, start, end in output)


def test_fixture_covers_all_types_and_validates() -> None:
    records = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]
    for record in records:
        validate(record)
    observed = {span["type"] for record in records for span in record["spans"]}
    assert observed == {label[2:] for label in LABELS if label.startswith("B-")}


def test_validate_rejects_spans_that_disagree_with_tags() -> None:
    record = json.loads(FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[0])
    record["spans"][0]["end"] += 1
    with pytest.raises(ValueError, match="spans disagree with tags"):
        validate(record)


@pytest.mark.parametrize("name", ["test", "heldout", "test_noparen"])
def test_final_splits_are_guarded_before_file_access(name: str) -> None:
    with pytest.raises(ValueError, match="without final=True"):
        load_split(name)
