"""Shared labels, tokenization, schema validation, and split loading for APILens."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


LABELS = [
    "O",
    "B-FUNCTION",
    "I-FUNCTION",
    "B-METHOD",
    "I-METHOD",
    "B-CLASS",
    "I-CLASS",
    "B-MODULE",
    "I-MODULE",
    "B-ATTRIBUTE",
    "I-ATTRIBUTE",
    "B-EXCEPTION",
    "I-EXCEPTION",
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

ENTITY_TYPES = tuple(label.removeprefix("B-") for label in LABELS if label.startswith("B-"))
SOURCE_TYPES = {"guide", "tutorial", "reference", "changelog"}
FINAL_ONLY_SPLITS = {"test", "heldout", "test_noparen"}
SPLIT_NAMES = {"train", "dev", *FINAL_ONLY_SPLITS}
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)*|\S")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Return regex tokens with half-open character offsets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def spans_to_tags(n_tokens: int, spans: Iterable[dict[str, Any]]) -> list[str]:
    """Convert non-overlapping, half-open token spans to a strict IOB2 sequence."""
    if not isinstance(n_tokens, int) or isinstance(n_tokens, bool) or n_tokens < 0:
        raise ValueError("n_tokens must be a non-negative integer")

    tags = ["O"] * n_tokens
    occupied = [False] * n_tokens
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError(f"span {index} must be an object")
        start = span.get("start")
        end = span.get("end")
        entity_type = span.get("type")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"span {index} start must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError(f"span {index} end must be an integer")
        if end <= start:
            raise ValueError(f"span {index} must satisfy end > start")
        if start < 0 or end > n_tokens:
            raise ValueError(f"span {index} [{start}, {end}) is outside {n_tokens} tokens")
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"span {index} has unknown entity type {entity_type!r}")
        if any(occupied[start:end]):
            raise ValueError(f"span {index} overlaps another span")

        tags[start] = f"B-{entity_type}"
        for token_index in range(start + 1, end):
            tags[token_index] = f"I-{entity_type}"
        occupied[start:end] = [True] * (end - start)
    return tags


def tags_to_spans(tags: Iterable[str]) -> list[dict[str, Any]]:
    """Convert a strict IOB2 sequence to half-open token spans."""
    tag_list = list(tags)
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for index, tag in enumerate(tag_list):
        if tag not in LABEL2ID:
            raise ValueError(f"unknown label {tag!r} at token {index}")
        if tag == "O":
            if current is not None:
                current["end"] = index
                spans.append(current)
                current = None
            continue

        prefix, entity_type = tag.split("-", 1)
        if prefix == "B":
            if current is not None:
                current["end"] = index
                spans.append(current)
            current = {"start": index, "end": index + 1, "type": entity_type}
            continue

        if current is None or current["type"] != entity_type:
            expected = f"B-{entity_type} or I-{entity_type}"
            raise ValueError(f"invalid IOB2 label {tag!r} at token {index}; expected {expected} before it")
        current["end"] = index + 1

    if current is not None:
        current["end"] = len(tag_list)
        spans.append(current)
    return spans


def _require_string(record: dict[str, Any], field: str) -> None:
    if not isinstance(record.get(field), str) or not record[field]:
        raise ValueError(f"{field} must be a non-empty string")


def validate(record: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a split record violates the project schema."""
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    for field in ("id", "library", "doc_path"):
        _require_string(record, field)

    source_type = record.get("source_type")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")

    tokens = record.get("tokens")
    tags = record.get("tags")
    spans = record.get("spans")
    if not isinstance(tokens, list) or any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("tokens must be a list of non-empty strings")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("tags must be a list of strings")
    if len(tokens) != len(tags):
        raise ValueError(f"len(tokens) ({len(tokens)}) != len(tags) ({len(tags)})")
    if not isinstance(spans, list):
        raise ValueError("spans must be a list")

    tag_spans = tags_to_spans(tags)
    schema_spans: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError(f"span {index} must be an object")
        for field in ("target", "role"):
            if not isinstance(span.get(field), str) or not span[field]:
                raise ValueError(f"span {index} {field} must be a non-empty string")
        schema_spans.append({key: span.get(key) for key in ("start", "end", "type")})

    spans_to_tags(len(tokens), schema_spans)
    if schema_spans != tag_spans:
        raise ValueError(f"spans disagree with tags: expected {tag_spans}, got {schema_spans}")


def load_split(name: str, final: bool = False) -> list[dict[str, Any]]:
    """Load and validate a processed split, guarding final evaluation data."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {sorted(SPLIT_NAMES)}")
    if name in FINAL_ONLY_SPLITS and not final:
        raise ValueError(f"refusing to load {name!r} without final=True")

    path = Path(__file__).resolve().parents[1] / "data" / "processed" / f"{name}.jsonl"
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            try:
                validate(record)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if record["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record['id']!r}")
            seen_ids.add(record["id"])
            records.append(record)
    return records


def main() -> None:
    """Validate one processed split from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=sorted(SPLIT_NAMES))
    parser.add_argument("--final", action="store_true", help="allow final-only evaluation splits")
    args = parser.parse_args()
    records = load_split(args.split, final=args.final)
    print(f"Validated {len(records)} records from {args.split}.")


if __name__ == "__main__":
    main()
