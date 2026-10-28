"""Check API mentions tagged in a text against the library inventories; usage: python src/hallucination_check.py answer.txt checkpoints/bilstm_crf_seed42.pt"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import sphobjinv
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import tags_to_spans, tokenize
from src.train_bilstm_crf import BiLSTMTagger, encode, predict

INVENTORY_DIR = ROOT / "data" / "raw" / "inventories"
ROLE_TO_TYPE = {
    "function": "FUNCTION",
    "method": "METHOD",
    "classmethod": "METHOD",
    "staticmethod": "METHOD",
    "class": "CLASS",
    "module": "MODULE",
    "attribute": "ATTRIBUTE",
    "property": "ATTRIBUTE",
    "data": "ATTRIBUTE",
    "exception": "EXCEPTION",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path, help="text file to check, for example an LLM answer")
    parser.add_argument("checkpoint", type=Path, help="saved BiLSTM-CRF checkpoint (.pt)")
    return parser.parse_args()


def load_inventory() -> dict[str, str]:
    """Map every Python object name in the local inventories to its entity type."""
    names: dict[str, str] = {}
    for path in sorted(INVENTORY_DIR.glob("*.inv")):
        for obj in sphobjinv.Inventory(str(path)).objects:
            if obj.domain == "py" and obj.role in ROLE_TO_TYPE:
                names.setdefault(obj.name, ROLE_TO_TYPE[obj.role])
    return names


def lookup(mention: str, names: dict[str, str]) -> list[str]:
    """Return the inventory names matching exactly, or else by dotted suffix."""
    if mention in names:
        return [mention]
    return [name for name in names if name.endswith("." + mention)]


def load_model(path: Path) -> tuple[BiLSTMTagger, dict[str, int], dict[str, int]]:
    """Rebuild the tagger and vocabularies from a checkpoint."""
    if not path.is_file():
        sys.exit(f"error: checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = BiLSTMTagger(len(checkpoint["words"]), len(checkpoint["chars"]), config["use_char"], config["use_crf"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["words"], checkpoint["chars"]


def find_mentions(text: str, model: BiLSTMTagger, words: dict[str, int], chars: dict[str, int]) -> list[tuple[str, str]]:
    """Return (mention, predicted type) for every tagged span, with "()" removed."""
    mentions = []
    for sentence in filter(None, (part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text))):
        tokens = [token for token, _, _ in tokenize(sentence)]
        record = {"tokens": tokens, "tags": ["O"] * len(tokens)}
        tags = predict(model, encode([record], words, chars), torch.device("cpu"))[0]
        for span in tags_to_spans(tags):
            name = "".join(tokens[span["start"] : span["end"]]).replace("()", "")
            mentions.append((name, span["type"]))
    return mentions


def main() -> None:
    """Tag the text, look every mention up in the inventories and print a table."""
    args = parse_args()
    if not args.text.is_file():
        sys.exit(f"error: text file not found: {args.text}")
    model, words, chars = load_model(args.checkpoint)
    names = load_inventory()
    mentions = find_mentions(args.text.read_text(encoding="utf-8"), model, words, chars)

    print(f"{'mention':<40} {'predicted':<10} {'status':<10} {'inventory':<10} agrees")
    missing = ambiguous = 0
    for mention, predicted in mentions:
        hits = lookup(mention, names)
        if not hits:
            missing += 1
            status, actual, agrees = "NOT FOUND", "-", "-"
        elif len(hits) > 1:
            ambiguous += 1
            status, actual, agrees = "AMBIGUOUS", str(len(hits)), "-"
        else:
            status, actual = "FOUND", names[hits[0]]
            agrees = "yes" if actual == predicted else "no"
        print(f"{mention:<40} {predicted:<10} {status:<10} {actual:<10} {agrees}")
    print(f"{len(mentions)} mentions, {missing} not found, {ambiguous} ambiguous")


if __name__ == "__main__":
    main()
