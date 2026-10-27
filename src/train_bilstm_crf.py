"""Train and evaluate the APILens BiLSTM-CRF tagger with a from-scratch CRF layer."""

from __future__ import annotations

import argparse
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import LABEL2ID, LABELS, load_split
from src.metrics import evaluate, write_preds, write_result
from src.viterbi import bio_allowed, viterbi


SEEDS = (13, 42, 2026)
FINAL_SPLITS = ("dev", "test", "heldout", "test_noparen")
CHECKPOINT_DIR = ROOT / "checkpoints"
PAD, UNK = "<pad>", "<unk>"
ILLEGAL = -1e4

CONFIG = {
    "word_dim": 100,
    "min_freq": 2,
    "char_dim": 30,
    "char_filters": 50,
    "char_width": 3,
    "hidden": 200,
    "dropout": 0.5,
    "lr": 1e-3,
    "batch_size": 32,
    "clip": 5.0,
    "patience": 5,
}


class CRF(nn.Module):
    """Linear-chain CRF whose illegal IOB2 transitions are fixed to a large negative score."""

    def __init__(self, allowed: np.ndarray, start_allowed: np.ndarray) -> None:
        super().__init__()
        num_tags = len(start_allowed)
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        self.register_buffer("allowed", torch.as_tensor(np.asarray(allowed), dtype=torch.bool))
        self.register_buffer("start_allowed", torch.as_tensor(np.asarray(start_allowed), dtype=torch.bool))

    def masked_scores(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return transition, start and end scores with illegal moves set to ILLEGAL."""
        transitions = self.transitions.masked_fill(~self.allowed, ILLEGAL)
        start = self.start.masked_fill(~self.start_allowed, ILLEGAL)
        return transitions, start, self.end

    def log_partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """Return log Z for each sentence of [B, T, K] emissions with the batched forward algorithm."""
        transitions, start, end = self.masked_scores()
        alpha = start + emissions[:, 0]
        for t in range(1, emissions.size(1)):
            step = torch.logsumexp(alpha.unsqueeze(2) + transitions + emissions[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), step, alpha)
        return torch.logsumexp(alpha + end, dim=1)

    def path_score(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        """Return the unnormalized score of each [B, T] tag path, ignoring padded positions."""
        transitions, start, end = self.masked_scores()
        batch = torch.arange(emissions.size(0), device=emissions.device)
        score = start[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        for t in range(1, emissions.size(1)):
            step = transitions[tags[:, t - 1], tags[:, t]] + emissions[batch, t, tags[:, t]]
            score = score + step * mask[:, t]
        last = tags[batch, mask.sum(dim=1) - 1]
        return score + end[last]

    def loss(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        """Return the batch mean of log Z minus the gold path score."""
        return (self.log_partition(emissions, mask) - self.path_score(emissions, tags, mask)).mean()

    def decode(self, emissions: Tensor, mask: Tensor) -> list[list[int]]:
        """Decode each sentence with the shared Viterbi on the learned, masked scores."""
        transitions, start, end = (score.detach().cpu().double().numpy() for score in self.masked_scores())
        allowed = self.allowed.cpu().numpy()
        start_allowed = self.start_allowed.cpu().numpy()
        scores = emissions.detach().cpu().double().numpy()
        lengths = mask.sum(dim=1).tolist()
        return [list(viterbi(scores[i, :n], transitions, start, end, allowed, start_allowed)) for i, n in enumerate(lengths)]


class BiLSTMTagger(nn.Module):
    """Word embeddings and an optional char CNN feeding a BiLSTM with a CRF or softmax output."""

    def __init__(self, n_words: int, n_chars: int, use_char: bool, use_crf: bool) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(n_words, CONFIG["word_dim"], padding_idx=0)
        self.char_emb = nn.Embedding(n_chars, CONFIG["char_dim"], padding_idx=0) if use_char else None
        self.char_cnn = (
            nn.Conv1d(CONFIG["char_dim"], CONFIG["char_filters"], CONFIG["char_width"], padding=1) if use_char else None
        )
        input_dim = CONFIG["word_dim"] + (CONFIG["char_filters"] if use_char else 0)
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.lstm = nn.LSTM(input_dim, CONFIG["hidden"], batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * CONFIG["hidden"], len(LABELS))
        self.crf = CRF(*bio_allowed(LABELS)) if use_crf else None

    def char_features(self, chars: Tensor, mask: Tensor) -> Tensor:
        """Return max-pooled char CNN features [B, T, F], ignoring padded characters."""
        batch, length, width = chars.shape
        flat = chars.view(batch * length, width)
        hidden = self.char_cnn(self.char_emb(flat).transpose(1, 2))
        hidden = hidden.masked_fill((flat == 0).unsqueeze(1), float("-inf")).max(dim=2).values
        return hidden.view(batch, length, -1).masked_fill(~mask.unsqueeze(2), 0.0)

    def emissions(self, words: Tensor, chars: Tensor, mask: Tensor) -> Tensor:
        """Return [B, T, K] emission scores."""
        features = self.word_emb(words)
        if self.char_emb is not None:
            features = torch.cat([features, self.char_features(chars, mask)], dim=2)
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(self.dropout(features), lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = self.lstm(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=words.size(1))
        return self.out(hidden)

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the CRF negative log-likelihood or the token cross-entropy."""
        emissions = self.emissions(batch["words"], batch["chars"], batch["mask"])
        if self.crf is not None:
            return self.crf.loss(emissions, batch["tags"], batch["mask"])
        return nn.functional.cross_entropy(emissions[batch["mask"]], batch["tags"][batch["mask"]])

    def predict(self, batch: dict[str, Tensor]) -> list[list[int]]:
        """Return label indices per sentence: Viterbi for the CRF, argmax for softmax."""
        emissions = self.emissions(batch["words"], batch["chars"], batch["mask"])
        if self.crf is not None:
            return self.crf.decode(emissions, batch["mask"])
        best = emissions.argmax(dim=2).cpu().tolist()
        return [row[:n] for row, n in zip(best, batch["mask"].sum(dim=1).tolist())]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=50, help="maximum number of training epochs")
    parser.add_argument("--final", action="store_true", help="evaluate all final splits and write results")
    ablation = parser.add_mutually_exclusive_group()
    ablation.add_argument("--no-char", action="store_true", help="ablation: drop the character CNN")
    ablation.add_argument("--softmax", action="store_true", help="ablation: per-token softmax instead of the CRF")
    return parser.parse_args()


def model_name(args: argparse.Namespace) -> str:
    """Return the model or ablation name used in result file names."""
    if args.no_char:
        return "bilstm_crf-nochar"
    if args.softmax:
        return "bilstm-softmax"
    return "bilstm_crf"


def set_seed(seed: int) -> None:
    """Seed every random number generator and force deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def build_vocab(sequences: Iterable[Iterable[str]], min_freq: int) -> dict[str, int]:
    """Map items seen at least min_freq times to ids after the <pad> and <unk> entries."""
    counts = Counter(item for sequence in sequences for item in sequence)
    vocab = {PAD: 0, UNK: 1}
    for item in sorted(item for item, count in counts.items() if count >= min_freq):
        vocab[item] = len(vocab)
    return vocab


def encode(records: list[dict[str, Any]], words: dict[str, int], chars: dict[str, int]) -> list[dict[str, Any]]:
    """Convert records to word, character and label ids."""
    return [
        {
            "words": [words.get(token, words[UNK]) for token in record["tokens"]],
            "chars": [[chars.get(char, chars[UNK]) for char in token] for token in record["tokens"]],
            "tags": [LABEL2ID[tag] for tag in record["tags"]],
        }
        for record in records
    ]


def collate(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Pad a list of encoded sentences into batch tensors."""
    length = max(len(item["words"]) for item in batch)
    width = max(len(chars) for item in batch for chars in item["chars"])
    words = torch.zeros(len(batch), length, dtype=torch.long)
    tags = torch.zeros(len(batch), length, dtype=torch.long)
    chars = torch.zeros(len(batch), length, width, dtype=torch.long)
    mask = torch.zeros(len(batch), length, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = len(item["words"])
        words[i, :n] = torch.tensor(item["words"])
        tags[i, :n] = torch.tensor(item["tags"])
        mask[i, :n] = True
        for j, token_chars in enumerate(item["chars"]):
            chars[i, j, : len(token_chars)] = torch.tensor(token_chars)
    return {"words": words, "chars": chars, "tags": tags, "mask": mask}


def to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    """Move every batch tensor to the device."""
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def predict(model: BiLSTMTagger, data: list[dict[str, Any]], device: torch.device) -> list[list[str]]:
    """Return predicted label strings for every encoded sentence."""
    model.eval()
    loader = DataLoader(data, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate)
    predictions: list[list[str]] = []
    for batch in loader:
        predictions.extend([LABELS[index] for index in row] for row in model.predict(to_device(batch, device)))
    return predictions


def train(
    model: BiLSTMTagger,
    train_data: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    dev_data: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, float]:
    """Train with early stopping on dev macro-F1, load the best weights and return (best_epoch, best_f1)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate, generator=generator)
    dev_gold = [record["tags"] for record in dev_records]
    best_epoch, best_f1, best_state = 0, -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.perf_counter(), 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model.loss(to_device(batch, device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip"])
            optimizer.step()
            total += loss.item()
        dev_f1 = evaluate(dev_gold, predict(model, dev_data, device))["macro_f1"]
        print(f"epoch {epoch:2d}  loss {total / len(loader):.4f}  dev macro-F1 {dev_f1:.4f}  {time.perf_counter() - started:.0f}s", flush=True)
        if dev_f1 > best_f1:
            best_epoch, best_f1 = epoch, dev_f1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= CONFIG["patience"]:
            print(f"early stop: no dev improvement for {CONFIG['patience']} epochs")
            break
    model.load_state_dict(best_state)
    return best_epoch, best_f1


def hardware(device: torch.device) -> str:
    """Describe the device used for training."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"cpu ({platform.system()} {platform.machine()})"


def run_final(
    model: BiLSTMTagger,
    words: dict[str, int],
    chars: dict[str, int],
    args: argparse.Namespace,
    run_info: dict[str, Any],
    device: torch.device,
) -> None:
    """Predict every final split, write predictions, the run file and the checkpoint."""
    name = model_name(args)
    splits: dict[str, Any] = {}
    for split in FINAL_SPLITS:
        records = load_split(split, final=True)
        predictions = predict(model, encode(records, words, chars), device)
        write_preds(name, args.seed, split, [record["id"] for record in records], predictions)
        splits[split] = evaluate([record["tags"] for record in records], predictions)
        print(f"{split}: macro-F1 {splits[split]['macro_f1']:.4f}")
    write_result(name, args.seed, splits, run_info)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint = {"config": run_info["config"], "words": words, "chars": chars, "state_dict": model.state_dict()}
    torch.save(checkpoint, CHECKPOINT_DIR / f"{name}_seed{args.seed}.pt")


def main() -> None:
    """Train on train, report dev, and with --final evaluate and save every final split."""
    args = parse_args()
    set_seed(args.seed)
    # MPS is skipped so that the same seed reproduces the same scores.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_records, dev_records = load_split("train"), load_split("dev")
    words = build_vocab((record["tokens"] for record in train_records), CONFIG["min_freq"])
    chars = build_vocab((token for record in train_records for token in record["tokens"]), 1)
    model = BiLSTMTagger(len(words), len(chars), use_char=not args.no_char, use_crf=not args.softmax).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"{model_name(args)} seed {args.seed} on {hardware(device)}: {n_params} parameters, {len(words)} words, {len(chars)} chars")

    started = time.perf_counter()
    best_epoch, best_f1 = train(model, encode(train_records, words, chars), dev_records, encode(dev_records, words, chars), args, device)
    train_time = time.perf_counter() - started
    print(f"best epoch {best_epoch}: dev macro-F1 {best_f1:.4f} ({train_time:.0f}s)")

    if args.final:
        config = {**CONFIG, "max_epochs": args.epochs, "use_char": not args.no_char, "use_crf": not args.softmax}
        run_info = {
            "config": config,
            "train_time_s": round(train_time, 1),
            "hardware": hardware(device),
            "n_params": n_params,
            "best_epoch": best_epoch,
        }
        run_final(model, words, chars, args, run_info, device)


if __name__ == "__main__":
    main()
