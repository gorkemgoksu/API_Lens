"""Fine-tune ModernBERT-base for APILens tagging with a softmax or CRF head."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import ID2LABEL, LABEL2ID, LABELS, load_split
from src.metrics import evaluate, write_preds, write_result
from src.train_bilstm_crf import CHECKPOINT_DIR, CRF, FINAL_SPLITS, SEEDS, hardware, set_seed
from src.viterbi import bio_allowed


ENCODER = "answerdotai/ModernBERT-base"
LR_GRID = (3e-5, 5e-5, 8e-5)
SWEEP_SEED = 42
PRECISIONS = ("bf16", "fp16", "fp32")
AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}

CONFIG = {
    "encoder": ENCODER,
    "max_length": 256,
    "batch_size": 32,
    "epochs": 5,
    "weight_decay": 0.01,
    "warmup": 0.1,
    "clip": 1.0,
    "crf_lr": 1e-3,
}


class SoftmaxTagger(nn.Module):
    """ModernBERT token classifier whose word scores are read at first sub-tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            ENCODER, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
        )

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the cross-entropy over first sub-tokens."""
        return self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss

    def predict(self, batch: dict[str, Tensor]) -> list[list[int]]:
        """Return the raw argmax label of every word, without IOB2 repair."""
        logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        best = word_scores(logits, batch["first"]).argmax(dim=2).cpu().tolist()
        return [row[:n] for row, n in zip(best, batch["word_mask"].sum(dim=1).tolist())]


class CRFTagger(nn.Module):
    """ModernBERT encoder with linear emissions and the shared CRF over first sub-tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER)
        self.emit = nn.Linear(self.encoder.config.hidden_size, len(LABELS))
        self.crf = CRF(*bio_allowed(LABELS))

    def emissions(self, batch: dict[str, Tensor]) -> Tensor:
        """Return float32 word-level emissions [B, W, K]."""
        hidden = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        return word_scores(self.emit(hidden), batch["first"]).float()

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the CRF negative log-likelihood, computed in float32."""
        emissions = self.emissions(batch)
        with torch.autocast(emissions.device.type, enabled=False):
            return self.crf.loss(emissions, batch["tags"], batch["word_mask"])

    def predict(self, batch: dict[str, Tensor]) -> list[list[int]]:
        """Return the Viterbi label path of every sentence."""
        return self.crf.decode(self.emissions(batch), batch["word_mask"])


def word_scores(scores: Tensor, first: Tensor) -> Tensor:
    """Gather sub-token scores [B, S, K] at each word's first sub-token, giving [B, W, K]."""
    return scores.gather(1, first.unsqueeze(2).expand(-1, -1, scores.size(2)))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=("softmax", "crf"), required=True, help="output layer")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42, help="random seed")
    parser.add_argument("--lr", type=float, default=8e-5, help="encoder learning rate")
    parser.add_argument("--precision", choices=PRECISIONS, default="bf16", help="CUDA autocast precision (CPU runs in fp32)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true", help="evaluate all final splits and write results")
    mode.add_argument("--lr-sweep", action="store_true", help=f"compare {LR_GRID} on dev with seed {SWEEP_SEED}")
    return parser.parse_args()


def encode(records: list[dict[str, Any]], tokenizer: Any, split: str) -> list[dict[str, Any]]:
    """Tokenize pre-split words and label only the first sub-token of each word."""
    # A leading space makes every word after the first look like it does in running text.
    words = [[token if i == 0 else f" {token}" for i, token in enumerate(record["tokens"])] for record in records]
    encoding = tokenizer(words, is_split_into_words=True)
    items, too_long = [], 0
    for index, record in enumerate(records):
        word_ids = encoding.word_ids(index)
        too_long += len(word_ids) > CONFIG["max_length"]
        first = [position for position, word in enumerate(word_ids) if word is not None and word_ids[position - 1] != word]
        if len(first) != len(record["tokens"]):
            raise ValueError(f"{record['id']}: {len(first)} first sub-tokens for {len(record['tokens'])} words")
        labels = [-100] * len(word_ids)
        tags = [LABEL2ID[tag] for tag in record["tags"]]
        for position, tag in zip(first, tags):
            labels[position] = tag
        items.append({"input_ids": encoding["input_ids"][index], "labels": labels, "first": first, "tags": tags})
    longest = longest_length(items)
    print(f"{split}: {too_long} of {len(records)} sentences longer than {CONFIG['max_length']} sub-tokens (longest {longest})")
    if too_long:
        raise SystemExit(f"max_length {CONFIG['max_length']} would truncate {too_long} sentences in {split}; {longest} is needed")
    return items


def longest_length(items: list[dict[str, Any]]) -> int:
    """Return the longest sub-token length among encoded sentences."""
    return max(len(item["input_ids"]) for item in items)


def make_collate(pad_id: int):
    """Return a collate function that pads sub-token and word tensors per batch."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        length = max(len(item["input_ids"]) for item in batch)
        n_words = max(len(item["first"]) for item in batch)
        out = {
            "input_ids": torch.full((len(batch), length), pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(len(batch), length, dtype=torch.long),
            "labels": torch.full((len(batch), length), -100, dtype=torch.long),
            "first": torch.zeros(len(batch), n_words, dtype=torch.long),
            "tags": torch.zeros(len(batch), n_words, dtype=torch.long),
            "word_mask": torch.zeros(len(batch), n_words, dtype=torch.bool),
        }
        for i, item in enumerate(batch):
            n, w = len(item["input_ids"]), len(item["first"])
            out["input_ids"][i, :n] = torch.tensor(item["input_ids"])
            out["attention_mask"][i, :n] = 1
            out["labels"][i, :n] = torch.tensor(item["labels"])
            out["first"][i, :w] = torch.tensor(item["first"])
            out["tags"][i, :w] = torch.tensor(item["tags"])
            out["word_mask"][i, :w] = True
        return out

    return collate


def autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """Return the autocast dtype for the requested precision, or None for fp32 and CPU."""
    if device.type != "cuda":
        return None
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("bf16 is not natively supported on this GPU; pass --precision fp16 or --precision fp32")
    return AUTOCAST_DTYPES[precision]


def autocast(device: torch.device, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    """Return the mixed-precision context for the device."""
    return torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()


@torch.no_grad()
def predict(model: nn.Module, data: list[dict[str, Any]], collate: Any, device: torch.device, dtype: torch.dtype | None) -> list[list[str]]:
    """Return predicted label strings for every encoded sentence."""
    model.eval()
    loader = DataLoader(data, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate)
    predictions: list[list[str]] = []
    for batch in loader:
        with autocast(device, dtype):
            rows = model.predict({key: value.to(device) for key, value in batch.items()})
        predictions.extend([LABELS[index] for index in row] for row in rows)
    for item, prediction in zip(data, predictions):
        assert len(prediction) == len(item["tags"]), "prediction length differs from word count"
    return predictions


def build_optimizer(model: nn.Module, lr: float, n_steps: int) -> tuple[torch.optim.Optimizer, Any]:
    """Return AdamW with a separate CRF learning rate and a warmup-then-linear-decay schedule."""
    crf_params = [p for name, p in model.named_parameters() if name.startswith("crf.")]
    other_params = [p for name, p in model.named_parameters() if not name.startswith("crf.")]
    groups = [{"params": other_params, "lr": lr, "weight_decay": CONFIG["weight_decay"]}]
    if crf_params:
        groups.append({"params": crf_params, "lr": CONFIG["crf_lr"], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(CONFIG["warmup"] * n_steps), n_steps)
    return optimizer, scheduler


def train(
    head: str,
    lr: float,
    seed: int,
    train_data: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    dev_data: list[dict[str, Any]],
    collate: Any,
    device: torch.device,
    dtype: torch.dtype | None,
) -> tuple[nn.Module, int, float]:
    """Fine-tune for CONFIG epochs and return the model at its best dev epoch with (epoch, macro-F1)."""
    set_seed(seed)
    # Some encoder kernels have no deterministic variant; repeat runs verify reproducibility instead.
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = (SoftmaxTagger() if head == "softmax" else CRFTagger()).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate, generator=generator)
    optimizer, scheduler = build_optimizer(model, lr, CONFIG["epochs"] * len(loader))
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    dev_gold = [record["tags"] for record in dev_records]
    best_epoch, best_f1, best_state = 0, -1.0, None
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        started, total = time.perf_counter(), 0.0
        for batch in loader:
            optimizer.zero_grad()
            with autocast(device, dtype):
                loss = model.loss({key: value.to(device) for key, value in batch.items()})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        dev = evaluate(dev_gold, predict(model, dev_data, collate, device, dtype))
        print(
            f"epoch {epoch}  loss {total / len(loader):.4f}  dev macro-F1 {dev['macro_f1']:.4f}  "
            f"invalid IOB2 {dev['invalid_iob2_rate']:.4f}  {time.perf_counter() - started:.0f}s",
            flush=True,
        )
        if dev["macro_f1"] > best_f1:
            best_epoch, best_f1 = epoch, dev["macro_f1"]
            best_state = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, best_f1


def lr_sweep(head: str, train_data: list, dev_records: list, dev_data: list, collate: Any, device: torch.device, dtype: torch.dtype | None) -> None:
    """Train once per learning rate in LR_GRID and print the dev table."""
    rows = []
    for lr in LR_GRID:
        print(f"--- lr {lr:g}")
        _, best_epoch, best_f1 = train(head, lr, SWEEP_SEED, train_data, dev_records, dev_data, collate, device, dtype)
        rows.append((lr, best_epoch, best_f1))
    print(f"\nlr sweep ({head} head, seed {SWEEP_SEED}, dev only)\n{'lr':>8}  best epoch  dev macro-F1")
    for lr, best_epoch, best_f1 in rows:
        print(f"{lr:>8g}  {best_epoch:>10d}  {best_f1:.4f}")


def run_final(model: nn.Module, name: str, final_data: dict[str, tuple[list, list]], collate: Any, args: argparse.Namespace, run_info: dict, device: torch.device, dtype: torch.dtype | None) -> None:
    """Predict every final split, write predictions, the run file and the checkpoint."""
    splits: dict[str, Any] = {}
    for split, (records, data) in final_data.items():
        predictions = predict(model, data, collate, device, dtype)
        write_preds(name, args.seed, split, [record["id"] for record in records], predictions)
        splits[split] = evaluate([record["tags"] for record in records], predictions)
        print(f"{split}: macro-F1 {splits[split]['macro_f1']:.4f}")
    write_result(name, args.seed, splits, run_info)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({"config": run_info["config"], "state_dict": model.state_dict()}, CHECKPOINT_DIR / f"{name}_seed{args.seed}.pt")


def main() -> None:
    """Fine-tune on train, report dev, and with --final evaluate and save every final split."""
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = autocast_dtype(args.precision, device)
    precision = args.precision if dtype else "fp32"
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    collate = make_collate(tokenizer.pad_token_id)
    train_records, dev_records = load_split("train"), load_split("dev")
    train_data = encode(train_records, tokenizer, "train")
    dev_data = encode(dev_records, tokenizer, "dev")
    final_data: dict[str, tuple[list, list]] = {"dev": (dev_records, dev_data)}
    if args.final:
        for split in FINAL_SPLITS[1:]:
            records = load_split(split, final=True)
            final_data[split] = (records, encode(records, tokenizer, split))

    if args.lr_sweep:
        print(f"precision {precision}")
        lr_sweep(args.head, train_data, dev_records, dev_data, collate, device, dtype)
        return

    name = "modernbert" if args.head == "softmax" else "modernbert_crf"
    print(f"{name} seed {args.seed} lr {args.lr:g} on {hardware(device)}, precision {precision}")
    started = time.perf_counter()
    model, best_epoch, best_f1 = train(args.head, args.lr, args.seed, train_data, dev_records, dev_data, collate, device, dtype)
    train_time = time.perf_counter() - started
    print(f"best epoch {best_epoch}: dev macro-F1 {best_f1:.4f} ({train_time:.0f}s)")

    if args.final:
        longest = {"train": longest_length(train_data)}
        longest.update({split: longest_length(data) for split, (_, data) in final_data.items()})
        config = {**CONFIG, "head": args.head, "lr": args.lr, "precision": precision, "longest_subtokens": longest}
        run_info = {
            "config": config,
            "train_time_s": round(train_time, 1),
            "hardware": hardware(device),
            "n_params": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
        }
        run_final(model, name, final_data, collate, args, run_info, device, dtype)


if __name__ == "__main__":
    main()
