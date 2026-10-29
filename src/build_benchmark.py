"""Validate every run in results/ and build benchmark.csv, ablations.csv and the figures.

Usage: python src/build_benchmark.py --final
Without --final only the dev split is checked and nothing is written. With --final every stored
number is recomputed from results/preds/ (a mismatch stops the script), then the CSVs and the SVG
figures in results/figures/ are written. Exit status 1 means some expected run is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from data_io import ENTITY_TYPES, load_split  # noqa: E402

RESULTS = metrics.RESULTS_DIR
FIGURES = RESULTS / "figures"
SPLITS = ("dev", "test", "heldout", "test_noparen")
SEEDS = (13, 42, 2026)
MODELS = {"rules": (0,), "hmm": (0,), "crf": (0,),
          "bilstm_crf": SEEDS, "modernbert": SEEDS, "modernbert_crf": SEEDS}
ABLATIONS = {"hmm-nomask": ("hmm", 0), "crf-nocontext": ("crf", 0),
             "bilstm_crf-nochar": ("bilstm_crf", 42), "bilstm-softmax": ("bilstm_crf", 42)}
NAMES = {"rules": "Rules", "hmm": "HMM", "crf": "CRF", "bilstm_crf": "BiLSTM-CRF",
         "modernbert": "ModernBERT", "modernbert_crf": "ModernBERT-CRF"}
# Categorical slots 1-6 of the validated palette, in fixed order (one colour per model).
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # sequential ramp, light -> dark
TOLERANCE = 1e-9
CSV_DIGITS = 4


def load_run(model: str, seed: int) -> dict[str, Any] | None:
    """Return the stored run file, or None if it does not exist."""
    path = RESULTS / f"{model}_seed{seed}.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_preds(model: str, seed: int, split: str, records: list[dict]) -> list[list[str]]:
    """Read one predictions file and check it against the split (ids, lengths, labels)."""
    path = RESULTS / "preds" / f"{model}_seed{seed}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing predictions: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return metrics.validate_preds(records, rows)


def assert_same(stored: Any, fresh: Any, where: str) -> None:
    """Fail loudly if a stored number differs from the recomputed one."""
    if isinstance(fresh, dict):
        for key, value in fresh.items():
            if key not in stored:
                raise ValueError(f"{where}: stored result lacks '{key}'")
            assert_same(stored[key], value, f"{where}/{key}")
    elif isinstance(fresh, list):
        if len(stored) != len(fresh):
            raise ValueError(f"{where}: length {len(stored)} stored vs {len(fresh)} recomputed")
        for i, (a, b) in enumerate(zip(stored, fresh)):
            assert_same(a, b, f"{where}[{i}]")
    elif isinstance(fresh, (int, float)):
        if abs(stored - fresh) > TOLERANCE:
            raise ValueError(f"{where}: stored {stored} but recomputed {fresh}")
    elif stored != fresh:
        raise ValueError(f"{where}: stored {stored!r} but recomputed {fresh!r}")


def collect(splits: tuple[str, ...], final: bool) -> tuple[dict, list[str]]:
    """Load, verify and score every expected run; return ({(model, seed): data}, missing)."""
    records = {s: load_split(s, final=final) for s in splits}
    gold = {s: [r["tags"] for r in records[s]] for s in splits}
    train_targets = {span["target"] for r in load_split("train") for span in r["spans"]}
    expected = [(m, seed) for m, seeds in MODELS.items() for seed in seeds]
    expected += [(name, seed) for name, (_, seed) in ABLATIONS.items()]
    runs, missing = {}, []
    for model, seed in expected:
        run = load_run(model, seed)
        if run is None:
            missing.append(f"{model}_seed{seed}.json")
            continue
        absent = [s for s in splits if s not in run["splits"]]
        if absent:
            raise ValueError(f"{model} seed {seed}: run file has no {absent}")
        data = {"run": run, "preds": {}, "eval": {}, "unseen": {}}
        for split in splits:
            pred = load_preds(model, seed, split, records[split])
            fresh = metrics.evaluate(gold[split], pred)
            assert_same(run["splits"][split], fresh, f"{model} seed {seed} {split}")
            data["preds"][split], data["eval"][split] = pred, fresh
            recall = metrics.slices(records[split], gold[split], pred, train_targets)
            data["unseen"][split] = recall["target_recall"]["unseen"]["recall"]
        runs[(model, seed)] = data
    return {"runs": runs, "gold": gold}, missing


def aggregate(model: str, split: str, bundle: dict) -> dict[str, Any] | None:
    """Mean and population std over seeds, bootstrap CI of the seed mean, other averages."""
    seeds = [s for s in MODELS[model] if (model, s) in bundle["runs"]]
    if not seeds:
        return None
    runs = [bundle["runs"][(model, s)] for s in seeds]
    evals = [r["eval"][split] for r in runs]
    ci = metrics.bootstrap_mean_ci(bundle["gold"][split], [r["preds"][split] for r in runs])
    macro = [e["macro_f1"] for e in evals]
    mean = lambda values: float(np.mean(values))  # noqa: E731
    return {
        "model": model, "split": split, "n_seeds": len(seeds),
        "macro_f1": mean(macro), "macro_f1_std": float(np.std(macro)),
        "macro_f1_ci_low": ci["macro_f1"][0], "macro_f1_ci_high": ci["macro_f1"][1],
        "micro_f1": mean([e["micro_f1"] for e in evals]),
        "micro_f1_ci_low": ci["micro_f1"][0], "micro_f1_ci_high": ci["micro_f1"][1],
        "span_f1": mean([e["span_f1"] for e in evals]),
        "unseen_target_recall": mean([r["unseen"][split] for r in runs]),
        "invalid_iob2_rate": mean([e["invalid_iob2_rate"] for e in evals]),
        "train_time_s": mean([r["run"]["train_time_s"] or 0.0 for r in runs]),
        "per_type_f1": {t: mean([e["per_type"][t]["f1"] for e in evals]) for t in ENTITY_TYPES},
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows with fixed columns; floats are rounded to CSV_DIGITS."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, CSV_DIGITS) if isinstance(v, float) else v
                             for k, v in row.items()})


def ablation_rows(bundle: dict) -> list[dict[str, Any]]:
    """Each ablation run next to the full model run with the same seed."""
    rows = []
    for name, (base, seed) in ABLATIONS.items():
        for variant, role in ((base, "full"), (name, "ablation")):
            data = bundle["runs"].get((variant, seed))
            for split in SPLITS if data else ():
                e = data["eval"][split]
                rows.append({"ablation": name, "role": role, "model": variant, "seed": seed,
                             "split": split, "macro_f1": e["macro_f1"], "micro_f1": e["micro_f1"],
                             "span_f1": e["span_f1"], "unseen_target_recall": data["unseen"][split],
                             "invalid_iob2_rate": e["invalid_iob2_rate"]})
    return rows


def _setup_figures():
    """Import matplotlib with deterministic SVG output and quiet axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "apilens",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.titleweight": "bold",
                         "axes.titlesize": 10, "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.facecolor": "white"})
    return plt


def _save(plt, fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)


def _bars(ax, agg: dict[str, dict], key: str, ci: bool = False) -> None:
    """One bar per model (own colour), value label above each bar, y axis fixed to 0-1."""
    names = [m for m in MODELS if m in agg]
    values = [agg[m][key] for m in names]
    ax.bar(range(len(names)), values, color=[COLORS[m] for m in names], width=0.65)
    if ci:
        low = [v - agg[m][f"{key}_ci_low"] for m, v in zip(names, values)]
        high = [agg[m][f"{key}_ci_high"] - v for m, v in zip(names, values)]
        ax.errorbar(range(len(names)), values, yerr=[low, high], fmt="none", ecolor="#0b0b0b",
                    elinewidth=1, capsize=3)
    for x, m, v in zip(range(len(names)), names, values):
        top = agg[m][f"{key}_ci_high"] if ci else v
        ax.text(x, top + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)), [NAMES[m] for m in names], rotation=35, ha="right")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def figure_ladder(agg: dict[str, dict[str, dict]]) -> None:
    """Macro-F1 per model on test with the 95% bootstrap interval."""
    plt = _setup_figures()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _bars(ax, {m: a["test"] for m, a in agg.items()}, "macro_f1", ci=True)
    ax.set_ylabel("Strict macro-F1 (test)")
    ax.set_title("Model ladder on the test split (bars: seed mean, whiskers: 95% CI)", loc="left")
    _save(plt, fig, "ladder_macro_f1.svg")


def figure_regimes(agg: dict[str, dict[str, dict]]) -> None:
    """Small multiples: the four test regimes, one panel each."""
    plt = _setup_figures()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    panels = [("test", "macro_f1", "Same domain: test macro-F1"),
              ("test", "unseen_target_recall", "Unseen APIs: test recall"),
              ("heldout", "macro_f1", "Unseen library: PyTorch macro-F1"),
              ("test_noparen", "macro_f1", "No '()' cue: macro-F1")]
    for ax, (split, key, title) in zip(axes, panels):
        _bars(ax, {m: a[split] for m, a in agg.items()}, key)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Score")
    _save(plt, fig, "regimes.svg")


def _heatmap(ax, matrix: np.ndarray, rows: list[str], cols: list[str], fmt: str, vmax: float,
             gamma: float = 1.0):
    """Annotated single-hue heatmap; gamma < 1 lifts small cells; text flips on dark cells."""
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("blues", BLUES),
              norm=PowerNorm(gamma, vmin=0, vmax=vmax))
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                color="white" if (v / vmax) ** gamma > 0.55 else "#0b0b0b")
    for side in ax.spines.values():
        side.set_visible(False)


def figure_per_type(agg: dict[str, dict[str, dict]]) -> None:
    """Heatmap of test F1 per model and entity type."""
    plt = _setup_figures()
    names = [m for m in MODELS if m in agg]
    matrix = np.array([[agg[m]["test"]["per_type_f1"][t] for t in ENTITY_TYPES] for m in names])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _heatmap(ax, matrix, [NAMES[m] for m in names], list(ENTITY_TYPES), ".2f", 1.0)
    ax.set_title("Strict F1 per entity type (test, seed mean)", loc="left")
    _save(plt, fig, "per_type_f1.svg")


def figure_confusion(best: str, bundle: dict) -> None:
    """Type confusion of the best model (seed 42, test): gold type vs. predicted type."""
    data = bundle["runs"].get((best, 42))
    if data is None:
        return
    plt = _setup_figures()
    conf = data["eval"]["test"]["confusion"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    matrix = np.array(conf["matrix"], dtype=float)
    _heatmap(ax, matrix, conf["rows"], conf["columns"], ".0f", float(matrix.max()),
             gamma=0.4)  # square-root-like scale so rare confusions stay visible
    ax.set_xlabel("Predicted type (or missed: boundary not found)")
    ax.set_ylabel("Gold type")
    ax.set_title(f"{NAMES[best]}, seed 42: gold spans on test", loc="left")
    _save(plt, fig, "confusion_best.svg")


def print_table(agg: dict[str, dict[str, dict]], splits: tuple[str, ...]) -> None:
    """Console summary: macro-F1 (mean +- std) per model and split."""
    print(f"{'model':<16}" + "".join(f"{s:>20}" for s in splits))
    for model, by_split in agg.items():
        cells = [f"{by_split[s]['macro_f1']:.4f} +- {by_split[s]['macro_f1_std']:.4f}" for s in splits]
        print(f"{NAMES[model]:<16}" + "".join(f"{c:>20}" for c in cells))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", action="store_true",
                        help="read the test splits, verify them and write CSVs and figures")
    return parser.parse_args()


def main() -> int:
    """Verify all runs, print the summary, and (with --final) write the outputs."""
    args = parse_args()
    splits = SPLITS if args.final else ("dev",)
    bundle, missing = collect(splits, args.final)
    agg = {m: {s: a for s in splits if (a := aggregate(m, s, bundle))} for m in MODELS}
    agg = {m: by_split for m, by_split in agg.items() if by_split}
    print(f"Verified {len(bundle['runs'])} runs against results/preds/ on: {', '.join(splits)}")
    print_table(agg, splits)
    if missing:
        print(f"\nINCOMPLETE: {len(missing)} expected run files are missing:")
        print("\n".join(f"  {name}" for name in missing))
    if not args.final:
        print("\nDev only; nothing written (use --final to write benchmark.csv and figures).")
        return 1 if missing else 0
    rows = [agg[m][s] for m in agg for s in SPLITS]
    columns = ["model", "split", "n_seeds", "macro_f1", "macro_f1_std", "macro_f1_ci_low",
               "macro_f1_ci_high", "micro_f1", "micro_f1_ci_low", "micro_f1_ci_high", "span_f1",
               "unseen_target_recall", "invalid_iob2_rate", "train_time_s"]
    write_csv(RESULTS / "benchmark.csv", rows, columns)
    write_csv(RESULTS / "ablations.csv", ablation_rows(bundle),
              ["ablation", "role", "model", "seed", "split", "macro_f1", "micro_f1", "span_f1",
               "unseen_target_recall", "invalid_iob2_rate"])
    figure_ladder(agg)
    figure_regimes(agg)
    figure_per_type(agg)
    best = max(agg, key=lambda m: agg[m]["test"]["macro_f1"])
    figure_confusion(best, bundle)
    print(f"\nWrote benchmark.csv, ablations.csv and 4 figures; best model on test: {NAMES[best]}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
