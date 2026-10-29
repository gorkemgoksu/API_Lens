"""Test APILens evaluation metrics against hand-computed examples."""

import json
import sys
from pathlib import Path

import pytest
from seqeval.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import metrics  # noqa: E402

# Three sentences, one error of each kind (hand-computed expectations in the tests below):
#   A: both entities correct          B: METHOD boundary too short      C: ATTRIBUTE typed FUNCTION
GOLD = [
    ["B-FUNCTION", "I-FUNCTION", "O", "B-CLASS"],
    ["B-METHOD", "I-METHOD", "I-METHOD", "O"],
    ["O", "B-ATTRIBUTE", "O", "O"],
]
PRED = [
    ["B-FUNCTION", "I-FUNCTION", "O", "B-CLASS"],
    ["B-METHOD", "I-METHOD", "O", "O"],
    ["O", "B-FUNCTION", "O", "O"],
]


def test_perfect_prediction():
    result = metrics.evaluate(GOLD, GOLD)
    assert result["macro_f1"] == 1.0 and result["micro_f1"] == 1.0
    assert result["span_f1"] == 1.0 and result["invalid_iob2_rate"] == 0.0


def test_hand_computed_scores():
    r = metrics.evaluate(GOLD, PRED)
    # FUNCTION: P=1/2 R=1/1 F1=2/3; CLASS: 1; METHOD: 0; ATTRIBUTE: 0 (MODULE/EXCEPTION unseen).
    assert r["per_type"]["FUNCTION"]["f1"] == pytest.approx(2 / 3)
    assert r["per_type"]["CLASS"]["f1"] == 1.0
    assert r["per_type"]["METHOD"]["f1"] == 0.0
    assert r["macro_f1"] == pytest.approx((2 / 3 + 1 + 0 + 0) / 4)
    assert r["micro_f1"] == pytest.approx(0.5)            # 2 correct of 4 gold, 4 predicted
    assert r["span_f1"] == pytest.approx(0.75)            # A x2 and C match; B boundary wrong
    assert r["token_accuracy"] == pytest.approx(10 / 12)  # two wrong tokens of twelve


def test_unseen_type_is_excluded_from_macro():
    r = metrics.evaluate(GOLD, PRED)
    assert r["per_type"]["MODULE"]["support"] == 0
    assert r["macro_f1"] > (2 / 3 + 1) / 6  # would be lower if MODULE/EXCEPTION counted as 0


def test_confusion_matrix():
    r = metrics.evaluate(GOLD, PRED)["confusion"]
    cell = lambda g, p: r["matrix"][r["rows"].index(g)][r["columns"].index(p)]  # noqa: E731
    assert cell("FUNCTION", "FUNCTION") == 1 and cell("CLASS", "CLASS") == 1
    assert cell("ATTRIBUTE", "FUNCTION") == 1  # boundary right, type wrong
    assert cell("METHOD", "missed") == 1       # boundary wrong -> missed


def test_strict_and_lenient_disagree():
    gold, pred = [["B-CLASS", "I-CLASS", "O"]], [["B-CLASS", "O", "O"]]
    r = metrics.evaluate(gold, pred)
    # Lenient (token overlap) would give credit for the first token; strict gives none.
    assert r["macro_f1"] == 0.0 and r["span_f1"] == 0.0
    assert r["token_accuracy"] == pytest.approx(2 / 3)


def test_invalid_iob2_rate():
    pred = [["O", "I-METHOD", "O"], ["B-METHOD", "I-FUNCTION", "O"], ["B-CLASS", "I-CLASS", "O"]]
    gold = [["O", "O", "O"]] * 3
    assert metrics.evaluate(gold, pred)["invalid_iob2_rate"] == pytest.approx(2 / 3)


def test_matches_seqeval_micro_f1():
    assert metrics.evaluate(GOLD, PRED)["micro_f1"] == pytest.approx(
        f1_score(GOLD, PRED, mode="strict", scheme=metrics.IOB2))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="tags"):
        metrics.evaluate([["O", "O"]], [["O"]])
    with pytest.raises(ValueError, match="unknown labels"):
        metrics.evaluate([["O"]], [["B-BANANA"]])


def test_writers_follow_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "RESULTS_DIR", tmp_path)
    path = metrics.write_preds("hmm", 0, "dev", ["a", "b"], PRED[:2])
    assert path.name == "hmm_seed0_dev.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0] == {"id": "a", "pred": PRED[0]}
    run = metrics.write_result("hmm", 0, {"dev": metrics.evaluate(GOLD, PRED)}, {"n_params": 5})
    data = json.loads(run.read_text())
    assert run.name == "hmm_seed0.json" and data["n_params"] == 5 and data["hardware"] is None
    assert set(data) == {"model", "seed", *metrics.RUN_FILE_KEYS, "splits"}
    with pytest.raises(ValueError, match="unknown keys"):
        metrics.write_result("hmm", 0, {}, {"lr": 1})
    with pytest.raises(ValueError, match="ids"):
        metrics.write_preds("hmm", 0, "dev", ["a"], PRED)


RECORDS = [
    {"id": "s0", "library": "numpy", "source_type": "guide", "tokens": ["a"] * 4,
     "spans": [{"start": 0, "end": 2, "type": "FUNCTION", "target": "f"},
               {"start": 3, "end": 4, "type": "CLASS", "target": "C"}]},
    {"id": "s1", "library": "pandas", "source_type": "reference", "tokens": ["a"] * 4,
     "spans": [{"start": 0, "end": 3, "type": "METHOD", "target": "m"}]},
    {"id": "s2", "library": "pandas", "source_type": "guide", "tokens": ["a"] * 4,
     "spans": [{"start": 1, "end": 2, "type": "ATTRIBUTE", "target": "a"}]},
]


def test_slices_by_library_and_target():
    s = metrics.slices(RECORDS, GOLD, PRED, train_targets={"f", "m"})
    assert s["by_library"]["numpy"]["macro_f1"] == 1.0   # sentence A only
    assert s["by_library"]["pandas"]["macro_f1"] == 0.0  # sentences B and C both wrong
    assert set(s["by_source_type"]) == {"guide", "reference"}
    # seen targets f, m: f recalled, m not -> 1/2; unseen C recalled, a not -> 1/2
    assert s["target_recall"]["seen"] == {"recall": 0.5, "support": 2}
    assert s["target_recall"]["unseen"] == {"recall": 0.5, "support": 2}


def test_bootstrap_ci_perfect_and_deterministic():
    assert metrics.bootstrap_ci(GOLD, GOLD, n=50)["macro_f1"] == [1.0, 1.0]
    a, b = metrics.bootstrap_ci(GOLD, PRED, n=200), metrics.bootstrap_ci(GOLD, PRED, n=200)
    assert a == b
    low, high = a["micro_f1"]
    assert 0.0 <= low <= 0.5 <= high <= 1.0  # point estimate (0.5) lies inside the interval


def test_bootstrap_uses_summed_counts():
    # One resample made only of sentence A has F1 1.0; the full-data micro F1 must be 0.5.
    counts = metrics._sentence_counts(GOLD, PRED)
    tp, n_pred, n_gold = counts.sum(axis=0)
    assert metrics._f1_from_counts(tp, n_pred, n_gold)[0] == pytest.approx(0.5)


def test_validate_preds_reorders_and_rejects():
    rows = [{"id": r["id"], "pred": p} for r, p in zip(RECORDS, PRED)]
    assert metrics.validate_preds(RECORDS, rows[::-1]) == PRED
    with pytest.raises(ValueError, match="no prediction"):
        metrics.validate_preds(RECORDS, rows[:2])
    with pytest.raises(ValueError, match="not in the split"):
        metrics.validate_preds(RECORDS, rows + [{"id": "zzz", "pred": []}])
    with pytest.raises(ValueError, match="duplicate"):
        metrics.validate_preds(RECORDS, rows + rows[:1])
    with pytest.raises(ValueError, match="tags for"):
        metrics.validate_preds(RECORDS, [{"id": "s0", "pred": ["O"]}, *rows[1:]])
    with pytest.raises(ValueError, match="unknown labels"):
        metrics.validate_preds(RECORDS, [{"id": "s0", "pred": ["X"] * 4}, *rows[1:]])


def test_bootstrap_mean_ci_matches_single_seed():
    assert metrics.bootstrap_mean_ci(GOLD, [PRED], n=100) == metrics.bootstrap_ci(GOLD, PRED, n=100)
    both = metrics.bootstrap_mean_ci(GOLD, [PRED, GOLD], n=100)
    assert both["micro_f1"][0] > metrics.bootstrap_ci(GOLD, PRED, n=100)["micro_f1"][0]


def test_benchmark_assert_same_detects_tampering():
    import build_benchmark as bench
    fresh = metrics.evaluate(GOLD, PRED)
    stored = json.loads(json.dumps(fresh))
    bench.assert_same(stored, fresh, "ok")  # identical numbers pass
    stored["per_type"]["CLASS"]["f1"] = 0.5
    with pytest.raises(ValueError, match="per_type/CLASS/f1"):
        bench.assert_same(stored, fresh, "run")
