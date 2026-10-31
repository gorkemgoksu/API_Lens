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


def test_cohen_kappa_hand_computed():
    a, b = list("yynny"), list("ynnny")
    # observed 4/5; expected 0.6*0.4 + 0.4*0.6 = 0.48; kappa = (0.8 - 0.48) / 0.52
    assert metrics.cohen_kappa(a, b) == pytest.approx(0.32 / 0.52)
    assert metrics.cohen_kappa(a, a) == 1.0
    assert metrics.cohen_kappa(list("yyyy"), list("nnnn")) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        metrics.cohen_kappa(["y"], ["y", "n"])


def _synthetic_records():
    types = ["FUNCTION", "METHOD", "CLASS", "MODULE", "ATTRIBUTE", "EXCEPTION"]
    records = []
    for i in range(400):
        spans = []
        if i % 2 == 0:  # half the sentences carry one entity; EXCEPTION is rare
            etype = types[i % 5] if i % 20 else "EXCEPTION"
            spans = [{"start": 0, "end": 1, "type": etype, "target": f"t{i}", "role": "obj"}]
        records.append({"id": f"s{i}", "library": "x", "tokens": ["a", "b"], "spans": spans})
    return records


def test_audit_sample_is_stratified_and_reproducible():
    import build_benchmark as bench
    records = _synthetic_records()
    picked = bench.audit_sample(records)
    assert len(picked) == len(set(picked)) == 150
    for etype in ("FUNCTION", "METHOD", "CLASS", "MODULE", "ATTRIBUTE", "EXCEPTION"):
        assert sum(s["type"] == etype for i in picked for s in records[i]["spans"]) >= 15
    assert picked == bench.audit_sample(records)  # same seed, same sample
    rows = bench.audit_rows(records, picked)
    assert sum(r["second"] for r in rows) == 50
    few = bench.audit_sample(records[:30])  # only 1 EXCEPTION here: it is taken, not skipped
    assert any(s["type"] == "EXCEPTION" for i in few for s in records[i]["spans"])
    with pytest.raises(ValueError, match="no mentions"):
        bench.audit_sample([r for r in records if not r["spans"]])


def test_detokenize_and_report(tmp_path, monkeypatch, capsys):
    import build_benchmark as bench
    assert bench.detokenize(["Use", "f", "(", ")", "and", "(", "x", ")", "."]) == "Use f() and (x)."
    header = ",".join(bench.AUDIT_COLUMNS)
    first = [header] + [f"s{i},lib,t,a|CLASS; b|METHOD,{'y' if i % 4 else 'n'},"
                        f"{'' if i % 4 else 1},{i % 2},ann,,{1 if i < 4 else 0}" for i in range(8)]
    second = [header] + [f"s{i},lib,t,a|CLASS,{'y' if i % 2 else 'n'},{'' if i % 2 else 1},0,b,,1"
                         for i in range(4)]
    (tmp_path / "a.csv").write_text("\n".join(first) + "\n")
    (tmp_path / "b.csv").write_text("\n".join(second) + "\n")
    monkeypatch.setattr(bench, "AUDIT_FILE", tmp_path / "a.csv")
    bench.audit_report(tmp_path / "b.csv")
    out = capsys.readouterr().out
    assert "Wrong-span rate: 2/16" in out          # two sentences with one wrong span each
    assert "Missing mentions per 100 sentences: 50.00" in out
    assert "Cohen's kappa on spans_ok over 4 shared sentences" in out
    (tmp_path / "a.csv").write_text("\n".join(first[:1] + [first[2].replace(",y,", ",?,")]) + "\n")
    with pytest.raises(ValueError, match="spans_ok must be y or n"):
        bench.audit_report(tmp_path / "b.csv")


def test_find_errors_groups_each_kind_once():
    import build_benchmark as bench
    rows = [("B-CLASS I-CLASS I-CLASS", "B-CLASS I-CLASS O"),      # too short
            ("B-CLASS I-CLASS O", "B-CLASS I-CLASS I-CLASS"),      # too long
            ("B-CLASS I-CLASS I-CLASS", "B-CLASS O B-CLASS"),      # split
            ("B-CLASS B-CLASS O", "B-CLASS I-CLASS O"),            # merged
            ("B-CLASS O O", "O O O"),                              # missed
            ("O O O", "B-METHOD O O"),                             # spurious
            ("B-CLASS I-CLASS O", "B-FUNCTION I-FUNCTION O")]      # type
    gold, pred = ([r[i].split() for r in rows] for i in (0, 1))
    events = bench.find_errors(gold, pred)
    assert [e["group"] for e in events] == ["too_short", "too_long", "split", "merged",
                                            "missed", "spurious", "type"]
    assert events[-1]["pair"] == ("CLASS", "FUNCTION")
