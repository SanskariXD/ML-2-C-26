import csv
import json
import sqlite3
from pathlib import Path
import numpy as np
import pytest
from ml2.data import Record, component_split, read_records, sample_queries, validate_source_ids
from ml2.features import FEATURE_NAMES, feature_matrix, context_features, pair_features
from ml2.index import Candidate, CandidateIndex
from ml2.metrics import macro_f05, query_f05, report, resolve, threshold_sweep
from ml2.text import fold_latin, normalize
from scripts.make_fixture import write_tsv

HEADER = ["entity_id", "business_name", "business_address", "country"]

def test_native_script_marks_preserved():
    assert normalize("राज कार्य") == "राज कार्य"
    assert fold_latin(normalize("Café Étoile राज")) == "cafe etoile राज"


def test_empty_fields_never_exact_match():
    r = Record("S1-e", "", "", "US")
    c = Candidate(Record("S2-e", "", "", "US"), 0, frozenset())
    x = pair_features(r, c)
    values = dict(zip(FEATURE_NAMES, x))
    assert values["name_exact"] == values["address_exact"] == values["name_ratio"] == 0
    assert values["name_missing"] == values["address_missing"] == 1
    assert np.isfinite(x).all()


def test_no_feature_padding():
    assert len(FEATURE_NAMES) == 34
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert not any("pad" in name for name in FEATURE_NAMES)


def test_tsv_quoted_tabs_and_trailing_empty(tmp_path):
    path = tmp_path / "q.tsv"
    write_tsv(path, HEADER, [["S1-1", "name\twith tab", "", ""]])
    r = list(read_records(path, "S1"))[0]
    assert r.business_name == "name\twith tab" and r.country == r.business_address == ""


def test_malformed_tsv_fails(tmp_path):
    path = tmp_path / "q.tsv"
    path.write_text("\t".join(HEADER) + "\nS1-1\tname\taddress\n")
    with pytest.raises(ValueError):
        list(read_records(path))


def test_duplicate_queries_fails(tmp_path):
    path = tmp_path / "q.tsv"
    write_tsv(path, HEADER, [["S1-1", "a", "b", "US"]] * 2)
    db = sqlite3.connect(":memory:")
    with pytest.raises(sqlite3.IntegrityError):
        validate_source_ids(path, db)
    db.close()


def test_hash_sample_not_order_dependent(tmp_path):
    rows = [[f"S1-{i}", "name", "", "US"] for i in range(100)]
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    write_tsv(a, HEADER, rows)
    write_tsv(b, HEADER, list(reversed(rows)))
    assert sample_queries(a, 10, 26) == sample_queries(b, 10, 26)


def test_shared_positive_components():
    gt = {"a": {"S2-x"}, "b": {"S2-x", "S3-y"}, "c": {"S3-y"}, "d": set()}
    split = component_split(gt)
    assert split["a"] == split["b"] == split["c"]


def test_singleton_metric():
    assert query_f05(set(), set()) == 1
    assert query_f05(set(), {"x"}) == 0
    assert query_f05({"x"}, set()) == 0
    assert query_f05({"x", "y"}, {"x"}) == pytest.approx(1.25 / 1.5)


def test_metric_requires_all_queries():
    with pytest.raises(ValueError):
        macro_f05({"a": set(), "b": {"x"}}, {"b": {"x"}})


@pytest.mark.parametrize("seed", range(12))
def test_threshold_sweep_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    gt = {f"q{i}": ({f"t{i}", f"missing{i}"} if i % 3 else set()) for i in range(12)}
    pairs = [(q, t, float(rng.choice([0., .1, .5, .9, 1.]))) for i, q in enumerate(gt) for t in [f"t{i}", f"n{i}"]]
    threshold, score = threshold_sweep(gt, pairs)
    values = {np.nextafter(1., 2.)} | {p for _, _, p in pairs}
    brute = max(macro_f05(gt, resolve(pairs, t, gt)) for t in values)
    assert score == pytest.approx(brute)
    assert macro_f05(gt, resolve(pairs, threshold, gt)) == pytest.approx(brute)


def test_more_than_ten_matches_are_preserved():
    pairs = [("q", f"t{i}", .99) for i in range(15)]
    assert len(resolve(pairs, .8, ["q"])["q"]) == 15


def test_optional_ownership_tie_is_deterministic():
    pairs = [("b", "t", .9), ("a", "t", .9)]
    assert resolve(pairs, .8, ["a", "b"], True) == {"a": {"t"}, "b": set()}
    assert resolve(pairs, .8, ["a", "b"]) == {"a": {"t"}, "b": {"t"}}


def test_retrieval_misses_not_hidden_by_metric():
    truth = {"q": {"t1", "t2"}, "empty": set()}
    candidates = {"q": {"t1"}, "empty": set()}
    result = report(truth, candidates, candidates)
    assert result["retrieved_positive_fraction"] == .5
    assert result["all_positives_retrieved_fraction"] == 0
    assert result["macro_f05"] < 1


def test_new_country_and_no_label_injection(tmp_path):
    p = tmp_path / "t.tsv"
    write_tsv(p, HEADER, [["S2-f", "Café Étoile", "10 Rue Neuve 75001", "France"], ["S3-x", "Unrelated", "44 Maple", "US"]])
    idx = CandidateIndex(tmp_path / "i.sqlite")
    idx.build([p])
    idx.assert_usable()
    q = Record("S1-f", "Cafe Etoile", "10 rue neuve 75001", "France")
    assert [c.record.entity_id for c in idx.retrieve(q)] == ["S2-f"]
    assert idx.retrieve(Record("S1-z", "", "", "France")) == []
    idx.close()


def test_probe_index_is_not_final_index(tmp_path):
    p = tmp_path / "t.tsv"
    write_tsv(p, HEADER, [["S2-f", "name", "address", "US"]])
    idx = CandidateIndex(tmp_path / "i.sqlite")
    idx.build([p], limit_per_file=1)
    with pytest.raises(ValueError, match="Sample/probe"):
        idx.assert_usable()
    idx.close()


def test_context_features_respond_to_competition_density():
    r = Record("S1-a", "Aster", "1 street", "US")
    c = Candidate(Record("S2-a", "Aster", "1 street", "US"), 3., frozenset({"name", "address"}))
    x = feature_matrix(r, [c])
    probs = np.array([[.8, .9, .85]])
    a = context_features(x, probs, np.array([.4, .35, .25]))
    b = context_features(np.repeat(x, 5, axis=0), np.repeat(probs, 5, axis=0), np.array([.4, .35, .25]))
    assert a[0, 4] != b[0, 4]
