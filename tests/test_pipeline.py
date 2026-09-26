import json
import sqlite3
from pathlib import Path
import numpy as np
import pytest
from ml2.index import CandidateIndex
from ml2.model import Ensemble
from ml2.pipeline import predict, train, validate_outputs
from scripts.make_fixture import make_fixture

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    data = make_fixture(root / "data", 240)
    index = root / "targets.sqlite"
    idx = CandidateIndex(index)
    idx.build([data / "train_source2.tsv", data / "train_source3.tsv"])
    idx.close()
    model, results, blocks = train(index, data / "train_source1.tsv", data / "train_ground_truth.tsv", root / "run",
                                   max_queries=240, max_pairs=20000, k=35, max_postings=1000, rounds=35, threads=2)
    return root, data, index, model, results, blocks


def test_real_models_train_and_reload(trained):
    root, _, _, model, results, blocks = trained
    loaded = Ensemble.load(root / "run/model")
    for b in blocks[:5]:
        for mode in ("pair", "care"):
            assert np.allclose(model.score_query(b.x, mode), loaded.score_query(b.x, mode), atol=1e-8)
    assert set(results["policies"]) == {"pair", "care"}
    assert sum(results["partition_query_counts"].values()) == 240


def test_saved_thresholds_are_used_and_batch_independent(trained):
    root, data, idx, _, _, _ = trained
    a, b = root / "prediction_a", root / "prediction_b"
    sa = predict(idx, data / "test_source1.tsv", root / "run/model", a, "care", batch_pairs=100)
    sb = predict(idx, data / "test_source1.tsv", root / "run/model", b, "care", batch_pairs=1000)
    assert sa["queries"] == 240
    assert sa["threshold"] == json.loads((root / "run/model/manifest.json").read_text())["thresholds"]["care"]
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    db = sqlite3.connect(a / "scored_pairs.sqlite")
    assert sa["pairs_scored"] == db.execute("select count(*) from scored").fetchone()[0]
    db.close()


def test_incompatible_artifact_rejected(trained, tmp_path):
    root = trained[0]
    manifest = json.loads((root / "run/model/manifest.json").read_text())
    manifest["schema_hash"] = "old-55-feature-model"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="feature schema"):
        Ensemble.load(tmp_path)


def test_candidate_subset_violation_is_hard_failure(trained, tmp_path):
    _, data, idxpath, _, _, _ = trained
    rows = (data / "test_source1.tsv").read_text().splitlines()[1:]
    qs = [line.split("\t")[0] for line in rows]
    (tmp_path / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" + "".join(q + "\tS2-p000001\n" for q in qs))
    (tmp_path / "candidate_pairs.tsv").write_text("source1_entity_id\tcandidate_entity_ids\n" + "".join(q + "\t\n" for q in qs))
    idx = CandidateIndex(idxpath)
    with pytest.raises(ValueError, match="never scored"):
        validate_outputs(tmp_path, data / "test_source1.tsv", idx)
    idx.close()


def test_prediction_preserves_zero_candidate_query(trained, tmp_path):
    from scripts.make_fixture import write_tsv
    root, _, idxpath, _, _, _ = trained
    q = tmp_path / "queries.tsv"
    write_tsv(q, ["entity_id", "business_name", "business_address", "country"], [["S1-empty", "", "", "UnseenCountry"]])
    out = tmp_path / "prediction"
    result = predict(idxpath, q, root / "run/model", out, "pair")
    assert result["queries"] == 1 and result["pairs_scored"] == 0
    assert (out / "matching_results.tsv").read_text().endswith("S1-empty\t\n")
    assert (out / "candidate_pairs.tsv").read_text().endswith("S1-empty\t\n")


def test_pair_memory_guard_fails_before_fit(trained, tmp_path):
    _, data, idxpath, _, _, _ = trained
    with pytest.raises(MemoryError, match="Pair limit"):
        train(idxpath, data / "train_source1.tsv", data / "train_ground_truth.tsv", tmp_path / "run",
              max_queries=240, max_pairs=1, k=35)
