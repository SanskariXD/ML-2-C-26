import json
import pytest
from ml2.profile import profile_dataset
from scripts.make_fixture import write_tsv

HEADER = ["entity_id", "business_name", "business_address", "country"]


def test_profile_counts_zero_and_multiple_matches(tmp_path):
    q, t, gt = [tmp_path / name for name in ("q.tsv", "t.tsv", "gt.tsv")]
    write_tsv(q, HEADER, [["S1-a", "राज कार्य", "", "IN"], ["S1-b", "", "10 Main", "US"]])
    write_tsv(t, HEADER, [["S2-a", "राज", "", "IN"], ["S3-b", "Work", "10 Main", "US"]])
    write_tsv(gt, ["source1_entity_id", "matched_entity_ids"],
              [["S1-a", "S2-a,S3-b"], ["S1-b", ""]])
    result = profile_dataset(str(q), [str(t)], str(gt), str(tmp_path / "out"))
    assert result["ground_truth"]["multi_match_queries"] == 1
    assert result["ground_truth"]["unmatched_queries"] == 1
    assert result["average_matches_per_query"] == 1
    assert result["files"][str(q)]["name_scripts"]["devanagari"] == 1
    assert json.loads((tmp_path / "out/data_profile.json").read_text()) == result


def test_profile_rejects_unknown_truth_target(tmp_path):
    q, t, gt = [tmp_path / name for name in ("q.tsv", "t.tsv", "gt.tsv")]
    write_tsv(q, HEADER, [["S1-a", "A", "", "US"]])
    write_tsv(t, HEADER, [["S2-a", "A", "", "US"]])
    write_tsv(gt, ["source1_entity_id", "matched_entity_ids"], [["S1-a", "S3-missing"]])
    with pytest.raises(ValueError, match="unknown target"):
        profile_dataset(str(q), [str(t)], str(gt), str(tmp_path / "out"))
