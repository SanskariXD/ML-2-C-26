"""Streaming, label-only dataset checks before building a full target index."""
from __future__ import annotations

import csv
import json
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

from .data import digest_file, read_records, write_json


def script_of(value: str) -> str:
    letters = [unicodedata.name(c, "") for c in value if c.isalpha()]
    for script in ("DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM",
                   "ARABIC", "CYRILLIC", "HAN", "HIRAGANA", "KATAKANA", "HANGUL", "HEBREW"):
        if any(script in name for name in letters):
            return script.lower()
    return "latin" if letters else "missing"


def profile_dataset(queries: str, targets: list[str], truth: str, out: str) -> dict:
    """Count every row and check IDs against GT with disk-backed membership.

    The result contains aggregates and file hashes, never raw entity names or labels.
    No sampling, trained statistics, or model scores are involved.
    """
    outpath = Path(out)
    outpath.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(outpath / "profile_ids.sqlite"))
    db.execute("CREATE TABLE ids(id TEXT PRIMARY KEY, kind TEXT NOT NULL)")
    summaries: dict[str, dict] = {}
    try:
        for kind, paths in (("query", [queries]), ("target", targets)):
            for path in paths:
                counts = Counter()
                countries = Counter()
                scripts = Counter()
                for record in read_records(path, "S1" if kind == "query" else None):
                    if kind == "target" and not record.entity_id.startswith(("S2-", "S3-")):
                        raise ValueError(f"{path}: unexpected target ID {record.entity_id}")
                    db.execute("INSERT INTO ids VALUES(?,?)", (record.entity_id, kind))
                    counts["rows"] += 1
                    counts["missing_name"] += not record.business_name.strip()
                    counts["missing_address"] += not record.business_address.strip()
                    counts["missing_country"] += not record.country
                    counts["numeric_address"] += any(c.isdigit() for c in record.business_address)
                    counts["both_fields_missing"] += not (record.business_name.strip() or record.business_address.strip())
                    countries[record.country or "<missing>"] += 1
                    scripts[script_of(record.business_name)] += 1
                    if counts["rows"] % 100000 == 0:
                        db.commit()
                db.commit()
                summaries[str(path)] = {
                    "kind": kind, "sha256": digest_file(path), "bytes": Path(path).stat().st_size,
                    "counts": dict(counts), "countries": dict(countries), "name_scripts": dict(scripts),
                }
        gt_stats = Counter()
        with open(truth, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None or not {"source1_entity_id", "matched_entity_ids"} <= set(reader.fieldnames):
                raise ValueError(f"{truth}: missing ground-truth columns")
            db.execute("CREATE TABLE seen_gt(id TEXT PRIMARY KEY)")
            for row in reader:
                if None in row or row["source1_entity_id"] is None or row["matched_entity_ids"] is None:
                    raise ValueError("Malformed ground-truth row")
                qid = row["source1_entity_id"].strip()
                if db.execute("SELECT kind FROM ids WHERE id=?", (qid,)).fetchone() != ("query",):
                    raise ValueError(f"Ground truth refers to unknown query: {qid}")
                db.execute("INSERT INTO seen_gt VALUES(?)", (qid,))
                target_ids = [t.strip() for t in row["matched_entity_ids"].split(",") if t.strip()]
                if len(target_ids) != len(set(target_ids)):
                    raise ValueError(f"Duplicate ground-truth target for {qid}")
                for tid in target_ids:
                    if db.execute("SELECT kind FROM ids WHERE id=?", (tid,)).fetchone() != ("target",):
                        raise ValueError(f"Ground truth refers to unknown target: {tid}")
                gt_stats["rows"] += 1
                gt_stats["matched_pairs"] += len(target_ids)
                gt_stats["unmatched_queries"] += not target_ids
                gt_stats["multi_match_queries"] += len(target_ids) > 1
                gt_stats["max_matches_per_query"] = max(gt_stats["max_matches_per_query"], len(target_ids))
                if gt_stats["rows"] % 100000 == 0:
                    db.commit()
        query_count = sum(s["counts"]["rows"] for s in summaries.values() if s["kind"] == "query")
        if gt_stats["rows"] != query_count:
            raise ValueError(f"Ground truth has {gt_stats['rows']} queries, expected {query_count}")
        result = {"evidence_kind": "real_data_profile" if query_count else "empty_data_profile",
                  "files": summaries, "ground_truth": {"sha256": digest_file(truth),
                      "bytes": Path(truth).stat().st_size, **gt_stats},
                  "average_matches_per_query": gt_stats["matched_pairs"] / query_count if query_count else None}
        write_json(outpath / "data_profile.json", result)
        lines = ["# Dataset profile", "", "Computed from supplied local TSVs; no business records are saved here.", "",
                 f"Queries: {query_count:,}; target records: {sum(s['counts']['rows'] for s in summaries.values() if s['kind'] == 'target'):,}.",
                 f"Unmatched queries: {gt_stats['unmatched_queries']:,}; multiple matches: {gt_stats['multi_match_queries']:,}.",
                 f"Average matches/query: {result['average_matches_per_query']:.3f}.", "",
                 "Per-file missingness, country/script counts, file sizes and hashes are in data_profile.json.",
                 "Postal availability needs a country-aware address parser; numeric-address counts are a rough proxy, not postal-code coverage.",
                 ""]
        (outpath / "data_profile.md").write_text("\n".join(lines), encoding="utf-8")
        return result
    finally:
        db.close()
