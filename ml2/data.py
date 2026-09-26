"""Strict streaming input, stable sampling and positive-component split isolation."""
from __future__ import annotations
import csv
import hashlib
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

FIELDS = ("entity_id", "business_name", "business_address", "country")

@dataclass(frozen=True, slots=True)
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: str, seed: int = 26) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "big")


def read_records(path: str | Path, prefix: str | None = None) -> Iterator[Record]:
    """Preserve quoted tabs and trailing empty fields; do not silently drop bad rows."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {FIELDS}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate header columns")
        for line, row in enumerate(reader, 2):
            if None in row or any(row[k] is None for k in FIELDS):
                raise ValueError(f"{path}:{line}: malformed TSV row")
            rid = row["entity_id"].strip()
            if not rid or any(c in rid for c in ",\t\r\n"):
                raise ValueError(f"{path}:{line}: invalid entity ID")
            if prefix and not rid.startswith(prefix + "-"):
                raise ValueError(f"{path}:{line}: expected {prefix}- ID, got {rid!r}")
            yield Record(rid, row["business_name"], row["business_address"], row["country"].strip())


def sample_queries(path: str | Path, limit: int, seed: int) -> list[Record]:
    """Bottom-hash sample over the entire file, bounded by limit (never prefix sampling).

    Whole-file duplicate detection is performed by `validate_source_ids`, not an
    unbounded Python ID set in this streaming sampler.
    """
    if limit < 1:
        raise ValueError("query limit must be positive")
    heap: list[tuple[int, str, Record]] = []
    for rec in read_records(path, "S1"):
        item = (-stable_hash(rec.entity_id, seed), rec.entity_id, rec)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    result = sorted((item[2] for item in heap), key=lambda r: r.entity_id)
    if len({r.entity_id for r in result}) != len(result):
        raise ValueError("Duplicate sampled query ID")
    return result


def validate_source_ids(path: str | Path, sqlite_connection, prefix: str = "S1") -> int:
    sqlite_connection.execute("CREATE TEMP TABLE IF NOT EXISTS input_ids(id TEXT PRIMARY KEY)")
    sqlite_connection.execute("DELETE FROM input_ids")
    n = 0
    for rec in read_records(path, prefix):
        sqlite_connection.execute("INSERT INTO input_ids VALUES(?)", (rec.entity_id,))
        n += 1
    sqlite_connection.commit()
    return n


def read_ground_truth(path: str | Path, keep: set[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing ground-truth columns")
        for row in reader:
            if None in row or any(row[k] is None for k in required):
                raise ValueError("Malformed ground-truth row")
            qid = row["source1_entity_id"].strip()
            if qid not in keep:
                continue
            if qid in result:
                raise ValueError(f"Duplicate GT row: {qid}")
            values = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate target in GT row: {qid}")
            if any(not t.startswith(("S2-", "S3-")) for t in values):
                raise ValueError(f"Invalid GT target: {qid}")
            result[qid] = set(values)
    if result.keys() != keep:
        raise ValueError(f"Missing GT for {len(keep - result.keys())} sampled queries")
    return result


def component_split(gt: dict[str, set[str]], seed: int = 26) -> dict[str, str]:
    """Put queries sharing a TRUE target in one partition; no negative-edge union.

    Stable hashing is not stratification. The caller reports country/singleton
    balance and refuses single-class model/calibration partitions.
    """
    parent = {q: q for q in gt}
    def root(q: str) -> str:
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q
    owners: dict[str, str] = {}
    for q in sorted(gt):
        for t in sorted(gt[q]):
            if t in owners:
                a, b = root(q), root(owners[t])
                parent[max(a, b)] = min(a, b)
            else:
                owners[t] = q
    names = ["fit"] * 6 + ["early", "calibration", "tune", "holdout"]
    return {q: names[stable_hash(root(q), seed) % 10] for q in gt}


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
