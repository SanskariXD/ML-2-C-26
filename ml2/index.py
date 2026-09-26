"""SQLite-backed multi-channel candidate retrieval with explicit bounded expansion.

This is a correctness-first index, not a proven multi-million-record speed claim.
Run probe-index before building a full index: postings can be large on disk.
"""
from __future__ import annotations
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from .data import Record, digest_file, read_records
from .text import CHANNELS, NORMALIZER_VERSION, blocking_keys, normalize, view

@dataclass(frozen=True, slots=True)
class Candidate:
    record: Record
    score: float
    routes: frozenset[str]


class CandidateIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA cache_size=-65536")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS records(
          id TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT);
        CREATE TABLE IF NOT EXISTS postings(
          country TEXT, channel TEXT, key TEXT, id TEXT,
          PRIMARY KEY(country,channel,key,id)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS counts(
          country TEXT, channel TEXT, key TEXT, n INTEGER,
          PRIMARY KEY(country,channel,key)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
        """)

    def close(self):
        self.db.close()

    def metadata(self) -> dict:
        return {k: json.loads(v) for k, v in self.db.execute("SELECT key,value FROM metadata")}

    def build(self, paths: list[str | Path], limit_per_file: int | None = None) -> dict:
        if self.db.execute("SELECT count(*) FROM records").fetchone()[0]:
            raise ValueError("Index already contains records; use a new path rather than silently mixing corpora")
        provenance, nrecords = [], 0
        for source in paths:
            nfile = 0
            for rec in read_records(source):
                if limit_per_file is not None and nfile >= limit_per_file:
                    break
                if not rec.entity_id.startswith(("S2-", "S3-")):
                    raise ValueError(f"Non-target ID in index: {rec.entity_id}")
                self.db.execute("INSERT INTO records VALUES(?,?,?,?)",
                                (rec.entity_id, rec.business_name, rec.business_address, rec.country))
                v = view(rec)
                self.db.executemany("INSERT INTO postings VALUES(?,?,?,?)", [
                    (v.country, channel, key, rec.entity_id)
                    for channel, keys in blocking_keys(v).items() for key in keys])
                nrecords += 1
                nfile += 1
                if nrecords % 5000 == 0:
                    self.db.commit()
            provenance.append({"path": str(Path(source).resolve()), "sha256": digest_file(source), "indexed_rows": nfile})
        self.db.execute("INSERT INTO counts SELECT country,channel,key,count(*) FROM postings GROUP BY country,channel,key")
        meta = {"normalizer": NORMALIZER_VERSION, "records": nrecords, "sources": provenance,
                "complete": limit_per_file is None, "channels": list(CHANNELS)}
        self.db.executemany("INSERT INTO metadata VALUES(?,?)", [(k, json.dumps(v)) for k, v in meta.items()])
        self.db.commit()
        return meta

    def assert_usable(self):
        meta = self.metadata()
        if meta.get("normalizer") != NORMALIZER_VERSION:
            raise ValueError("Index normalizer mismatch")
        if not meta.get("complete"):
            raise ValueError("Sample/probe index is not permitted for training or final inference")

    def get(self, target_id: str) -> Record | None:
        row = self.db.execute("SELECT id,name,address,country FROM records WHERE id=?", (target_id,)).fetchone()
        return Record(*row) if row else None

    def retrieve(self, rec: Record, k: int = 100, max_postings: int = 2000, keys_per_channel: int = 4) -> list[Candidate]:
        if k < 1 or max_postings < 1 or keys_per_channel < 1:
            raise ValueError("Retrieval budgets must be positive")
        v = view(rec)
        scores: dict[str, float] = {}
        routes: dict[str, set[str]] = {}
        weights = {"name": 6., "root": 4., "address": 4., "postal_number": 3., "name_word": 2., "address_word": 1., "gram": .5}
        for channel, keys in blocking_keys(v).items():
            available = []
            for key in keys:
                row = self.db.execute("SELECT n FROM counts WHERE country=? AND channel=? AND key=?", (v.country, channel, key)).fetchone()
                if row and row[0] <= max_postings:
                    available.append((row[0], key))
            for count, key in sorted(available)[:keys_per_channel]:
                contribution = weights[channel] / math.log2(2 + count)
                for (tid,) in self.db.execute("SELECT id FROM postings WHERE country=? AND channel=? AND key=? ORDER BY id", (v.country, channel, key)):
                    scores[tid] = scores.get(tid, 0.) + contribution
                    routes.setdefault(tid, set()).add(channel)
        selected = sorted(scores, key=lambda t: (-scores[t], t))[:k]
        return [Candidate(self.get(t), scores[t], frozenset(routes[t])) for t in selected]
