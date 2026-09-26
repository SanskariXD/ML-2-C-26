from __future__ import annotations
import argparse
import json
from pathlib import Path
try:
    import resource
except ImportError:  # Windows: RSS is not available through the stdlib.
    resource = None
import time
from .index import CandidateIndex
from .pipeline import train, predict
from .data import write_json


def main():
    parser = argparse.ArgumentParser(description="Plan 2 business entity resolution")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "probe-index"):
        p = sub.add_parser(name)
        p.add_argument("--targets", nargs="+", required=True)
        p.add_argument("--out", required=True)
        if name == "probe-index":
            p.add_argument("--limit-per-file", type=int, default=10000)
    p = sub.add_parser("train")
    for arg in ("index", "queries", "truth", "out"):
        p.add_argument("--" + arg, required=True)
    for arg, value in [("max-queries", 25000), ("max-pairs", 1500000), ("k", 100), ("max-postings", 2000),
                       ("keys-per-channel", 4), ("seed", 26), ("rounds", 250), ("threads", 2)]:
        p.add_argument("--" + arg, type=int, default=value)
    p = sub.add_parser("predict")
    for arg in ("index", "queries", "model", "out"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--mode", choices=["pair", "care"], default="pair")
    p.add_argument("--batch-pairs", type=int, default=4096)
    a = parser.parse_args()
    if a.command in ("index", "probe-index"):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        if Path(a.out).exists():
            parser.error("Refusing to overwrite an index; provide a new --out")
        start = time.perf_counter()
        idx = CandidateIndex(a.out)
        try:
            result = idx.build(a.targets, getattr(a, "limit_per_file", None))
            result.update({"elapsed_seconds": time.perf_counter() - start, "disk_bytes": Path(a.out).stat().st_size,
                           "rss_max_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
                           "postings": idx.db.execute("SELECT count(*) FROM postings").fetchone()[0]})
            result["bytes_per_record"] = result["disk_bytes"] / max(1, result["records"])
            write_json(str(a.out) + ".benchmark.json", result)
        finally:
            idx.close()
    elif a.command == "train":
        _, result, _ = train(a.index, a.queries, a.truth, a.out, max_queries=a.max_queries,
                 max_pairs=a.max_pairs, k=a.k, max_postings=a.max_postings, keys_per_channel=a.keys_per_channel,
                 seed=a.seed, rounds=a.rounds, threads=a.threads)
    else:
        result = predict(a.index, a.queries, a.model, a.out, a.mode, a.batch_pairs)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
