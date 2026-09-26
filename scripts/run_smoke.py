"""Run all real tree learners on synthetic fixtures and validate output formatting.

No competition dataset is used. 'test' fixture rows intentionally mirror the synthetic
source: test predictions check plumbing only; only the split holdout exercises isolation.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time
from pathlib import Path
# Allow both python -m scripts.run_smoke and python scripts/run_smoke.py.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_fixture import make_fixture
from ml2.data import digest_file, write_json
from ml2.index import CandidateIndex
from ml2.pipeline import train, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/smoke")
    p.add_argument("--queries", type=int, default=360)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error("Smoke output already exists; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data = make_fixture(out / "synthetic_data", args.queries)
    idx_path = out / "targets.sqlite"
    idx = CandidateIndex(idx_path)
    meta = idx.build([data / "train_source2.tsv", data / "train_source3.tsv"])
    postings = idx.db.execute("SELECT count(*) FROM postings").fetchone()[0]
    idx.close()
    _, metrics, _ = train(idx_path, data / "train_source1.tsv", data / "train_ground_truth.tsv", out / "training",
                         max_queries=args.queries, k=40, max_pairs=40000, max_postings=1000, rounds=50, threads=2)
    prediction = predict(idx_path, data / "test_source1.tsv", out / "training/model", out / "prediction", "care", batch_pairs=1000)
    validation = subprocess.run([sys.executable, str(ROOT / "reference/source_snapshot/student_resource/utils/validate_submission.py"),
                  "--matching", str(out / "prediction/matching_results.tsv"), "--candidate", str(out / "prediction/candidate_pairs.tsv"),
                  "--test-dir", str(data), "--check-ids"], text=True, capture_output=True)
    (out / "official_validator.txt").write_text(validation.stdout + validation.stderr)
    if validation.returncode:
        raise RuntimeError(f"Supplied validator failed:\n{validation.stdout}\n{validation.stderr}")
    report = {"evidence_kind": "SYNTHETIC_SOFTWARE_SMOKE_NOT_COMPETITION_PERFORMANCE", "fixture_queries": args.queries,
              "elapsed_seconds": time.perf_counter() - start,
              "python": platform.python_version(), "platform": platform.platform(),
              "versions": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn", "xgboost", "lightgbm", "catboost", "rapidfuzz"]},
              "rss_max_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
              "index_bytes": idx_path.stat().st_size, "index_records": meta["records"], "index_postings": postings,
              "fixture_sha256": digest_file(data / "train_source1.tsv"), "training": metrics, "prediction": prediction,
              "supplied_validator_exit_code": validation.returncode}
    write_json(out / "smoke_report.json", report)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
