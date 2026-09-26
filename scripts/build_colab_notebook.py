"""Regenerate the Colab orchestrator without embedding model logic in notebook cells."""
from __future__ import annotations

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Plan 2 — Google Drive → Colab training

Run from the top, one cell at a time. The original 1.09 GB dataset ZIP remains in your Drive. Colab extracts only the seven required TSVs to its temporary disk for speed. GitHub contains code; Drive stores the dataset and completed checkpoints.

**What this notebook does:** clone the private Plan 2 repository; install and test it; mount Drive; extract and validate the dataset; save a complete data profile; build 10k and optionally 100k index probes; then, after reviewing disk/RAM estimates, build the full target index and run one honest baseline. It never claims a real score until the real cells finish.

For private GitHub access, add a **Colab Secret named `GITHUB_TOKEN`** with read access to `SanskariXD/ML-2-C-26`, or paste a token into the hidden prompt in the clone cell. The token is used only in memory during Git authentication. Never put it into a notebook cell or Git URL.
"""),
    code("""from pathlib import Path
import json, os, shutil, subprocess, sys

REPO_URL = "https://github.com/SanskariXD/ML-2-C-26.git"
PROJECT = Path("/content/ML-2-C-26")
DRIVE_ZIP = Path("/content/drive/MyDrive/6ab10eb3b23ba_student_resource.zip")
DATA = Path("/content/plan2_data/dataset")
WORK = Path("/content/plan2_runs")
DRIVE_RUNS = Path("/content/drive/MyDrive/ML-2-C-26-runs")
MAX_QUERIES, MAX_PAIRS, K = 25000, 1500000, 100
RUN_100K_PROBE = False  # Set True after reading the 10k benchmark.
RUN_FULL_BASELINE = False  # Set True only after reading both resource probes.
RUN_TEST_PREDICTION = False  # Set True only after reviewing the real holdout.
for folder in (WORK, DATA):
    folder.mkdir(parents=True, exist_ok=True)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

def run(*args):
    print("Running:", " ".join(map(str, args)))
    subprocess.run([sys.executable, *map(str, args)], cwd=PROJECT, check=True)

print("Temporary space:", round(shutil.disk_usage("/content").free / 1024**3, 2), "GiB")
"""),
    code("""from google.colab import drive
drive.mount("/content/drive")
assert DRIVE_ZIP.is_file(), f"Dataset ZIP missing: {DRIVE_ZIP}. Edit DRIVE_ZIP above if its location differs."
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
print("Drive ZIP:", DRIVE_ZIP.name, round(DRIVE_ZIP.stat().st_size / 1024**3, 2), "GiB")
"""),
    code("""# Clone/update the private repository without storing a token in Git config.
import getpass, tempfile
from google.colab import userdata
try:
    token = userdata.get("GITHUB_TOKEN")
except Exception:
    token = None
if not token:
    token = getpass.getpass("GitHub read token (hidden input): ")
assert token, "A GitHub token with read access to this private repository is required."
with tempfile.TemporaryDirectory() as private_dir:
    askpass = Path(private_dir) / "askpass.sh"
    askpass.write_text('#!/bin/sh\\ncase "$1" in *Username*) printf "%s\\\\n" "x-access-token";; *) printf "%s\\\\n" "$ML2_GITHUB_TOKEN";; esac\\n')
    askpass.chmod(0o700)
    git_env = os.environ.copy()
    git_env.update({"GIT_ASKPASS": str(askpass), "GIT_TERMINAL_PROMPT": "0", "ML2_GITHUB_TOKEN": token})
    try:
        if PROJECT.exists():
            assert (PROJECT / ".git").is_dir(), f"{PROJECT} exists but is not a Git checkout"
            origin = subprocess.check_output(["git", "-C", str(PROJECT), "remote", "get-url", "origin"], text=True).strip()
            assert origin == REPO_URL, f"Unexpected Git origin: {origin}"
            subprocess.run(["git", "-C", str(PROJECT), "fetch", "origin", "main"], env=git_env, check=True)
            subprocess.run(["git", "-C", str(PROJECT), "merge", "--ff-only", "FETCH_HEAD"], env=git_env, check=True)
        else:
            subprocess.run(["git", "clone", "--branch", "main", REPO_URL, str(PROJECT)], env=git_env, check=True)
    finally:
        del git_env, token
print("Plan 2 commit:", subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip())
"""),
    code("""run("-m", "pip", "install", "-r", "requirements.txt")
run("-m", "pytest", "-q")
"""),
    markdown("""## Prepare the official dataset

The extractor reads the ZIP from Drive, checks for all four train and three test TSVs, checks free disk, and copies only those members to `/content`. It validates ZIP CRCs and reuses verified extractions if a cell is rerun. The original ZIP is never changed.
"""),
    code("""run("-m", "scripts.prepare_colab_data", "--zip", DRIVE_ZIP, "--out", DATA)
TRAIN, TEST = DATA / "train", DATA / "test"
for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"):
    assert (TRAIN / name).is_file(), name
print("Train folder:", TRAIN)
"""),
    code("""# Profile every official training row and every ground-truth reference.
PROFILE_DIR = WORK / "profile"
profile_saved = DRIVE_RUNS / "profile"
profile_saved.mkdir(parents=True, exist_ok=True)
archive_hash = json.loads((DATA / "extraction_manifest.json").read_text())["source_zip_sha256"]
fingerprint_file = profile_saved / "source_zip_sha256.txt"
if (profile_saved / "data_profile.json").exists():
    assert fingerprint_file.is_file() and fingerprint_file.read_text().strip() == archive_hash, (
        "Saved profile belongs to a different ZIP; use a fresh DRIVE_RUNS directory."
    )
else:
    run("-m", "ml2", "profile-data",
        "--queries", TRAIN / "train_source1.tsv",
        "--targets", TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv",
        "--truth", TRAIN / "train_ground_truth.tsv", "--out", PROFILE_DIR)
    for name in ("data_profile.json", "data_profile.md"):
        shutil.copy2(PROFILE_DIR / name, profile_saved / name)
    fingerprint_file.write_text(archive_hash + "\\n")
profile = json.loads((profile_saved / "data_profile.json").read_text())
print((profile_saved / "data_profile.md").read_text())
print("Countries/scripts and file hashes: ", profile_saved / "data_profile.json")
"""),
    markdown("""## Resource gate: sample target indexes

The first probe indexes up to 10,000 records **per target file**. Set `RUN_100K_PROBE=True` in the settings cell and rerun this cell for the 100,000 row probe. Probe samples currently use file prefixes; use these estimates as a warning signal, not proof that the full index fits.

These are intentionally probe indexes; training refuses to use them.
"""),
    code("""from ml2.index import CandidateIndex

def persist_file(local, saved):
    saved.parent.mkdir(parents=True, exist_ok=True)
    temporary = saved.with_name(saved.name + ".partial")
    shutil.copy2(local, temporary)
    os.replace(temporary, saved)

def check_probe(path, limit):
    idx = CandidateIndex(path)
    try:
        meta = idx.metadata()
        assert meta.get("complete") is False, "A probe must never be treated as a complete index"
        assert all(s["indexed_rows"] <= limit for s in meta["sources"])
        hashes = [s["sha256"] for s in meta["sources"]]
        expected = [profile["files"][str(TRAIN / name)]["sha256"] for name in ("train_source2.tsv", "train_source3.tsv")]
        assert hashes == expected, "Checkpoint index was built from different targets"
        assert idx.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        idx.close()

def probe(limit):
    path = WORK / f"probe{limit}.sqlite"
    saved = DRIVE_RUNS / "indexes" / path.name
    benchmark = Path(str(path) + ".benchmark.json")
    saved_benchmark = Path(str(saved) + ".benchmark.json")
    if saved.exists() and saved_benchmark.exists() and not path.exists():
        print("Restoring probe", limit, "from Drive")
        shutil.copy2(saved, path)
        shutil.copy2(saved_benchmark, benchmark)
    if not path.exists():
        run("-m", "ml2", "probe-index", "--targets",
            TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv",
            "--limit-per-file", limit, "--out", path)
    check_probe(path, limit)
    assert benchmark.is_file(), f"Missing benchmark for {path}"
    if not saved.exists():
        persist_file(path, saved)
        persist_file(benchmark, saved_benchmark)
    result = json.loads(benchmark.read_text())
    print(f"Probe {limit:,}/file: {result['records']:,} targets; {result['disk_bytes']/1024**3:.3f} GiB index; {result['elapsed_seconds']:.1f}s; RSS max {result['rss_max_kib_linux']/1024**2:.2f} GiB")
    return result

probe10k = probe(10000)
probe100k = probe(100000) if RUN_100K_PROBE else None
if not RUN_100K_PROBE:
    print("Read the 10k result, then set RUN_100K_PROBE=True in settings and rerun this cell.")
"""),
    markdown("""## Full index and baseline — manually enabled after both probes

The full index uses **every supplied S2/S3 training target**. The baseline fits on a bounded, stable hash sample of S1 queries and evaluates against the full target catalog. This can take hours and much more disk than the probes. The model, reports, and complete index are copied to `ML-2-C-26-runs` in Drive after each stage completes, so they can be restored after a Colab disconnect.

Set `RUN_FULL_BASELINE=True` only after reviewing both probe reports and available disk. The notebook refuses to start if the 100k probe or the data profile is missing.
"""),
    code("""def verify_full(path):
    idx = CandidateIndex(path)
    try:
        idx.assert_usable()
        hashes = [s["sha256"] for s in idx.metadata()["sources"]]
        expected = [profile["files"][str(TRAIN / name)]["sha256"] for name in ("train_source2.tsv", "train_source3.tsv")]
        assert hashes == expected, "Full index checkpoint was built from different target files"
        assert idx.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        idx.close()

BASELINE = WORK / "baseline_v1"
FULL_INDEX = WORK / "train_targets.sqlite"
if RUN_FULL_BASELINE:
    assert probe100k is not None, "Set RUN_100K_PROBE=True and finish the 100k probe first"
    targets = sum(v["counts"]["rows"] for v in profile["files"].values() if v["kind"] == "target")
    estimated = 1.5 * targets * probe100k["bytes_per_record"]
    available = shutil.disk_usage(WORK).free
    print(f"Estimated full index (1.5x safety factor): {estimated/1024**3:.2f} GiB; local free: {available/1024**3:.2f} GiB")
    assert available > estimated + 2 * 1024**3, "Not enough local disk for full index and training headroom"
    saved_index = DRIVE_RUNS / "indexes" / FULL_INDEX.name
    if saved_index.exists() and not FULL_INDEX.exists():
        print("Restoring complete index from Drive")
        shutil.copy2(saved_index, FULL_INDEX)
    if not FULL_INDEX.exists():
        run("-m", "ml2", "index", "--targets",
            TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv", "--out", FULL_INDEX)
    verify_full(FULL_INDEX)
    if not saved_index.exists():
        persist_file(FULL_INDEX, saved_index)
    saved_baseline = DRIVE_RUNS / BASELINE.name
    if saved_baseline.exists():
        assert (saved_baseline / "model/manifest.json").is_file() and (saved_baseline / "metrics.json").is_file()
        manifest = json.loads((saved_baseline / "model/manifest.json").read_text())
        assert manifest["training_queries_sha256"] == profile["files"][str(TRAIN / "train_source1.tsv")]["sha256"]
        assert manifest["ground_truth_sha256"] == profile["ground_truth"]["sha256"]
        assert manifest["config"]["k"] == K, "Saved baseline used another K"
        print("Baseline checkpoint exists:", saved_baseline)
    else:
        run("-m", "ml2", "train", "--index", FULL_INDEX,
            "--queries", TRAIN / "train_source1.tsv", "--truth", TRAIN / "train_ground_truth.tsv",
            "--max-queries", MAX_QUERIES, "--max-pairs", MAX_PAIRS, "--k", K,
            "--threads", 2, "--out", BASELINE)
        shutil.copytree(BASELINE, saved_baseline)
    metrics = json.loads((saved_baseline / "metrics.json").read_text())
    print("REAL held-out query macro F0.5 (pair):", metrics["policies"]["pair"]["holdout"]["macro_f05"])
    print("Experimental CARE:", metrics["policies"]["care"]["holdout"]["macro_f05"])
else:
    print("Full index/training paused. Review the 100k probe, then set RUN_FULL_BASELINE=True.")
"""),
    markdown("""## Optional test prediction

Run this after examining the real holdout and locking the pair policy. It indexes all supplied **test** targets, scores every test query, runs the supplied validator, and keeps the exact scored-pair log with the outputs in Drive. Test records have no labels here; no test F0.5 is reported.
"""),
    code("""if RUN_TEST_PREDICTION:
    assert RUN_FULL_BASELINE and (DRIVE_RUNS / "baseline_v1/model/manifest.json").is_file()
    TEST_INDEX = WORK / "test_targets.sqlite"
    saved_test_index = DRIVE_RUNS / "indexes" / TEST_INDEX.name
    if saved_test_index.exists() and not TEST_INDEX.exists():
        shutil.copy2(saved_test_index, TEST_INDEX)
    if not TEST_INDEX.exists():
        run("-m", "ml2", "index", "--targets",
            TEST / "test_source2.tsv", TEST / "test_source3.tsv", "--out", TEST_INDEX)
    test_index = CandidateIndex(TEST_INDEX)
    try:
        test_index.assert_usable()
        assert test_index.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert [Path(s["path"]).name for s in test_index.metadata()["sources"]] == [
            "test_source2.tsv", "test_source3.tsv"]
    finally:
        test_index.close()
    if not saved_test_index.exists():
        persist_file(TEST_INDEX, saved_test_index)
    prediction = WORK / "prediction_v1"
    saved_prediction = DRIVE_RUNS / prediction.name
    if not saved_prediction.exists():
        run("-m", "ml2", "predict", "--index", TEST_INDEX,
            "--queries", TEST / "test_source1.tsv", "--model", DRIVE_RUNS / "baseline_v1/model",
            "--mode", "pair", "--out", prediction)
        run("reference/source_snapshot/student_resource/utils/validate_submission.py",
            "--matching", prediction / "matching_results.tsv",
            "--candidate", prediction / "candidate_pairs.tsv",
            "--test-dir", TEST, "--check-ids")
        shutil.copytree(prediction, saved_prediction)
    print("Saved test outputs and complete scored-pair log:", saved_prediction)
else:
    print("Prediction paused until the real baseline and threshold are reviewed.")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "Plan2_Training.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    target = Path(__file__).resolve().parents[1] / "notebooks" / "Plan2_Training.ipynb"
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    old = target.with_name("Plan_2.ipynb")
    redirect = {
        **notebook,
        "cells": [markdown("# Plan 2 notebook moved\n\nOpen [Plan2_Training.ipynb](Plan2_Training.ipynb) for the complete Google Drive → Colab workflow.\n")],
        "metadata": {**notebook["metadata"], "colab": {"name": old.name, "provenance": []}},
    }
    old.write_text(json.dumps(redirect, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
