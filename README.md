# ML 2 C 26

**Plan 2: a separate, single-source business entity resolution track.**

The uploaded repo's tree-ensemble idea is retained. Its missing modules, padded features and validation/threshold inconsistencies are replaced with an auditable first implementation. No other competitor code is imported. Plan 1 is not modified.

**Status:** working synthetic-tested milestone; not a trained competition solution. New GitHub repository creation is pending authorization through a connector that supports repository creation. This local package is ready to publish as a private `SanskariXD/ML-2-C-26` repository after access is authorized.

## Start here

- [Workflow and acceptance gates](WORKFLOW.md)
- [Audit with original file/cell references](docs/AUDIT.md)
- [Primary-source research and priorities](docs/RESEARCH.md)
- [CARE experimental method](docs/NEW_METHOD.md)
- [Executed tests and limitations](evidence/STATUS.md)

## What is implemented

Seven-route SQLite candidate index; Unicode-preserving normalization and Latin accent folding; 34 real named features; early-stopped XGBoost / LightGBM / CatBoost; separate fit/early/calibration/tune/holdout partitions; a pair-only calibrated baseline and a 14-input CARE context model; exact macro-F0.5 threshold selection; batched predictions with exact scored-candidate logs; strict output checks; model manifests and regression tests.

There is no full general-purpose Indic transliterator, learned dense retriever, train-only hard-negative miner, nested cross-validation system or full-scale benchmark yet. The root source artifacts are not silently reused. See limitations before interpreting any metric.

## Quick smoke test

Python 3.11+ is required. Development was exercised on Linux with Python 3.13; Linux/Colab/WSL is recommended for reproducibility. A GPU is not needed for this milestone.

```bash
python -m venv .venv
# Linux/Colab/WSL shell; on Windows activate .venv\Scripts\activate instead.
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m scripts.run_smoke --out runs/smoke
```

The smoke generator creates synthetic records, trains all three actual tree libraries, reloads the models, writes TSVs and invokes the supplied validator with ID checks. These are not official competition records. Synthetic `test` files deliberately mirror fixture inputs for plumbing checks and must not be presented as a test-generalization score. The split holdout is a separate correctness exercise, also synthetic.

The Colab runbook is `notebooks/Plan_2.ipynb`. It runs smoke tests first and leaves real-data execution disabled until paths and the benchmark gate are reviewed.

## Real dataset workflow

The ZIP does not include the real TSVs. Reuse the already obtained official dataset; do not download business records from external sources. Replace the following local paths with the actual extracted official files. File names are explicit CLI arguments, not hardcoded discovery guesses.

First benchmark a sample index, including storage. A probe index is rejected by training/prediction so that it cannot accidentally become the evaluation target universe.

```bash
python -m ml2 probe-index \
  --targets /data/train/train_source2.tsv /data/train/train_source3.tsv \
  --limit-per-file 10000 --out runs/probe10k.sqlite
# Inspect the JSON benchmark, then repeat at 100000 into a new index path.
```

Only after estimating full-run disk/time/RAM from the sample measurements:

```bash
python -m ml2 index \
  --targets /data/train/train_source2.tsv /data/train/train_source3.tsv \
  --out runs/train_targets.sqlite

python -m ml2 train --index runs/train_targets.sqlite \
  --queries /data/train/train_source1.tsv \
  --truth /data/train/train_ground_truth.tsv \
  --max-queries 25000 --max-pairs 1500000 --k 100 \
  --threads 2 --out runs/baseline_v1
```

`--max-queries` is a reproducible hash sample across the entire query file; it does not reduce the target pool. `--max-pairs` is a hard safety limit and aborts before fitting if exceeded. Reducing it may require reducing the query sample. Do not replace the all-target index with a truth-conditioned pool to get around memory pressure.

Read `metrics.json`, `splits.json`, the model manifest and both held-out output files. Pair-only is the default policy; CARE is not automatically promoted based on synthetic or one lucky holdout result. The initial split is known-catalog/query-disjoint, not strict unseen-target validation.

For a locked policy, build a **separate test-target index**:

```bash
python -m ml2 index \
  --targets /data/test/test_source2.tsv /data/test/test_source3.tsv \
  --out runs/test_targets.sqlite

python -m ml2 predict --index runs/test_targets.sqlite \
  --queries /data/test/test_source1.tsv \
  --model runs/baseline_v1/model --mode pair --out runs/prediction_v1

python reference/source_snapshot/student_resource/utils/validate_submission.py \
  --matching runs/prediction_v1/matching_results.tsv \
  --candidate runs/prediction_v1/candidate_pairs.tsv \
  --test-dir /data/test --check-ids
```

The supplied validator's `--check-ids` can use substantial RAM; the runner additionally performs ID and candidate-subset checks via its disk-backed target index. Large `candidate_pairs.tsv` and score-log files also consume disk. Do not submit the development ZIP as the organizer's final submission: the final packager/model card is a later gate described in the workflow.

## Source attribution and safety

Original uploaded materials remain unchanged under `reference/source_snapshot`. The source methodology credits DataResolvers / Aamod; those credits are not replaced with the current user's name. The source has no LICENSE file even though its README makes license references. Preserve privately and resolve reuse/redistribution permission before making source materials public or submitting inherited work. A library's license does not by itself grant rights to a third-party repository.

No passwords, tokens, real dataset files or competition-trained checkpoints are included. Never load a model from an untrusted source merely because it uses a familiar file extension. The new runner rejects the uploaded 55/72-feature artifact mismatch.

To publish after GitHub authorization, use `bash scripts/publish_github.sh` in a shell with an authenticated GitHub CLI. It verifies the account, creates a private repo only, and refuses to overwrite an existing one. It does not modify Plan 1.
