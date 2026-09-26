# Colab 12.7 GiB execution profile

## Hardware and status

Observed runtime: Tesla T4, 12.7 GiB system RAM, 55.5 GiB free local disk. T4 VRAM and system RAM are separate. The old 28 GiB profile remains available; this notebook selects `--profile lowram` and a fresh `ber_lowram_v2` run.

This is an execution refactor. It does not promise a leaderboard gain or an uninterrupted Colab session. Synthetic equivalence tests have passed; the user's real dataset and GPU have not been run in this development environment. The notebook runs a real-data regression gate before the full job. Full-scale peak RAM/disk and completion time remain to be measured on Colab.

## What changed

| Stage | Memory change | Quality invariant |
|---|---|---|
| Read/normalize | 25,000-row batches; SQLite ID/ownership lookups; Parquet output | Same TSV repair, first-ID dedupe, row order, country partitions and dev selection |
| Embeddings | Disk-backed first-occurrence text dedupe | Same E5 model, text, unique-text order, batch size, max length and fp16 storage |
| Exact dense retrieval | Separate process; one preallocated FAISS flat index; streamed index filling | All original country targets, original CPU search batches and tie behavior |
| Key retrieval | Token families and indexes backed by temporary files; smaller joins | Global token counts, first-occurrence codes, key caps, ordering and top-K unchanged |
| BM25 | Stream tokenization and sparse construction; bound dense query-score tiles | Full target corpus, original vocabulary/DF/IDF, formula and candidate budgets |
| Features | Compact Arrow strings; compute sparse rows on demand; fit IDF on all partition rows | Same feature values and global group context |
| LightGBM | `Sequence` reads exact selected rows in batches | Same folds, hard-negative selection, 30M cap, max_bin, tree parameters and seeds |
| Second-stage features | Write global context one column at a time; compute pair support in chunks | Same competition across entities and reference candidates |
| Disk lifecycle | Remove consumed training caches after the train-stage checkpoint transition | Test needs saved models/calibrator; original dataset stays intact |

Feature chunks are 25,000 pairs, raw join budget 500,000 rows, worker count at most 2. These are execution settings, not sample rates. Standard mode remains an independent control branch.

## Dense retrieval limit

The largest remaining unavoidable allocation in this implementation is the **complete float32 exact FAISS index for one country**: `target_rows × 384 × 4` bytes. There is no duplicate full float32 input matrix in the worker. A >9 GiB index is rejected; query/result buffers and Python also need memory. This does not prove that every partition below 9 GiB fits under all runtime conditions.

A trial that searched separate target shards was rejected: equal-score boundary/tie handling changed some neighbor IDs. It is not in the shipped workflow.

## Score protection

`colab/validate_low_memory.py` prepares a deterministic 3% entity-consistent training slice using the streamed reader. It shares preparation/embeddings, independently computes candidates/features for standard and low-memory paths, checks exact array equality, trains both, and checks nondecreasing overall and per-country holdout metrics. Results bind to dataset hashes and source-code hash. The full runner requires a passing current report.

The dataset fraction is used only for this comparison. The subsequent full job uses `dev_frac=1.0`. Slice scores are optimistic due to fewer distractors: they are regression evidence, not leaderboard forecasts. Preparing the slice once means parser/encoder equivalence is additionally covered by dedicated fixtures, not independently measured by this gate.

The user's existing instruction requests execution equivalence. Therefore flat accuracy with measured memory savings is a valid optimization result; we do not use the upstream positive-score-gain gate to claim this is a modeling improvement.

## Resume and disk

Keep `RUN_NAME='ber_lowram_v2'`. A fresh name locks the revised commit; the old `ber_full_v1` lock still refers to old code. Completed main stages restore from Drive after resets. The unfinished stage can repeat; comparison work within a lost runtime can repeat too. The passing comparison report is saved to Drive and reused only for the same code/data. No artificial keep-alive or runtime-limit bypass is used.

40 GiB local capacity is the full-job planning guard, reduced by streaming and deletion of consumed train caches. Temporary files stay under the run directory. Keep Drive space for completed-stage backups; Drive quota or Colab interruption can still stop a job. If a guard or allocation fails, send the error and stage log; do not lower candidate/training budgets to force it through.

## Research used

- [LightGBM 4.5 Sequence API](https://lightgbm.readthedocs.io/en/v4.5.0/pythonapi/lightgbm.Sequence.html) and [data interface](https://lightgbm.readthedocs.io/en/v4.5.0/Python-Intro.html): batches during Dataset construction, one global binned dataset rather than training independent chunk models.
- [NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html): access segments of large arrays without loading the whole file. Advanced indexing still copies; selected matrices must also be streamed.
- [Arrow ParquetFile.iter_batches](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html): bounded record batches and selected columns.
- [FAISS FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ): numerical and tie behavior matters. We retained the original full flat index/search semantics after the sharding regression failed.

These are storage and execution techniques. No external entity data, new model, smaller candidate set, quantized retrieval index, or reduced training cap is introduced.
