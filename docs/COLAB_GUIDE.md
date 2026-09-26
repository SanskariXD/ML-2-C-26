# Current Colab profile

The notebook now defaults to **T4 + 12.7 GiB RAM**, `RUN_NAME='ber_lowram_v2'`, and a required real-data comparison before full mode. See [LOW_MEMORY.md](LOW_MEMORY.md) for the current memory strategy, resource limits and verification status. The guide below describes the original profile; its 28 GiB/60 GiB requirements apply to `--profile standard` only. The new profile plans for 40 GiB local capacity and at least 11 GiB system RAM, with no full-scale resource guarantee yet.

---

# Colab setup and recovery

## Dataset already configured

The supplied Drive file was checked using metadata only:

- Name: `6ab10eb3b23ba_student_resource.zip`
- Location: My Drive root
- Colab path: `/content/drive/MyDrive/6ab10eb3b23ba_student_resource.zip`
- File ID: `1IDOadTPN0lM9XnuYjNyJuOwFJRIxFfYQ`
- Size reported by Drive: 1,094,823,222 bytes

The dataset was **not downloaded or inspected during this migration**. At runtime the notebook validates that the archive contains the required seven TSVs, allowing a surrounding `student_resource/dataset/` folder. It extracts only those files into `/content/ber_runs/<RUN_NAME>/dataset`. Your original ZIP stays in Drive. Mount the account holding that file; otherwise adjust DATASET_PATH. Do not publish competition data into this public repository.

## Start

Open [Amazon_ML_2026_Colab.ipynb](https://colab.research.google.com/github/SanskariXD/ML-2-C-26/blob/main/colab/Amazon_ML_2026_Colab.ipynb).

1. Select T4 GPU. TPU is unsupported by this implementation.
2. Select High-RAM if available. The full runner stops below **28 GiB system RAM** to avoid launching the original large in-memory preprocessing/training on a standard small runtime. This is a conservative minimum, not a promise that 28 GiB always suffices.
3. Allow **60 GiB local capacity** for the run plus installed packages and the dataset. The upstream docs estimate ~40 GB peak work caches, but this is not a Colab benchmark.
4. Keep **80–100 GB free in Drive** for safe replacement of large checkpoints. During an update, old and new versions of changed files coexist. Drive storage is separate from Colab RAM/local disk.
5. Run all cells. The notebook defaults to `MODE='full'`, `RUN_NAME='ber_full_v1'`, batch 512.

A T4 alone does not establish full-dataset feasibility. We cannot see your Colab RAM, disk, compute balance or session limits from GitHub. No fixed runtime or leaderboard score is promised. Upstream's 4–9 hour estimate excludes this migration's Drive backup overhead and is not measured on your T4.

## What is installed

An isolated `/content/ber-py311` Python 3.11 environment, PyTorch 2.7.1 CUDA 12.6, original pinned requirements, sentence-transformers 3.3.1, transformers 4.46.3 and faiss-cpu 1.9.0.post1. CUDA uses the host driver already supplied by Colab. Colab's notebook Python and preinstalled torch are not replaced. The first run needs Internet for GitHub, dependencies and `intfloat/multilingual-e5-small`.

The notebook runs the original CPU synthetic smoke test and a tiny GPU encoder check before starting the real data pipeline. These verify wiring, not full-scale memory or accuracy.

## Full flow

1. Prepare training records.
2. Build training embeddings, candidate pairs and features.
3. Fit the original two-stage LightGBM models and calibration.
4. Prepare test records.
5. Build test embeddings, candidate pairs and features.
6. Predict all test entities, run the official validator and publish outputs.

The matching code and model defaults are unchanged. The launcher pins T4 batch 512, feature chunk 1,000,000, join budget 20,000,000, at most four preprocessing workers, and the original 30,000,000 candidate-row training cap. Full mode explicitly uses `--dev-frac 1.0`. The cap controls model-fitting pairs, not source-file coverage; the original hard-negative sampling policy remains in effect.

LightGBM and much of retrieval/preprocessing use CPU and system RAM. GPU acceleration mainly supports embeddings and the dense retrieval implementation; low GPU utilization during other phases is expected.

For a setup experiment, use `MODE='dev'` with **a different run name** such as `ber_dev_v1`. It trains on a 3% entity subset and stops after the training report. This is not a leaderboard submission; dev scores are optimistic due to the smaller target pool.

## Checkpoints and interruptions

Work happens on local disk. After each successful stage the runner copies completed files to `MyDrive/ML-2-C-26/runs/<RUN_NAME>/checkpoint_blobs/`, with a hash-addressed manifest in `checkpoint.json`.

- A manifest is published only after that stage's file copies finish. Wait for `SAVED TO DRIVE`.
- Obsolete blobs are deleted only after publishing the replacement manifest.
- A restarted session restores verified files from the latest complete checkpoint and skips its completed stages.
- Partial files from a failed stage are discarded before restore; the unfinished stage repeats. This avoids trusting partially written feature/embedding memmaps.
- A kill **within** a long stage can lose hours of that stage. This is stage-level recovery, not per-batch/per-tree checkpointing.
- Logs are copied after a stage returns; a sudden VM loss can also lose its newest log lines.
- Keep MODE and RUN_NAME unchanged when resuming. Dataset/code changes are rejected for an existing checkpoint. Use a new run name for changed settings or data.
- The first resolved Git commit is saved in `code_commit.txt` on Drive. Resume uses that exact commit, even after the repository changes.
- After CUDA OOM you may reduce DENSE_BATCH from 512 to 256 and rerun; completed stages remain reusable. System-RAM OOM needs a larger RAM runtime or a separately tested pipeline optimization.
- Backups add substantial disk I/O and elapsed time. Mounted Drive errors/quota exhaustion can stop a backup; the previous completed manifest remains the recovery point.

Do not run two sessions simultaneously against the same RUN_NAME/Drive checkpoint. Disconnecting a Colab runtime is still possible on paid plans. No keep-alive workaround is included.

## Outputs

Under `MyDrive/ML-2-C-26/runs/<RUN_NAME>/`:

- `output/matching_results.tsv`: the leaderboard upload.
- `output/candidate_pairs.tsv`: candidates required for the final package.
- `ber_submission.zip`: the two TSVs under `output/`; **not** the complete final code/documentation package.
- `report.json`: model validation report.
- `logs/`: completed or gracefully failed stage logs and the official validation log.
- `checkpoint.json`, `checkpoint_blobs/`: completed-stage recovery data.
- `code_commit.txt`, `run_settings.json`: reproducibility information.

Do not delete checkpoints until you have downloaded/verified the final outputs and no longer need to resume. See the original [problem statement](../student_resource/README.md) for final documentation/code packaging requirements.
