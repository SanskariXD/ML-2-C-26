# Amazon ML Challenge 2026 — Colab

[Open the Colab notebook](https://colab.research.google.com/github/SanskariXD/ML-2-C-26/blob/main/colab/Amazon_ML_2026_Colab.ipynb)

1. Select **T4 GPU**. The notebook uses the new **12.7 GiB RAM profile**.
2. Run cells in order; mount the Google Drive account containing your dataset.
3. Keep `RUN_NAME = 'ber_lowram_v2'`. The old run name locks the previous code.
4. Let the real-data comparison pass, then run training and prediction.

Full mode uses all train/test files with the existing 30M training candidate-pair cap. Outputs and completed-stage checkpoints go to `MyDrive/ML-2-C-26/runs/ber_lowram_v2/`.

**Validation status:** synthetic equivalence tested; full-dataset RAM, runtime, and leaderboard score still need Colab verification. Interruptions can repeat the unfinished stage.

[Run guide](docs/COLAB_GUIDE.md) · [Memory changes, score checks and research](docs/LOW_MEMORY.md)
