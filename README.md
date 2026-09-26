# Amazon ML Challenge 2026

Business entity resolution with multilingual embeddings, candidate retrieval, and LightGBM.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SanskariXD/ML-2-C-26/blob/main/colab/Amazon_ML_2026_Colab.ipynb)

## Run

1. Open the Colab notebook above and select **T4 GPU**; use **High-RAM** for the full dataset.
2. Mount the Google Drive account containing your dataset. Its existing ZIP path is already filled in.
3. Run all cells. Full mode trains, predicts every test entity, validates, and saves outputs to Drive.

Outputs: `MyDrive/ML-2-C-26/runs/ber_full_v1/output/`
- **matching_results.tsv** — leaderboard upload.
- **candidate_pairs.tsv** — candidate set for the final submission.

Completed stages are backed up to Drive. Rerun with the same run name to restore them; an interrupted stage may repeat. Colab runtime continuity is not guaranteed.

[Colab setup & recovery](docs/COLAB_GUIDE.md) · [Technical run guide](code/business_entity_resolution/README.md) · [Research & source notes](docs/INDEX.md)
