# Technical and research notes

## Current execution

- [Colab setup, storage, outputs and recovery](COLAB_GUIDE.md)
- [Original CLI guide](../code/business_entity_resolution/README.md)
- [Problem statement and submission rules](../student_resource/README.md)
- [Migration verification](MIGRATION.md)

## Supplied source material

The source ZIP `amazon-ml-challenge-2026-main.zip` reports upstream commit `5eec1cd169dd84ba6cb5ed370e308c04ce5545ca`. Its notebook points to [Karthikatstuffmama/amazon-ml-challenge-2026](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026). All supplied files are retained; the previous root README is archived as [UPSTREAM_README.md](UPSTREAM_README.md) (its relative links were originally rooted at the repository root).

- [Experiments](../EXPERIMENTS.md)
- [Findings and validation caveats](../FINDINGS.md)
- [Hardware estimates](../HARDWARE.md)
- [Plan](../PLAN.md), [research](../research.md), [strategy](../strategy.md), [literature](../LITERATURE.md)
- [Original Kaggle notebook](../kaggle/ber_full_pipeline.ipynb) — legacy platform entry point; current recommended entry point is Colab.

Reported leaderboard scores, expected scores, timing and scale claims in these original documents are **upstream claims**, not newly reproduced results of this migration. The supplied ZIP contains no dataset, trained model weights, or archived leaderboard outputs. This migration does not claim a new accuracy result or resolve all validation risks in upstream research.

The target repository's pre-existing `notebooks/Plan2_Training.ipynb` and `reference/source_snapshot/` files are preserved for history. They are unrelated to the imported BER pipeline and are not loaded by the Colab notebook.
