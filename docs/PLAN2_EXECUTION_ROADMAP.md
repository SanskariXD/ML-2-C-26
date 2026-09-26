# Plan 2 execution roadmap

Promote changes only against a fixed full-catalog evaluation, with query-macro F0.5 and resource cost reported together. Store each experiment with input and split fingerprints. Keep the final holdout untouched after the predeclared comparison.

| Phase | Work and reusable modules | New/changed files | Gate and expected compute |
|---|---|---|---|
| 0. Audit | Verify source and reconstructed code; run tests/smoke | `docs/ASTRA_INITIAL_AUDIT.md`, `evidence/STATUS.md` | 37 tests pass; minutes on CPU |
| 1. Real data profile | Stream full official TSVs; verify IDs, columns, country/script/missingness and match cardinality | `ml2/profile.py`, `ml2/cli.py`, local `reports/data_profile.*` | Every GT query/target exists; duplicate IDs fail; disk-backed IDs; minutes to hours, disk scales with ID count |
| 2. Resource probe | Representative 10k/100k per country; measure index size, time, RSS, posting tails, retrieval latency | `ml2/index.py`, `scripts/profile_resources.py`, local `reports/resource_profile.md` | Measured rather than assumed fit in available RAM/disk; hours on CPU |
| 3. Full index and baseline | Reuse `data/index/features/model/pipeline/metrics`, full target index, bounded query sample | `ml2/pipeline.py`, `experiments/registry.csv` | All-target validation, complete candidate logs and exact macro F0.5; hours to days on CPU |
| 4. Retrieval diagnostics | Route ablations and K=50/100/200; manually classify sampled misses | `ml2/index.py`, `scripts/retrieval_ablation.py`, `reports/retrieval_error_analysis.md` | Positive recall, extra candidates, latency and RAM per route; hours |
| 5. Training experiments | Fit-only model-mined hard negatives then partition-safe augmentations | `ml2/mining.py`, `ml2/augment.py`, tests | Paired locked-eval gain or revert; hours to days |
| 6. Decision policy | Tune ensemble/calibration; compare CARE on identical candidates; no holdout-driven selection | `ml2/model.py`, `ml2/pipeline.py`, `reports/ablation_table.md` | Macro F0.5 or substantial resource Pareto gain; hours |
| 7. CARE-AR | Conditional second lexical pass; keep zero/one/many outputs and exact union log | `ml2/adaptive.py`, integration tests | Better score/candidate/runtime/RAM frontier; hours |
| 8. Optional learned retrieval/verifier | Only after measured lexical misses or ambiguity; cached target embeddings, licensed compact model | isolated experimental modules and notebook orchestration | End-to-end locked-eval improvement within measured hardware; potentially GPU days |
| 9. Domain shift/final lock | Held-out-country stress, final ablations, saved threshold and submission checks | `MODEL_CARD.md`, `reports/ablation_table.md`, notebook | Validator passes; complete evidence, model card and reproducible artifact |

Dependencies: phases 1→2→3 are hard gates. Phases 4→5→6 precede phase 7. Phase 8 is optional and requires lexical miss evidence. The current `notebooks/Plan_2.ipynb` is a smoke-first runbook; upgrade it as a CLI orchestrator when real-data paths are usable.

**Current execution path:** the official archive is 1,094,823,222 bytes. This agent's connected Drive transfer endpoint refused it with a 268,435,456-byte limit. Colab can mount the owner's Drive directly: `notebooks/Plan2_Training.ipynb` reads the ZIP there and extracts the TSVs to Colab's temporary disk. Phase 1 onward remains unexecuted until the user runs the notebook; the connector limit is not a Colab restriction.
