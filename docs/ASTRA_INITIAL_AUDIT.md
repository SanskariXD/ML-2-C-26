# Plan 2 initial audit — 26 September 2026

## Repository reality and provenance

At the start of this work, remote `main` contained **only** a 172-byte placeholder `README.md`; it had one branch, `main`. The intended Plan 2 implementation was in the previously uploaded `ML-2-C-26.zip`, not in the remote. This commit imports that archive into **this** repository and applies the small changes listed below. Plan 1 was neither read as a source of implementation nor modified.

The archive contains 47 files: Python modules in `ml2/`, tests, scripts, a Colab runbook, workflow/research/audit documents, evidence from a prior synthetic run, and an attributed source snapshot under `reference/source_snapshot/`. **One original source PDF was excluded from GitHub by automatic upload review**, so the remote contains 46 of those 47 files at import. The uploaded standalone `ML-2-C-26_Workflow_and_Research.md` describes the same project, but does not replace the repository's `WORKFLOW.md`, `docs/RESEARCH.md`, or executable code. `reference/manifest.json` records the expected archive provenance, including the excluded PDF; it is not a list of files all present remotely.

## Architecture actually implemented

`ml2.data` streams strict TSVs, samples S1 queries by stable hash, and separates sampled queries linked by shared positive targets. `ml2.index` builds a seven-channel, country-scoped SQLite index over all supplied S2/S3 targets; query retrieval is capped by posting/key budgets and K. `ml2.features` computes 34 named pair features. `ml2.model` fits XGBoost, LightGBM and CatBoost, then fits separate pair and CARE logistic scorers. `ml2.pipeline` trains on fit/early/calibration/tune partitions, scores a holdout, and writes scored pairs and TSVs at inference. `ml2.metrics` implements macro per-query F0.5 with unmatched and multi-match queries.

The target catalog remains searchable during all partitions. This is **known-catalog, sampled-query-disjoint validation**, not fully unseen-entity validation. The archive's synthetic smoke and test evidence are software checks, not real challenge performance. A complete index only means all *supplied* target files were read.

## Evidence and missing work

| Item | Evidence now | State |
|---|---|---|
| Regression tests | 37 pass in this environment after import and two new profile tests | TESTED_SYNTHETIC |
| End-to-end smoke | Reran locally: 360 synthetic queries, 1,336 targets, 14,400 scored pairs; supplied validator exit 0 | TESTED_SYNTHETIC |
| Saved source weights | Feature dimensions conflict with archived notebook; never loaded | REJECTED |
| Full official dataset | Prior Drive ZIP identified at about 1.09 GB; connected transfer rejects files above 256 MB | BLOCKED |
| Real profile/index/baseline | No real TSVs in repository; no run performed | BLOCKED |
| CARE/CARE-AR gain | CARE implemented, CARE-AR not implemented; no real paired comparison | PLANNED |
| Dense retrieval, hard negatives, augmentation | Described in research/workflow; no implementation | PLANNED |

The official dataset must remain outside GitHub. The new `profile-data` command writes aggregate `data_profile.json` and `data_profile.md` locally after checking duplicate IDs, label coverage and target references. No real report has been manufactured.

## Code/document inconsistencies and risks

1. Archived README said remote creation was pending; the remote now exists but had no implementation. Update its status.
2. The pre-import threshold sweep initialized “predict nothing” with a threshold *below* 1.0. A false candidate scored exactly 1.0 would be accepted. Fixed to a representable threshold above 1.0 and regression-tested.
3. `component_split` isolates only links visible among sampled queries. Links through unsampled queries can cross partitions.
4. `train` retains per-query arrays up to `max_pairs`; its guard does not guarantee a particular RAM budget.
5. SQLite postings have a fixed per-key cap, so a frequent legitimate token can disappear completely from candidate generation.
6. Retrieval blocks strictly by country. Mislabelled or missing country can make a true match unreachable.
7. Probe indexing takes the first rows of each target file; ordering may distort resource extrapolation.
8. Optional target ownership is tested as a helper but not calibrated or enabled in production.
9. Source documentation's historical scores have not been reproduced; none should be called a Plan 2 baseline.
10. The archived source contains attributed third-party material without a source LICENSE; keep the remote private and review redistribution before public release.
11. `reference/source_snapshot/docs/amazon_ml_challenge_problem_statement.pdf` is not in GitHub: an automatic review rejected its upload for possible private communications/access links. The original attachment remains outside the repository. Do not infer it is present from the archive manifest.

## Ten highest-value next improvements

1. Obtain official TSVs via a transfer method that handles the >256 MB ZIP; validate competition rules and file schema.
2. Run `profile-data` and inspect missingness, scripts, cardinality, duplicate IDs and label coverage.
3. Profile representative 10k/100k indexes with country-stratified samples and measured walltime/RSS/disk.
4. Build the complete allowed target index and run one honest baseline with full target universe.
5. Save exact candidates and per-query errors; report retrieval recall at K=50/100/200.
6. Add route-level retrieval ablation and inspect missed positive aliases.
7. Build a strict full-positive-graph or unseen-entity stress split, clearly separate from known-catalog validation.
8. Mine fit-only false positives, compare retraining on the locked evaluation protocol.
9. Ablate training-only perturbations and calibrators after retrieval is measured.
10. Compare CARE, adaptive second pass, and optional compact embeddings only when observed errors and compute justify them.

## Test record for this import

`python -m pytest -q` → 37 passed on Python 3.12 with pinned dependencies. `python -m scripts.run_smoke` → synthetic end-to-end success and validator exit 0. The threshold regression covers score 1.0; the profile tests cover zero/multiple matches, Unicode script counts and unknown GT targets. No real-data score or leaderboard result is claimed.
