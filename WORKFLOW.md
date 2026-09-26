# ML 2 C 26 — Plan 2 workflow

Research and implementation snapshot: 26 September 2026. This is a separate project from Plan 1. The only inherited code foundation is the uploaded Business Entity Resolution ZIP. No other competitor implementation is imported.

## Objective and evidence standard

Improve end-to-end macro per-query F0.5, including unmatched queries, within measured resource limits. Do not optimize a sampled pairwise validation score and present it as the competition metric. Uploaded scores are unverified; synthetic tests establish software behavior, not competition performance.

## Execution order

| Stage | Deliverable | Acceptance gate |
|---|---|---|
| 0. Audit and provenance | File manifest, notebook-vs-document discrepancy register | Every archived file inventoried; no incompatible source model loaded |
| 1. Correctness baseline | Real named features, strict TSV reader, disk-backed retrieval, tri-tree scorer | Tests pass; empty fields cannot trigger exact-match acceptance; no arbitrary match-count cap |
| 2. Honest training | Component-isolated fit / early-stop / calibration / threshold / holdout partitions | No sampled query or directly/shared-positive connected target group crosses partitions; no validation positive injection; same retrieval policy everywhere |
| 3. Reproducible decisions | Saved preprocessing, model versions, feature schema, calibration and threshold | One inference path; validation and production use identical decisions |
| 4. CARE experiment | Candidate-Aware Risk Estimation meta-model compared with calibrated pair-only baseline | Fit only on separate calibration records; tune only on threshold partition; paired holdout evaluation without choosing on it |
| 5. Full-data feasibility | 10k then 100k index/throughput/RAM/disk benchmark | Extrapolated storage and walltime fit actual available machine; all-target index is complete before final validation |
| 6. Retrieval improvements | Route ablations, larger K, learned contrastive retrieval if feasible | Recall/precision-resource Pareto improvement, measured by country/script/missingness |
| 7. Training improvements | Train-only model-mined hard negatives, grouped cross-fitting, augmentation | Same locked evaluation pool; no validation-derived dictionary, augmentation or target encodings |
| 8. Final lock | Selected configuration, untouched audit, exact candidate log, model card, submission files | Internal strict checks plus supplied validator; no external entity enrichment; licensed models |

## Architecture

```text
Official TSVs -> streaming schema/ID validation -> immutable input fingerprints
            -> disk-backed country/channel index (unlabeled target data only)
Training GT -> connected-query split by shared positive target
            -> fit | early-stop | calibration | threshold-tune | holdout
Each query -> same multi-route candidate generator -> real named pair features
            -> XGBoost + LightGBM + CatBoost -> pair-only calibration OR CARE
            -> stored global threshold -> optional deterministic target ownership
            -> exact scored candidates + accepted matches -> strict validation
```

The index may contain all allowed target records, including held-out targets: they must be searchable at deployment. No held-out labels enter features, dictionaries, model fitting or calibration. Corpus posting counts are explicitly transductive **retrieval** statistics, not an inductive learned-label claim. The baseline does not learn a transliteration dictionary.

## Training and evaluation contract

Assign positive-connected groups within the sampled queries deterministically to five disjoint partitions (60/10/10/10/10). This is an initial, easy-to-audit alternative to more complex nested cross-fitting. The training sample is a stable-hash sample across the entire source, not a file prefix or country-specific row offset. Training and all evaluation use the full supplied target index. The initial implementation sets an explicit pair-count memory guard; it does not claim out-of-core GBDT training.

Base trees fit only on `fit` and early-stop on `early`. A pair-only sigmoid and CARE context model fit on `calibration`. Each receives its own macro-F0.5 threshold on `tune`. Both locked policies are evaluated once on `holdout`; this is a predeclared comparison, not an invitation to repeatedly select on holdout. Do not refit on calibration/tune/holdout and reuse the old calibrator. A later fold-ensemble finalization requires another clean calibration design.

Report: macro F0.5; singleton correctness; pair precision/recall; per-query candidate recall; fraction of queries with all positives retrieved; candidate-count p50/p95/max; runtime; peak RSS; disk size; country/script/missing-field slices; all seeds, input fingerprints and configuration. Predictions must include queries with no candidates. A rule-based override is not exempt from evaluation or candidate logging.

## Resource policy

Default: CPU-first, bounded candidate buffers, SQLite on local SSD, two model threads in smoke tests. This is not a claim of 16 GB full-data feasibility. Large posting tables can move a RAM problem to disk; measure 10k/100k samples and extrapolate before a multi-million-row build. Training has an explicit maximum number of retained queries and pairs. Neural retrieval is optional: encode each record once, cache embeddings, then ANN; do not confuse this with running a cross-encoder on every possible pair. Check model license and the supplied 8B parameter limit before any checkpoint download.

## Proposed experiments, in priority order

1. Rebuilt honest baseline versus the uploaded notebook's architecture, without reproducing its validation shortcuts.
2. Multi-channel lexical retrieval versus name/postal-only; K = 50/100/200 with all-target negatives.
3. Separate calibration versus fixed country thresholds. Unknown countries use the learned global policy, not a fabricated country threshold.
4. CARE context versus pair-only calibration, same base models and candidates.
5. Hard negatives mined using fit-only or out-of-fold training scores. Keep all realistic validation candidates unchanged.
6. Training-only perturbations (typos, reordered tokens, accents, missing address) with original and augmented examples in the same split.
7. Compact contrastive retriever as an additional candidate route, not an immediate replacement. Mine false-positive-like negatives while avoiding known positive aliases.
8. Budgeted cross-encoder or adaptive second-pass retrieval only for uncertain cases, after cost and licensing checks.

Do not silently add test pseudo-labels, external POI lookups, geocoding, public business registries, other competition datasets, or a universal top-one-per-query constraint. Multiple valid matches are part of this task.

## Current status

This document was created before implementation. Final executed steps and test results are recorded in `evidence/STATUS.md`; planned stages are not implied to be completed. Remote repository creation requires a repository-creation-capable authorized connection. No changes are authorized to Plan 1.

## Protocol limitations

This first milestone is query-disjoint with a known searchable target catalog. Target records can appear as negatives for training queries and as positives for held-out queries; this is not strict unseen-target evaluation. Full graph components through unsampled queries, script/missingness evaluation slices, paired bootstrap intervals and final packaging remain work to do. A complete index denotes all explicitly supplied files, not independent verification of the official corpus completeness.
