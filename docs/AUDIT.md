# Uploaded repository audit

Audit date: 26 September 2026. Scope: the single uploaded `Amazon-ML-Challenge-2026-Business-Entity-Resolution-main(7).zip`. Cell references below are zero-based notebook cell indices. Archived originals are under `reference/source_snapshot/`; per-file hashes and sizes are in `reference/manifest.json`.

## Finding

The uploaded snapshot contains a useful architecture idea—country-aware blocking, fuzzy pair features, and a tree ensemble—but is **not a reproducible implementation of everything its documentation claims**. Rebuilding missing modules from that design is necessary. This does not establish that another, unpublished version never implemented the claims.

There are 14 regular files and six directory entries in the archive. It contains one notebook, four saved model/metadata files, three top-level Markdown documents, two competition PDFs, requirements, a `.gitignore`, the student-resource README, and the supplied submission validator. The documented modular `code/business_entity_resolution/src/` files, experiment logs, train/test TSVs and LICENSE are absent. The root README begins inside a mathematical expression. The original notebook has 24 cells, 11 code cells, no executed-cell counts and no stored outputs.

## Priority discrepancy register

| Priority | Observed evidence | Consequence | Plan 2 treatment |
|---|---|---|---|
| P0 | Architecture guide links local `file:///f:/AmazonML/code/...` modules; none are in this archive | README commands are not reproducible from this snapshot | New modules are explicitly a reconstruction, not a recovered original implementation |
| P0 | Notebook cell 12 builds 28 values, then pads to 72 using zeros | Claimed 72-dimensional representation overstates computed information | 34 named real features; missingness explicit; feature-name hash checked on load |
| P0 | XGBoost model header says 55 features; LightGBM max_feature_idx is 54; metadata lists 55 feature names | Saved weights do not match the notebook's 72-column vectors | Do not load source models; retain only as provenance; new model manifest |
| P0 | Cells 8/14 make smaller label-conditioned target pools, inject positives and sample negatives before OOF calibration | Candidate-miss and false-positive pressure differ from deployment; reported OOF decision score is not end-to-end | All supplied targets searchable; realistic retrieval in every partition; no injected validation positives |
| P0 | Cell 14 finds `best_t_global`, but cells 16/20 use CONFIG thresholds | Threshold selection and actual decisions disagree | Save threshold for each policy; both validation and inference read the same artifact |
| P0 | Cell 14 fits a logistic meta-model on base OOF predictions and scores those same rows | Base OOF is not automatically second-level OOF; meta evaluation is in-sample | Separate calibration partition; no in-sample meta score claimed |
| P0 | Cells 16/20 accept exact name/address matches as a rule | Empty-normalized fields can satisfy equality; rule overrides bypass uncertainty | Nonempty exact features only; no unconditional acceptance overrides |
| P1 | Cell 6 uses all six train/test sources for token statistics before splitting | Transductive retrieval information is conflated with leakage-free inductive claims | Clearly label allowed unlabeled target-index counts; no test labels or learned dictionaries |
| P1 | Cell 8 selects fixed row offsets and uses S1-only grouping | File ordering, shared targets and country proportions may bias validation | Whole-file bottom-hash sampling; selected queries sharing positive targets grouped; split balance logged |
| P1 | Cell 10 implements name-token and postal index routes; many routes in prose are absent | Claimed 12-route recall cannot be attributed to this notebook | Seven concrete routes, bounded and named; full-data recall still unmeasured |
| P1 | Cell 6 uses punctuation cleanup, with no implemented multi-script transliteration despite descriptions | Cross-script matches may be missed; Indic combining marks may be lost by naive regex | Preserve Unicode letters/marks; separate Latin accent folding; general Indic transliteration remains planned |
| P1 | Training retrieval is 50 candidates while inference uses up to 75 | Calibration sees a different candidate distribution | One stored retrieval configuration, reused across stages |
| P1 | CONFIG caps matches per query at 10 while the notebook comments GT can reach 11 | Valid matches can be discarded without support in the task specification | No arbitrary accepted-match cap |
| P1 | All-country scans and large candidate/match dictionaries in cell 20 | Batched model calls do not make the whole pipeline memory-bounded | SQLite index, batched scoring and disk pair log; measure disk and training RAM separately |
| P1 | Notebook embedding rejection uses an assumed per-pair cost | Does not evaluate cached per-record embeddings plus ANN | Compact contrastive retrieval is a later measured experiment, not a blanket rejection |
| P1 | Saved model metadata contains no linked threshold, split fingerprint or preprocessor version | Old artifact cannot be safely reproduced or identified | New manifest includes versions, features, calibrators, thresholds, inputs and split hash |
| P2 | Requirements use minimum versions rather than exact versions | Results can drift between installations | Pin exercised core dependencies; manifest rejects runtime version mismatch |
| P2 | Cell 20 parses rows with `strip().split('\t')` | Trailing missing fields may be lost | Strict CSV/TSV parser with proper quoting and empty columns |
| P2 | README references MIT/Apache licensing but no LICENSE exists | Public redistribution rights of source code are not established by library licenses | Preserve attribution, keep private by default; no invented source license |

## Documentation claims versus notebook implementation

The guide discusses 55 features; the methodology alternates between 27, 55 and 72; the notebook produces 28 computed values in 72 slots. The guide describes TF-IDF routes, transliteration bridges, cross-source transitive expansion, detailed benchmark tables, a richer module tree and saved training artifacts. Those components are not established by the uploaded notebook. LightGBM is described as using GOSS, but the visible constructor does not enable the relevant sampling mode.

Reported scores also differ across files: examples include 0.98006 in the root README, 0.96880 in the notebook's prose and higher experiment claims in the methodology. These are **source self-reports**, not independently reproduced scores. No saved notebook output, split manifest or complete runnable code supports comparing them fairly to another repository. Plan 2 begins without adopting any of them as its baseline.

A source-only code audit cannot prove the original author's intent or identify what ran elsewhere. The correct conclusion is an evidence/reproducibility gap, not an accusation of fabricated results.

## Competition contract extracted from the supplied PDFs

The problem statement describes a deduplicated S1 reference and zero/one/many S2/S3 matches. It requires all S1 query rows, proper TSV headers, no duplicate IDs within lists, and candidate lists corresponding to the final pairs the ML model actually scores. Final matches should be a subset of that candidate list. The supplied validator merely warns on a subset violation; Plan 2 treats it as a failure. See problem statement pages 1–5 and `student_resource/utils/validate_submission.py`.

Evaluation is macro per-query F0.5, including special singleton handling. A correct empty prediction for an unmatched query scores 1; a false merge on it scores 0. For a matched query the equivalent count formula is:

`F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP)`

The PDFs specify the final model license/size constraint (MIT or Apache 2.0; at most 8B parameters) and prohibit external business-data enrichment. Internet literature research is not the same as calling business registries, geocoders or search engines to resolve test records. Plan 2 has no external enrichment code.

The source's generic guidelines and specific statement differ on methodology length and leaderboard wording. A short executive summary plus full technical appendix is safer than relying on either contradictory prose passage; verify the organizer's live instructions before final submission. No source assertion of a CPU-only or fixed-hour budget is accepted as an official hardware restriction here.

## Limits of this audit and implementation

The actual dataset was not supplied in this ZIP and no real-data score, full-scale memory measurement or training run has been completed. No model binary from the source was executed. Header/metadata inspection establishes schema mismatch, not the binaries' training provenance. The source documentation's measured dataset statistics, cardinality law and runtime claims still need independent checks.

The new implementation intentionally remains a first milestone: its SQLite index needs full-scale performance profiling, its training arrays have a configured memory guard rather than out-of-core boosting, its country blocking assumes country labels are reliable, and its postal-like feature is not a full address parser. It does not yet implement learned transliteration, hard-negative mining, dense retrieval, nested OOF stacking or the full organizer submission packager.

The initial split isolates sampled S1 queries and shared positive target IDs. Full target records still appear in the searchable catalog and can appear as negatives for other training queries. This is a **known-catalog query-disjoint protocol**, not a claim of fully unseen-target/entity-disjoint generalization. Cross-query components connected only through unsampled queries are not inferred. A stricter full-graph split is a separate planned experiment.
