# Research review and implementation decisions

Reviewed 26 September 2026. Only primary research papers, official technical documentation and competitors' own write-ups support the decisions below. No other repository's code was imported. Benchmark results are not interchangeable with this competition's macro F0.5. Published gains are not predicted gains for this project.

## 1. CaRL-EM — ACL 2026; arXiv posting 1 September 2026

**Primary source:** [ACL paper and proceedings record](https://aclanthology.org/2026.acl-long.1258/), [PDF](https://aclanthology.org/2026.acl-long.1258.pdf). Reviewed the main formulation and experimental setup; PDF pages 2 and 5 were also visually inspected. The proceedings are dated July 2026; September is the arXiv posting date, not a second publication.

The paper models candidate-aware entity matching as a cost-aware sequence of operations. It uses a controller to decide when extra matching or comparison work is worthwhile. Crucially, it studies clean-clean matching: choose one candidate or NONE. Its evaluation uses small retrieved lists and a different metric, with substantial neural backends in the experiments.

**Adopt:** measure which ambiguous cases deserve extra computation; use candidate-set context rather than only isolated pair scores.

**Do not adopt directly:** a top-one output rule, its exact reward/metric, its model sizes or its reported cost savings. Our task allows multiple correct targets and has distinct licensing/size constraints. The first CARE prototype is a small context scorer, not a reproduction of this RL controller.

## 2. UCL-Blocker — Applied Soft Computing, May 2026

**Primary source:** [publisher record and abstract](https://www.sciencedirect.com/science/article/pii/S1568494626002826), DOI `10.1016/j.asoc.2026.114834`. Review level: publisher abstract/highlights, not full-text experimental reproduction.

This work combines unsupervised contrastive blocking with multi-granularity representations, dynamic temperature and hard-negative emphasis. Its relevance is the blocker training problem, not simply replacing the pair classifier with a larger model.

**Adopt as a later experiment:** compact learned retrieval, training-only corruption/augmentation, and negative mining that focuses on confusable records. Use known matching aliases to avoid false negatives. Our raw index routes and tree training are the starting point; the dense contrastive stage has not been implemented.

**Gate:** measure all-target candidate recall at fixed candidate budget and include embedding build time, ANN memory and cached embedding storage. A benchmark win on other datasets is insufficient to promote it.

## 3. IDP-EM — IEEE Access, 30 July 2026

**Primary source:** [IEEE publication record](https://ieeexplore.ieee.org/abstract/document/11630618/), DOI `10.1109/ACCESS.2026.3718155`. Review level: primary abstract and publication metadata.

IDP-EM combines instance-aware prompt tuning, contrastive pseudo-label learning and uncertainty-aware weighting in low-resource entity matching. It treats unreliable pseudo-labels as a training problem rather than assuming every confident model output is correct.

**Adopt the caution:** uncertainty and noisy supervision need explicit treatment. Initially, use genuine provided labels and real hard negatives; do not turn test predictions into trusted labels.

**Defer:** prompt tuning and pseudo-labeling. This challenge may have abundant labeled examples, and pseudo-labels could amplify systematic cross-country mistakes. No self-training module is included in this milestone. CARE disagreement features do not reproduce the paper's uncertainty method or inherit its guarantees.

## 4. SC-Block — supervised contrastive blocking

**Primary source:** [paper](https://arxiv.org/abs/2303.03132), 2023 preprint; [ESWC 2024 publication record](https://madoc.bib.uni-mannheim.de/67274/).

SC-Block learns a record embedding space using supervised contrastive learning and retrieves nearest neighbors to form candidate pairs. It explicitly evaluates the blocker together with downstream matchers, instead of measuring classifier quality alone.

**Adopt:** separate blocker recall from matcher quality; compare complete-pipeline cost; consider cached per-record embeddings as an additional route. This directly challenges the uploaded notebook's blanket rejection of embeddings using an assumed cost for scoring every pair.

**Gate:** index and embedding resource measurements on this dataset, license verification, and grouped training without label-derived normalization leakage. Do not claim the speedups from the paper apply to this challenge.

## 5. WDC Products — realistic entity-matching benchmark design

**Primary source:** [paper](https://arxiv.org/abs/2301.09521).

WDC Products varies corner-case prevalence, unseen entities and development-set size. It shows why one convenient random pair split does not describe all generalization settings.

**Adopt:** separate known-catalog query-disjoint evaluation from stronger unseen-entity/target evaluation. Report nuisance slices such as missing addresses, scripts, near-duplicate brands and country shift. The first code milestone isolates sampled query labels and shared positive targets but intentionally keeps a full searchable target catalog. It must not be mislabeled strict unseen-entity validation.

**Next:** full-graph split construction and an unseen-country stress test using the labeled training countries as proxies. France has no supplied training labels in the source description, so a France score cannot be invented.

## 6. Fine-tuning Large Language Models for Entity Matching — revised May 2025

**Primary source:** [paper](https://arxiv.org/abs/2409.08185).

The study finds that fine-tuning can benefit smaller models and in-domain generalization, while cross-domain transfer can degrade. The effect of generated explanations and example selection is not uniform across models.

**Adopt:** treat cross-country/domain shift as a separate acceptance gate. A larger or more heavily fine-tuned model is not automatically safer for unseen France. Structured field serialization and training-only augmentations are plausible controlled experiments.

**Do not adopt automatically:** closed APIs, externally generated business facts or a checkpoint whose license violates the competition. This milestone uses no LLM data generation and no external business enrichment.

## 7. Foursquare Location Matching — competitor's own 2022 ninth-place write-up

**Primary source:** [author write-up on Kaggle](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/taksai-9th-place-solution).

The author used POI-grouped validation, broad candidate retrieval, a cheaper model stage, richer CatBoost/LightGBM models and post-processing. This is a relevant real-world competition architecture because place/business identity errors can be subtle and the all-pairs space is enormous.

**Adopt:** group related identities, measure retrieval first, and spend richer computation on a manageable shortlist. Preserve the uploaded repository's tree-ensemble direction instead of rebuilding everything around a huge language model.

**Do not transfer:** coordinates or categories absent from our schema, unsafe target encodings, IoU results as F0.5 estimates, or any implementation code. The write-up is historical evidence for an approach, not a verified current rank in Amazon's competition.

## 8. Established training mechanics that solve concrete bugs

**Primary sources:** [LightGBM 4.6 early-stopping documentation](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.early_stopping.html), [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

Validation-based early stopping and held-out calibration are directly applicable to the notebook's fixed-round trees and threshold inconsistency. The calibration data must be distinct from the data used to fit the base model, and its candidate distribution should match the deployed scoring stage.

**Implemented:** early stopping for all three trees, equal-total-query training weights, disjoint calibration and threshold partitions, stored calibration parameters, exact end-to-end macro-F0.5 threshold sweeps, and versioned artifacts. The calibrators optimize logistic loss, while the separate threshold sweep optimizes the competition metric. This is not end-to-end differentiable F0.5 training.

## Prioritized change list

| Order | Change | Reason to do it before a bigger model | Status |
|---|---|---|---|
| 1 | Fix code/document/model mismatch and unsafe empty-field rules | Makes a result interpretable and reproducible | Implemented + tests |
| 2 | Full supplied target index, honest partitions and exact metric | Removes a major source of optimistic validation | Implemented harness; real-data run pending |
| 3 | Seven bounded lexical routes and missingness-aware features | Directly addresses reachable positives and false merges | Implemented; real recall unmeasured |
| 4 | Early stopping, query-balanced loss and saved calibration/thresholds | Aligns training/decision stages without adding large compute | Implemented + synthetic training |
| 5 | CARE candidate-context model | Tests context usefulness cheaply | Implemented prototype; no gain established |
| 6 | Model-mined training hard negatives and split-safe augmentation | Focuses learning on realistic mistakes | Planned |
| 7 | Compact contrastive retriever | Recovers semantic/cross-script cases missed by lexical routes | Planned, resource/license gated |
| 8 | Budgeted verification or second-pass retrieval | Reserves expensive work for difficult cases | Planned, after measured need |

## What this research does not establish

It does not establish a leaderboard target, a percentage-point improvement, a machine-memory guarantee or global novelty of CARE. Newer literature has been screened through the task's one-to-many semantics, offline data restrictions and practical resource limits. An implementation is promoted only by reproducible real-data experiments on the locked evaluation protocol.
