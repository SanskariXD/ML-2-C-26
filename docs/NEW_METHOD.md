# CARE: Candidate-Aware Risk Estimation

**Status:** implemented experimental decision model, synthetic-tested. Not a proven competition improvement, not a claim of worldwide research novelty. This is a new method within this repository, assembled from established calibration and candidate-context ideas. The default deployed policy remains pair-only until real validation supports changing it.

## Why add this instead of immediately using a larger model?

A pair score hides useful context. A score of 0.88 with agreement across three tree learners and several independent retrieval channels is different from 0.88 produced by one uncertain route in a crowded candidate set. Missing addresses and contradictory number fields also change the reliability of apparent name agreement. The proposed hypothesis is that a small context model can learn some of these reliability differences without a large neural inference bill.

This is a hypothesis, not a hard rule that crowded candidate sets are always bad. A business may legitimately have many S2/S3 records. CARE must not penalize every second match simply because another match already exists.

## Implemented version

Let the three base models produce p_X, p_L and p_C for candidate c of query q. The baseline score uses p = 0.40 p_X + 0.35 p_L + 0.25 p_C, followed by a sigmoid fitted on a disjoint calibration partition.

CARE uses a standardized, L2-regularized logistic model over 14 inputs:

- Pair log-odds, ensemble standard deviation, minimum score and maximum score.
- Log candidate count, mean candidate score for this query, and fraction of candidates scoring at least 0.5.
- Count of retrieval channels supporting this candidate.
- Missing name, missing address, numeric contradiction and postal-like contradiction indicators.
- Retrieval-channel count × pair score, and ensemble disagreement × log candidate count.

Its learned score is `sigmoid(b + w · standardized_context)`. It is a decision score, not a calibrated uncertainty guarantee. Correlated tree errors can produce confident agreement; all three learners agreeing does not establish truth. The implementation does not call external services or use a pretrained language model.

Both calibration variants fit on the same separate calibration partition. They receive separate thresholds chosen on the threshold-tuning partition by exact macro per-query F0.5, including singletons and unretrieved true targets. The full held-out candidate set is retained; there is no injected true candidate and no negative downsampling there.

## Why this preserves the actual task

Each query may accept zero, one or many candidates. There is no arbitrary cap of ten and no within-query softmax or top-one selection. A zero-candidate query outputs an empty list and remains in metric denominators. All candidates actually scored are logged, even those rejected.

Target ownership is a **different** constraint: one S2/S3 fragment may be restricted to one S1 owner after the ground truth supports that assumption. A deterministic ownership helper is tested, but it is not enabled in the released training/inference policy. It needs end-to-end retuning with the same full-query population. Do not present that helper as a global optimum.

## Evaluation protocol

The primary comparison is CARE versus a pair-only calibrated ensemble, with identical retrieval, base learners, sample, and target pool. Compare per-query score differences, singleton false merges, low-address-information records, and countries. Bootstrap query groups for uncertainty on real data; do not report synthetic bootstrap intervals as a competition finding. A future entity-heldout split should eliminate target-record exposure across training and evaluation, unlike the initial known-catalog query-disjoint protocol.

CARE must earn its complexity. Keep the baseline if CARE adds false merges, has unstable threshold sensitivity, or shows no reproducible benefit. The synthetic smoke resulted in an equal holdout score for the two variants; it does **not** establish a CARE gain.

## Later extension: budgeted second-pass retrieval

Not implemented. Route low-support or contradiction-heavy cases to extra character retrieval or a compact offline contrastive index, then recompute context and calibrate the complete two-pass policy. The candidate log must include the union actually scored in both passes. Learn the routing decision and its budget on training/development data, never by looking at test identities online.

CaRL-EM motivates cost-aware use of expensive operations, but its published setup is clean-clean matching with at most one correct candidate. That selection constraint is incompatible with this competition. We borrow the design question—where is extra compute useful?—not its top-one output rule, model choices or reported performance. See the primary-paper review in `RESEARCH.md`.
