"""Named, non-padded numeric features with an artifact compatibility hash."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from rapidfuzz import fuzz
from .data import Record
from .index import Candidate
from .text import CHANNELS, NORMALIZER_VERSION, trigrams, view

FEATURE_NAMES = (
    "name_ratio", "name_token_sort", "name_token_set", "name_jaccard", "name_exact", "root_exact",
    "address_ratio", "address_token_sort", "address_jaccard", "address_exact",
    "name_length_ratio", "address_length_ratio", "number_jaccard", "number_conflict",
    "postal_like_agree", "postal_like_conflict", "postal_like_both_present",
    "name_missing", "address_missing", "country_equal", "latin_name_ratio", "latin_address_ratio",
    "name_trigram_jaccard", "retrieval_log_score", "route_count", "name_address_product",
    "minimum_name_address", *[f"route_{c}" for c in CHANNELS],
)
SCHEMA_HASH = hashlib.sha256(json.dumps([NORMALIZER_VERSION, FEATURE_NAMES]).encode()).hexdigest()


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a and b else 0.


def similarity(a: str, b: str, fn=fuzz.ratio) -> float:
    return fn(a, b) / 100. if a and b else 0.


def exact(a: str, b: str) -> float:
    return float(bool(a and b) and a == b)


def length_ratio(a: str, b: str) -> float:
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.


def pair_features(query: Record, candidate: Candidate) -> np.ndarray:
    a, b = view(query), view(candidate.record)
    nr, ar = similarity(a.name, b.name), similarity(a.address, b.address)
    vals = [nr, similarity(a.name, b.name, fuzz.token_sort_ratio), similarity(a.name, b.name, fuzz.token_set_ratio),
            jaccard(a.name_tokens, b.name_tokens), exact(a.name, b.name), exact(a.root, b.root),
            ar, similarity(a.address, b.address, fuzz.token_sort_ratio), jaccard(a.address_tokens, b.address_tokens),
            exact(a.address, b.address), length_ratio(a.name, b.name), length_ratio(a.address, b.address),
            jaccard(a.numbers, b.numbers), float(bool(a.numbers and b.numbers) and not (a.numbers & b.numbers)),
            float(bool(a.postals & b.postals)), float(bool(a.postals and b.postals) and not (a.postals & b.postals)),
            float(bool(a.postals and b.postals)), float(not a.name or not b.name), float(not a.address or not b.address),
            exact(a.country, b.country), similarity(a.latin_name, b.latin_name), similarity(a.latin_address, b.latin_address),
            jaccard(trigrams(a.name), trigrams(b.name)), np.log1p(candidate.score), len(candidate.routes), nr * ar, min(nr, ar)]
    vals.extend(float(channel in candidate.routes) for channel in CHANNELS)
    x = np.asarray(vals, dtype=np.float32)
    if len(x) != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Feature schema/finite-value violation")
    return x


def feature_matrix(query: Record, candidates: list[Candidate]) -> np.ndarray:
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([pair_features(query, c) for c in candidates])


CARE_NAMES = ("pair_logit", "ensemble_std", "ensemble_min", "ensemble_max", "log_candidate_count",
              "query_mean_score", "query_high_score_fraction", "route_count", "name_missing", "address_missing",
              "number_conflict", "postal_like_conflict", "route_agreement_x_pair", "uncertainty_x_density")


def context_features(x: np.ndarray, model_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """One query at a time. Candidate context is unlabeled and never top-one-only.

    Ensemble disagreement is a heuristic, not a calibrated uncertainty bound.
    The output is a learned decision score, not a guaranteed probability under shift.
    """
    if len(x) == 0:
        return np.empty((0, len(CARE_NAMES)), dtype=np.float32)
    p = np.clip(model_probs @ weights, 1e-6, 1 - 1e-6)
    std = model_probs.std(axis=1)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    routes = x[:, idx["route_count"]]
    return np.column_stack([
        np.log(p / (1 - p)), std, model_probs.min(axis=1), model_probs.max(axis=1),
        np.full(len(x), np.log1p(len(x))), np.full(len(x), p.mean()), np.full(len(x), (p >= .5).mean()),
        routes, x[:, idx["name_missing"]], x[:, idx["address_missing"]], x[:, idx["number_conflict"]],
        x[:, idx["postal_like_conflict"]], routes * p, std * np.log1p(len(x)),
    ]).astype(np.float32)
