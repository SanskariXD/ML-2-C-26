"""Exact query-macro F0.5 and efficient threshold sweep over all retrieved pairs."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def query_f05(truth: set[str], prediction: set[str]) -> float:
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    fp, fn = len(prediction - truth), len(truth - prediction)
    denominator = 1.25 * tp + .25 * fn + fp
    return 1.25 * tp / denominator if denominator else 0.


def macro_f05(truth: dict[str, set[str]], predictions: dict[str, set[str]]) -> float:
    if not truth:
        raise ValueError("Cannot evaluate an empty query set")
    if set(predictions) != set(truth):
        raise ValueError("Prediction queries must match truth exactly, including singletons")
    return float(np.mean([query_f05(truth[q], predictions[q]) for q in truth]))


def resolve(pairs: list[tuple[str, str, float]], threshold: float, query_ids, exclusive: bool = False) -> dict[str, set[str]]:
    result = {q: set() for q in query_ids}
    winners: dict[str, tuple[float, str]] = {}
    for q, target, p in pairs:
        if q not in result or not np.isfinite(p):
            raise ValueError("Unknown query or nonfinite score")
        if p < threshold:
            continue
        if not exclusive:
            result[q].add(target)
        elif target not in winners or (-p, q) < (-winners[target][0], winners[target][1]):
            winners[target] = (p, q)
    if exclusive:
        for t, (_, q) in winners.items():
            result[q].add(t)
    return result


def threshold_sweep(truth: dict[str, set[str]], pairs: list[tuple[str, str, float]]) -> tuple[float, float]:
    """O(P log P) sweep; no injected positives, no capped match count.

    Threshold policy is independent-edge acceptance. Target-exclusive resolution is
    a separately tested optional policy and is not silently added at inference.
    """
    if not truth:
        raise ValueError("Empty tuning partition")
    unique = {(q, t) for q, t, _ in pairs}
    if len(unique) != len(pairs):
        raise ValueError("Duplicate scored pair")
    if any(q not in truth or not np.isfinite(p) or p < 0 or p > 1 for q, _, p in pairs):
        raise ValueError("Invalid tuning pair")
    tp, fp = defaultdict(int), defaultdict(int)
    current = {q: float(not actual) for q, actual in truth.items()}
    total = sum(current.values())
    best_score, best_threshold = total / len(truth), float(np.nextafter(1., 2.))
    ordered = sorted(pairs, key=lambda row: (-row[2], row[0], row[1]))
    i = 0
    while i < len(ordered):
        threshold = ordered[i][2]
        while i < len(ordered) and ordered[i][2] == threshold:
            q, t, _ = ordered[i]
            total -= current[q]
            tp[q] += int(t in truth[q])
            fp[q] += int(t not in truth[q])
            denominator = 1.25 * tp[q] + .25 * (len(truth[q]) - tp[q]) + fp[q]
            current[q] = 1.25 * tp[q] / denominator if truth[q] else 0.
            total += current[q]
            i += 1
        score = total / len(truth)
        if score > best_score + 1e-12:
            best_score, best_threshold = score, threshold
    return float(best_threshold), float(best_score)


def report(truth, predictions, candidates) -> dict:
    if set(candidates) != set(truth):
        raise ValueError("Candidate query coverage mismatch")
    singleton = [q for q in truth if not truth[q]]
    positive = [q for q in truth if truth[q]]
    counts = np.array([len(candidates[q]) for q in truth])
    tp = sum(len(truth[q] & predictions[q]) for q in truth)
    fp = sum(len(predictions[q] - truth[q]) for q in truth)
    fn = sum(len(truth[q] - predictions[q]) for q in truth)
    return {
        "queries": len(truth), "macro_f05": macro_f05(truth, predictions),
        "singleton_accuracy": float(np.mean([not predictions[q] for q in singleton])) if singleton else None,
        "pair_precision": tp / (tp + fp) if tp + fp else None,
        "pair_recall": tp / (tp + fn) if tp + fn else None,
        "retrieved_positive_fraction": sum(len(truth[q] & candidates[q]) for q in truth) / (tp + fn) if tp + fn else None,
        "query_macro_candidate_recall": float(np.mean([len(truth[q] & candidates[q]) / len(truth[q]) for q in positive])) if positive else None,
        "all_positives_retrieved_fraction": float(np.mean([truth[q] <= candidates[q] for q in positive])) if positive else None,
        "candidate_count_p50": float(np.quantile(counts, .5)), "candidate_count_p95": float(np.quantile(counts, .95)),
        "candidate_count_max": int(counts.max()), "zero_candidate_queries": int((counts == 0).sum()),
    }
