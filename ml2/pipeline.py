"""Bounded training harness and batch-scored, disk-logged inference."""
from __future__ import annotations
import csv
import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .data import (Record, component_split, digest_file, read_ground_truth,
                   read_records, sample_queries, validate_source_ids, write_json)
from .features import FEATURE_NAMES, feature_matrix
from .index import CandidateIndex
from .metrics import report, resolve, threshold_sweep
from .model import Ensemble, fit_logistic, logistic_predict

@dataclass
class QueryBlock:
    record: Record
    targets: list[str]
    x: np.ndarray
    y: np.ndarray
    split: str


def flatten(blocks: list[QueryBlock]):
    nonempty = [b for b in blocks if len(b.x)]
    if not nonempty:
        raise ValueError("Partition has no retrieved pairs; inspect blocking")
    x = np.vstack([b.x for b in nonempty])
    y = np.concatenate([b.y for b in nonempty])
    # Equal total weight per nonempty query, not equal per pair.
    w = np.concatenate([np.full(len(b.x), 1. / len(b.x)) for b in nonempty])
    w *= len(w) / w.sum()
    return x, y, w


def designs(model: Ensemble, blocks: list[QueryBlock], mode: str):
    nonempty = [b for b in blocks if len(b.x)]
    x, y, w = flatten(nonempty)
    probabilities = model.probabilities(x)
    outputs, offset = [], 0
    for block in nonempty:
        n = len(block.x)
        outputs.append(model.design(block.x, probabilities[offset:offset + n], mode))
        offset += n
    return np.vstack(outputs), y, w


def scored_pairs(model: Ensemble, blocks: list[QueryBlock], mode: str):
    if not any(len(b.x) for b in blocks):
        return []
    z, _, _ = designs(model, blocks, mode)
    p = logistic_predict(z, model.calibrators[mode])
    pairs, offset = [], 0
    for b in blocks:
        pairs.extend((b.record.entity_id, t, float(score)) for t, score in zip(b.targets, p[offset:offset + len(b.targets)]))
        offset += len(b.targets)
    return pairs


def train(index_path, queries_path, truth_path, out, *, max_queries=25000, max_pairs=1500000,
          k=100, max_postings=2000, keys_per_channel=4, seed=26, rounds=250, threads=2):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Training output is not empty; use a new experiment directory")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    index = CandidateIndex(index_path)
    try:
        index.assert_usable()
        validate_source_ids(queries_path, index.db)
        queries = sample_queries(queries_path, max_queries, seed)
        gt = read_ground_truth(truth_path, {q.entity_id for q in queries})
        splits = component_split(gt, seed)
        split_names = ("fit", "early", "calibration", "tune", "holdout")
        counts = Counter(splits.values())
        if any(counts[s] == 0 for s in split_names):
            raise ValueError(f"Empty query partition: {dict(counts)}. Increase max_queries.")
        for query in queries:
            for target in gt[query.entity_id]:
                if index.get(target) is None:
                    raise ValueError(f"GT target {target} absent from full supplied index")
        blocks, npairs = [], 0
        for query in queries:
            cand = index.retrieve(query, k, max_postings, keys_per_channel)
            npairs += len(cand)
            if npairs > max_pairs:
                raise MemoryError(f"Pair limit {max_pairs} exceeded. Reduce max_queries; do not shrink validation target pool.")
            targets = [c.record.entity_id for c in cand]
            blocks.append(QueryBlock(query, targets, feature_matrix(query, cand),
                                     np.array([t in gt[query.entity_id] for t in targets], dtype=np.int32), splits[query.entity_id]))
        groups = {s: [b for b in blocks if b.split == s] for s in split_names}
        xf, yf, wf = flatten(groups["fit"])
        xe, ye, we = flatten(groups["early"])
        model = Ensemble().fit(xf, yf, wf, xe, ye, we, rounds=rounds, threads=threads)
        model.config = {"k": k, "max_postings": max_postings, "keys_per_channel": keys_per_channel,
                        "seed": seed, "target_exclusive": False, "default_policy": "pair",
                        "query_weighting": "equal-total-weight-per-nonempty-query"}
        results = {"evidence_kind": "dataset_evaluation_not_leaderboard", "pair_count": npairs,
                   "target_index_records": index.metadata()["records"], "partition_query_counts": dict(counts),
                   "partition_slices": {}, "policies": {}}
        for split in split_names:
            results["partition_slices"][split] = {
                "countries": dict(Counter(b.record.country for b in groups[split])),
                "singletons": sum(not gt[b.record.entity_id] for b in groups[split]),
                "zero_candidate_queries": sum(not b.targets for b in groups[split]),
                "retrieved_positive_pairs": sum(int(b.y.sum()) for b in groups[split]),
                "retrieved_negative_pairs": sum(int((1 - b.y).sum()) for b in groups[split]),
            }
        # Predeclared pair-only and CARE comparison. No holdout-based automatic selection.
        for mode in ("pair", "care"):
            z, y, w = designs(model, groups["calibration"], mode)
            model.calibrators[mode] = fit_logistic(z, y, w)
            tuning_truth = {b.record.entity_id: gt[b.record.entity_id] for b in groups["tune"]}
            tune_pairs = scored_pairs(model, groups["tune"], mode)
            threshold, score = threshold_sweep(tuning_truth, tune_pairs)
            model.thresholds[mode] = threshold
            held = groups["holdout"]
            held_truth = {b.record.entity_id: gt[b.record.entity_id] for b in held}
            held_candidates = {b.record.entity_id: set(b.targets) for b in held}
            held_pairs = scored_pairs(model, held, mode)
            held_predictions = resolve(held_pairs, threshold, held_truth)
            evaluation = report(held_truth, held_predictions, held_candidates)
            evaluation["by_country"] = {}
            for country in sorted({b.record.country for b in held}):
                ids = {b.record.entity_id for b in held if b.record.country == country}
                evaluation["by_country"][country] = report({q: held_truth[q] for q in ids},
                     {q: held_predictions[q] for q in ids}, {q: held_candidates[q] for q in ids})
            results["policies"][mode] = {"threshold": threshold, "tune_macro_f05": score, "holdout": evaluation}
            write_json(out / f"holdout_{mode}.json", {q: sorted(v) for q, v in held_predictions.items()})
        results["elapsed_seconds"] = time.perf_counter() - start
        results["constant_fit_features"] = [name for i, name in enumerate(FEATURE_NAMES) if np.ptp(xf[:, i]) == 0]
        model.manifest = {"training_queries_sha256": digest_file(queries_path), "ground_truth_sha256": digest_file(truth_path),
                          "training_index": index.metadata(), "sampled_query_count": len(queries),
                          "split_hash": __import__("hashlib").sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
                          "audit_note": "Fit/early/calibration/tune/holdout isolated among sampled queries and shared positive targets; no scored holdout labels used for fitting."}
        model.save(out / "model")
        write_json(out / "splits.json", splits)
        write_json(out / "metrics.json", results)
        return model, results, blocks
    finally:
        index.close()


def predict(index_path, queries_path, model_path, out, mode="pair", batch_pairs=4096):
    """All queries logged on disk; exact candidates = every scored pair.

    Batches are only execution units. Context is computed independently per query,
    so changing batch size must not change decisions.
    """
    if batch_pairs < 1:
        raise ValueError("batch_pairs must be positive")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prediction output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    model = Ensemble.load(model_path)
    if mode not in model.thresholds:
        raise ValueError("Policy lacks a stored threshold")
    index = CandidateIndex(index_path)
    log = sqlite3.connect(str(out / "scored_pairs.sqlite"))
    log.executescript("""CREATE TABLE queries(n INTEGER PRIMARY KEY, id TEXT UNIQUE);
    CREATE TABLE scored(q TEXT, t TEXT, p REAL, PRIMARY KEY(q,t)) WITHOUT ROWID;""")
    batch, count, order = [], 0, 0
    threshold = model.thresholds[mode]
    def flush():
        if not batch:
            return
        nonempty = [b for b in batch if len(b.x)]
        if nonempty:
            x, _, _ = flatten(nonempty)
            probs = model.probabilities(x)
            offset = 0
            rows = []
            for b in nonempty:
                n = len(b.x)
                p = logistic_predict(model.design(b.x, probs[offset:offset + n], mode), model.calibrators[mode])
                rows.extend((b.record.entity_id, t, float(v)) for t, v in zip(b.targets, p))
                offset += n
            log.executemany("INSERT INTO scored VALUES(?,?,?)", rows)
        log.commit()
        batch.clear()
    try:
        index.assert_usable()
        if model.config.get("target_exclusive"):
            raise ValueError("This released runner has no production target-exclusive calibration policy")
        for query in read_records(queries_path, "S1"):
            log.execute("INSERT INTO queries VALUES(?,?)", (order, query.entity_id))
            order += 1
            cand = index.retrieve(query, model.config["k"], model.config["max_postings"], model.config["keys_per_channel"])
            batch.append(QueryBlock(query, [c.record.entity_id for c in cand], feature_matrix(query, cand), np.zeros(len(cand)), "test"))
            count += len(cand)
            if count >= batch_pairs or len(batch) >= 1024:
                flush()
                count = 0
        flush()
        with open(out / "matching_results.tsv", "w", newline="", encoding="utf-8") as fm, open(out / "candidate_pairs.tsv", "w", newline="", encoding="utf-8") as fc:
            wm, wc = csv.writer(fm, delimiter="\t", lineterminator="\n"), csv.writer(fc, delimiter="\t", lineterminator="\n")
            wm.writerow(["source1_entity_id", "matched_entity_ids"])
            wc.writerow(["source1_entity_id", "candidate_entity_ids"])
            for (qid,) in log.execute("SELECT id FROM queries ORDER BY n"):
                pairs = list(log.execute("SELECT t,p FROM scored WHERE q=? ORDER BY t", (qid,)))
                wc.writerow([qid, ",".join(t for t, _ in pairs)])
                wm.writerow([qid, ",".join(t for t, p in pairs if p >= threshold)])
        validate_outputs(out, queries_path, index)
        summary = {"queries": order, "pairs_scored": log.execute("SELECT count(*) FROM scored").fetchone()[0],
                   "mode": mode, "threshold": threshold, "model_manifest_sha256": digest_file(Path(model_path) / "manifest.json"),
                   "queries_sha256": digest_file(queries_path), "index": index.metadata()}
        write_json(out / "prediction_manifest.json", summary)
        return summary
    finally:
        log.close()
        index.close()


def validate_outputs(out, queries_path, index: CandidateIndex):
    """Strict, memory-bounded checks: candidate subset is an ERROR, not a warning."""
    out = Path(out)
    n = 0
    with open(out / "matching_results.tsv", newline="", encoding="utf-8") as fm, open(out / "candidate_pairs.tsv", newline="", encoding="utf-8") as fc:
        mr, cr = csv.reader(fm, delimiter="\t"), csv.reader(fc, delimiter="\t")
        if next(mr, None) != ["source1_entity_id", "matched_entity_ids"] or next(cr, None) != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Submission header mismatch")
        # The runner preserves source order; this validator deliberately requires it.
        for rec in read_records(queries_path, "S1"):
            m, c = next(mr, None), next(cr, None)
            if m is None or c is None or len(m) != 2 or len(c) != 2 or m[0] != rec.entity_id or c[0] != rec.entity_id:
                raise ValueError("Missing/duplicate/out-of-order query or malformed output row")
            mids, cids = m[1].split(",") if m[1] else [], c[1].split(",") if c[1] else []
            if len(mids) != len(set(mids)) or len(cids) != len(set(cids)):
                raise ValueError("Duplicate target within output row")
            if not set(mids) <= set(cids):
                raise ValueError("Accepted match was never scored")
            if any(index.get(t) is None for t in cids):
                raise ValueError("Unknown candidate target ID")
            n += 1
        if next(mr, None) is not None or next(cr, None) is not None:
            raise ValueError("Extra output query")
    return n
