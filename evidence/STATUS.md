# Execution status — 26 September 2026

## Completed in this session

- Inspected the single uploaded archive and inventoried all 14 regular files with SHA-256 hashes. Preserved source materials unchanged, including original attribution.
- Wrote the workflow before implementation, then completed the discrepancy audit, primary-source research review and CARE design.
- Implemented a modular first milestone with seven retrieval channels, 34 named features, real XGBoost/LightGBM/CatBoost training, separate calibration and threshold partitions, and batch-scored exact candidate logging.
- Ran **34 pytest test cases**, all passing, with no test warnings in the final run. See `pytest.txt` and `pytest.xml`.
- Ran a separate end-to-end synthetic smoke using **360 queries**, **1336 target records** and **14400 scored training/evaluation pairs**. All three base learners trained; both calibration policies ran; artifacts were saved/reloaded; TSV outputs were generated.
- The supplied validator with `--check-ids` returned **exit code 0**. The runner also enforces candidate-subset violations as hard failures.
- Python compile checks and shell syntax checks passed. Notebook JSON was validated; the Colab environment itself was not executed.

## Synthetic measurements — NOT competition performance

One synthetic smoke took 4.43 seconds on this environment. Its maximum Linux RSS counter was 262,964 KiB, and its small SQLite index occupied 1,810,432 bytes. These figures are not estimates for the real dataset and do not establish a 16 GB memory guarantee. The smoke index had 34,245 postings.

Both pair-only and CARE had the same synthetic split-holdout macro F0.5 (0.994949). This is deliberately simple generated data and **does not show CARE improves real entity resolution**. Synthetic fixture `test` rows mirror the synthetic inputs solely to check output plumbing; their predictions must not be treated as unseen-test evaluation.

## Not completed / not claimed

No real competition dataset was trained or scored. No full-scale target index was built. No leaderboard result was verified or submission uploaded. No learned dense retrieval, general Indic transliterator, model-mined hard negatives, nested cross-fitting, full-graph unseen-target split, bootstrap uncertainty study or final competition packager is implemented.

The first index is correctness-oriented SQLite. Large posting lists/tables may be too slow or large; 10k/100k probes and full-data checks remain necessary. Probe builds use prefixes of each supplied target file, so extrapolations can be biased by file ordering and must be followed by representative/country-stratified profiling. A 'complete' index means all records from the explicitly supplied input files were read, not that missing official files or truncated upstream downloads have been independently ruled out.

The default decision policy is independent-edge thresholding. Optional target-exclusive conflict resolution exists as a tested helper, but is not enabled in the production runner because it needs a separately calibrated full-query evaluation.

## GitHub status

The existing GitHub connection successfully read `SanskariXD/Amazon-ML-C-26`. That repository was not modified. The native tool interface did not expose repository creation; a separate creation-capable connection was initiated but remained INITIATED after the authorization wait timed out. **No new remote repository or remote commit is claimed.** Intended new name: `ML-2-C-26`, private. `scripts/publish_github.sh` is a guarded manual publishing path after authorization.

## Reproducibility evidence

`reference/manifest.json` identifies the exact uploaded snapshot. `requirements.txt` pins exercised core versions; `smoke_report.json` records runtime, input hashes, partition counts, thresholds and synthetic outputs. The regression suite covers Unicode marks, empty fields, TSV parsing, duplicate IDs, stable sampling, shared-positive split isolation, metric calculation, threshold sweep versus brute force, more than ten matches, target ties, new countries, retrieval misses, artifact compatibility, batch invariance, zero-candidate output and memory guard behavior.
