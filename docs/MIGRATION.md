# Colab migration — verification record

Date: 2026-09-27 (India).

## Changes

- Imported the complete supplied source tree into SanskariXD/ML-2-C-26.
- Added `colab/Amazon_ML_2026_Colab.ipynb` and the stdlib-based `colab/runner.py` wrapper.
- Set the verified existing Drive ZIP path; no dataset download or upload was performed here.
- Added an isolated Python 3.11 environment and pinned CUDA PyTorch installation.
- Added explicit full/dev modes, hardware checks, code-version locking, dataset hashes,
  completed-stage Drive backups, checked restore, streamed logs, and explicit output validation.
- Replaced the root README with minimal run instructions; archived the original README in docs.
- Preserved both pre-existing unrelated target-repository files and all original source files.

## Checks completed

- All 28 original files under `code/` are byte-identical to the supplied ZIP.
- All Colab notebook code cells parse, and the notebook passes nbformat schema validation.
- Eight migration tests pass: nested ZIP extraction, missing-file rejection, directory staging,
  interrupted-stage recovery, failed-backup preservation, incompatible-run rejection,
  corrupt-backup rejection, and full/dev command configuration.
- Original synthetic pipeline smoke test passes: 300 test entities, 803 matched IDs,
  11,711 candidate IDs, structurally valid TSV outputs.
- CPU integration test of the new wrapper passes all six stages, the official validator,
  output publication, and a second invocation skipping every completed stage.
  This harness mocks GPU/resource checks, disables sandbox-incompatible telemetry, and disables
  dense retrieval only for this synthetic CPU test. It does not validate GPU behavior.
- Python 3.11 installation and the pinned dependency resolution were checked locally.

## Not claimed

No real competition dataset run, GPU benchmark, full-scale memory measurement, new leaderboard
score, or live mounted-Drive checkpoint test was performed. Drive transfer tests use ordinary
local directories. The notebook performs a real GPU/model check when opened in Colab.

Upstream research scores and time estimates remain unverified historical claims. The migration
changes execution/storage wrappers, not the matching model or its accuracy configuration.
