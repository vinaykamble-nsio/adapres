# 2026-11-22 Q-ROUTE implementation: local checks (CPU only, synthetic fixtures)

All commands are local-safe: synthetic fixtures of a few KB, no dataset, no GPU.

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -m "slow or not slow" -q -p no:cacheprovider` | 245 passed in 168 s |
| `.venv/Scripts/python.exe -m ruff check src tests scripts paper` | All checks passed |
| `.venv/Scripts/python.exe scripts/make_notebooks.py` | wrote notebooks/QROUTE_COMPLETE_EXPERIMENT.ipynb (others: cell ids only, restored) |
| SHA-256 of configs/prereg_qroute.yaml | 1d90635d76e44b700ccef3ce64ef7d59efe7eaf2ff44882f4efa88e022fc0a62 |

## Issues found by the dry-run and fixed before the freeze
- prereg_qroute.yaml did not parse as YAML: a flow list containing `{...}`, and `": "` inside a plain
  scalar. Both are fixed; the content is unchanged.
- New-test stratification: the fake dataset carries time-of-day labels. Now follows the Step 1 rule: label
  attribute first, luma/Otsu fallback.
- Live stage: a tuple-unpacking bug for the fixed baselines in the contention loop.
- Energy-cost ablation: needed per-config energy before controller tuning. It is now measured in S11, and
  S12 runs after S11.
- Live vs replay: with hardware adaptation, measured latency feeds back into decisions. The implementation
  check is now the no-hw variant (1.0 agreement, 0.0 mAP difference on fakes), and both are reported.
- ContentionGenerator: added a CUDA-ready handshake and per-matmul synchronisation. Without the sync,
  asynchronous kernel queueing would saturate the GPU at any duty cycle.
- tests/test_v2_dryrun.py::test_v2_requires_frozen_addendum had been failing since the real addendum hash
  was committed: the fixture copies configs/, hash included. The test now removes the copied hash first,
  and the Q-ROUTE test does the same.

## Not executed locally (Kaggle-only by CLAUDE.md)
Engine builds, INT8 calibration, dataset inference, contention and energy measurements: everything in
QROUTE_COMPLETE_EXPERIMENT.ipynb. No Q-ROUTE result exists yet.
