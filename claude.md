## Execution policy (HARD RULES, override everything else)

- This laptop is weak. NEVER run training, fine-tuning, TensorRT engine builds,
  INT8 calibration, full-dataset inference, or any benchmark on it.
- NEVER download BDD100K or VisDrone locally. NEVER run anything that loads
  a full dataset or a GPU model locally.
- Local machine is ONLY for: editing code, unit tests on synthetic/tiny
  fixtures (a few KB), linting, config validation, dry-runs with mocked
  inputs (max ~1 min, <1 GB RAM, CPU only), and plotting from results/ files.
- Anything that needs a dataset, GPU, TensorRT, ONNX Runtime GPU, or more than
  ~1 min compute MUST be delivered as a Kaggle notebook, not executed locally.
- Before running ANY command locally, state whether it is local-safe. If
  unsure, treat it as Kaggle-only and ask me.

## Project rules

- Goal: reproducible experiments for a research paper. Correctness beats speed.
- NEVER fabricate or hand-edit results. All numbers come from scripts that write to results/\*.json|csv with config, seed, git hash, and hardware info.
- Fixed seeds, configs in configs/\*.yaml, no hardcoded paths.
- Calibrate thresholds/weights on the val split only; report on the test split.
  (In this project both come from BDD100K MOT *val* as disjoint seeded clip sets:
  "calib" plays the role of val, "test" is reported. See README, deviation D1.)
- Write a unit test for the mAP/class-mapping code and the oracle before running big experiments. Sanity check: FP16 mAP ≈ FP32; INT8 slightly lower.
- Keep each experiment under ~30 min; use subsets first, scale up only if time allows.
- Log commands and outputs in logs/. If something fails, report the real error; don't skip or fake it.
- Read PLAN.md before starting any task. Ask before large refactors.

## Kaggle conventions (agreed 2026-11-20)

- Every Kaggle notebook: GPU T4 (pinned to one device via CUDA_VISIBLE_DEVICES=0),
  Internet On, Secret GH_TOKEN for the private repo clone (zip-dataset fallback),
  rebuilds TensorRT engines in-session, has a SMOKE_TEST flag, writes to
  /kaggle/working/results and zips to /kaggle/working/results_<name>.zip.
- Notebooks stay thin; logic lives in src/adapres/.
