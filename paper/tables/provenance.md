| Item | Value |
|---|---|
| BDD subsets (Step 1) | 098a7257 |
| Step 2 eval / 02b diagnostic | 18f62138 / 4d37cc15 |
| Notebook 04 run (clean checkout) | 4e35b3e1 (dirty=False) |
| GPU / driver / CUDA | Tesla T4 / 580.178.04 / 13.0 |
| torch / ultralytics / tensorrt | 2.11.0+cu130 / 8.4.216 / 10.16.1.11 |
| rebuilt-engine drift vs Step 2, calib mAP50-95 (max over slices, pts) | fp32 0.000, fp16 0.083, int8 0.213 |
| engine SHA-256 (FP32 / FP16 / INT8) | 7d32b8d1ce78 / 2ad8fc60a77a / 2cc6f201db4a |
| GPU memory, engine alone (MiB) FP32 / FP16 / INT8 | 124 / 76 / 66 |
| GPU memory, adaptive set INT8+FP16 (MiB) | 140 |
| live vs replay (test subset): level agreement / \|Δ mAP50-95\| pts | 1.000 / 0.000 (1924 frames) |

Provenance and measurement details (all values from results/).
