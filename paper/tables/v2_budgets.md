| Budget | Policy | mAP50-95 | 95% CI | INT8 frac. | switches/100 |
|---|---|---|---|---|---|
| – | Fixed FP16 | 26.69 | [22.96, 36.77] | 0.00 | 0.00 |
| – | Fixed INT8 | 25.90 | [22.19, 35.93] | 1.00 | 0.00 |
| – | Fixed FP32 | 26.70 | – | 0.00 | 0.00 |
| 25% | v2 (learned risk) | 26.56 | [22.90, 36.81] | 0.18 | 2.17 |
| 25% | Matched random | 26.58 ± 0.10 | [22.85, 36.59] | 0.17 | 2.14 |
| 25% | v1 (r04 cues) | 26.42 | – | 0.36 | 7.89 |
| 25% | Lagged oracle (GT, t−1) | 26.61 | – | 0.22 | 30.55 |
| 25% | Oracle (GT, upper bound) | 26.86 | – | 0.25 | 33.16 |
| 50% | v2 (learned risk) | 26.30 | [22.64, 36.38] | 0.41 | 21.41 |
| 50% | Matched random | 26.32 ± 0.09 | [22.65, 36.32] | 0.42 | 21.47 |
| 50% | v1 (r04 cues) | 26.17 | [22.46, 35.97] | 0.66 | 7.66 |
| 50% | Lagged oracle (GT, t−1) | 26.43 | – | 0.66 | 35.77 |
| 50% | Oracle (GT, upper bound) | 26.99 | – | 0.50 | 35.52 |
| 75% | v2 (learned risk) | 26.10 | [22.48, 36.22] | 0.60 | 3.32 |
| 75% | Matched random | 26.13 ± 0.19 | [22.42, 36.38] | 0.60 | 3.34 |
| 75% | v1 (r04 cues) | 26.14 | – | 0.86 | 3.82 |
| 75% | Lagged oracle (GT, t−1) | 26.27 | – | 0.74 | 32.44 |
| 75% | Oracle (GT, upper bound) | 26.59 | – | 0.75 | 31.84 |

Controller v2 follow-up (pre-registered addendum), test split, 40 clips. Budget = target INT8 share set on calib; INT8 frac. = realised test share. Matched random has the realised share and switch rate of v2 at the same budget (mean ± SD over 10 seeds). Oracle and lagged oracle use ground truth (references, not deployable). 95% cluster-bootstrap CI over clips. Generated from results/ at commit 2f0c780d.
