| Policy | mAP50-95 | 95% CI | mAP50 | easy | crowded | night | scene-chg.† | INT8 frac. | switches/100 |
|---|---|---|---|---|---|---|---|---|---|
| Fixed FP32 | 26.70 | [22.99, 36.93] | 42.95 | 33.98 | 24.50 | 12.59 | 21.80 | 0.00 | 0.00 |
| Fixed FP16 | 26.71 | [22.99, 36.93] | 42.95 | 34.01 | 24.50 | 12.55 | 21.61 | 0.00 | 0.00 |
| Fixed INT8 | 25.76 | [22.08, 36.10] | 41.27 | 33.91 | 23.53 | 11.61 | 18.68 | 1.00 | 0.00 |
| Adaptive (locked) | 25.81 | [22.18, 36.06] | 41.23 | 33.91 | 23.82 | 11.64 | 20.65 | 0.96 | 1.48 |
| Matched random (10 seeds) | 25.77 ± 0.06 | [22.08, 36.12] | 41.26 | 33.85 | 23.55 | 11.63 | 18.91 | 0.96 | 1.45 |
| i.i.d. random (10 seeds) | 26.16 ± 0.11 | [22.46, 36.40] | 42.04 | 33.84 | 23.95 | 12.05 | 20.33 | 0.50 | 50.13 |
| Oracle-cheap (uses GT) | 26.67 | [22.92, 37.09] | 43.28 | 34.24 | 24.34 | 12.31 | 21.56 | 0.63 | 34.08 |

Test split (40 clips, 7796 frames; easy 3841, crowded 768, night 2295; scene-change† 160 frames of 2080 in 20 spliced streams). mAP in points (pycocotools, single-label predict outputs); 95% cluster-bootstrap CI over clips (1000 resamples); random policies: mean ± SD over 10 seeds. †Scene changes are synthetic (spliced clips). Generated from results/ at commit 4e35b3e1.
