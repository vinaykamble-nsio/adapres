| Difference | Δ mAP50-95 | boot. mean | 95% CI | excludes 0 | Δ mAP50 | 95% CI |
|---|---|---|---|---|---|---|
| Adaptive (locked) − Adaptive K=1 | 0.000 | 0.000 | [0.000, 0.000] | no | 0.000 | [0.000, 0.000] |
| Adaptive (locked) − Fixed FP16 | -0.895 | -0.998 | [-2.420, -0.127] | yes | -1.711 | [-3.046, -0.395] |
| Adaptive (locked) − Fixed INT8 | 0.050 | 0.030 | [-0.531, 0.267] | no | -0.038 | [-0.683, 0.299] |
| Adaptive (locked) − i.i.d. random (10 seeds) | -0.343 | -0.378 | [-1.375, 0.212] | no | -0.801 | [-1.832, 0.126] |
| Adaptive (locked) − Matched random (10 seeds) | 0.044 | 0.007 | [-0.702, 0.270] | no | -0.024 | [-0.894, 0.336] |
| Oracle-cheap (uses GT) − Fixed FP16 | -0.034 | -0.097 | [-0.549, 0.482] | no | 0.334 | [-0.884, 0.937] |

Paired differences on the test split (points): full-data point estimate, bootstrap mean and 95% cluster-bootstrap CI over the 40 clips (1000 resamples, seed 2027); random groups use the per-resample mean over 10 seeds. Adaptive − K=1 is identically 0 because the locked controller has K = 1.
