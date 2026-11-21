| ID | Variant | mAP50-95 | Δ vs main | INT8 frac. | switches/100 | calib |
|---|---|---|---|---|---|---|
| A-asym | asymmetric persistence (K_up = 1) | 25.84 | 0.02 | 0.95 | 1.73 | feasible |
| A-c | leave out clutter | 25.81 | 0.00 | 0.96 | 1.48 | feasible |
| A-ema | no EMA (alpha = 1) | 25.81 | 0.00 | 0.96 | 1.48 | feasible |
| A-hyst | no persistence (K = 1) | 25.81 | 0.00 | 0.96 | 1.48 | feasible |
| A-m | leave out motion | 25.89 | 0.07 | 0.90 | 2.24 | feasible |
| A-n | leave out count | 25.86 | 0.05 | 0.58 | 1.94 | feasible |
| A-s | leave out scene-change cue | 25.82 | 0.00 | 0.96 | 1.48 | feasible |
| A-sc-override | scene-change override off | 25.81 | -0.01 | 0.98 | 1.06 | feasible |
| A-u | leave out uncertainty | 25.81 | 0.00 | 0.96 | 1.48 | feasible |
| S-c | clutter only | 25.58 | -0.23 | 0.78 | 1.35 | feasible |
| S-m | motion only | 25.75 | -0.07 | 0.64 | 1.81 | feasible |
| S-n | count only | 25.87 | 0.05 | 0.92 | 1.92 | feasible |
| S-s | scene-change cue only | 26.71 | 0.90 | 0.00 | 0.00 | feasible |
| S-u | uncertainty only | 26.25 | 0.44 | 0.61 | 4.16 | feasible |
| delta0.25 | locked at δ = 0.25 pts | 25.88 | 0.06 | 0.95 | 1.68 | infeasible |
| delta0.5 | locked at δ = 0.5 pts | 25.81 | 0.00 | 0.96 | 1.48 | feasible |
| delta1.0 | locked at δ = 1.0 pts | 25.82 | 0.01 | 0.96 | 1.09 | feasible |
| delta2.0 | locked at δ = 2.0 pts | 25.82 | 0.01 | 0.96 | 1.09 | feasible |

Ablations on the test split. Each variant was re-calibrated on calib with the same procedure and locked before test (δ = 0.5 pts); δ rows are the Pareto locks. Δ vs the locked main controller, in points.
