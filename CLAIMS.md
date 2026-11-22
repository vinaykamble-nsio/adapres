# Claims register (response to review, 2026-11-21)

What the paper may and may not claim, point by point. Numbers are cited in the manuscript only
through `paper/numbers.tex` macros (generated from `results/`). Wording for v2 is fixed for
**both** outcomes before notebook 05 runs (addendum `configs/prereg_addendum.yaml`, SHA-256 in
`configs/prereg_addendum.yaml.sha256`).

## Scope sentence (abstract, introduction, conclusion)
> "We study per-frame precision switching (TensorRT INT8/FP16, post-training quantisation) for
> YOLO11s on BDD100K MOT driving video (val-derived calib/test clips) on a single NVIDIA T4."

No claims about embedded devices (the Jetson claim is removed), other detectors, other datasets,
other GPUs, quantisation-aware training, or adaptive inference in general.

**Title and core idea follow the abstract submitted to the conference** ("Content- and
Hardware-Aware Adaptive Precision for Real-Time Video Object Detection"; corresponding author Vinay
Kamble). The hardware-aware (device-load) term is part of the controller's formulation, but
**`w_h = 0` in every experiment: it is not evaluated.** 4-bit engines, VisDrone and the Jetson board
from the submitted abstract were not run. The paper states this in its abstract and limitations,
and claims nothing about them.

## 1. "The algorithm does not beat its strongest baselines"
- **Claim (r04, pre-registered primary):** the content-cue controller does not outperform fixed
  INT8 or matched random switching; H1 and H2 are not supported.
- **Do not claim** that adaptive precision improves the accuracy–cost trade-off.
- **v2 follow-up, wording fixed in advance:**
  - *If P1 holds* (CI of v2 − matched random at 50% INT8 entirely above 0): "a learned risk
    estimator places INT8 better than chance at a fixed budget (+X pts, CI [a, b]), recovering Y%
    of the oracle headroom." State that it is a pre-registered follow-up designed after the
    primary result, and evaluated once.
  - *If P1 fails:* "even a learned estimator using causal cues does not place INT8 better than
    chance at a fixed budget." The paper is then a bounded negative result.

- **Outcome (2026-11-22):** P1 failed (−0.03 pts, CI [−0.49, +0.38]), so the paper uses the
  pre-written "P1 fails" wording. That is a bounded negative result for both controllers.

## 2. "The controller collapses to INT8"
- **State the cause, not just the symptom.** Maximising the saving within δ, with a constant cue
  cost, made the objective equivalent to "most INT8 within δ". The locked weights (motion 0.25,
  clutter 0, count 0.50, uncertainty 0, scene 0.25) and 96% INT8 follow from that.
- **Do not describe r04 as "adaptive"** in any sense beyond the mechanism: it rarely adapted.
- v2 removes the degeneracy by design: fixed INT8 budgets force a choice of which frames run INT8.

## 3. "Limited generalisation"
- Use the scope sentence above everywhere. The limitations section lists: one detector (YOLO11s),
  one dataset (BDD100K MOT, val-derived splits), one GPU (T4), one quantisation recipe (TensorRT
  PTQ), board-level GPU energy only.
- Future work: a second model size, VisDrone, an embedded device.

## 4. "Energy does not reflect representative switching"
- **r04:** the energy subset (2 clips) ran 100% INT8, so it measures the cost of the adaptive
  infrastructure (two resident engines plus the cue computation), not mixed switching. Say exactly
  that.
- **v2:** energy is measured on 6 seeded clips at a 50% budget. Report the realised INT8 share and
  switch rate inside the measurement windows, plus a forced-alternation configuration as the
  worst-case switching cost. E_total is primary, E_dyn secondary (DEVIATIONS.md V1).

## 5. "The oracle is optimistic"
- Call it **"a ground-truth-informed upper bound under the per-frame F1@0.5 adequacy rule"**,
  never a baseline or an achievable target. Per-frame F1 optimisation is not guaranteed to be
  mAP-optimal.
- The **lagged oracle** (ground truth one frame late) is also a reference that uses ground truth,
  not a deployable method. It shows how much of the headroom is temporally predictable.
- "Headroom captured" = (v2 − matched random) / (oracle − matched random) is descriptive only.

## Q-ROUTE (r06, second pre-registered follow-up): status
- **Implemented, not yet run.** It has been checked only by local dry-runs with fake engines,
  which test wiring, gates, locks and live-equals-replay, not performance.
- **No Q-ROUTE result may appear in the paper** until `results_QROUTE_COMPLETE_EXPERIMENT.zip` from a
  clean-commit Kaggle run is unpacked into `results/`.
- When results exist, the wording follows Q1 (primary: Q-ROUTE vs best fixed config at B_int8, with
  the bootstrap CI over the 40 NEW test clips), whatever its sign.
- Q4/Q5 and every live number must be labelled *measured*; replay costs must be labelled *estimated*.
- The hardware-aware term is evaluated only by Q-ROUTE (the r04/r05 claims keep `w_h = 0`).
- INT4, Jetson and VisDrone end-to-end stay future work (see DEVIATIONS.md F).

## Statements that stay as they are
- Fixed INT8 vs FP16: −0.95 pts mAP50-95 on test, with 13% lower latency and 15% lower E_total
  per frame (T4, measured live). This is the strongest practical finding.
- Scene-change benefit (r04): only on **spliced, synthetic** cuts, where the cue is correlated with
  the splice by construction.
- Night: the controller ran INT8 on 98% of night frames although INT8 loses 0.94 pts there. This
  contradicts H4's premise that night needs FP16; H4's mechanical pass is narrow.
- The r04 lock sat at the δ limit on calib (−0.44 pts) and did not hold on test (−0.90).
