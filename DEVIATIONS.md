# Deviations from the pre-registration and PLAN.md

`configs/preregistration.yaml` was frozen on 2026-11-20 and is **not edited**. Every departure
from it or from `PLAN.md` is listed here, split by **when** it was decided relative to seeing
test results. Numbers quoted below are copied from the named `results/` files, which are the
source of truth. The paper cites them only through `paper/numbers.tex`.

## A. Decided before any test-split result (2026-11-20)

Full text in `README.md`, section "Deviations from PLAN.md".

| ID | Deviation |
|---|---|
| D1 | Calib and test are disjoint, seeded clip sets from BDD100K MOT **val**: the Kaggle mirror has no MOT train images |
| D2 | Time-of-day strata from median-frame luma: `box_track` labels have no `timeofday` attribute; dawn/dusk merged into day |
| D3 | Ignore classes become `iscrowd=1` boxes for every evaluated class |
| D4 | AP read from `COCOeval.eval['precision']`: pycocotools 2.0.11 `stats[0]` hardcodes maxDets=100 |
| D5 | Software stack: Python 3.13, CUDA 13, TensorRT cu13 wheels, ORT as fallback only |
| D6 | The `model.val()` agreement check uses 200 BDD calib frames, not a COCO mini-set |
| D7 | Evaluation uses single-label `predict()`. The `model.val()` gap is multi-label NMS (notebook 02b); the binding check is identical-prediction agreement plus the NMS explanation |
| D8 | Notebook 04 choices: engines rebuilt in-session; idle measured once per repeat; settle wait capped at 60 s; θ from open-loop quantiles; locks in `results/locked/` |

## B. Decided AFTER seeing the notebook 04 test results (2026-11-21)

These were decided with knowledge of the results. Both the original and the revised values are
reported, and nothing was re-run.

### V1. Primary energy metric: E_total/frame (was E_dyn/frame)
- **Pre-registered / PLAN §7.3:** the primary metric was E_dyn/frame = (ΔE − P_idle·T)/N.
- **Now:** **E_total/frame = ΔE/N is primary.** E_dyn/frame is secondary, and both appear in every
  energy table and figure.
- **Reason: an idle-subtraction artifact.** E_dyn subtracts idle board power over the whole
  window, so a configuration that leaves the GPU waiting for part of each frame gets a larger
  subtraction and looks cheaper.
  - The adaptive controller with **sequential** cues takes 8.85 ms/frame vs 5.03 ms for fixed INT8,
    because the GPU idles while the CPU computes the cues.
  - It shows E_dyn = 79 mJ/frame vs 167 for INT8, while its E_total is **369 vs 333** mJ/frame.
    E_total is the energy actually drawn per processed frame.
  - Source: `results/bdd100k_mot/r04_test_energy_latency/energy.json`.

### V2. H3 evaluated on the K-sweep, reported as a **sensitivity analysis**
- **Revision 1** (in notebook 04) compared the locked K with K = 1. The locked K is 1, so the test
  compared a configuration with itself and returned "not supported" by construction.
- **Revision 2** compares each K ≥ 2 of the K-sweep with K = 1 (θ fixed at the locked value, no
  re-calibration), using the pre-registered criterion: fewer switches, and |Δ mAP| < 0.3 pts.
- Because the K-sweep keeps θ fixed and was not part of the lock, it is labelled a sensitivity
  analysis wherever it appears.

### V3. H4 "share of saving from easy frames" uses engine latency only
- **Revision 1** used the replay latency including the constant sequential cue cost (2.70 ms/frame).
  That made every slice's "saving" vs FP16 negative, so the ratio (0.489) was meaningless.
- **Revision 2** uses engine latency only (microbench medians): easy share = **0.505**.

### V4. Record keeping for V1–V3
- The original output `results/bdd100k_mot/r04_test_hypotheses/hypotheses.json` (revision 1) is
  **unchanged**. A `SUPERSEDED.txt` next to it points to the replacement.
- Revision 2 is `results/bdd100k_mot/r04_test_hypotheses_v2/hypotheses.json`. It was recomputed
  locally by `scripts/recompute_hypotheses.py` from existing `results/` files only. Its `meta.json`
  lists the SHA-256 of every input file. No new data, no re-locking, no re-tuning, no additional
  test runs.
- `adapres.pipeline.full._hypotheses()` now calls `adapres.hypotheses.compute()` (revision 2), so
  the notebook code matches for the record.

## C. Limitations to report

- **L1. The lock sat at the δ limit and did not hold on test.**
  - On calib the locked controller was **0.44 pts** below FP16 (δ = 0.5); on test it was
    **0.90 pts** below.
  - Choosing the configuration with the largest saving subject to mAP ≥ FP16 − δ selects
    configurations near the constraint boundary, so calib-to-test regression should be expected.
  - Per the 2026-11-21 decision there is no re-locking, re-tuning or additional test run.
  - Sources: `r04_calibration/calibration.json`, `r04_test_adaptive_main/metrics.json`.
- **L2. The objective degenerated.** The sequential cue cost (2.70 ms) is a constant per frame, so
  maximising the replay saving within δ reduces to maximising the INT8 fraction. The locked
  controller runs INT8 on 96% of test frames with K = 1, and is statistically indistinguishable
  from fixed INT8 and from matched random switching.
- **L3. INT8 rebuild drift.** The notebook 04 engines differ from the Step 2 engines on calib by
  **0.21 pts** mAP50-95 for INT8 (FP16 0.08, FP32 0.00; max over slices).
  - That is above PLAN's 0.1-pt threshold for reusing caches across sessions, and below notebook
    04's gate (0.5). Every number from notebook 04 uses the rebuilt engines consistently.
  - Source: `results/r04_stages/stages.json`.
- **L4. The scene-change slice is synthetic.** It comes from 20 spliced streams (160 scene-change
  frames). The controller's histogram cue and override are correlated with the splice points by
  construction, so its advantage there (adaptive 20.65 vs fixed INT8 18.68 vs matched random 18.91
  mAP50-95) must be stated as spliced/synthetic. Natural scene changes are rare (31 test frames).
- **L5. The night result contradicts H4's premise.**
  - The controller ran INT8 on **98.4%** of night frames, because dark frames score low on motion
    and clutter. On test, INT8 is **0.94 pts** below FP16 at night (11.61 vs 12.55).
  - H4's mechanical criterion still passes, but only narrowly: FP16 is chosen on 1.6% of night
    frames vs 1.1% of easy frames, and the easy saving share is 0.505 vs the 0.5 threshold.
- **L6. The energy/latency measurement set** is 2 test clips (seeded). On these the adaptive
  controller chose INT8 on 100% of frames, vs 96% across the test split.
- **L7. General** (PLAN §1.4):
  - one cloud GPU (Tesla T4), GPU board energy only, no embedded device;
  - COCO-pretrained weights, so absolute mAP is low;
  - cluster-bootstrap CIs are wide with 40 clips;
  - replay was validated against a live run on 10 test clips (1,924 frames: identical decisions
    and mAP).

## E. Pre-registered follow-up: controller v2 (decided 2026-11-21, after r04 results)

- **Why:** a review found that the r04 controller beats neither fixed INT8 nor matched random,
  collapses to 96% INT8, has energy measured without mixed switching, and leans on an optimistic
  oracle.
- **What:** a learned per-frame risk estimator under fixed INT8 budgets (25/50/75%). The design,
  features, model grid, decision rule, baselines (matched random, v1@budget, oracle@budget, lagged
  oracle), the primary test (P1) and the reporting rules are frozen in
  `configs/prereg_addendum.yaml`. Its SHA-256 (`configs/prereg_addendum.yaml.sha256`) is verified by
  notebook 05 and recorded in every v2 lock.
- **Integrity rules:**
  - v2 is fitted and locked on calib only, and the test split is evaluated once.
  - **The r04 result stays the pre-registered primary outcome.** v2 is reported as a follow-up
    designed after seeing r04, whatever its result.
  - No re-locking or re-running on test.
- **Implementation note, decided before the freeze:** detections are canonicalised to float32 before
  computing features, so live and cached detections give identical decisions. A local dry-run found
  72% live-vs-replay agreement without this, and 100% with it.
- The claims, with v2 wording fixed for both outcomes, are in `CLAIMS.md`.
- **Outcome (notebook 05, commit `2f0c780`, full run, all gates passed):**
  - **P1 not supported**: v2 − matched random at the 50% budget = −0.03 pts, CI [−0.49, +0.38].
  - **P2, P3 and P5 not supported** either.
  - Source: `results/bdd100k_mot/r05_test_v2_hypotheses/`.
- **Notes for reporting v2:**
  - Realised test INT8 shares were below the calib budgets (0.18 / 0.41 / 0.60 for 25 / 50 / 75%).
    Matched random is matched to the realised share, so P1 compares equal costs.
  - v1@budget and the lagged oracle realised different shares from v2, so **P3 is not an equal-cost
    comparison**.
  - The rebuilt INT8 engine differed from Step 2 by 0.41 pts of calib mAP. That is inside the 0.5 gate,
    but larger than r04's 0.21: TensorRT INT8 calibration varies between builds.
  - The v2 evaluation did not score the spliced scene-change streams, so that slice is reported only
    for r04.
  - v2's latency was dominated by its controller step (feature statistics plus model prediction,
    2.29 ms per frame, on the critical path even with pipelined cues).

## F. Second pre-registered follow-up: Q-ROUTE (decided 2026-11-22, after the r04 and r05 results)

**Why.** v1 and v2 switched only between INT8 and FP16 at 640 px. Two r05 results motivated a redesign:
- the per-frame INT8 regret was barely predictable (held-out Spearman 0.06);
- the decision cost (2.3 ms) was larger than the INT8 saving (0.7 ms).

**What.** Q-ROUTE routes each frame among 6 engines (FP16/INT8 × 640/480/320) by predicted regret
+ λ·cost.
- The cost charges the profiled latency, an online contention slowdown, and the measured controller
  overhead.
- The design is frozen in `configs/prereg_qroute.yaml`, with its SHA-256 in
  `configs/prereg_qroute.yaml.sha256`. It covers:
  - splits, action space, features and feature gating;
  - predictors and the controller;
  - budgets (B_int8 primary, B_mid);
  - every baseline and ablation;
  - Q1–Q5, the contention protocol and the reporting rules.
- `QROUTE_COMPLETE_EXPERIMENT.ipynb` verifies the hash, and every Q-ROUTE lock records it.

**New untouched test set.** The 40 former test clips have been seen through r04/r05 results, so they
are no longer untouched.
- They and the 20 calib clips (60 in all) become **development** data.
- **40 NEW final-test clips** are drawn once (seed 2027) from the BDD100K MOT val clips that are not
  in the Step 1 selection.
- They are stratified 28 day / 12 night with the Step 1 rule: the label attribute if present,
  otherwise median-frame luma vs the Step 1 Otsu threshold. Step 1's 24/12/4 mix used a dawn/dusk
  stratum; this rule classifies only day/night, so the new set has no dawn/dusk stratum.
- Image ids start at 2,000,000.

**Integrity rules.**
- All fitting, model selection, feature gating and λ/K tuning use development clips only.
- The final test clips are evaluated once, with hashed locks.
- r04 stays the primary pre-registered result; r05 and r06 are follow-ups, reported whatever they show.

**Edits before the freeze (no data touched):** two YAML syntax fixes, found by the local dry-run:
- a flow list containing `{…}`, quoted;
- a `": "` inside a plain scalar, reworded.

The design content is unchanged. The hash was written after the code was final.

**Implementation notes, decided before any real Q-ROUTE run.**
- **Budget B_int8.** It is the profiled cost of fixed INT8-640 *without* controller overhead, while
  Q-ROUTE pays its measured overhead. Variants restricted to {FP16-640, INT8-640} (precision-only,
  hardware-only) therefore cannot reach B_int8. Each lock records `budget_met_on_dev`; this is a result,
  not an error.
- **Live vs replay.** With hardware adaptation the live controller reacts to measured latency, which
  isolated replay does not model. The implementation check is therefore the no-hardware variant,
  which must match replay; the main controller's agreement is reported as well.
- **Contention generator.** It waits until its CUDA context is ready, and synchronises after every
  matmul so that GPU busy time tracks the duty cycle.
- **`full_features_no_gating` ablation.** It is skipped, and recorded as skipped, when image cues are
  gated in, because it would then be identical to the main controller.

**Not run (blocked; stated, never claimed):**
- **INT4 engines:** the Ultralytics TensorRT export offers FP32/FP16/INT8 only, so the notebook
  records INT4 as unsupported.
- **Jetson / embedded hardware:** not available.
- **VisDrone end-to-end:** only parsed and validated if attached. The pre-registered pipeline is
  BDD100K only.

**Outcome:** pending. Q-ROUTE has not been run on Kaggle yet.

## D. Provenance (clean commits)

| Stage | Commit |
|---|---|
| Step 1 subsets | `098a725` |
| Step 2 evaluation | `18f6213` |
| 02b diagnostic | `4d37cc1` |
| Notebook 04 run | `4e35b3e` (clean checkout, `git_dirty: false`) |

`results/index.csv` records the commit of every run.
