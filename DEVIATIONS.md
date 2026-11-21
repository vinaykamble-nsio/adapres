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

## D. Provenance (clean commits)

| Stage | Commit |
|---|---|
| Step 1 subsets | `098a725` |
| Step 2 evaluation | `18f6213` |
| 02b diagnostic | `4d37cc1` |
| Notebook 04 run | `4e35b3e` (clean checkout, `git_dirty: false`) |

`results/index.csv` records the commit of every run.
