# PLAN.md — AdaPres: Content- and Hardware-Aware Adaptive Precision for Real-Time Video Object Detection

Target: IRIS 2027 (Springer LNEE). **Final paper due Wed 25 Nov 2026.** Plan written Fri 20 Nov 2026.
Audience: the implementing engineer (Claude Code). Read sections 0–2 before writing code.

---

## 0. Ground rules (non-negotiable)

1. **No invented numbers.** Every number in the paper comes from a file under `results/`, rendered into LaTeX by `paper/make_tables.py` / `paper/make_figs.py` / `paper/make_numbers.py`. Nobody types a result into the manuscript by hand.
2. **No invented citations.** Any reference not checked against the publisher/DOI page carries `[VERIFY]` in `paper/refs.bib` (as a `note` field) and in this plan. Remove the tag only after checking the DOI.
3. **Test split is touched exactly once per locked config.** All weights, thresholds, K, δ, INT8 calibration images and slice thresholds come from the calibration split. The locked config is hashed and logged before any test run (§3.5).
4. **Figures:** only plots/diagrams we generate ourselves from our own results or code. No third-party figures. Dataset frames in figures only if the dataset licence allows it (§10); default is **no dataset imagery in the paper**.
5. **Hardware honesty:** we have Colab/Kaggle only. The abstract currently promises an "embedded platform such as NVIDIA Jetson" — **that claim must be removed or moved to future work** unless a Jetson becomes available. tegrastats is documented (§7.5) but not used.
6. If a hypothesis fails (e.g. INT8 is not faster than FP16 on our GPU), report it. A clean negative/bounded result with an oracle headroom analysis is publishable; a fudged one is not.

---

## 1. Research questions, hypotheses, ablations, threats

### 1.1 Research questions

- **RQ1 (headroom).** On real driving/aerial video, how often do lower-precision engines give per-frame detections as good as FP16, and what is the oracle's best-case cost reduction at matched accuracy?
- **RQ2 (effectiveness).** Does the content-aware controller improve the accuracy–cost trade-off over fixed FP16, fixed INT8, and random switching with the same precision mix?
- **RQ3 (cues).** Which cues (motion, clutter, count, confidence, scene change, hardware) carry the signal?
- **RQ4 (stability and overhead).** How much does persistence (hysteresis) reduce switching, and what does adaptivity cost in controller time, switch latency and resident memory (multiple engines loaded)?
- **RQ5 (conditions).** How do accuracy, cost and chosen precision vary across easy / crowded / night / scene-change slices?
- **RQ6 (hardware-aware, nice-to-have).** Under injected GPU contention, does the hardware term keep tail latency within budget at lower accuracy cost than fixed INT8?

### 1.2 Hypotheses (pre-registered; written into `configs/preregistration.yaml` on Day 1 before any test run)

- **H1.** On the BDD100K test split, the locked adaptive controller has mAP@0.5:0.95 within δ (default δ = 0.5 points absolute) of fixed FP16 and lower mean per-frame latency and energy than fixed FP16.
- **H2.** At the same fraction of frames per precision and the same switch rate, adaptive beats random switching in mAP@0.5:0.95 (cluster-bootstrap 95% CI of the difference excludes 0).
- **H3.** Persistence with K ≥ 2 reduces switch rate versus K = 1 with an mAP@0.5:0.95 change smaller than 0.3 points.
- **H4.** The controller picks FP16 more often on night/crowded/scene-change slices than on easy, and most of the cost saving comes from the easy slice.
- **H5 (nice-to-have).** Under contention, the hardware-aware variant has lower p95 latency than content-only adaptive.

Pre-registered decision rule for model size (Day 1): benchmark YOLO11s at 640 px on the session GPU. If INT8 median engine latency is < 10% below FP16, also run YOLO11m and use whichever has the larger FP16→INT8 gap as primary. Record the benchmark in `results/microbench/`.

### 1.3 Ablations (all via offline replay, §2.3, so they're cheap)

| ID                             | Variant                                                                                 | What it tests                       |
| ------------------------------ | --------------------------------------------------------------------------------------- | ----------------------------------- |
| A-m, A-c, A-n, A-u, A-s        | Leave-one-cue-out: weight set to 0, others renormalised, θ/K **re-calibrated on calib** | marginal value of each cue          |
| S-m … S-s                      | Single-cue-only controller (re-calibrated)                                              | standalone value of each cue        |
| A-hyst                         | K = 1 (no persistence)                                                                  | RQ4 / H3                            |
| K-sweep                        | K ∈ {1, 2, 3, 5, 8} with θ fixed at locked value                                        | stability–accuracy curve            |
| A-sc-override                  | scene-change override disabled                                                          | value of the immediate-upgrade rule |
| A-asym                         | K_up = K_down vs K_up < K_down                                                          | asymmetric persistence              |
| A-ema                          | EMA smoothing off (α = 1)                                                               |                                     |
| A-hw (live only, nice-to-have) | hardware term on/off under contention levels                                            | RQ6                                 |

The same re-calibration is used for every ablation, so the comparison is fair. All ablation numbers are reported on test **after** the main config is locked; ablation configs are locked the same way.

### 1.4 Threats to validity (put a condensed version in the paper)

- **Domain shift:** COCO-pretrained weights on BDD100K/VisDrone give low absolute mAP (VisDrone especially). Claims are about **relative** differences between precision policies on identical detections, not SOTA accuracy.
- **Frame rate:** BDD100K MOT labels are at 5 fps (frames ~200 ms apart; [VERIFY] in the BDD100K docs). Motion cues are coarser than at 30 fps, so "real-time" means per-frame latency, not live 30 fps capture.
- **Single GPU class:** results come from one cloud GPU (expected T4); there's no embedded device. TensorRT INT8/FP16 speedups depend on the GPU, so don't generalise to Jetson.
- **Cloud measurement noise:** clocks can't be locked, thermal state varies, and Colab can reassign GPU types between sessions. Mitigation: record GPU name, driver, clocks and temperature; do all timing/energy runs for one table in one session on one GPU; interleave configs; 3 repeats.
- **Power sensor limits:** the NVML board power is averaged and coarse-grained; we can't see CPU/DRAM energy. Energy is **GPU board energy only**.
- **Replay vs live:** most policy numbers come from replaying cached per-precision outputs. Validated by one live end-to-end run per main policy (§2.3); report the discrepancy.
- **Synthetic scene changes:** the scene-change slice is built by splicing clips (§4.4) and is not naturally occurring. Also, the controller's scene-change cue (histogram distance) is correlated with splice points by construction, so we report it separately and don't overclaim.
- **Slice definitions:** crowded uses GT counts while the controller uses predicted counts. These are related but not circular (the controller never sees GT).
- **Small subsets:** a few thousand frames, so we use cluster-bootstrap CIs over sequences and treat slice results as indicative.
- **Calibration overfitting:** the grid search is over a small space and tested on held-out sequences from a different official split.
- **INT8 calibration set:** INT8 accuracy depends on the calibration images (taken from the calib split only).
- **Software versions:** Ultralytics/TensorRT versions change engine behaviour. Pin them and record them.
- **Multiple comparisons:** many slices × ablations. Hypotheses H1–H5 are primary; everything else is exploratory.

---

## 2. Repo layout and module responsibilities

```
adapres/
├── PLAN.md
├── README.md                    # how to reproduce on Colab/Kaggle, licences
├── LICENSE                      # AGPL-3.0 (see §10: we import Ultralytics)
├── requirements.txt             # pinned: ultralytics==X, tensorrt==Y, pycocotools, pynvml(nvidia-ml-py), opencv-python-headless, pandas, pyarrow, numpy, pyyaml
├── configs/
│   ├── default.yaml             # all tunables (weights, θ grid, K grid, δ, paths, imgsz=640, model)
│   ├── preregistration.yaml     # H1–H5, δ, primary metrics, model-size rule — frozen Day 1
│   └── locked/                  # locked_<dataset>_<name>.yaml + .sha256 (written by calibrate.py)
├── adapres/
│   ├── data/
│   │   ├── bdd_mot.py           # parse BDD100K MOT (box_track) JSON → per-frame records
│   │   ├── visdrone_vid.py      # parse VisDrone-VID txt annotations
│   │   ├── subsets.py           # deterministic sequence selection (seeded), writes manifests
│   │   ├── splice.py            # builds spliced scene-change streams
│   │   ├── slices.py            # objective slice labels per frame (§4)
│   │   └── classmap.py          # COCO→dataset mapping tables (§5)
│   ├── models/
│   │   ├── export.py            # Ultralytics → TensorRT engines FP32/FP16/INT8 (+INT4 stub)
│   │   └── engines.py           # EnginePool: loads all engines once, run(level, frame) → dets, timings
│   ├── features.py              # cues: motion, clutter, hist-distance, prev-frame count/conf
│   ├── controller.py            # score, level mapping, persistence, overrides, hw term
│   ├── baselines.py             # fixed, random-matched (Markov), oracle-cheap, oracle-best
│   ├── cache.py                 # run every engine on every frame → detections + timings parquet
│   ├── replay.py                # simulate any policy on the cache (causal), produce per-frame log
│   ├── live.py                  # true end-to-end online run with EnginePool + controller + energy meter
│   ├── calibrate.py             # grid search on calib split, writes locked config + hash
│   ├── eval/
│   │   ├── coco_gt.py           # dataset GT → COCO-format JSON (with ignore/crowd boxes)
│   │   ├── map_eval.py          # pycocotools mAP@0.5, @0.5:0.95, per slice (imgIds filter)
│   │   ├── perframe.py          # per-frame F1@0.5 for oracle/proxy
│   │   └── bootstrap.py         # cluster (sequence) bootstrap CIs
│   ├── energy/
│   │   ├── nvml_meter.py        # background thread: energy counter + power samples + util/clock/temp
│   │   └── protocol.py          # warm-up, idle measurement, interleaving, repeats (§7)
│   ├── hw/contention.py         # (nice) synthetic GPU load process with duty cycle d
│   └── results.py               # schema writer/validator (§8); refuses to write without meta
├── scripts/                     # numbered, idempotent, resumable (Colab disconnects!)
│   ├── 00_env.sh                # install pinned deps, print versions → results/env/
│   ├── 01_download_bdd.md       # manual steps (registration required) + checksums
│   ├── 02_build_subsets.py
│   ├── 03_export_engines.py
│   ├── 04_microbench.py         # latency per engine; applies model-size rule
│   ├── 05_cache_runs.py         # per dataset/split/precision; resumes per sequence
│   ├── 06_calibrate.py
│   ├── 07_eval_test.py          # fixed/random/oracle/adaptive/ablations via replay
│   ├── 08_energy.py             # per-engine energy + live adaptive energy
│   ├── 09_live_validate.py      # live runs vs replay
│   └── 10_contention.py         # nice-to-have
├── paper/
│   ├── make_tables.py  make_figs.py  make_numbers.py   # read results/ ONLY
│   ├── numbers.tex  tables/  figs/                     # generated, never hand-edited
│   ├── refs.bib                                         # [VERIFY] notes
│   └── main.tex                                         # Springer template
├── tests/                       # controller hysteresis unit tests, classmap tests, schema tests, tiny mAP sanity test
├── data/        (gitignored)    # raw datasets — never committed or redistributed
├── cache/       (gitignored)    # detections parquet, engines
└── results/                     # committed: small JSON/CSV only
```

### 2.1 Key module contracts

- `EnginePool(levels=["fp32","fp16","int8"])` keeps every engine resident. `run(level, img)` returns `(dets[N,6] = x1,y1,x2,y2,score,coco_cls, timings{pre,infer,post,total_ms})`. Use Ultralytics `YOLO("<engine>.engine")` with `predict(..., imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False)`. Timings: Ultralytics `result.speed` plus our own `torch.cuda.synchronize()` + `perf_counter()` wall clock around the whole call.
- `Controller.step(frame_gray_small, prev_dets, hw_state) -> level` is pure and causal. It's deterministic given its inputs and keeps internal state (current level, candidate, counter, EMA, override timer).
- `replay(policy, cache, seq)` must call exactly the same `Controller.step` as `live.py`, with `prev_dets` taken from the cache entry **of the level chosen at t−1**. That's what makes replay causal and faithful.

### 2.2 Engines

- Build with Ultralytics `model.export(format="engine", imgsz=640, batch=1, dynamic=False, half=...)` / `int8=True, data=<calib yaml>`. Check the INT8 calibration arguments (`data`, `fraction`) against the docs for the pinned Ultralytics version [VERIFY].
- **INT8 calibration images = calibration split only** (≤ 500 frames, seeded sample). Never test frames.
- FP32 baseline: a TensorRT FP32 engine, so all levels share one runtime. Also record PyTorch FP32 `.pt` accuracy once as a sanity reference.
- TensorRT engines are tied to the GPU and version. Build once per session GPU type, store a SHA-256 of each engine, and never mix engines across GPU types within one table.
- **INT4 (nice-to-have):** as far as I know, TensorRT has no general INT4 path for conv-heavy detectors (INT4 support targets weight-only GEMM/LLM use) [VERIFY against current TensorRT docs]. If confirmed, INT4 can only be **simulated** (fake-quant weights in PyTorch). That gives accuracy only, with no real speed/energy, and the paper must say so. Skip unless all must-haves are done.

### 2.3 Cache-and-replay architecture (the core time-saver)

1. `05_cache_runs.py` runs every engine on every frame once and stores detections (`cache/dets_<ds>_<split>_<level>.parquet`) and per-frame timings (`cache/time_<ds>_<split>_<level>.parquet`). The frame order is preserved. A warm-up of 50 frames per engine is discarded.
2. Fixed baselines, random, oracle, adaptive, every ablation and calibration then **select per-frame detections from the cache**. Each is a numpy pass plus one mAP evaluation.
3. Replay latency = Σ cached engine latency of the chosen level + measured controller overhead (`features.py` time per frame, measured live) + measured switch penalty × number of switches.
4. Replay energy = Σ per-level energy/frame (§7) for the chosen levels. It's labelled **"estimated"** everywhere.
5. `09_live_validate.py` runs FP16, INT8 and locked adaptive live on the test split (or a fixed 25% subset if time is short) with the energy meter on. These **measured** numbers go in the main table, next to replay estimates, with the replay–live gap reported.

---

## 3. Complexity score, controller, calibration

### 3.1 Per-frame raw cues (computed on a downscaled grayscale frame, width 320, causal)

Let g*t be the grayscale frame (uint8), and D*{t−1} the detections of the level actually used at t−1, filtered at conf ≥ τ_c (default 0.25).

| Cue                     | Raw definition                                                                                                                                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------- |
| motion m_t              | mean(                                                                                                                                                                                                                                                                           | g*t − g*{t−1} | ) / 255 |
| clutter c_t             | fraction of pixels marked by Canny(g_t, 100, 200) (thresholds in config)                                                                                                                                                                                                        |
| count n_t               |                                                                                                                                                                                                                                                                                 | D\_{t−1}      |         |
| uncertainty u_t         | 1 − mean(score over D*{t−1}); if D*{t−1} is empty, u_t = u_empty (config, default 0)                                                                                                                                                                                            |
| scene change s_t        | Bhattacharyya distance between 32-bin HSV (H,S) histograms of t and t−1; binary s_t = 1[dist ≥ θ_sc]                                                                                                                                                                            |
| hardware h_t (optional) | slowdown ratio: h*t = clip((ℓ̄_t / ℓ_iso(L*{t−1})) − 1, 0, 1), where ℓ̄_t is the EMA (β = 0.3) of observed engine latency and ℓ_iso is that engine's isolated median latency from microbench. NVML GPU utilisation is also logged but not used, because it includes our own work. |

The first frame of each sequence gets m = 0, s = 0 and n, u from u_empty, and the controller starts at FP16.

### 3.2 Normalisation (fit on calib split only, stored in the locked config)

For continuous cues x ∈ {m, c, n, u}: x̃ = clip((x − p5_x) / (p95_x − p5_x), 0, 1), where p5/p95 are percentiles over all calibration frames (for n and u, computed from FP16 cache detections). s_t is already binary.

### 3.3 Score

```
S_t  = w_m·m̃_t + w_c·c̃_t + w_n·ñ_t + w_u·ũ_t + w_s·s_t        with w ≥ 0, Σw = 1
Ŝ_t  = α·S_t + (1−α)·Ŝ_{t−1}                                    (EMA; α default 0.6; α=1 disables)
Ŝ'_t = Ŝ_t − w_h·h_t                                             (hardware term; w_h = 0 when off)
```

Default weights are equal (0.2 each), and the calibration grid can move them (§3.5).

### 3.4 Level mapping, persistence, overrides

Levels are ordered low→high cost: `[int4?] < int8 < fp16`. FP32 is a baseline only and is never chosen by the controller.

- Two-level (must-have): proposed L\*\_t = INT8 if Ŝ'\_t < θ, else FP16.
- Three-level (if INT4): INT4 if Ŝ' < θ₁; INT8 if θ₁ ≤ Ŝ' < θ₂; else FP16.
- **Persistence:** state (L_cur, cand, k). If L*\_t = L_cur, set k ← 0. Else if L*\_t = cand, k ← k+1; else cand ← L\*\_t, k ← 1. Switch L_cur ← cand when k ≥ K_up (if cand is a higher level) or k ≥ K_down (if lower). Default K_up = K_down = K.
- **Scene-change override:** if s_t = 1, set L_cur ← FP16 immediately (bypassing persistence), reset k, and hold FP16 for H frames (default H = 3).
- **Hardware override (nice-to-have):** if h_t > h_hi (default 0.5) for K frames, cap L_cur at one level below FP16.
- Unit tests: oscillating proposals never switch with K > 1; a constant proposal switches after exactly K frames; the override fires on the same frame.

### 3.5 Calibration (calib split only) and locking

- **Splits:** calibration sequences come from the official BDD100K MOT **train** split, and test sequences from the official MOT **val** split, so they're disjoint by construction (§4.1). VisDrone uses VID train for calib and VID val for test.
- **Objective:** maximise latency saving 1 − ℓ̄/ℓ̄_FP16 (replay latency) subject to mAP@0.5:0.95(policy) ≥ mAP@0.5:0.95(FP16) − δ on calib, with δ = 0.5 pts (pre-registered). If several configs tie, prefer the lower switch rate, then larger K.
- **Search space:**
  - θ ∈ quantiles {0.05, 0.10, …, 0.95} of Ŝ on calib (thresholds are quantiles, which makes them interpretable);
  - K ∈ {1, 2, 3, 5}; α ∈ {0.6, 1.0}; θ_sc ∈ {0.3, 0.4, 0.5} (Bhattacharyya);
  - weights: equal-weights first; then the simplex grid with step 0.25 over the 5 content weights (70 combos), but only if time permits.
- **Two-stage to keep it fast.** Stage A ranks all configs by the proxy Σ per-frame F1@0.5 (§6) and replay latency. Stage B computes true pycocotools mAP for the top 20 and picks by the objective. (Optionally use `faster-coco-eval` for speed [VERIFY package/API].)
- Also save the full calib Pareto front (saving vs ΔmAP for δ ∈ {0.25, 0.5, 1, 2}). The test-time Pareto plot re-runs those δ-locked configs on test.
- **Locking:** `calibrate.py` writes `configs/locked/locked_bdd_main.yaml` (normalisation percentiles, weights, θ, K, α, θ_sc, H, τ_c, engine hashes, git commit) plus `.sha256`. `07_eval_test.py` refuses to run unless the locked file's hash matches and the hash is recorded in each result's `meta.json`.

---

## 4. Dataset subsets and objective slice labels

### 4.1 BDD100K (primary)

- Use the **BDD100K MOT 2020 (box tracking)** set, not the 100K detection set: the detection set labels only one keyframe per video, while MOT has dense per-frame boxes with track IDs. From memory: ~1,400 train / 200 val / 400 test sequences, ~40 s each, labelled at 5 fps, with 8 evaluated classes [VERIFY all counts against doc.bdd100k.com]. Test labels are not public, so we never use the official test split.
- Download requires registration on the BDD100K site and acceptance of the licence (§10). Download only `MOT 2020 images (val)`, the val/train box_track labels, and **only the train sequences we sample** if per-sequence download is possible; otherwise download the train zip, extract the 20 sampled sequences, and delete the rest. Record archive names and SHA-256 in `results/env/data_manifest.json`.
- **Subset (seeded, `seed=2027`):**
  - **Test:** 40 sequences from MOT val, stratified by sequence-level time of day (target ~24 daytime, ~12 night, ~4 dawn/dusk, subject to availability). Use the label JSON's `attributes.timeofday` if present [VERIFY field exists in box_track labels]; otherwise use the luma rule from §4.3 on the median frame.
  - **Calib:** 20 sequences from MOT train, with the same stratification.
  - **INT8 calib images:** 500 frames sampled uniformly from the 20 calib sequences.
  - Expected size: about 8k test frames and 4k calib frames (verify after build and record actual counts).
- Manifest: `data/manifests/bdd_{calib,test}.csv` with seq, frame_idx, path, image_id, W, H, plus slice columns.

### 4.2 VisDrone (nice-to-have)

- Use **VisDrone2019-VID** (video, per-frame boxes with track IDs). Use val sequences as test and a few train sequences as calib. Sequence counts per split (from memory: 56 train / 7 val / 17 test-dev) [VERIFY].
- Annotation format per line: `frame, track_id, x, y, w, h, score, category, truncation, occlusion`. category 0 = ignored region and 11 = others; score 0 marks a box to ignore [VERIFY against the VisDrone toolkit README].
- Subset: all 7 val sequences (test), and 6 train sequences (calib) chosen with a seed. If frames are > 30 fps-dense, subsample every 3rd frame and record the rate.

### 4.3 Slice labels (objective, frame-level, thresholds fixed on calib before test)

All slice columns are written to the manifest by `slices.py`. A frame can belong to several slices, except `easy`, which is exclusive.

| Slice            | Rule                                                                                                                                                                                                                                                                             |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------ | ----- | -------------- | ----- | ------------------------- |
| **night**        | mean luma Y (BT.601) of the full-res frame < τ_night. For BDD, τ_night is chosen on calib to maximise agreement with `timeofday == night` (if the attribute exists); otherwise it's the valley of the calib luma histogram (Otsu on frame-mean luma). Report the agreement rate. |
| **crowded**      | number of evaluated GT objects (non-ignore) in the frame ≥ N_crowd = 90th percentile of the per-frame GT count on calib (per dataset)                                                                                                                                            |
| **scene-change** | frames t ∈ [t_splice, t_splice + W−1] in spliced streams (§4.4), W = 5. Secondary, natural: frames with new-track ratio                                                                                                                                                          | IDs*t \ IDs*{t−1} | / max( | IDs_t | , 1) ≥ 0.5 and | IDs_t | ≥ 3 (reported separately) |
| **easy**         | not night, not crowded, not scene-change, and GT count ≤ median calib GT count                                                                                                                                                                                                   |
| **all**          | every test frame                                                                                                                                                                                                                                                                 |

### 4.4 Spliced scene-change streams

Splice the test sequences into 20 streams (seeded): each stream concatenates 2–3 segments of 40 frames from different test sequences with different `timeofday` where possible, so day↔night cuts occur. Splices are evaluated as additional streams. Their GT is the original frame GT, so no new labels are needed. Main "all"-slice numbers are computed on **unspliced** sequences; the scene-change slice comes from the spliced streams only.

---

## 5. Class mapping and mAP

### 5.1 COCO → BDD100K MOT (eval classes)

| BDD class                            | COCO class (id in the Ultralytics 80-class index)                                |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| pedestrian                           | person (0)                                                                       |
| rider                                | person (0) — see note                                                            |
| car                                  | car (2)                                                                          |
| truck                                | truck (7)                                                                        |
| bus                                  | bus (5)                                                                          |
| train                                | train (6)                                                                        |
| motorcycle                           | motorcycle (3)                                                                   |
| bicycle                              | bicycle (1)                                                                      |
| other person, other vehicle, trailer | **ignore** (GT added as `iscrowd=1` so matching detections aren't counted as FP) |

Note: COCO has no separate "rider" class, so pedestrian and rider are merged into **person** for evaluation. That gives 7 evaluated categories: person, car, truck, bus, train, motorcycle, bicycle. All other COCO predictions are dropped before evaluation. Also respect BDD's per-box `crowd` attribute, if present, as `iscrowd=1` [VERIFY attribute name].

### 5.2 COCO → VisDrone

| VisDrone                          | COCO                                         |
| --------------------------------- | -------------------------------------------- |
| pedestrian (1), people (2)        | person                                       |
| bicycle (3)                       | bicycle                                      |
| car (4), van (5)                  | car (design decision: COCO has no van class) |
| truck (6)                         | truck                                        |
| bus (9)                           | bus                                          |
| motor (10)                        | motorcycle                                   |
| tricycle (7), awning-tricycle (8) | **ignore** (no COCO equivalent)              |
| ignored region (0), others (11)   | **ignore** (`iscrowd=1`)                     |

Evaluated: person, bicycle, car, truck, bus, motorcycle (6 classes). State the merges in the paper.

### 5.3 mAP computation

- Convert GT to COCO JSON (`coco_gt.py`): one image entry per frame, with category IDs = our evaluated set and ignore boxes as `iscrowd=1, area=w·h`.
- Predictions: all detections with conf ≥ 0.001, NMS IoU 0.7, max_det 300 (Ultralytics val-style), mapped classes only, converted to xywh in original image coordinates.
- Evaluate with `pycocotools.COCOeval(iouType="bbox")`. Report `stats[1]` = mAP@0.5 and `stats[0]` = mAP@0.5:0.95. For slices, set `params.imgIds` to the slice frames. Also save per-class AP.
- **Sanity check (Day 1):** run the FP32 `.pt` model through our pipeline and through `model.val()` on a 200-frame COCO-format mini set. The numbers must agree to within 0.5 pts; if they don't, fix the pipeline before going on. Save the check to `results/sanity/`.
- **CIs:** cluster bootstrap over sequences (1,000 resamples, seed fixed) for mAP and for the difference adaptive − baseline (paired: same resampled sequences).

---

## 6. Oracle

The oracle works from the cache, with full knowledge of GT, per frame, with no persistence and zero switch cost. It's an upper reference, not an achievable policy.

1. For each frame t and level ℓ ∈ {int8, fp16} (+int4 if present), compute **per-frame F1@0.5**: class-aware greedy matching of detections at conf ≥ τ_c (0.25) to non-ignore GT at IoU ≥ 0.5 (detections matching ignore boxes are discarded). If a frame has no GT and no detections, F1 = 1; if it has no GT but some detections, F1 = 0.
2. **Oracle-cheap (main):** choose the cheapest level whose F1_ℓ(t) ≥ F1_fp16(t) − ε (ε = 0). This answers "how much cheaper could we go without losing per-frame quality".
3. **Oracle-best:** argmax*ℓ F1*ℓ(t), with ties going to the cheapest level. This answers "how much better could accuracy get by mixing".
4. Then compute dataset-level mAP of the oracle's chosen detections like any other policy, along with replay latency and energy.
5. Caveat for the paper: per-frame F1 is a proxy, so a per-frame oracle is not guaranteed to be mAP-optimal at the dataset level. State this.
6. **Headroom for RQ1:** the fraction of frames where INT8 F1 ≥ FP16 F1, and the oracle-cheap saving vs FP16.

### 6.1 Random baseline (matched)

Random switching is a Markov chain over levels whose stationary distribution and transition probabilities are estimated from the adaptive policy's test log. That way it matches the adaptive policy's precision mix **and** switch rate, and only the _placement_ of switches differs. Run 10 seeds and report mean ± SD. Also run a naive i.i.d. 50/50 random as a secondary point.

---

## 7. Energy, latency and memory measurement (Colab / Kaggle)

### 7.1 Environment capture (every run → `meta.json`)

GPU name, UUID, driver, CUDA, TensorRT, Ultralytics, torch versions; `nvidia-smi -q` snapshot (clocks, power limit, persistence mode); Colab vs Kaggle; date/time. Use **one GPU type for all timing/energy tables**; expect a T4 (Turing, FP16/INT8 tensor cores). If Kaggle assigns a P100, don't use it for INT8 timing, because Pascal sm_60 has no INT8 tensor/DP4A path [VERIFY]. On Kaggle "T4 ×2", pin `CUDA_VISIBLE_DEVICES=0`.

### 7.2 Meter (`nvml_meter.py`)

- **Primary:** `nvmlDeviceGetTotalEnergyConsumption` (mJ counter, Volta and newer). Read it at the start and end of each window: ΔE = E_end − E_start. Check on Day 1 that it works inside the Colab/Kaggle VM [VERIFY]; log the result in `results/env/`.
- **Fallback:** a background thread samples `nvmlDeviceGetPowerUsage` (mW) every 20 ms, with trapezoidal integration. Note that the board power reading is internally averaged (on recent drivers `power.draw` is averaged over ~1 s, and `power.draw.instant` may exist [VERIFY]), so **only long windows are meaningful**.
- Also sample utilisation, SM clock, memory used and temperature at 10 Hz for the logs.
- Cross-check once: run both methods on the same window and report agreement.

### 7.3 Protocol (`protocol.py`)

1. Load all engines needed for the configuration (they stay resident, as in deployment).
2. **Pre-decode and pre-resize** the frames of the measurement set into pinned host memory, so disk/JPEG decode is excluded. End-to-end latency including decode is measured separately and reported as such.
3. **Warm-up:** 30 s of continuous inference on the configuration, discarded. Wait until GPU temperature changes < 1 °C over 20 s (max 120 s).
4. **Idle baseline:** 30 s with engines resident and no work. P_idle = median power, or ΔE_idle/30 s.
5. **Measure:** loop over the measurement frames continuously for ≥ 60 s (repeat the sequence list as needed); N = frames processed and T = elapsed wall time.
6. Report per frame:
   - E_total/frame = ΔE / N
   - E_dyn/frame = (ΔE − P_idle·T) / N (idle-subtracted; the primary metric)
   - plus mean power and throughput N/T.
7. **Interleave and repeat:** run order is randomised per repeat ([fp32, fp16, int8, adaptive] shuffled), 3 repeats, in one session. Report the median and range across repeats.
8. **Latency:** per-frame engine latency (CUDA-synchronised), report median and p95 over frames; FPS = 1000 / mean end-to-end latency at batch 1. Also report the breakdown: pre / infer / post / controller.
9. **Switch penalty:** a live run alternating levels every frame vs a constant level over the same frames. Penalty = (mean latency difference per switch). Measured once and stored in `results/microbench/switch_penalty.json`; it's used by replay.
10. **Controller overhead:** median time of `features.py` + `Controller.step` per frame, measured live.

### 7.4 Memory

- GPU memory via NVML `memory.used`: measured after CUDA context init (baseline), then after loading each engine alone, then after loading the adaptive set (INT8+FP16). Report the deltas. The adaptive policy's resident memory is the **sum** of its engines; that's a real cost and must be shown.
- Also report engine file sizes and host RSS.

### 7.5 tegrastats (not available; for completeness only)

If a Jetson becomes available: `tegrastats --interval 100 --logfile ...`, parse the `VDD_IN` (total module power) rail [VERIFY rail names per Jetson model], with the same warm-up / idle / ≥ 60 s protocol and `nvpmodel`/`jetson_clocks` state recorded. **Not in scope for 25 Nov.**

---

## 8. Results schema (the paper reads only these files)

Every experiment writes `results/<dataset>/<exp_id>/`:

**`meta.json`** (required; `results.py` refuses to write without it)

```json
{
  "exp_id": "bdd_test_adaptive_main_replay",
  "dataset": "bdd100k_mot",
  "split": "test",
  "subset_manifest_sha256": "...",
  "policy": "adaptive|fixed_fp32|fixed_fp16|fixed_int8|random_matched|random_iid|oracle_cheap|oracle_best|ablation_<id>",
  "mode": "replay|live",
  "locked_config": "configs/locked/locked_bdd_main.yaml",
  "locked_config_sha256": "...",
  "model": "yolo11s",
  "imgsz": 640,
  "engine_sha256": { "fp16": "...", "int8": "..." },
  "git_commit": "...",
  "seed": 2027,
  "env": {
    "gpu": "...",
    "driver": "...",
    "cuda": "...",
    "tensorrt": "...",
    "ultralytics": "...",
    "torch": "...",
    "platform": "colab|kaggle"
  },
  "timestamp_utc": "...",
  "energy_method": "nvml_energy_counter|nvml_power_sampling|none",
  "notes": ""
}
```

**`per_frame.csv`** — one row per frame:
`stream_id, seq, frame_idx, image_id, level, switched(0/1), score_S, score_hat, m, c, n, u, s, h, latency_pre_ms, latency_infer_ms, latency_post_ms, latency_ctrl_ms, latency_total_ms, energy_est_mJ, n_dets_025, f1_05, slice_night, slice_crowded, slice_scene_change, slice_easy`

**`metrics.json`** — per slice ∈ {all, easy, crowded, night, scene_change}:

```json
{
  "all": {
    "n_frames": 0,
    "map50": 0.0,
    "map50_95": 0.0,
    "map50_ci": [0, 0],
    "map50_95_ci": [0, 0],
    "lat_mean_ms": 0.0,
    "lat_median_ms": 0.0,
    "lat_p95_ms": 0.0,
    "fps": 0.0,
    "energy_dyn_mJ_per_frame": 0.0,
    "energy_total_mJ_per_frame": 0.0,
    "energy_kind": "measured|estimated",
    "gpu_mem_MiB": 0.0,
    "switch_rate_per_100": 0.0,
    "frac_level": { "int8": 0.0, "fp16": 0.0 },
    "per_class_ap50_95": {}
  }
}
```

(zeros are placeholders for the schema only)

**`energy.json`** (energy runs): per repeat: ΔE, T, N, P_idle, warm-up duration, temperatures, clocks.

**`results/index.csv`** — one row per exp: `exp_id, dataset, split, policy, mode, path, locked_config_sha256, git_commit, timestamp`. `make_tables.py` selects runs only through this index and fails loudly on missing or duplicate entries.

Also: `results/microbench/*.json`, `results/sanity/*.json`, `results/env/*.json`, `results/calib/<dataset>/grid.csv` (every calib config tried, so the search is transparent).

**Paper generation:** `make_numbers.py` emits `\newcommand{\AdaMapAll}{...}`-style macros into `paper/numbers.tex`; the text uses only these macros. `make_figs.py` produces vector PDFs (Pareto plot, K-sweep, per-slice bars, level-over-time trace, latency CDF) with colour-blind-safe palettes, markers/line styles so colour isn't the only carrier (Springer requirement), and an alt-text string per figure written to `paper/figs/alt_text.json`.

---

## 9. Schedule (Fri 20 → Wed 25 Nov 2026)

### Must-have (paper is not submittable without these)

- BDD100K MOT subsets (calib/test), slices, splice streams
- TensorRT FP32/FP16/INT8 engines (YOLO11s, or 11m per rule), mAP sanity check
- Cache runs for all levels on calib + test
- Two-level controller (INT8/FP16) with persistence + scene-change override; calibration + lock
- Baselines: FP32, FP16, INT8, random-matched, oracle-cheap/best
- Ablations: leave-one-cue-out, K-sweep, no override
- Latency/memory/switch-rate for all; **measured** energy for FP32/FP16/INT8/adaptive (live)
- Live validation of adaptive vs replay
- Cluster-bootstrap CIs; per-slice tables
- Paper with generated tables/figures; references verified

### Nice-to-have (in priority order)

1. VisDrone-VID replication (calib + test, main table only)
2. Single-cue-only ablations, weight-simplex search
3. Hardware-aware term under synthetic contention (RQ6/H5)
4. Second model size (if not already triggered by the rule)
5. INT4 (simulated accuracy only, clearly labelled)

### Day by day

| Day                                   | Work                                                                                                                                                                                                                                                                                                                       | Exit criterion                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Fri 20 (today, afternoon/evening)** | Register for BDD100K and start downloads (MOT val images + box_track labels; train sequences). Repo skeleton, `00_env.sh`, pinned versions. Write `preregistration.yaml`. Parse BDD MOT → manifests; class map + COCO GT converter; unit tests for classmap. Start the VisDrone-VID download in parallel (cheap to start). | Manifests for calib/test exist; GT JSON validates in pycocotools          |
| **Sat 21**                            | Export FP32/FP16/INT8 engines (INT8 calibrated on calib frames). Microbench → apply model-size rule. mAP sanity check. Check the NVML energy counter. `cache.py` on calib and test for all levels (resumable per sequence). Slices + splice streams.                                                                       | Cache parquet complete; sanity ≤ 0.5 pt gap; `results/microbench` written |
| **Sun 22**                            | `features.py`, `controller.py` (+tests), `replay.py`, `baselines.py`, `perframe.py`, `bootstrap.py`. Calibration on calib → **lock config**. Switch penalty + controller overhead microbench.                                                                                                                              | `locked_bdd_main.yaml` + hash; calib grid saved                           |
| **Mon 23**                            | Test eval (replay): all baselines, adaptive, ablations, K-sweep, random 10 seeds, oracle. Energy protocol for FP32/FP16/INT8 + live adaptive (3 interleaved repeats). Live validation. `make_tables/figs/numbers`. **Nice-to-have:** VisDrone through the same pipeline if the cache finishes overnight.                   | All must-have results in `results/`; figures render                       |
| **Tue 24**                            | Write the full paper draft (intro, related work, method, setup, results, threats, conclusion). Update the abstract to reflect the actual hardware (remove the Jetson claim). **Code freeze 18:00 IST** — after this, only re-runs of locked configs. Nice-to-haves only if the draft is complete by noon.                  | Complete draft within page limit                                          |
| **Wed 25**                            | Verify every reference (DOI), clear all [VERIFY] tags, alt text, licence/acknowledgement section, Springer permissions checklist (no third-party content expected), final compile, submit.                                                                                                                                 | Submitted                                                                 |

**Risk triggers:**

- If the BDD100K download isn't done by Sat 12:00, start caching on whatever val sequences are present and cut test to 25 sequences.
- If INT8 export fails, fall back to ONNX Runtime TensorRT EP or the TensorRT Python API directly (time-boxed to 3 h); if that also fails, report FP16 vs FP32 adaptivity and state the limitation.
- If Colab is reassigned a different GPU mid-way, redo all timing/energy on one GPU type (the cache detections stay valid if engines are rebuilt from the same weights — verify that mAP matches within 0.1 pt).

### Paper constraints

- Paper length is set by the conference, not by Springer, according to the Springer proceedings guidance page. **IRIS 2027 page limit: [VERIFY on the IRIS 2027 call for papers].** Working assumption until verified: plan for ~10–12 pages incl. references [VERIFY].
- Springer asks for alt text on all figures and tables, adequate colour contrast, and colour that isn't the sole carrier of information; third-party content only if strictly necessary, with a permissions checklist if used; and authors must check AI-generated content for fabricated references.
- Use the Springer LaTeX template specified in the IRIS author kit [VERIFY which class].

---

## 10. Licences and their implications

- **Ultralytics (code + pretrained COCO weights): AGPL-3.0.** Ultralytics states that free use under AGPL-3.0 requires open-sourcing the entire project under AGPL-3.0, and that an Enterprise licence is needed otherwise. Implications for us:
  - Release the AdaPres repo under AGPL-3.0 (or don't distribute it). A research paper itself isn't affected; distributing our code/engines is.
  - Don't publish TensorRT engines built from Ultralytics weights unless that's under AGPL terms too (and engines are GPU-specific anyway; publish export scripts instead).
  - Mention in the paper that any deployment of this pipeline inherits AGPL-3.0 or needs a commercial licence.
  - Cite Ultralytics per its `CITATION.cff` for the pinned version [VERIFY].
- **BDD100K:** the toolkit code is BSD-3-Clause, but the **data** is under the BDD100K licence (doc.bdd100k.com/license.html), which, as quoted in redistributions, grants use for educational, research and not-for-profit purposes [VERIFY full text]. Implications:
  - Non-commercial research only; don't redistribute frames or labels (keep `data/` gitignored; publish only sequence IDs/manifests without images).
  - No dataset frames in figures unless the licence text explicitly permits reproduction in publications [VERIFY]; default is no frames.
  - Cite the BDD100K paper as the dataset authors request.
- **VisDrone:** released for academic/research use; check the terms on the VisDrone GitHub/website before download [VERIFY]. Same no-redistribution and no-frames-in-figures defaults.
- **COCO weights:** trained on COCO images (mixed Flickr licences); weight licence follows Ultralytics AGPL-3.0. COCO annotations are CC BY 4.0 [VERIFY].
- **Springer:** the publishing agreement transfers exclusive publication rights. Keep code licensing (AGPL) separate from the paper; the paper links to the repo.

---

## 11. Candidate references (all [VERIFY] before use; add only those actually cited)

- Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning," CVPR 2020. [VERIFY pages/DOI]
- Zhu et al., "Detection and Tracking Meet Drones Challenge," IEEE TPAMI 2021 (VisDrone). [VERIFY]
- Lin et al., "Microsoft COCO: Common Objects in Context," ECCV 2014. [VERIFY DOI]
- Jocher et al., Ultralytics YOLO (software; YOLO11), per `CITATION.cff`. [VERIFY version/year]
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018. [VERIFY]
- Nagel et al., "A White Paper on Neural Network Quantization," arXiv 2021. [VERIFY]
- Wang et al., "HAQ: Hardware-Aware Automated Quantization with Mixed Precision," CVPR 2019. [VERIFY]
- Chin et al., "AdaScale: Towards Real-time Video Object Detection Using Adaptive Scaling," MLSys 2019. [VERIFY]
- Jiang et al., "Chameleon: Scalable Adaptation of Video Analytics," SIGCOMM 2018. [VERIFY]
- Xu et al., "ApproxDet: Content and Contention-Aware Approximate Object Detection for Mobiles," SenSys 2020. [VERIFY] (closest prior work on content + contention awareness, so positioning against it is mandatory)
- Liu et al., "Instance-Aware Dynamic Neural Network Quantization," CVPR 2022. [VERIFY]
- NVIDIA TensorRT Developer Guide (INT8 calibration, precision support), version matching the pinned TensorRT. [VERIFY]
- NVIDIA NVML API Reference (`nvmlDeviceGetTotalEnergyConsumption`, `nvmlDeviceGetPowerUsage`). [VERIFY]

Related-work search to run on Day 3–4 (the author, not the engineer): "dynamic precision video object detection", "adaptive quantization per input", "content-aware model switching edge video analytics". Add only papers that were actually opened and read.
