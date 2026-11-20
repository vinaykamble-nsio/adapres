"""Build BDD100K MOT calib/test subsets, slice labels, splice streams and COCO GT.

KAGGLE-ONLY when run on real data (reads dataset images). Unit tests drive it with a
tiny synthetic dataset and a mocked image reader via `read_rgb`.

Outputs
  results/bdd100k_mot/bdd_subsets/   (small, commit these)
    meta.json, selection.json, frames_calib.csv, frames_test.csv, splice_streams.csv,
    int8_calib_frames.csv, slice_thresholds.json, report.json
  data/bdd/                          (derived, NOT committed: contains GT boxes)
    coco_gt_calib.json, coco_gt_test.json, coco_gt_splice.json, luma/<seq>.csv
"""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import config_sha256, resolve_out
from ..data import bdd_mot, slices, splice, subsets
from ..eval.coco_gt import build_coco_gt, gt_counts
from ..eval.coco_metrics import ap_from_eval
from ..repro import capture_env, set_seed
from ..results import ExperimentWriter

log = logging.getLogger("adapres.build_bdd_subsets")

ReadRGB = Callable[[Path], np.ndarray]


class BuildError(RuntimeError):
    pass


def cv2_read_rgb(path: Path) -> np.ndarray:
    import cv2  # lazy: not needed for local unit tests

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise BuildError(f"cv2 could not read image: {path}")
    return img[:, :, ::-1]


def _luma_and_size(path: Path, read_rgb: ReadRGB) -> tuple[float, int, int]:
    rgb = read_rgb(path)
    return slices.luma_bt601(rgb), rgb.shape[1], rgb.shape[0]


def _seq_luma(seq: str, frames: list[bdd_mot.Frame], image_root: Path, cache_dir: Path,
              read_rgb: ReadRGB, workers: int) -> dict[int, tuple[float, int, int]]:
    """Per-frame (luma, W, H) for one sequence, checkpointed to cache_dir/<seq>.csv."""
    cache = cache_dir / f"{seq}.csv"
    if cache.exists():
        out = {}
        with open(cache, newline="") as f:
            for r in csv.DictReader(f):
                out[int(r["frame_idx"])] = (float(r["luma"]), int(r["W"]), int(r["H"]))
        if set(out) == {fr.frame_idx for fr in frames}:
            return out
        log.warning("luma cache for %s incomplete; recomputing", seq)
    paths = [image_root / seq / fr.name for fr in frames]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        vals = list(ex.map(lambda p: _luma_and_size(p, read_rgb), paths))
    out = {fr.frame_idx: v for fr, v in zip(frames, vals, strict=True)}
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "luma", "W", "H"])
        for k in sorted(out):
            w.writerow([k, f"{out[k][0]:.4f}", out[k][1], out[k][2]])
    tmp.replace(cache)
    return out


def validate_coco_gt(gt: dict) -> dict:
    """Load with pycocotools and evaluate GT-as-detections. Non-crowd GT used as
    detections (score 1) must give AP ~ 1.0; anything else means a converter bug."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO()
    coco.dataset = json.loads(json.dumps(gt))
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
            for a in gt["annotations"] if not a["iscrowd"]]
    if not dets:
        raise BuildError("COCO GT has no non-crowd annotations")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = coco.loadRes(dets)
        ev = COCOeval(coco, res, iouType="bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
    out = {"n_images": len(gt["images"]), "n_annotations": len(gt["annotations"]),
           "n_crowd_or_ignore": sum(a["iscrowd"] for a in gt["annotations"]),
           "gt_as_det_map50_95": ap_from_eval(ev), "gt_as_det_map50": ap_from_eval(ev, iou=0.5)}
    if out["gt_as_det_map50_95"] < 0.99:
        raise BuildError(f"GT-as-detection mAP {out['gt_as_det_map50_95']:.4f} < 0.99: converter bug\n{buf.getvalue()}")
    return out


def run(cfg: dict, smoke: bool, repo_dir: str | Path, search_root: str | Path | None = None,
        read_rgb: ReadRGB = cv2_read_rgb, workers: int = 8) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)
    bcfg = cfg["bdd"]
    sub_cfg = bcfg["smoke"] if smoke else bcfg["subsets"]
    results_root = resolve_out(cfg, "results_dir", smoke)
    data_dir = resolve_out(cfg, "data_dir", smoke) / "bdd"
    (data_dir / "luma").mkdir(parents=True, exist_ok=True)

    # ---- 1. layout
    if search_root is None:
        pref = Path(cfg["paths"]["input_root"]) / cfg["paths"]["bdd_dataset_slug"]
        search_root = pref if pref.exists() else Path(cfg["paths"]["input_root"])
    layout = bdd_mot.discover_layout(search_root, bcfg["images_dir_name"], bcfg["labels_dir_name"],
                                     split=bcfg["label_split"])
    log.info("layout: labels=%s (%d json) images=%s (%d video dirs)", layout.label_dir,
             layout.n_label_files, layout.image_root, layout.n_image_dirs)

    # ---- 2. parse labels, check images exist
    videos: dict[str, list[bdd_mot.Frame]] = {}
    for p in sorted(layout.label_dir.glob("*.json")):
        frames = bdd_mot.load_video_labels(p)
        if not frames:
            continue
        videos[frames[0].seq] = frames
    if not videos:
        raise BuildError(f"no frames parsed from {layout.label_dir}")
    missing: dict[str, int] = {}
    for seq, frames in videos.items():
        n_miss = sum(not (layout.image_root / seq / fr.name).exists() for fr in frames)
        if n_miss:
            missing[seq] = n_miss
    eligible = {s: f for s, f in videos.items() if s not in missing}
    if missing:
        log.warning("%d sequences have missing images and are excluded: %s", len(missing),
                    dict(list(missing.items())[:10]))
    if len(missing) > 0.1 * len(videos):
        raise BuildError(f"{len(missing)}/{len(videos)} sequences have missing images (>10%); "
                         f"check the dataset mirror. Examples: {dict(list(missing.items())[:10])}")

    # ---- 3. time of day per sequence
    tod: dict[str, str] = {}
    tod_source: dict[str, str] = {}
    unresolved = []
    for seq, frames in eligible.items():
        t = bdd_mot.normalise_timeofday(bdd_mot.sequence_timeofday(frames))
        if t is None:
            unresolved.append(seq)
        else:
            tod[seq], tod_source[seq] = t, "label_attribute"
    luma_fallback = None
    if unresolved:
        med_luma = {}
        for seq in unresolved:
            fr = eligible[seq][len(eligible[seq]) // 2]
            med_luma[seq] = slices.luma_bt601(read_rgb(layout.image_root / seq / fr.name))
        thr = slices.otsu_threshold(np.array(list(med_luma.values())))
        for seq, y in med_luma.items():
            tod[seq], tod_source[seq] = ("night" if y < thr else "daytime"), "median_frame_luma_otsu"
        luma_fallback = {"n_sequences": len(unresolved), "otsu_threshold": thr}
    have_attr = any(v == "label_attribute" for v in tod_source.values())

    # ---- 4. select
    targets = {sp: dict(sub_cfg[sp]["strata_targets"]) for sp in ("calib", "test")}
    if not have_attr:
        # luma cannot identify dawn/dusk: fold that target into daytime (recorded)
        for sp in targets:
            targets[sp]["daytime"] = targets[sp].get("daytime", 0) + targets[sp].pop("dawn_dusk", 0)
    selection = subsets.select_disjoint_splits(tod, targets, seed)
    selection["timeofday_source"] = "label_attribute" if have_attr else "median_frame_luma_otsu"
    selection["seq_timeofday"] = {s: tod[s] for sp in selection["splits"] for s in selection["splits"][sp]}
    selection["smoke"] = smoke
    selection["source_split"] = f"bdd100k_mot_2020/{bcfg['label_split']}"

    # ---- 5. per-frame luma/size + counts for selected sequences
    frame_rows: dict[str, list[dict]] = {"calib": [], "test": []}
    frames_by_key: dict[tuple[str, int], bdd_mot.Frame] = {}
    size_by_key: dict[tuple[str, int], tuple[int, int]] = {}
    image_id = 1
    for sp in ("calib", "test"):
        for seq in selection["splits"][sp]:
            frames = eligible[seq]
            lum = _seq_luma(seq, frames, layout.image_root, data_dir / "luma", read_rgb, workers)
            nsc = slices.natural_scene_change(
                [{b.track_id for b in fr.boxes if b.kind == "eval"} for fr in frames],
                cfg["slices"]["natural_scene_change"]["new_track_ratio"],
                cfg["slices"]["natural_scene_change"]["min_tracks"])
            for fr, ns in zip(frames, nsc, strict=True):
                y, w, h = lum[fr.frame_idx]
                frames_by_key[(seq, fr.frame_idx)] = fr
                size_by_key[(seq, fr.frame_idx)] = (w, h)
                frame_rows[sp].append({
                    "split": sp, "seq": seq, "frame_idx": fr.frame_idx, "image_id": image_id,
                    "rel_path": f"{seq}/{fr.name}", "W": w, "H": h, "luma": round(y, 4),
                    "gt_count": gt_counts(fr),
                    "n_ignore": sum(b.kind == "ignore" or b.crowd for b in fr.boxes),
                    "timeofday_seq": tod[seq], "timeofday_source": tod_source[seq],
                    "natural_sc": int(ns)})
                image_id += 1

    # ---- 6. thresholds from calib ONLY
    cal = frame_rows["calib"]
    cal_luma = np.array([r["luma"] for r in cal])
    cal_night = np.array([r["timeofday_seq"] == "night" for r in cal]) if have_attr else None
    thr = slices.fit_night_threshold(cal_luma, cal_night)
    thr.update(slices.fit_count_thresholds(np.array([r["gt_count"] for r in cal]),
                                           cfg["slices"]["crowded_percentile"]))
    thr["fit_on"] = "calib"
    thr["n_calib_frames"] = len(cal)

    for sp in ("calib", "test"):
        rows = frame_rows[sp]
        lab = slices.label_frames(np.array([r["luma"] for r in rows]), np.array([r["gt_count"] for r in rows]),
                                  np.zeros(len(rows), bool), np.array([r["natural_sc"] for r in rows], bool),
                                  thr, cfg["slices"]["easy_excludes_natural_scene_change"])
        for i, r in enumerate(rows):
            for k, v in lab.items():
                r[k] = int(v[i])

    # ---- 7. splice streams from TEST sequences
    scfg = cfg["splice"]
    test_seqs = selection["splits"]["test"]
    stream_rows = splice.build_streams(
        {s: [fr.frame_idx for fr in eligible[s]] for s in test_seqs}, tod,
        scfg["smoke_n_streams"] if smoke else scfg["n_streams"], scfg["segment_len"],
        scfg["min_segments"], min(scfg["max_segments"], len(test_seqs)), scfg["window"], seed)
    test_by_key = {(r["seq"], r["frame_idx"]): r for r in frame_rows["test"]}
    splice_items = []
    sid0 = 1_000_000
    for i, r in enumerate(stream_rows):
        src = test_by_key[(r["seq"], r["frame_idx"])]
        r["image_id"] = sid0 + i
        r["src_image_id"] = src["image_id"]
        r["rel_path"] = src["rel_path"]
        lab = slices.label_frames(np.array([src["luma"]]), np.array([src["gt_count"]]),
                                  np.array([bool(r["is_splice_window"])]), np.array([False]), thr,
                                  cfg["slices"]["easy_excludes_natural_scene_change"])
        for k, v in lab.items():
            r[k] = int(v[0])
        w, h = size_by_key[(r["seq"], r["frame_idx"])]
        splice_items.append((r["image_id"], frames_by_key[(r["seq"], r["frame_idx"])], w, h))

    # ---- 8. INT8 calibration frames (calib only)
    n_int8 = min(sub_cfg["int8_calib_frames"], len(cal))
    int8 = subsets.sample_frames([(r["seq"], r["frame_idx"]) for r in cal], n_int8, seed)
    cal_by_key = {(r["seq"], r["frame_idx"]): r for r in cal}
    int8_rows = [{"seq": s, "frame_idx": f, "image_id": cal_by_key[(s, f)]["image_id"],
                  "rel_path": cal_by_key[(s, f)]["rel_path"]} for s, f in int8]

    # ---- 9. COCO GT (data dir, not results) + validation
    coco_report = {}
    for name, items in (
        ("calib", [(r["image_id"], frames_by_key[(r["seq"], r["frame_idx"])], r["W"], r["H"]) for r in frame_rows["calib"]]),
        ("test", [(r["image_id"], frames_by_key[(r["seq"], r["frame_idx"])], r["W"], r["H"]) for r in frame_rows["test"]]),
        ("splice", splice_items),
    ):
        gt = build_coco_gt(items, cfg["classmap"]["ignore_mode"], description=f"bdd100k_mot {name}")
        p = data_dir / f"coco_gt_{name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(gt, f)
        coco_report[name] = validate_coco_gt(gt)
        coco_report[name]["path_relative_to_work_root"] = str(p.relative_to(Path(cfg["paths"]["work_root"]))).replace("\\", "/")

    # ---- 10. write results
    env = capture_env(repo_dir, seed)
    meta = {"exp_id": "bdd_subsets", "dataset": "bdd100k_mot", "split": "calib+test", "seed": seed,
            "git_commit": env["git"]["commit"], "git_dirty": env["git"]["dirty"], "env": env,
            "config_sha256": config_sha256(cfg), "config": cfg, "timestamp_utc": env["timestamp_utc"],
            "smoke": smoke, "mode": "data_prep"}
    wr = ExperimentWriter(results_root, meta)
    selection["selection_sha256"] = subsets.selection_sha256(
        {"splits": selection["splits"], "seed": seed})
    wr.write_json("selection.json", selection)
    frame_fields = list(frame_rows["calib"][0].keys())
    for sp in ("calib", "test"):
        wr.write_csv(f"frames_{sp}.csv", frame_rows[sp], frame_fields)
    wr.write_csv("splice_streams.csv", stream_rows, list(stream_rows[0].keys()))
    wr.write_csv("int8_calib_frames.csv", int8_rows, list(int8_rows[0].keys()))
    wr.write_json("slice_thresholds.json", thr)

    def _slice_counts(rows):
        return {k: int(sum(r[k] for r in rows)) for k in rows[0] if k.startswith("slice_")} | {"n_frames": len(rows)}

    report = {
        "layout": {"label_dir": str(layout.label_dir), "image_root": str(layout.image_root),
                   "n_label_files": layout.n_label_files, "n_image_dirs": layout.n_image_dirs},
        "n_videos_parsed": len(videos), "n_eligible": len(eligible),
        "sequences_with_missing_images": missing,
        "timeofday_counts_all_eligible": {st: sum(v == st for v in tod.values()) for st in subsets.STRATA_ORDER},
        "timeofday_source": selection["timeofday_source"], "luma_fallback": luma_fallback,
        "selection_strata_counts": selection["strata_counts"], "shortfalls": selection["shortfalls"],
        "frames": {"calib": _slice_counts(frame_rows["calib"]), "test": _slice_counts(frame_rows["test"]),
                   "splice": _slice_counts(stream_rows)},
        "n_streams": len({r["stream_id"] for r in stream_rows}),
        "int8_calib_frames": len(int8_rows), "coco_gt": coco_report, "smoke": smoke,
    }
    wr.write_json("report.json", report)
    wr.register(policy="", mode="data_prep", replace=True)
    log.info("wrote %s", wr.dir)
    return {"results_dir": str(wr.dir), "report": report}
