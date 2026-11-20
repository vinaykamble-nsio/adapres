"""Step 2 / notebook 02: fixed FP32, FP16, INT8 (+ PyTorch FP32 reference) on the
calib clips only. KAGGLE-ONLY with real backends; dry-run locally with fakes.

Stages (each a method, called by its own notebook cell; caches make them resumable):
  prepare -> int8_dataset -> export -> microbench -> cache_runs -> evaluate -> val_check -> sanity

Outputs (results/, small, commit):
  bdd100k_mot/calib_fixed_<level>/  meta.json, metrics.json, per_frame.csv
  bdd100k_mot/calib_oracle_{cheap,best}_dev/  (development only, calib)
  bdd100k_mot/engines_<model>/  engines.json (sha256, build time, export args, metadata)
  microbench/microbench_<model>/  microbench.json (+ model-size rule decision)
  sanity/sanity_02_<model>/  sanity.json, val_check.json
Work dir (notebook output, NOT committed): work02/{engines,cache,int8_calib,val_mini}
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from .. import baselines
from ..config import config_sha256, resolve_out
from ..data import bdd_mot, subsets
from ..data.yolo_format import write_yolo_dataset
from ..eval.coco_gt import build_coco_gt
from ..eval.map_eval import CocoEvaluator, dets_to_coco_results
from ..eval.perframe import gt_arrays_from_coco, perframe_f1
from ..models import engines
from ..repro import capture_env, set_seed
from ..results import ExperimentWriter

log = logging.getLogger("adapres.eval_fixed")

SLICE_COLS = {"easy": "slice_easy", "crowded": "slice_crowded", "night": "slice_night",
              "scene_change_natural": "slice_scene_change_natural"}
PER_FRAME_FIELDS = ["seq", "frame_idx", "image_id", "level", "latency_pre_ms", "latency_infer_ms",
                    "latency_post_ms", "latency_total_ms", "n_dets_025", "tp", "fp", "fn", "f1_05",
                    "slice_night", "slice_crowded", "slice_scene_change_natural", "slice_easy"]


class Step2Error(RuntimeError):
    pass


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stats(x) -> dict:
    a = np.asarray([v for v in x if v is not None], float)
    if a.size == 0:
        return {"n": 0}
    return {"median": float(np.median(a)), "mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
            "n": int(a.size)}


class Step2:
    def __init__(self, cfg: dict, smoke: bool, repo_dir: str | Path, search_root: str | Path | None = None,
                 exporter: Callable | None = None, backend_factory: Callable | None = None,
                 image_reader: Callable | None = None, val_fn: Callable | None = None):
        self.cfg, self.smoke, self.repo = cfg, smoke, Path(repo_dir)
        self.seed = int(cfg["seed"])
        set_seed(self.seed)
        self.m, self.s2 = cfg["model"], cfg["step2"]
        self.model_name = self.m["name"]
        self.levels: list[str] = list(self.s2["levels"])
        self.results_root = resolve_out(cfg, "results_dir", smoke)
        self.results_root.mkdir(parents=True, exist_ok=True)
        base = Path(cfg["paths"]["work_root"]) / ("smoke" if smoke else "")
        self.work = base / "work02"
        self.engines_dir = self.work / "engines"
        self.cache_dir = self.work / "cache" / self.model_name
        self.search_root = search_root
        self.exporter = exporter or engines.export_engine
        self.backend_factory = backend_factory or self._default_backend
        self.image_reader = image_reader or self._cv2_read
        self.val_fn = val_fn or self._ultralytics_val
        self.env = capture_env(self.repo, self.seed)
        if self.env["git"]["commit"] is None:
            raise Step2Error("no git commit available; refusing to produce unattributable results")
        self.engine_info: dict[str, dict] = {}
        # keep earlier experiments in index.csv (the zip replaces the repo's index on unzip)
        repo_index = self.repo / "results" / "index.csv"
        if repo_index.exists() and not (self.results_root / "index.csv").exists():
            shutil.copy(repo_index, self.results_root / "index.csv")

    # ------------------------------------------------------------------ defaults (Kaggle)
    def _default_backend(self, name: str, path: str | Path):
        return engines.UltralyticsBackend(name, path, self.m["imgsz"], self.m["conf"], self.m["iou"],
                                          self.m["max_det"], self.m["rect"])

    @staticmethod
    def _cv2_read(path: Path) -> np.ndarray:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            raise Step2Error(f"cannot read {path}")
        return img  # BGR, as Ultralytics expects for numpy input

    def _ultralytics_val(self, data_yaml: Path) -> dict:
        from ultralytics import YOLO

        r = YOLO(self.weights_path).val(data=str(data_yaml), imgsz=self.m["imgsz"], batch=1, conf=self.m["conf"],
                                        iou=self.m["iou"], max_det=self.m["max_det"], rect=False, plots=False,
                                        save_json=False, verbose=False, device=0, half=False)
        return {"map50_95": float(r.box.map), "map50": float(r.box.map50)}

    def _meta(self, exp_id: str, policy: str, extra: dict | None = None, dataset: str = "bdd100k_mot") -> dict:
        m = {"exp_id": exp_id, "dataset": dataset, "split": self.s2["split"], "policy": policy, "mode": "cache",
             "seed": self.seed, "git_commit": self.env["git"]["commit"], "git_dirty": self.env["git"]["dirty"],
             "env": self.env, "config_sha256": config_sha256(self.cfg), "config": self.cfg,
             "timestamp_utc": self.env["timestamp_utc"], "smoke": self.smoke, "model": self.model_name,
             "imgsz": self.m["imgsz"], "subset_manifest_sha256": self.selection_sha,
             "sequences": self.sequences,
             "engine_sha256": {k: v.get("sha256") for k, v in self.engine_info.items()},
             "energy_method": "none", "notes": "calib split only (Step 2); test untouched"}
        m.update(extra or {})
        return m

    # ------------------------------------------------------------------ stages
    def prepare(self) -> dict:
        sub = self.repo / "results" / "bdd100k_mot" / "bdd_subsets"
        rows = _read_csv(sub / f"frames_{self.s2['split']}.csv")
        report = json.loads((sub / "report.json").read_text())
        sel = json.loads((sub / "selection.json").read_text())
        self.selection_sha = sel["selection_sha256"]
        if sel.get("smoke"):
            raise Step2Error("committed bdd_subsets is a SMOKE selection; commit the full Step 1 results first")
        seqs = sorted({r["seq"] for r in rows})
        n = self.s2["smoke_n_sequences"] if self.smoke else self.s2["n_sequences"]
        if n > len(seqs):
            raise Step2Error(f"want {n} sequences, manifest has {len(seqs)}")
        if n < len(seqs):
            rng = np.random.default_rng(self.seed)
            seqs = sorted(seqs[i] for i in rng.choice(len(seqs), size=n, replace=False))
        self.sequences = seqs
        rows = [r for r in rows if r["seq"] in set(seqs)]
        rows.sort(key=lambda r: (r["seq"], int(r["frame_idx"])))
        self.rows = rows

        if self.search_root is None:
            pref = Path(self.cfg["paths"]["input_root"]) / self.cfg["paths"]["bdd_dataset_slug"]
            self.search_root = pref if pref.exists() else Path(self.cfg["paths"]["input_root"])
        b = self.cfg["bdd"]
        self.layout = bdd_mot.discover_layout(self.search_root, b["images_dir_name"], b["labels_dir_name"],
                                              split=b["label_split"])
        frames_by_key = {}
        for seq in seqs:
            for fr in bdd_mot.load_video_labels(self.layout.label_dir / f"{seq}.json"):
                frames_by_key[(seq, fr.frame_idx)] = fr
        self.frames_by_key = frames_by_key
        items = []
        for r in rows:
            key = (r["seq"], int(r["frame_idx"]))
            if key not in frames_by_key:
                raise Step2Error(f"manifest frame {key} not found in labels")
            items.append((int(r["image_id"]), frames_by_key[key], int(r["W"]), int(r["H"])))
        self.gt = build_coco_gt(items, self.cfg["classmap"]["ignore_mode"], description=f"calib step2 {n} seqs")
        check = {"n_images": len(self.gt["images"]), "n_annotations": len(self.gt["annotations"])}
        if not self.smoke:  # must reproduce Step 1's GT exactly
            exp = report["coco_gt"][self.s2["split"]]
            if (check["n_images"], check["n_annotations"]) != (exp["n_images"], exp["n_annotations"]):
                raise Step2Error(f"rebuilt GT {check} != Step 1 report {exp}: labels or code changed")
            check["matches_step1_report"] = True
        self.gt_index = gt_arrays_from_coco(self.gt)
        self.evaluator = CocoEvaluator(self.gt, self.cfg["eval"]["max_dets"])
        self.image_ids = [int(r["image_id"]) for r in rows]
        self.seq_of = {int(r["image_id"]): r["seq"] for r in rows}
        self.slices = {"all": self.image_ids}
        for name, col in SLICE_COLS.items():
            self.slices[name] = [int(r["image_id"]) for r in rows if r[col] == "1"]
        self.paths = {int(r["image_id"]): self.layout.image_root / r["rel_path"] for r in rows}
        info = {"sequences": seqs, "n_frames": len(rows), **check,
                "slice_sizes": {k: len(v) for k, v in self.slices.items()}}
        log.info("prepared: %s", info)
        return info

    def int8_dataset(self, names: dict[int, str]) -> Path:
        sub = self.repo / "results" / "bdd100k_mot" / "bdd_subsets"
        rows = _read_csv(sub / "int8_calib_frames.csv")
        calib_seqs = set(json.loads((sub / "selection.json").read_text())["splits"]["calib"])
        if any(r["seq"] not in calib_seqs for r in rows):
            raise Step2Error("INT8 calibration list contains a non-calib sequence")
        if self.smoke:
            keep = set(subsets.sample_frames([(r["seq"], int(r["frame_idx"])) for r in rows],
                                             min(self.cfg["bdd"]["smoke"]["int8_calib_frames"], len(rows)), self.seed))
            rows = [r for r in rows if (r["seq"], int(r["frame_idx"])) in keep]
        need = {r["seq"] for r in rows} - {k[0] for k in self.frames_by_key}
        for seq in sorted(need):  # INT8 frames may come from calib clips outside a smoke subset
            for fr in bdd_mot.load_video_labels(self.layout.label_dir / f"{seq}.json"):
                self.frames_by_key[(seq, fr.frame_idx)] = fr
        sizes = {(r["seq"], int(r["frame_idx"])): (int(r["W"]), int(r["H"]))
                 for r in _read_csv(sub / "frames_calib.csv")}
        items = [(r["rel_path"], self.frames_by_key[(r["seq"], int(r["frame_idx"]))],
                  *sizes[(r["seq"], int(r["frame_idx"]))]) for r in rows]
        self.n_int8_frames = len(items)
        return write_yolo_dataset(self.work / "int8_calib", items, self.layout.image_root, names)

    def export(self, weights: str | Path, int8_yaml: Path) -> dict:
        self.weights_path = str(weights)
        from ..results import sha256_file

        self.weights_sha = sha256_file(weights)
        for level in self.levels:
            log.info("exporting %s", level)
            self.engine_info[level] = self.exporter(weights, level, self.m["imgsz"], self.engines_dir,
                                                    int8_data=int8_yaml if level == "int8" else None,
                                                    batch=self.s2["int8_batch"])
        w = ExperimentWriter(self.results_root, self._meta(f"engines_{self.model_name}", "engine_export",
                                                           {"weights_sha256": self.weights_sha,
                                                            "int8_calib_frames": self.n_int8_frames}))
        w.write_json("engines.json", self.engine_info)
        w.register(policy="engine_export", mode="export", replace=True)
        return self.engine_info

    def backends(self) -> dict:
        out = {}
        if self.s2["pt_reference"]:
            out["pt_fp32"] = self.backend_factory("pt_fp32", self.weights_path)
        for level in self.levels:
            out[level] = self.backend_factory(level, self.engine_info[level]["path"])
        return out

    def microbench(self, backends: dict) -> dict:
        img = self.image_reader(self.paths[self.image_ids[0]])
        res = {name: engines.microbench(be, img, self.s2["microbench_iters"], self.s2["microbench_warmup"])
               for name, be in backends.items()}
        out = {"per_backend": res, "image_id": self.image_ids[0],
               "note": "same pre-decoded frame repeated; JPEG decode excluded; Ultralytics speed + sync wall"}
        if "fp16" in res and "int8" in res:
            out["size_rule"] = engines.size_rule(res["fp16"]["infer_ms"]["median"], res["int8"]["infer_ms"]["median"],
                                                 self.m["size_rule_min_int8_speedup"])
        w = ExperimentWriter(self.results_root, self._meta(f"microbench_{self.model_name}", "microbench",
                                                           dataset="microbench"))
        w.write_json("microbench.json", out)
        w.register(policy="microbench", mode="microbench", replace=True)
        return out

    def _cache_file(self, name: str, seq: str) -> Path:
        return self.cache_dir / name / f"{seq}.npz"

    def cache_runs(self, backends: dict) -> dict:
        """Run every backend on every selected frame once; one .npz per (backend, seq).
        Existing files are reused (resume after a crash)."""
        done = {}
        for name, be in backends.items():
            (self.cache_dir / name).mkdir(parents=True, exist_ok=True)
            warm = self.image_ids[: self.s2["warmup_frames"]]
            warmed = False
            for seq in self.sequences:
                f = self._cache_file(name, seq)
                if f.exists():
                    continue
                if not warmed:
                    for iid in warm:
                        be.predict(str(self.paths[iid]))
                    warmed = True
                dets, tims = [], []
                for iid in [i for i in self.image_ids if self.seq_of[i] == seq]:
                    d, t = be.predict(str(self.paths[iid]))
                    d = np.asarray(d, np.float32).reshape(-1, 6)
                    dets.append(np.column_stack([np.full(len(d), iid, np.float64), d]))
                    tims.append([iid, t["pre_ms"], t["infer_ms"], t["post_ms"], t["wall_ms"]])
                tmp = f.with_name(f.stem + ".tmp.npz")
                np.savez_compressed(tmp, dets=np.concatenate(dets) if dets else np.zeros((0, 7)),
                                    timings=np.asarray(tims, np.float64))
                tmp.replace(f)
            done[name] = len(list((self.cache_dir / name).glob("*.npz")))
            log.info("cache %s: %d/%d sequences", name, done[name], len(self.sequences))
        return done

    def load_cache(self, name: str) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        dets: dict[int, list] = {i: [] for i in self.image_ids}
        tim: dict[int, dict] = {}
        for seq in self.sequences:
            f = self._cache_file(name, seq)
            if not f.exists():
                raise Step2Error(f"missing cache {f}; run cache_runs first")
            z = np.load(f)
            for row in z["dets"]:
                dets[int(row[0])].append(row[1:])
            for iid, pre, inf, post, wall in z["timings"]:
                tim[int(iid)] = {"pre_ms": pre, "infer_ms": inf, "post_ms": post, "wall_ms": wall}
        missing = set(self.image_ids) - set(tim)
        if missing:
            raise Step2Error(f"cache {name} lacks {len(missing)} frames")
        return {k: (np.asarray(v, float).reshape(-1, 6)) for k, v in dets.items()}, tim

    def _write_policy(self, exp_id: str, policy: str, name_per_frame: list[str], dets_by_image: dict,
                      timings: dict | None, extra: dict | None = None) -> dict:
        res = dets_to_coco_results(dets_by_image)
        metrics = self.evaluator.evaluate_slices(res, self.slices)
        tau_c = self.m["conf_cue"]
        pf = perframe_f1(dets_by_image, self.gt_index, tau_c, self.cfg["eval"]["perframe_iou"],
                         self.cfg["eval"]["ignore_ioa"])
        rows = []
        for iid, r, lv in zip(self.image_ids, self.rows, name_per_frame, strict=True):
            t = (timings or {}).get(iid, {})
            d = dets_by_image[iid]
            rows.append({"seq": r["seq"], "frame_idx": r["frame_idx"], "image_id": iid, "level": lv,
                         "latency_pre_ms": t.get("pre_ms"), "latency_infer_ms": t.get("infer_ms"),
                         "latency_post_ms": t.get("post_ms"), "latency_total_ms": t.get("wall_ms"),
                         "n_dets_025": int((d[:, 4] >= tau_c).sum()) if len(d) else 0,
                         "tp": pf[iid]["tp"], "fp": pf[iid]["fp"], "fn": pf[iid]["fn"],
                         "f1_05": round(pf[iid]["f1"], 6),
                         "slice_night": r["slice_night"], "slice_crowded": r["slice_crowded"],
                         "slice_scene_change_natural": r["slice_scene_change_natural"],
                         "slice_easy": r["slice_easy"]})
        if timings:
            for name, ids in self.slices.items():
                metrics[name]["timing_cache"] = {k: _stats([timings[i][k] for i in ids])
                                                 for k in ("pre_ms", "infer_ms", "post_ms", "wall_ms")}
                metrics[name]["timing_note"] = ("cache-run timings (wall includes JPEG decode, single pass); "
                                                "NOT the Step 4 latency protocol")
        for name, ids in self.slices.items():
            idset = set(ids)
            metrics[name]["mean_f1_05"] = float(np.mean([pf[i]["f1"] for i in ids])) if ids else None
            metrics[name]["frac_level"] = {lv: float(np.mean([p == lv for i, p in zip(self.image_ids, name_per_frame, strict=True) if i in idset]))
                                           for lv in sorted(set(name_per_frame))} if ids else {}
            metrics[name]["lat_mean_ms"] = None  # filled only by the Step 4 protocol
            metrics[name]["energy_kind"] = "none"
        w = ExperimentWriter(self.results_root, self._meta(exp_id, policy, extra))
        w.write_json("metrics.json", metrics)
        w.write_csv("per_frame.csv", rows, PER_FRAME_FIELDS)
        w.register(policy=policy, mode="cache", replace=True)
        return metrics

    def evaluate(self, names: list[str] | None = None) -> dict:
        names = names or ((["pt_fp32"] if self.s2["pt_reference"] else []) + self.levels)
        out, f1 = {}, {}
        caches = {}
        for name in names:
            dets, tim = self.load_cache(name)
            caches[name] = dets
            out[name] = self._write_policy(f"calib_fixed_{name}", f"fixed_{name}", [name] * len(self.image_ids),
                                           dets, tim)
        # development-only oracle on calib (validates the oracle on real data; not a test result)
        if "int8" in caches and "fp16" in caches:
            lv = ("int8", "fp16")
            tau_c = self.m["conf_cue"]
            for lev in lv:
                pfl = perframe_f1(caches[lev], self.gt_index, tau_c, self.cfg["eval"]["perframe_iou"],
                                  self.cfg["eval"]["ignore_ioa"])
                f1[lev] = [pfl[i]["f1"] for i in self.image_ids]
            for kind, fn in (("cheap", lambda: baselines.oracle_cheap(f1, lv, "fp16")),
                             ("best", lambda: baselines.oracle_best(f1, lv))):
                choice = fn()
                sel = baselines.select_dets(choice, self.image_ids, lv, caches)
                seqs = [self.seq_of[i] for i in self.image_ids]
                extra = {"frac_level": baselines.level_fractions(choice, lv),
                         "switch_rate_per_100": baselines.switch_rate_per_100(choice, seqs),
                         "frac_int8_f1_ge_fp16": float(np.mean(np.asarray(f1["int8"]) >= np.asarray(f1["fp16"]))),
                         "notes": "DEVELOPMENT ONLY: oracle on calib; uses GT; not a deployable policy"}
                out[f"oracle_{kind}_dev"] = self._write_policy(
                    f"calib_oracle_{kind}_dev", f"oracle_{kind}", [lv[c] for c in choice], sel, None, extra)
        return out

    def val_check_set(self, out_name: str = "val_mini") -> dict:
        """The seeded mini-set for the model.val() comparison (same frames in notebooks 02
        and 02b): image ids, COCO GT without crowd/ignore boxes, YOLO-format dataset."""
        n = self.s2["smoke_val_check_frames"] if self.smoke else self.s2["val_check_frames"]
        keys = subsets.sample_frames([(self.seq_of[i], i) for i in self.image_ids], min(n, len(self.image_ids)),
                                     self.seed)
        ids = [i for _, i in keys]
        row_of = {int(r["image_id"]): r for r in self.rows}
        items_gt, items_y = [], []
        for i in ids:
            r = row_of[i]
            fr = self.frames_by_key[(r["seq"], int(r["frame_idx"]))]
            items_gt.append((i, fr, int(r["W"]), int(r["H"])))
            items_y.append((r["rel_path"], fr, int(r["W"]), int(r["H"])))
        gt = build_coco_gt(items_gt, "none", description="val_check", drop_crowd=True)
        from ..data.classmap import EVAL_CATEGORIES
        from ..data.yolo_format import flat_name

        names = getattr(self, "coco_names", None) or {c: n for _, (n, c) in EVAL_CATEGORIES.items()}
        yaml_path = write_yolo_dataset(self.work / out_name, items_y, self.layout.image_root, names)
        stem_to_id = {Path(flat_name(r)).stem: i for (r, *_), i in zip(items_y, ids, strict=True)}
        return {"ids": ids, "gt": gt, "yaml": yaml_path, "stem_to_id": stem_to_id,
                "paths": {i: self.paths[i] for i in ids}}

    def val_check(self) -> dict:
        """PLAN §5.3: our pycocotools pipeline vs Ultralytics model.val() on the same frames
        with the PyTorch FP32 model (D6: BDD calib frames, crowd/ignore boxes removed in both).
        Both mAP50-95 and mAP50 are compared (sanity requires both)."""
        vs = self.val_check_set()
        ev = CocoEvaluator(vs["gt"], self.cfg["eval"]["max_dets"])
        dets, _ = self.load_cache("pt_fp32")
        ours = ev.evaluate(dets_to_coco_results({i: dets[i] for i in vs["ids"]}))
        theirs = self.val_fn(vs["yaml"])
        out = {"n_frames": len(vs["ids"]), "ours": {k: ours[k] for k in ("map50_95", "map50")},
               "ultralytics_val": theirs,
               "abs_diff_pts": {k: 100 * abs(ours[k] - theirs[k]) for k in ("map50_95", "map50")},
               "note": "pt FP32, crowd/ignore removed from GT in both pipelines (YOLO labels cannot hold them)"}
        self.val_check_result = out
        return out

    def sanity(self, metrics: dict) -> dict:
        s = self.s2["sanity"]
        g = {k: v["all"]["map50_95"] * 100 for k, v in metrics.items() if not k.startswith("oracle")}
        checks = []

        def add(name, value, ok, rule):
            checks.append({"check": name, "value_pts": None if value is None else round(value, 3), "rule": rule,
                           "pass": bool(ok)})

        for k, v in g.items():
            add(f"{k} mAP50-95 valid", v, v >= 0, ">= 0 (not -1: GT categories present)")
        if "fp16" in g and "fp32" in g:
            d = g["fp16"] - g["fp32"]
            add("FP16 vs FP32 (TRT) |diff|", d, abs(d) <= s["fp16_vs_fp32_max_abs_pts"],
                f"|diff| <= {s['fp16_vs_fp32_max_abs_pts']} pts")
        if "fp32" in g and "pt_fp32" in g:
            d = g["fp32"] - g["pt_fp32"]
            add("TRT FP32 vs PyTorch FP32 |diff|", d, abs(d) <= s["trt_fp32_vs_pt_max_abs_pts"],
                f"|diff| <= {s['trt_fp32_vs_pt_max_abs_pts']} pts")
        if "int8" in g and "fp16" in g:
            d = g["int8"] - g["fp16"]
            add("INT8 - FP16", d, -s["int8_max_drop_pts"] <= d <= s["int8_max_gain_pts"],
                f"-{s['int8_max_drop_pts']} <= diff <= +{s['int8_max_gain_pts']} pts (slightly lower)")
        vc = getattr(self, "val_check_result", None)
        if vc:  # both metrics required (decision 2026-11-20 after the mAP50 gap in notebook 02)
            for k, label in (("map50_95", "mAP50-95"), ("map50", "mAP50")):
                d = vc["abs_diff_pts"][k]
                add(f"pipeline vs model.val() |diff| {label}", d, d <= s["val_check_max_abs_pts"],
                    f"<= {s['val_check_max_abs_pts']} pts")
        out = {"checks": checks, "all_pass": all(c["pass"] for c in checks), "map50_95_pts": g,
               "map50_pts": {k: v["all"]["map50"] * 100 for k, v in metrics.items() if not k.startswith("oracle")}}
        w = ExperimentWriter(self.results_root, self._meta(f"sanity_02_{self.model_name}", "sanity", dataset="sanity"))
        w.write_json("sanity.json", out)
        if vc:
            w.write_json("val_check.json", vc)
        w.register(policy="sanity", mode="check", replace=True)
        return out
