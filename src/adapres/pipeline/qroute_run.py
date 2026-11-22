"""QROUTE_COMPLETE_EXPERIMENT.ipynb orchestration (configs/prereg_qroute.yaml). One run, gated stages:

  S3  data       : dev = the 60 earlier clips; NEW untouched test = 40 never-used BDD100K MOT val clips
  S4  models     : TensorRT engines for 6 precision x resolution configs (+ FP32 640), INT4 support check,
                   output-consistency check, isolated cost profiling
  S5/6 caches    : counterfactual prediction cache (every config on every dev/test frame), per-frame F1, regret
  S7  headroom   : fixed operating points and budgeted oracles per action space (dev)
  S8  features   : causal features, their cost, usefulness, gating decision (dev)
  S9  predictor  : clip-grouped CV over ridge / gradient-boosted trees, OOF predicted vs observed regret (dev)
  S10 controller : lambda (budget) and K tuning on cross-fitted dev replay; locks (+ ablations, variants)
  S11 hardware   : per-config latency under controlled GPU contention -> slowdown ratios; deadline
  S12 overhead   : measured feature / predictor cost (light vs full; ridge vs trees)
  S13-15 final   : frozen configs on the untouched test clips; all baselines + ablations; bootstrap; Q1-Q3, Q5 replay
  S16 live       : latency / p95 / throughput / energy / memory / allocation; contention runs (Q4, Q5 live)
  S17-19         : figures, export, validation summary (paper/make_qroute.py renders from results/)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .. import baselines
from .. import calibrate as CAL
from .. import qroute as Q
from .. import risk as R
from ..config import config_sha256, resolve_out
from ..data import bdd_mot, slices, subsets
from ..data import visdrone as VD
from ..data.yolo_format import write_yolo_dataset
from ..energy import protocol as PROTO
from ..energy.nvml_meter import NvmlMeter
from ..eval.coco_gt import build_coco_gt, gt_counts
from ..eval.fast_ap import FastAP, cluster_bootstrap
from ..eval.map_eval import CocoEvaluator, ComposedEvaluator, dets_to_coco_results
from ..eval.perframe import gt_arrays_from_coco, perframe_f1
from ..features import FrameFeaturizer
from ..live import FixedStepper, QRouteStepper
from ..models import engines
from ..repro import capture_env, set_seed
from ..results import ExperimentWriter
from .full import GateError, _jsonable

log = logging.getLogger("adapres.qroute")
SLICES = {"easy": "slice_easy", "crowded": "slice_crowded", "night": "slice_night",
          "scene_change_natural": "slice_scene_change_natural"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class QRun:
    K_GRID = (1, 2, 3)          # prereg_qroute controller.persistence
    DEADLINE_FACTOR = 1.25      # prereg_qroute hardware_contention.deadline_ms
    def __init__(self, cfg: dict, smoke: bool, repo_dir, search_root=None, exporter: Callable | None = None,
                 backend_factory: Callable | None = None, image_reader: Callable | None = None,
                 featurizer_factory: Callable | None = None, meter_factory: Callable | None = None,
                 contention_factory: Callable | None = None, release_fn: Callable | None = None):
        self.cfg, self.smoke, self.repo = json.loads(json.dumps(cfg)), smoke, Path(repo_dir)
        self.Qc = self.cfg["qroute"]
        self.S = self.Qc["smoke"]
        self.prefix = self.Qc["exp_prefix"]
        self.seed = int(cfg["seed"])
        set_seed(self.seed)
        pp = self.repo / self.Qc["prereg"]
        hp = pp.with_suffix(pp.suffix + ".sha256")
        if not hp.exists():
            raise GateError("prereg_qroute hash missing: the design must be frozen before this run")
        self.prereg_sha = _sha(pp.read_bytes())
        if self.prereg_sha != hp.read_text().strip():
            raise GateError("configs/prereg_qroute.yaml changed after it was frozen (hash mismatch)")
        self.D = yaml.safe_load(pp.read_text(encoding="utf-8"))
        self.configs = list(self.D["action_space"]["configs"])
        self.ref = self.D["action_space"]["reference"]
        self.extra = self.D["action_space"]["extra_fixed_reference"]
        self.all_cfgs = self.configs + [self.extra]
        self.search_root = search_root
        self.exporter = exporter or engines.export_engine
        m = self.cfg["model"]
        self.backend_factory = backend_factory or (lambda name, path, imgsz: engines.UltralyticsBackend(
            name, path, imgsz, m["conf"], m["iou"], m["max_det"], m["rect"]))
        self.read = image_reader or self._cv2_read
        f = self.cfg["features"]
        self.ff = featurizer_factory or (lambda: FrameFeaturizer(f["width"], f["canny_low"], f["canny_high"], f["hist_bins"]))
        self.meter_factory = meter_factory or (lambda: NvmlMeter())
        self.contention_factory = contention_factory
        self.release_fn = release_fn or (lambda: None)
        self.tau_c = m["conf_cue"]
        self.results_root = resolve_out(self.cfg, "results_dir", smoke)
        self.results_root.mkdir(parents=True, exist_ok=True)
        base = Path(self.cfg["paths"]["work_root"]) / ("smoke" if smoke else "")
        self.work = base / "work06"
        self.lock_dir = self.results_root / "locked"
        self.env = capture_env(self.repo, self.seed)
        if self.env["git"]["commit"] is None:
            raise GateError("no git commit; refusing unattributable results")
        ri = self.repo / "results" / "index.csv"
        if ri.exists() and not (self.results_root / "index.csv").exists():
            import shutil

            shutil.copy(ri, self.results_root / "index.csv")
        self.stages: dict = {}
        self.backends: dict = {}
        self.energy_cost: dict | None = None

    # ------------------------------------------------------------------ infra
    def _contention(self):
        if self.contention_factory is not None:
            return self.contention_factory()
        from ..hw.contention import ContentionGenerator

        return ContentionGenerator()

    @staticmethod
    def _cv2_read(path):
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            raise GateError(f"cannot read {path}")
        return img

    def n(self, key, full):
        return self.S[key] if (self.smoke and key in self.S) else full

    def _meta(self, exp_id, policy, split, dataset="bdd100k_mot", extra=None):
        m = {"exp_id": self.prefix + exp_id, "dataset": dataset, "split": split, "policy": policy, "seed": self.seed,
             "git_commit": self.env["git"]["commit"], "git_dirty": self.env["git"]["dirty"], "env": self.env,
             "config_sha256": config_sha256(self.cfg), "config": self.cfg, "timestamp_utc": self.env["timestamp_utc"],
             "smoke": self.smoke, "prereg_qroute_sha256": self.prereg_sha, "model": self.cfg["model"]["name"],
             "engine_sha256": {k: v.get("sha256") for k, v in getattr(self, "engine_info", {}).items()}}
        m.update(extra or {})
        return m

    def _write(self, exp_id, policy, split, files: dict, dataset="bdd100k_mot", extra=None, csvs: dict | None = None):
        w = ExperimentWriter(self.results_root, self._meta(exp_id, policy, split, dataset, extra))
        for name, obj in files.items():
            w.write_json(name, _jsonable(obj))
        for name, rows in (csvs or {}).items():
            if rows:
                w.write_csv(name, rows, list(dict.fromkeys(k for r in rows for k in r)))
        w.register(policy=policy, mode=(extra or {}).get("mode", "replay"), replace=True)
        return w.dir

    def gate(self, stage: str, checks: list[dict], soft: list[dict] | None = None) -> dict:
        ok = all(c["pass"] for c in checks)
        rec = {"stage": stage, "pass": ok, "checks": checks, "soft_checks": soft or []}
        self.stages[stage] = rec
        d = self.results_root / f"{self.prefix}stages"
        d.mkdir(parents=True, exist_ok=True)
        (d / "stages.json").write_text(json.dumps(_jsonable(self.stages), indent=2))
        for c in checks + (soft or []):
            log.info("[%s] %s %s", "PASS" if c["pass"] else ("FAIL" if c in checks else "WARN"), c["check"], c.get("detail", ""))
        if not ok:
            raise GateError(f"gate '{stage}' failed: " + "; ".join(c["check"] for c in checks if not c["pass"]))
        return rec

    # ------------------------------------------------------------------ S3 data
    def s3_data(self) -> dict:
        sub = self.repo / "results" / "bdd100k_mot" / "bdd_subsets"
        sel = json.loads((sub / "selection.json").read_text())
        rep = json.loads((sub / "report.json").read_text())
        self.thr = json.loads((sub / "slice_thresholds.json").read_text())
        used = set(sel["splits"]["calib"]) | set(sel["splits"]["test"])
        if self.search_root is None:
            pref = Path(self.cfg["paths"]["input_root"]) / self.cfg["paths"]["bdd_dataset_slug"]
            self.search_root = pref if pref.exists() else Path(self.cfg["paths"]["input_root"])
        b = self.cfg["bdd"]
        self.layout = bdd_mot.discover_layout(self.search_root, b["images_dir_name"], b["labels_dir_name"], b["label_split"])
        # development clips (earlier calib + earlier test): manifest rows reused
        dev_rows = _read_csv(sub / "frames_calib.csv") + _read_csv(sub / "frames_test.csv")
        dev_seqs = sorted({r["seq"] for r in dev_rows})
        if self.smoke:
            rng = np.random.default_rng(self.seed)
            dev_seqs = sorted(dev_seqs[i] for i in rng.choice(len(dev_seqs), self.S["dev_clips"], replace=False))
        self.dev_rows = sorted([r for r in dev_rows if r["seq"] in set(dev_seqs)], key=lambda r: (r["seq"], int(r["frame_idx"])))
        # NEW untouched test clips
        cand = sorted(p.stem for p in self.layout.label_dir.glob("*.json") if p.stem not in used)
        labels = {}
        ok = []
        for s in cand:
            fr = bdd_mot.load_video_labels(self.layout.label_dir / f"{s}.json")
            if fr and all((self.layout.image_root / s / f.name).exists() for f in fr):
                labels[s] = fr
                ok.append(s)
        # Step 1 rule: label attribute if present, else median-frame BT.601 luma vs the Step 1 Otsu threshold
        otsu = (rep.get("luma_fallback") or {}).get("otsu_threshold")
        strata, tod_src = {}, {}
        for s in ok:
            t = bdd_mot.normalise_timeofday(bdd_mot.sequence_timeofday(labels[s]))
            if t is not None:
                strata[s], tod_src[s] = t, "label_attribute"
                continue
            if otsu is None:
                raise GateError("clip without timeofday label and no Step 1 luma threshold to stratify it")
            f = labels[s][len(labels[s]) // 2]
            y = slices.luma_bt601(self.read(self.layout.image_root / s / f.name)[:, :, ::-1])
            strata[s], tod_src[s] = ("night" if y < otsu else "daytime"), "median_frame_luma_otsu"
        targets = dict(self.n("new_test_strata", self.Qc["new_test_strata"]))
        pick = subsets.select_disjoint_splits(strata, {"test": targets}, self.seed, split_order=("test",))
        self.test_seqs = pick["splits"]["test"]
        off = int(self.Qc["image_id_offset"])
        self.test_frames = {}
        rows, iid = [], off
        for s in self.test_seqs:
            for fr in labels[s]:
                rows.append({"split": "final_test", "seq": s, "frame_idx": fr.frame_idx, "image_id": iid,
                             "rel_path": f"{s}/{fr.name}", "gt_count": gt_counts(fr)})
                self.test_frames[(s, fr.frame_idx)] = fr
                iid += 1
        self.test_rows = rows
        self.dev_frames = {}
        for s in dev_seqs:
            for fr in bdd_mot.load_video_labels(self.layout.label_dir / f"{s}.json"):
                self.dev_frames[(s, fr.frame_idx)] = fr
        vd = {"attached": False}
        try:
            v = VD.discover(Path(self.cfg["paths"]["input_root"]))
            if v["n_annotation_files"]:
                f0 = sorted(Path(v["annotation_dirs"][0]).glob("*.txt"))[0]
                frs = VD.parse_sequence(f0)
                vd = {"attached": True, **v, "example_sequence": f0.stem, "example_frames": len(frs),
                      "example_eval_boxes": sum(b.kind == "eval" for f in frs for b in f.boxes),
                      "note": "validated only; the pre-registered Q-ROUTE pipeline runs on BDD100K"}
        except Exception as e:  # noqa: BLE001
            vd = {"attached": False, "error": repr(e)}
        manifest = {"dev_clips": dev_seqs, "final_test_clips": self.test_seqs, "final_test_strata": pick["strata_counts"],
                    "candidates_untouched": len(cand), "candidates_with_all_images": len(ok), "luma_threshold": otsu,
                    "timeofday_source": {s: tod_src[s] for s in self.test_seqs},
                    "image_id_offset": off, "seed": self.seed, "visdrone": vd,
                    "final_test_sha256": _sha(json.dumps(sorted(self.test_seqs)).encode())}
        self._write("split", "data_split", "dev+final_test", {"split.json": manifest},
                    csvs={"final_test_frames.csv": self.test_rows}, extra={"mode": "data_prep"})
        checks = [{"check": "final test clips are disjoint from every earlier clip", "pass": not (set(self.test_seqs) & used)},
                  {"check": "dev and final test disjoint", "pass": not (set(self.test_seqs) & set(dev_seqs))},
                  {"check": "final test clip count as pre-registered", "pass": len(self.test_seqs) == sum(targets.values()),
                   "detail": len(self.test_seqs)}]
        soft = [{"check": "VisDrone attached (optional benchmark)", "pass": vd.get("attached", False), "detail": vd}]
        self.gate("S3_data", checks, soft)
        return manifest

    # ------------------------------------------------------------------ S4 models
    def s4_models(self, weights, names) -> dict:
        lay = self.layout
        sub = self.repo / "results" / "bdd100k_mot" / "bdd_subsets"
        rows = _read_csv(sub / "int8_calib_frames.csv")
        if self.smoke:
            rows = rows[:50]
        sizes = {(r["seq"], int(r["frame_idx"])): (int(r["W"]), int(r["H"])) for r in _read_csv(sub / "frames_calib.csv")}
        fr_cache = {}
        items = []
        for r in rows:
            key = (r["seq"], int(r["frame_idx"]))
            if r["seq"] not in fr_cache:
                fr_cache[r["seq"]] = {f.frame_idx: f for f in bdd_mot.load_video_labels(lay.label_dir / f"{r['seq']}.json")}
            items.append((r["rel_path"], fr_cache[r["seq"]][key[1]], *sizes[key]))
        int8_yaml = write_yolo_dataset(self.work / "int8_calib", items, lay.image_root, names)
        self.engine_info = {}
        for c in self.all_cfgs:
            lv, res = Q.parse_config(c)
            self.engine_info[c] = self.exporter(weights, lv, res, self.work / "engines",
                                                int8_data=int8_yaml if lv == "int8" else None, tag=c)
            self.engine_info[c]["imgsz"] = res
        self.backends = {c: self.backend_factory(c, self.engine_info[c]["path"], self.engine_info[c]["imgsz"]) for c in self.all_cfgs}
        try:
            import tensorrt as trt

            int4 = {"tensorrt": trt.__version__, "builder_flag_INT4": hasattr(trt.BuilderFlag, "INT4"),
                    "supported_for_this_detector": False,
                    "reason": "Ultralytics TensorRT export exposes FP32/FP16/INT8 only; no INT4 path for conv detectors"}
        except Exception as e:  # noqa: BLE001
            int4 = {"supported_for_this_detector": False, "reason": repr(e)}
        # output consistency + isolated cost profiling on pre-decoded dev frames
        probe = [self.read(lay.image_root / r["rel_path"]) for r in self.dev_rows[:5]]
        cons = {}
        ref_n = [len(self.backends[self.ref].predict(img)[0]) for img in probe]
        for c in self.all_cfgs:
            outs = [self.backends[c].predict(img)[0] for img in probe]
            cons[c] = {"finite": bool(all(np.isfinite(np.asarray(o, float)).all() for o in outs)),
                       "mean_dets": float(np.mean([len(o) for o in outs])), "ref_mean_dets": float(np.mean(ref_n))}
        iters, warm = (20, 5) if self.smoke else (200, 30)
        self.profile = {c: engines.microbench(self.backends[c], probe[0], iters, warm) for c in self.all_cfgs}
        self.cost_iso = {c: self.profile[c]["wall_ms"]["median"] for c in self.all_cfgs}
        self._write("models", "models", "dev", {"engines.json": self.engine_info, "int4.json": int4,
                                                "consistency.json": cons, "profile.json": self.profile},
                    dataset="microbench", extra={"mode": "export"})
        checks = [{"check": f"{c}: finite outputs", "pass": cons[c]["finite"]} for c in self.all_cfgs]
        checks += [{"check": f"{c}: produces detections", "pass": cons[c]["mean_dets"] > 0} for c in self.all_cfgs]
        soft = [{"check": "INT4 engine supported", "pass": int4["supported_for_this_detector"], "detail": int4.get("reason")}]
        self.gate("S4_models", checks, soft)
        return {"cost_iso_ms": self.cost_iso, "int4": int4}

    # ------------------------------------------------------------------ S5/S6 caches
    def _cache(self, split: str, frames_by_seq: dict[str, list[tuple[int, Path]]], cfgs: list[str]) -> dict:
        d = self.work / "cache" / split
        d.mkdir(parents=True, exist_ok=True)
        out = {"dets": {c: {} for c in cfgs}, "cues": {}, "luma": {}, "wh": {}, "lat": {c: {} for c in cfgs}}
        for s, items in sorted(frames_by_seq.items()):
            f = d / f"{s}.npz"
            if not f.exists():
                fz = self.ff()
                arr = {c: [] for c in cfgs}
                lat = {c: [] for c in cfgs}
                ids, cm, cc, cd, lu, W, H = [], [], [], [], [], [], []
                for iid, path in items:
                    img = self.read(path)
                    cue = fz(img)
                    ids.append(iid)
                    cm.append(cue["m"])
                    cc.append(cue["c"])
                    cd.append(cue["d"])
                    lu.append(slices.luma_bt601(img[:, :, ::-1]))
                    H.append(img.shape[0])
                    W.append(img.shape[1])
                    for c in cfgs:
                        det, t = self.backends[c].predict(img)
                        det = np.asarray(det, np.float32).reshape(-1, 6)
                        arr[c].append(np.column_stack([np.full(len(det), iid), det]))
                        lat[c].append(t["wall_ms"])
                tmp = f.with_name(f.stem + ".tmp.npz")
                np.savez_compressed(tmp, ids=ids, m=cm, c=cc, d=cd, luma=lu, W=W, H=H,
                                    **{f"dets_{c}": (np.concatenate(arr[c]) if arr[c] else np.zeros((0, 7))) for c in cfgs},
                                    **{f"lat_{c}": lat[c] for c in cfgs})
                tmp.replace(f)
            z = np.load(f)
            ids = [int(i) for i in z["ids"]]
            for k, i in enumerate(ids):
                out["cues"][i] = {"m": float(z["m"][k]), "c": float(z["c"][k]), "d": float(z["d"][k])}
                out["luma"][i] = float(z["luma"][k])
                out["wh"][i] = (int(z["W"][k]), int(z["H"][k]))
            for c in cfgs:
                per = {i: [] for i in ids}
                for row in z[f"dets_{c}"]:
                    per[int(row[0])].append(row[1:])
                for k, i in enumerate(ids):
                    out["dets"][c][i] = np.asarray(per[i], float).reshape(-1, 6)
                    out["lat"][c][i] = float(z[f"lat_{c}"][k])
        return out

    def s5_caches(self) -> dict:
        lay = self.layout
        dev_by = {}
        for r in self.dev_rows:
            dev_by.setdefault(r["seq"], []).append((int(r["image_id"]), lay.image_root / r["rel_path"]))
        test_by = {}
        for r in self.test_rows:
            test_by.setdefault(r["seq"], []).append((int(r["image_id"]), lay.image_root / r["rel_path"]))
        self.dev = self._cache("dev", dev_by, self.configs)
        self.test = self._cache("final_test", test_by, self.all_cfgs)
        ig = self.cfg["classmap"]["ignore_mode"]
        self.dev_gt = build_coco_gt([(int(r["image_id"]), self.dev_frames[(r["seq"], int(r["frame_idx"]))],
                                      *self.dev["wh"][int(r["image_id"])]) for r in self.dev_rows], ig, "dev")
        self.test_gt = build_coco_gt([(r["image_id"], self.test_frames[(r["seq"], r["frame_idx"])], *self.test["wh"][r["image_id"]])
                                      for r in self.test_rows], ig, "final_test")
        # slices for the new test clips with the Step 1 (calib-fitted) thresholds
        for s in self.test_seqs:
            rs = [r for r in self.test_rows if r["seq"] == s]
            nsc = slices.natural_scene_change(
                [{b.track_id for b in self.test_frames[(s, r["frame_idx"])].boxes if b.kind == "eval"} for r in rs],
                self.cfg["slices"]["natural_scene_change"]["new_track_ratio"], self.cfg["slices"]["natural_scene_change"]["min_tracks"])
            lab = slices.label_frames(np.array([self.test["luma"][r["image_id"]] for r in rs]), np.array([r["gt_count"] for r in rs]),
                                      np.zeros(len(rs), bool), nsc, self.thr, self.cfg["slices"]["easy_excludes_natural_scene_change"])
            for k, r in enumerate(rs):
                for key, v in lab.items():
                    r[key] = int(v[k])
        self.dev_seq = {}
        for r in self.dev_rows:
            self.dev_seq.setdefault(r["seq"], []).append(int(r["image_id"]))
        self.test_seq = {}
        for r in self.test_rows:
            self.test_seq.setdefault(r["seq"], []).append(r["image_id"])
        self.dev_ids = [i for s in sorted(self.dev_seq) for i in self.dev_seq[s]]
        self.test_ids = [i for s in sorted(self.test_seq) for i in self.test_seq[s]]
        e = self.cfg["eval"]
        self.dev_f1 = {c: {i: v["f1"] for i, v in perframe_f1(self.dev["dets"][c], gt_arrays_from_coco(self.dev_gt), self.tau_c,
                                                               e["perframe_iou"], e["ignore_ioa"]).items()} for c in self.configs}
        self.test_f1 = {c: {i: v["f1"] for i, v in perframe_f1(self.test["dets"][c], gt_arrays_from_coco(self.test_gt), self.tau_c,
                                                                e["perframe_iou"], e["ignore_ioa"]).items()} for c in self.all_cfgs}
        self.dev_area = {i: float(w * h) for i, (w, h) in self.dev["wh"].items()}
        self.test_area = {i: float(w * h) for i, (w, h) in self.test["wh"].items()}
        self.dev_ev = ComposedEvaluator(self.dev_gt, {c: self.dev["dets"][c] for c in self.configs}, e["max_dets"])
        checks = [{"check": "dev cache complete", "pass": all(i in self.dev["dets"][c] for c in self.configs for i in self.dev_ids)},
                  {"check": "final test cache complete", "pass": all(i in self.test["dets"][c] for c in self.all_cfgs for i in self.test_ids)},
                  {"check": "no dev image id in final test", "pass": not (set(self.dev_ids) & set(self.test_ids))}]
        self.gate("S5_caches", checks)
        return {"dev_frames": len(self.dev_ids), "final_test_frames": len(self.test_ids)}

    # ------------------------------------------------------------------ helpers
    def _map(self, ev, choice: dict[int, str], ids) -> float:
        return ev.evaluate(choice, {"all": ids})["all"]["map50_95"]

    def _choice(self, ids_by_seq, seq_order, ch_list, cfgs):
        return {i: cfgs[c] for s, ch in zip(seq_order, ch_list, strict=True) for i, c in zip(ids_by_seq[s], ch, strict=True)}

    # ------------------------------------------------------------------ S5/S6 views (dev only)
    def fixed_dev_table(self) -> list[dict]:
        """Section 5: every fixed precision x resolution config on the development clips (exact mAP, profiled cost)."""
        ids = self.dev_ids
        self.dev_fixed_map = {c: self._map(self.dev_ev, {i: c for i in ids}, ids) for c in self.configs}
        return [{"config": c, "dev_map50_95": self.dev_fixed_map[c], "dev_mean_f1": float(np.mean([self.dev_f1[c][i] for i in ids])),
                 "cost_iso_ms": self.cost_iso[c], "engine_infer_ms": self.profile[c]["infer_ms"]["median"]} for c in self.configs]

    def regret_summary(self) -> dict:
        """Section 6: counterfactual regret F1(ref) - F1(c) per config on development frames."""
        R_ = np.array([[self.dev_f1[self.ref][i] - self.dev_f1[c][i] for c in self.configs] for i in self.dev_ids])
        out = {c: {"mean": float(R_[:, j].mean()), "frac_no_worse": float(np.mean(R_[:, j] <= 0)),
                   "frac_better": float(np.mean(R_[:, j] < 0)), "p90": float(np.percentile(R_[:, j], 90))}
               for j, c in enumerate(self.configs)}
        self._write("dev_regret", "regret", "dev", {"regret_summary.json": out})
        return out

    # ------------------------------------------------------------------ S10 controller implementation check (dev)
    def s10_controller_check(self) -> dict:
        """Section 10: the step-by-step QRouteController and the fast closed-loop replay used for tuning make identical
        decisions (isolated and under a synthetic slowdown schedule, with a deadline) on development clips."""
        qd = self.qd_dev
        tables = Q.prediction_table(qd, lambda si: self.model, self.use_img)
        lens = [len(self.dev_seq[s]) for s in qd.seq_ids]
        synth = {0.0: {c: 1.0 for c in self.configs}, 1.0: {c: 1.0 + 0.5 * (k % 3) for k, c in enumerate(self.configs)}}
        sl = Q.contention_slowdown(lens, self.configs, synth, [0.0, 1.0], 7)
        checks = []
        for name, kw, slow in (("isolated", {}, None), ("slowdown+deadline", {"deadline": 1.1 * self.cost_iso[self.ref]}, sl)):
            p = Q.QParams(configs=self.configs, ref=self.ref, lam=1.0, K=2, cost_iso=dict(self.cost_iso), overhead=self.overhead_ms,
                          use_img=self.use_img, **kw)
            ch = Q.fast_replay_q(tables, p, slow)
            ok = True
            for si, s in enumerate(qd.seq_ids):
                ctl = Q.QRouteController(self.model, p, self.tau_c)
                got = []
                for t, iid in enumerate(self.dev_seq[s]):
                    o = ctl.step(self.dev["cues"][iid])
                    c = o["config"]
                    got.append(self.configs.index(c))
                    f = 1.0 if slow is None else slow[si][t, self.configs.index(c)]
                    ctl.observe(self.dev["dets"][c][iid], self.dev_area[iid], observed_cost=self.cost_iso[c] * f)
                ok &= got == ch[si].tolist()
            checks.append({"check": f"controller == fast replay on dev ({name})", "pass": bool(ok)})
        self.gate("S10_controller_check", checks)
        return {c["check"]: c["pass"] for c in checks}

    # ------------------------------------------------------------------ S7 headroom (dev)
    def s7_headroom(self) -> dict:
        ids = self.dev_ids
        if not getattr(self, "dev_fixed_map", None):
            self.fixed_dev_table()
        self.budgets = {"B_int8": self.cost_iso["int8_640"], "B_mid": (self.cost_iso["int8_640"] + self.cost_iso[self.ref]) / 2}
        spaces = {"joint": self.configs, "precision_only": [self.ref, "int8_640"],
                  "resolution_only": [self.ref, "fp16_480", "fp16_320"]}
        out = {"fixed_dev_map": self.dev_fixed_map, "budgets_ms": self.budgets, "oracle": {}}
        for bname, B in self.budgets.items():
            feas = [c for c in self.configs if self.cost_iso[c] <= B + 1e-9]
            bf = max(feas, key=lambda c: self.dev_fixed_map[c]) if feas else min(self.configs, key=lambda c: self.cost_iso[c])
            out["oracle"][bname] = {"best_fixed": bf, "best_fixed_map": self.dev_fixed_map[bf]}
            for sp, cf in spaces.items():
                Rm = np.array([[self.dev_f1[self.ref][i] - self.dev_f1[c][i] for c in cf] for i in ids])
                ch, cst = Q.oracle_choices(Rm, np.array([self.cost_iso[c] for c in cf]), B)
                mp = self._map(self.dev_ev, {i: cf[k] for i, k in zip(ids, ch, strict=True)}, ids)
                out["oracle"][bname][sp] = {"map": mp, "mean_cost": cst, "headroom_vs_best_fixed": mp - self.dev_fixed_map[bf]}
        self.headroom = out
        self._write("dev_headroom", "headroom", "dev", {"headroom.json": out})
        soft = [{"check": f"oracle headroom over best fixed at {b} (joint) > 0",
                 "pass": out["oracle"][b]["joint"]["headroom_vs_best_fixed"] > 0,
                 "detail": out["oracle"][b]["joint"]["headroom_vs_best_fixed"]} for b in self.budgets]
        self.gate("S7_headroom", [{"check": "headroom computed", "pass": True}], soft)
        return out

    # ------------------------------------------------------------------ S8 + S9 features and predictor (dev)
    def s8_s9_predictor(self) -> dict:
        qd = Q.build_qdata(self.dev_seq, self.configs, self.dev["dets"], self.dev_area, self.dev["cues"], self.dev_f1, self.tau_c)
        self.qd_dev = qd
        ref_i = self.configs.index(self.ref)
        cands = self.S["candidates"] if self.smoke else [{"kind": "ridge", "alpha": a} for a in (0.1, 1.0, 10.0)] + \
            [{"kind": "gbt", "max_depth": d, "max_iter": 100, "learning_rate": 0.1} for d in (2, 3)]
        nf = self.S["n_folds"] if self.smoke else 5
        res = {}
        for use_img in (False, True):
            X, Y, G = Q.training_rows(qd, self.ref, use_img)
            res[use_img] = Q.select_predictor(X, Y, G, cands, ref_i, self.seed, nf)
            res[use_img]["XYG"] = (X, Y, G)
        gain = res[True]["best_spearman"] - res[False]["best_spearman"]
        self.use_img = bool(gain >= self.Qc["feature_gate_min_gain"])
        sel = res[self.use_img]
        X, Y, G = sel["XYG"]
        # feature usefulness: standardized ridge coefficients (mean |coef| over outputs)
        rg = Q.RidgeMulti(1.0).fit(X, Y)
        names = Q.feature_names(self.use_img)
        useful = {n_: float(np.mean(np.abs(rg.W[k]))) for k, n_ in enumerate(names)}
        # OOF predictions (predicted vs observed regret)
        oof = np.zeros_like(Y)
        for tr, te in sel["folds"]:
            oof[te] = Q.make_predictor(sel["spec"], self.seed).fit(X[tr], Y[tr]).predict(X[te])
        # feature cost (CPU): det stats + vector, image cues measured from the featurizer
        t0 = time.perf_counter()
        for i in self.dev_ids[:300]:
            st = R.det_stats(self.dev["dets"][self.ref][i], self.dev_area[i], self.tau_c)
            Q.feature_vector(st, st, self.ref, self.dev["cues"][i], False)
        light_ms = (time.perf_counter() - t0) * 1000 / min(300, len(self.dev_ids))
        fz = self.ff()
        img = self.read(self.layout.image_root / self.dev_rows[0]["rel_path"])
        t0 = time.perf_counter()
        for _ in range(20):
            fz(img)
        img_ms = (time.perf_counter() - t0) * 1000 / 20
        self.model = sel["model"]
        self.sel = sel
        if isinstance(self.model, Q.GBTMulti):
            err = self.model.validate(X[:200])
            if err > 1e-6:
                raise GateError(f"numpy tree export disagrees with sklearn (max err {err})")
        t0 = time.perf_counter()
        for x in X[:500]:
            self.model.predict_one(x)
        pred_ms = (time.perf_counter() - t0) * 1000 / min(500, len(X))
        self.overhead_ms = light_ms + pred_ms + (img_ms if self.use_img else 0.0)
        self.feature_cost = {"light_features_ms": light_ms, "image_cues_ms": img_ms, "predictor_ms": pred_ms,
                             "total_overhead_ms": self.overhead_ms}
        self.dev_regret_mean = Y.mean(0)
        out = {"cv_light": res[False]["table"], "cv_full": res[True]["table"], "spearman_light": res[False]["best_spearman"],
               "spearman_full": res[True]["best_spearman"], "gate_gain": gain, "use_image_cues": self.use_img,
               "chosen": sel["spec"], "feature_usefulness": useful, "feature_cost": self.feature_cost,
               "per_config_oof_spearman": {c: float(__import__("scipy").stats.spearmanr(oof[:, j], Y[:, j]).statistic)
                                           for j, c in enumerate(self.configs) if j != ref_i}}
        rows = [{"config": c, "pred": float(oof[k, j]), "true": float(Y[k, j])} for j, c in enumerate(self.configs)
                if j != ref_i for k in range(0, len(Y), max(1, len(Y) // 2000))]
        self.predictor_report = out
        self._write("dev_predictor", "predictor", "dev", {"predictor.json": out}, csvs={"oof_sample.csv": rows})
        self.gate("S8_S9_features_predictor", [{"check": "predictor fitted", "pass": True}],
                  [{"check": "held-out Spearman > 0", "pass": sel["best_spearman"] > 0, "detail": sel["best_spearman"]}])
        return out

    # ------------------------------------------------------------------ S10 controller tuning + locks (dev)
    def _tables(self, qd, model_for_seq, use_img, use_temporal=True):
        return Q.prediction_table(qd, model_for_seq, use_img, use_temporal)

    def _tune(self, name, cfgs, budget, predictor_kind="model", use_img=None, use_temporal=True, use_hw=True,
              Ks=None, cost=None, overhead=None) -> dict:
        """Tune (lambda, K) on cross-fitted closed-loop dev replay for one variant and lock it."""
        use_img = self.use_img if use_img is None else use_img
        cost = cost or self.cost_iso
        qd = self.qd_dev if cfgs == self.configs else Q.build_qdata(self.dev_seq, cfgs, self.dev["dets"], self.dev_area,
                                                                     self.dev["cues"], self.dev_f1, self.tau_c)
        if predictor_kind == "constant":
            Ym = np.array([np.mean([self.dev_f1[self.ref][i] - self.dev_f1[c][i] for i in self.dev_ids]) for c in cfgs])
            final = Q.ConstantPredictor(Ym)
            mfs = lambda si: final  # noqa: E731
            spec = {"kind": "constant"}
        else:
            X, Y, G = Q.training_rows(qd, self.ref, use_img, use_temporal)
            spec = self.sel["spec"]
            from sklearn.model_selection import GroupKFold

            k = min(self.S["n_folds"] if self.smoke else 5, len(np.unique(G)))
            fold_models, fog = [], {}
            for fi, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, Y, G)):
                fold_models.append(Q.make_predictor(spec, self.seed).fit(X[tr], Y[tr]))
                for g in np.unique(G[te]):
                    fog[int(g)] = fi
            final = Q.make_predictor(spec, self.seed).fit(X, Y)
            mfs = lambda si: fold_models[fog[si]]  # noqa: E731
        tables = self._tables(qd, mfs, use_img, use_temporal)
        ov = self.overhead_ms if overhead is None else overhead
        cands = []
        for K in (Ks or self.K_GRID):
            p = Q.QParams(configs=cfgs, ref=self.ref, lam=0.0, K=int(K), cost_iso={c: cost[c] for c in cfgs}, overhead=ov,
                          use_hw=use_hw, use_img=use_img, use_temporal=use_temporal)
            lam, c_ = Q.tune_lambda(tables, p, budget)
            p.lam = lam
            ch = Q.fast_replay_q(tables, p)
            mp = self._map(self.dev_ev, self._choice(self.dev_seq, qd.seq_ids, ch, cfgs), self.dev_ids)
            cands.append({"K": int(K), "lam": lam, "dev_mean_cost": c_, "dev_map": mp, "params": p})
        best = sorted(cands, key=lambda c: (-c["dev_map"], -c["K"]))[0]
        mb = pickle.dumps(final)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        (self.lock_dir / f"qroute_{name}_model.pkl").write_bytes(mb)
        body = {"prereg_qroute_sha256": self.prereg_sha, "variant": name, "configs": cfgs, "ref": self.ref, "lam": best["lam"],
                "K": best["K"], "cost_iso": {c: cost[c] for c in cfgs}, "overhead": ov, "use_hw": use_hw, "use_img": use_img,
                "use_temporal": use_temporal, "budget": budget, "predictor": spec, "model_sha256": _sha(mb),
                "budget_met_on_dev": bool(abs(best["dev_mean_cost"] - budget) <= 0.01 * budget),
                "dev": {k: v for k, v in best.items() if k != "params"},
                "candidates": [{k: v for k, v in c.items() if k != "params"} for c in cands]}
        text = yaml.safe_dump(json.loads(json.dumps(_jsonable(body), default=float)), sort_keys=True)
        lp = self.lock_dir / f"locked_qroute_{name}.yaml"
        lp.write_text(text, encoding="utf-8")
        lp.with_suffix(".yaml.sha256").write_text(_sha(text.encode()) + "\n")
        return body

    def _load_lock(self, name):
        lp = self.lock_dir / f"locked_qroute_{name}.yaml"
        text = lp.read_text(encoding="utf-8")
        if _sha(text.encode()) != lp.with_suffix(".yaml.sha256").read_text().strip():
            raise CAL.LockError(f"{lp} modified after locking")
        body = yaml.safe_load(text)
        mb = (self.lock_dir / f"qroute_{name}_model.pkl").read_bytes()
        if _sha(mb) != body["model_sha256"]:
            raise CAL.LockError(f"model for {name} does not match its lock")
        if body["prereg_qroute_sha256"] != self.prereg_sha:
            raise CAL.LockError("lock made under a different prereg_qroute")
        p = Q.QParams(configs=body["configs"], ref=body["ref"], lam=body["lam"], K=body["K"], cost_iso=body["cost_iso"],
                      overhead=body["overhead"], use_hw=body["use_hw"], use_img=body["use_img"], use_temporal=body["use_temporal"])
        return p, pickle.loads(mb), body

    def s12_controller(self) -> dict:
        B = self.budgets
        prec = [self.ref, "int8_640"]
        reso = [self.ref, "fp16_480", "fp16_320"]
        self.variants = {
            "main_B_int8": dict(cfgs=self.configs, budget=B["B_int8"]),
            "main_B_mid": dict(cfgs=self.configs, budget=B["B_mid"]),
            "precision_only": dict(cfgs=prec, budget=B["B_int8"]),
            "resolution_only": dict(cfgs=reso, budget=B["B_int8"]),
            "hardware_only": dict(cfgs=self.configs, budget=B["B_int8"], predictor_kind="constant"),
            "abl_no_temporal": dict(cfgs=self.configs, budget=B["B_int8"], use_temporal=False),
            "abl_no_persistence": dict(cfgs=self.configs, budget=B["B_int8"], Ks=[1]),
            "abl_full_features": dict(cfgs=self.configs, budget=B["B_int8"], use_img=True),
            "abl_no_hw": dict(cfgs=self.configs, budget=B["B_int8"], use_hw=False),
        }
        if self.energy_cost:
            self.variants["abl_energy_cost"] = dict(cfgs=self.configs, budget=self.energy_cost["int8_640"], cost=self.energy_cost, overhead=0.0)
        locks, skipped = {}, {}
        for name, v in self.variants.items():
            if name == "abl_full_features" and self.use_img:
                skipped[name] = "identical to the main controller: image cues were gated in on development"
                continue
            locks[name] = self._tune(name, **v)
        if not self.energy_cost:
            skipped["abl_energy_cost"] = "per-config energy unavailable (NVML energy not measured in S11)"
        self.locks = locks
        self.skipped_variants = skipped
        self._write("controller_locks", "controller", "dev", {"locks.json": locks, "skipped.json": skipped})
        checks = []
        for name in locks:
            try:
                self._load_lock(name)
                checks.append({"check": f"lock {name} verified", "pass": True})
            except CAL.LockError as e:
                checks.append({"check": f"lock {name}", "pass": False, "detail": str(e)})
        soft = [{"check": f"{n}: dev mean cost within 1% of budget",
                 "pass": abs(locks[n]["dev"]["dev_mean_cost"] - locks[n]["budget"]) <= 0.01 * locks[n]["budget"],
                 "detail": [locks[n]["dev"]["dev_mean_cost"], locks[n]["budget"]]} for n in locks]
        self.gate("S12_controller_locks", checks, soft)
        return {n: {"K": b["K"], "lam": b["lam"], "dev_map": b["dev"]["dev_map"], "dev_cost": b["dev"]["dev_mean_cost"]} for n, b in locks.items()}

    # ------------------------------------------------------------------ S11 hardware contention profiling
    def s11_hardware(self) -> dict:
        levels = list(self.D["hardware_contention"]["levels"])
        img = self.read(self.layout.image_root / self.dev_rows[0]["rel_path"])
        iters = int(self.n("contention_iters", self.Qc["contention_iters"]))
        lat = {}
        with self._contention() as g:
            for lv in levels:
                g.set_level(lv)
                time.sleep(0.5)
                lat[lv] = {}
                for c in self.configs:
                    ts = []
                    for _ in range(iters):
                        t0 = time.perf_counter()
                        self.backends[c].predict(img)
                        ts.append((time.perf_counter() - t0) * 1000)
                    lat[lv][c] = float(np.median(ts))
            g.set_level(0.0)
        self.ratios = {lv: {c: lat[lv][c] / lat[levels[0]][c] for c in self.configs} for lv in levels}
        self.deadline = self.DEADLINE_FACTOR * self.cost_iso[self.ref]
        # isolated GPU-board energy per frame and config (cost of the energy-objective ablation)
        meter = self.meter_factory()
        secs = float(self.n("energy_probe_s", self.Qc["energy_probe_s"]))
        energy = {}
        for c in self.configs:
            for _ in range(5):
                self.backends[c].predict(img)
            n = 0
            with meter.window() as w:
                t_end = time.perf_counter() + secs
                while time.perf_counter() < t_end:
                    self.backends[c].predict(img)
                    n += 1
            energy[c] = {"E_total_mJ_per_frame": (w["energy_mJ"] / n) if (w["energy_mJ"] is not None and n) else None,
                         "frames": n, "T_s": w["T_s"], "method": w["method"]}
        self.energy_cost = ({c: energy[c]["E_total_mJ_per_frame"] for c in self.configs}
                            if all(energy[c]["E_total_mJ_per_frame"] for c in self.configs) else None)
        out = {"latency_ms": lat, "slowdown_ratios": self.ratios, "deadline_ms": self.deadline, "kind": "measured (live)",
               "energy_isolated": energy}
        self._write("hardware_profile", "hardware", "dev", {"contention_profile.json": out}, dataset="microbench",
                    extra={"mode": "live"})
        soft = [{"check": "heavier contention slows the reference engine", "pass": self.ratios[levels[-1]][self.ref] > 1.0,
                 "detail": self.ratios[levels[-1]][self.ref]},
                {"check": "per-config energy measured (energy-objective ablation runs)", "pass": self.energy_cost is not None,
                 "detail": self.energy_cost}]
        self.gate("S11_hardware", [{"check": "contention profile measured", "pass": True}], soft)
        return out

    # ------------------------------------------------------------------ S13-15 final held-out evaluation
    def s15_final_eval(self) -> dict:
        e = self.cfg["eval"]
        ev = ComposedEvaluator(self.test_gt, {c: self.test["dets"][c] for c in self.all_cfgs}, e["max_dets"])
        fast = FastAP(ev)
        ids = self.test_ids
        seqs = sorted(self.test_seq)
        test_slices = {"all": ids, **{n: [r["image_id"] for r in self.test_rows if r.get(c) == 1] for n, c in SLICES.items()}}
        pol, info = {}, {}
        for c in self.all_cfgs:
            pol[f"fixed_{c}"] = {i: c for i in ids}
        for bname, B in self.budgets.items():
            feas = [c for c in self.configs if self.cost_iso[c] <= B + 1e-9]
            bf = max(feas, key=lambda c: self.dev_fixed_map[c]) if feas else min(self.configs, key=lambda c: self.cost_iso[c])
            pol[f"best_fixed_{bname}"] = {i: bf for i in ids}
            info[f"best_fixed_{bname}"] = {"config": bf}
        sl = Q.contention_slowdown([len(self.test_seq[s]) for s in seqs], self.configs, self.ratios,
                                   list(self.D["hardware_contention"]["levels"]), int(self.Qc["frames_per_phase"]))
        replay_check = True
        logs = {}
        for name in self.locks:
            p, model, body = self._load_lock(name)
            qd = Q.build_qdata(self.test_seq, p.configs, self.test["dets"], self.test_area, self.test["cues"], self.test_f1, self.tau_c)
            tables = Q.prediction_table(qd, lambda si, m=model: m, p.use_img, p.use_temporal)
            ch = Q.fast_replay_q(tables, p)
            pol[f"qroute_{name}"] = self._choice(self.test_seq, qd.seq_ids, ch, p.configs)
            logs[name] = (p, ch, qd)
            info[f"qroute_{name}"] = {"K": p.K, "lam": p.lam, "budget": body["budget"], "configs": p.configs}
            if name == "main_B_int8":     # reference: step-by-step controller on the cache
                for si, s in enumerate(qd.seq_ids):
                    ctl = Q.QRouteController(model, p, self.tau_c)
                    got = []
                    for iid in self.test_seq[s]:
                        o = ctl.step(self.test["cues"][iid])
                        got.append(p.configs.index(o["config"]))
                        ctl.observe(self.test["dets"][o["config"]][iid], self.test_area[iid], observed_cost=p.cost_iso[o["config"]])
                    replay_check &= got == ch[si].tolist()
            if name in ("main_B_int8", "abl_no_hw", "hardware_only"):    # contention replay (estimated)
                p2 = Q.QParams(**{**p.__dict__, "deadline": self.deadline})
                chc = Q.fast_replay_q(tables, p2, sl)
                pol[f"contention_qroute_{name}"] = self._choice(self.test_seq, qd.seq_ids, chc, p.configs)
                iso = np.array([p.cost_iso[c] for c in p.configs])
                lat = np.concatenate([iso[c] * s_[np.arange(len(c)), c] for c, s_ in zip(chc, sl, strict=True)])
                info[f"contention_qroute_{name}"] = {"deadline_ms": self.deadline, "miss_rate": float(np.mean(lat > self.deadline)),
                                                     "mean_latency_ms_est": float(lat.mean()), "kind": "estimated (replay with measured slowdown ratios)"}
        # v1 / v2 on the new test clips (two-level, 640)
        two = {"int8": self.test["dets"]["int8_640"], "fp16": self.test["dets"][self.ref]}
        f1two = {"int8": self.test_f1["int8_640"], "fp16": self.test_f1[self.ref]}
        try:
            v1p, _, _ = CAL.load_locked(self.repo / "results" / "locked" / "locked_bdd_main.yaml")
            arr = CAL.build_arrays(self.test_seq, self.test["cues"], two, f1two, self.tau_c, self.cfg["controller"]["u_empty"])
            pol["v1_r04"] = {int(i): ("int8_640" if c == 0 else self.ref) for i, c in zip(arr.image_ids, CAL.fast_replay(arr, v1p), strict=True)}
        except Exception as ex:  # noqa: BLE001
            info["v1_r04"] = {"error": repr(ex)}
        try:
            vb = yaml.safe_load((self.repo / "results" / "locked" / "locked_v2_rho0.5.yaml").read_text())
            vm = pickle.loads((self.repo / "results" / "locked" / "v2_model.pkl").read_bytes())
            sd = R.build_seqdata(self.test_seq, self.test["cues"], two, self.test_area, None, self.tau_c)
            ch = R.fast_replay_v2(R.risk_table(vm, sd), [np.array([self.test["cues"][i]["d"] for i in s]) for s in sd.image_ids],
                                  R.V2Params(vb["tau"], vb["K"], vb["theta_sc"], vb["H"]))
            pol["v2_r05"] = {i: ("int8_640" if c == 0 else self.ref) for s, cc in zip(sd.seq_ids, ch, strict=True) for i, c in zip(self.test_seq[s], cc, strict=True)}
        except Exception as ex:  # noqa: BLE001
            info["v2_r05"] = {"error": repr(ex)}
        # oracle@B (upper bound) and matched random@B_int8
        Rm = np.array([[self.test_f1[self.ref][i] - self.test_f1[c][i] for c in self.configs] for i in ids])
        for bname, B in self.budgets.items():
            chb, cst = Q.oracle_choices(Rm, np.array([self.cost_iso[c] for c in self.configs]), B)
            pol[f"oracle_{bname}"] = {i: self.configs[k] for i, k in zip(ids, chb, strict=True)}
            info[f"oracle_{bname}"] = {"mean_cost": cst, "kind": "upper bound, uses GT"}
        p, ch, qd = logs["main_B_int8"]
        mk = baselines.estimate_markov({s: c.tolist() for s, c in zip(qd.seq_ids, ch, strict=True)}, len(self.configs))
        nrs = self.n("random_seeds", self.Qc["random_seeds"])
        groups = {"random_B_int8": []}
        for k in range(nrs):
            rc = baselines.markov_random({s: len(self.test_seq[s]) for s in seqs}, mk["P"], mk["init"], self.seed + k)
            nm = f"random_B_int8_s{self.seed + k}"
            pol[nm] = {i: self.configs[c] for s in seqs for i, c in zip(self.test_seq[s], rc[s], strict=True)}
            groups["random_B_int8"].append(nm)
        # metrics
        out = {}
        for name, choice in pol.items():
            m = ev.evaluate(choice, test_slices)
            lv = [choice[i] for i in ids]
            m["all"]["allocation"] = {c: float(np.mean([x == c for x in lv])) for c in self.all_cfgs}
            m["all"]["mean_cost_ms_est"] = float(np.mean([self.cost_iso[x] for x in lv])) + (
                self.locks.get(name.replace("qroute_", ""), {}).get("overhead", 0.0) if name.startswith("qroute_") else 0.0)
            chi = np.array([self.all_cfgs.index(x) for x in lv])
            starts = np.cumsum([0] + [len(self.test_seq[s]) for s in seqs])[:-1]
            same = np.ones(len(chi), bool)
            same[starts] = False
            m["all"]["switch_rate_per_100"] = 100.0 * float(np.sum((chi[1:] != chi[:-1]) & same[1:])) / len(chi)
            m["info"] = info.get(name, {})
            out[name] = m
        agg = {}
        for g, members in groups.items():
            vals = [out[x]["all"]["map50_95"] for x in members]
            agg[g] = {"map50_95": float(np.mean(vals)), "map50_95_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                      "mean_cost_ms_est": float(np.mean([out[x]["all"]["mean_cost_ms_est"] for x in members]))}
        nb = self.n("bootstrap_resamples", self.Qc["bootstrap_resamples"])
        boot_pol = {k: pol[k] for k in ("qroute_main_B_int8", "qroute_main_B_mid", "best_fixed_B_int8", "best_fixed_B_mid",
                                         "fixed_fp16_640", "fixed_int8_640") if k in pol}
        for mmb in groups["random_B_int8"]:
            boot_pol[mmb] = pol[mmb]
        pairs = [("qroute_main_B_int8", "best_fixed_B_int8"), ("qroute_main_B_int8", "random_B_int8"),
                 ("qroute_main_B_mid", "best_fixed_B_mid"), ("qroute_main_B_int8", "fixed_int8_640"),
                 ("qroute_main_B_int8", "fixed_fp16_640")]
        boot = cluster_bootstrap(fast, dict(self.test_seq), boot_pol, pairs, nb, self.seed, random_groups=groups)
        hyp = self._hypotheses(out, boot, info)
        self.final = {"metrics": out, "aggregates": agg, "bootstrap": boot, "hypotheses": hyp, "info": info}
        for name, m in out.items():
            if name.startswith("random_B_int8_s"):
                continue
            self._write(f"final_{name}", name, "final_test", {"metrics.json": m})
        self._write("final_random_B_int8", "random_B_int8", "final_test", {"metrics.json": agg["random_B_int8"]})
        self._write("final_bootstrap", "bootstrap", "final_test", {"bootstrap.json": boot})
        self._write("final_hypotheses", "hypotheses", "final_test", {"hypotheses.json": hyp})
        checks = [{"check": "Q-ROUTE main: fast replay == step-by-step controller on final test", "pass": bool(replay_check)},
                  {"check": "every policy scored on all final-test frames", "pass": all(out[n_]["all"]["n_images"] == len(ids) for n_ in out)},
                  {"check": f"bootstrap ran ({nb})", "pass": boot["n_resamples"] == nb}]
        self.gate("S15_final_eval", checks)
        return {"hypotheses": hyp, "main": out["qroute_main_B_int8"]["all"]}

    def _hypotheses(self, out, boot, info) -> dict:
        def pr(a, b):
            v = boot["pairs"][f"{a} - {b}"]["map50_95"]
            return {"mean_pts": 100 * v["mean"], "ci95_pts": [100 * x for x in v["ci95"]], "supported": bool(v["ci95"][0] > 0)}

        h = {"Q1_primary": pr("qroute_main_B_int8", "best_fixed_B_int8"), "Q2": pr("qroute_main_B_int8", "random_B_int8"),
             "Q3": pr("qroute_main_B_mid", "best_fixed_B_mid")}
        c1, c2 = info.get("contention_qroute_main_B_int8"), info.get("contention_qroute_abl_no_hw")
        if c1 and c2:
            dm = 100 * (out["contention_qroute_main_B_int8"]["all"]["map50_95"] - out["contention_qroute_abl_no_hw"]["all"]["map50_95"])
            h["Q5_replay"] = {"miss_rate_hw": c1["miss_rate"], "miss_rate_no_hw": c2["miss_rate"], "map_diff_pts": dm,
                              "supported": bool(c1["miss_rate"] < c2["miss_rate"] and dm > -0.3), "kind": "estimated (replay)"}
        return h

    # ------------------------------------------------------------------ S16 live benchmarking + contention
    def s16_live(self) -> dict:
        nclips = self.n("live_clips", self.Qc["live_clips"])
        nfr = self.n("live_frames_per_clip", self.Qc["live_frames_per_clip"])
        rng = np.random.default_rng(self.seed)
        seqs = sorted(self.test_seq)
        pick = sorted(seqs[i] for i in rng.choice(len(seqs), min(nclips, len(seqs)), replace=False))
        frames, starts, fids = [], [], []
        for s in pick:
            starts.append(len(frames))
            ids = self.test_seq[s][:nfr]
            fids += ids
            frames += [self.read(self.layout.image_root / next(r["rel_path"] for r in self.test_rows if r["image_id"] == i)) for i in ids]
        area = float(np.median(list(self.test_area.values())))
        meter = self.meter_factory()
        p_main, m_main, _ = self._load_lock("main_B_int8")
        p_mid, m_mid, _ = self._load_lock("main_B_mid")
        steppers = {f"fixed_{c}": FixedStepper(self.backends[c], frames) for c in self.configs}
        steppers["qroute_main_B_int8"] = QRouteStepper(self.backends, m_main, p_main, self.ff, frames, starts, area, self.tau_c)
        steppers["qroute_main_B_mid"] = QRouteStepper(self.backends, m_mid, p_mid, self.ff, frames, starts, area, self.tau_c)
        T = self.cfg["full"]["smoke_timing" if self.smoke else "timing"]
        prot = PROTO.run_protocol(steppers, meter, T["warmup_s"], T["settle_s"], T["idle_s"], T["measure_s"], T["repeats"],
                                  self.seed, log=log.info)
        # memory: all six engines resident vs one
        mem = {"used_mib_all_engines_resident": meter.memory_used_mib()}
        # live vs replay decisions (one pass, no contention). With hardware adaptation the live controller reacts to
        # measured latency, which isolated replay does not model, so the implementation check is the no-hw variant.
        ev_live = CocoEvaluator(self.test_gt, self.cfg["eval"]["max_dets"])

        def agreement(name):
            p_, m_, _ = self._load_lock(name)
            st = QRouteStepper(self.backends, m_, p_, self.ff, frames, starts, area, self.tau_c)
            rows = [st.step() for _ in range(len(frames))]
            qd = Q.build_qdata({s: self.test_seq[s][:nfr] for s in pick}, p_.configs, self.test["dets"], self.test_area,
                               self.test["cues"], None, self.tau_c)
            rep = np.concatenate(Q.fast_replay_q(Q.prediction_table(qd, lambda si: m_, p_.use_img, p_.use_temporal), p_))
            ag = float(np.mean([p_.configs[c] == r["level"] for c, r in zip(rep, rows, strict=True)]))
            m_live = ev_live.evaluate(dets_to_coco_results({i: r["dets"] for i, r in zip(fids, rows, strict=True)}), fids)["map50_95"]
            m_rep = ev_live.evaluate(dets_to_coco_results({i: self.test["dets"][p_.configs[c]][i] for i, c in zip(fids, rep, strict=True)}),
                                     fids)["map50_95"]
            return {"decision_agreement": ag, "map50_95_live": m_live, "map50_95_replay": m_rep, "map_diff_pts": 100 * (m_live - m_rep)}

        lv_main = agreement("main_B_int8")
        lv_nohw = agreement("abl_no_hw")
        agree, agree_nohw = lv_main["decision_agreement"], lv_nohw["decision_agreement"]
        # contention (live): schedule of phases, deadline-aware variants
        phase = self.n("contention_phase_s", self.Qc["contention_phase_s"])
        levels = list(self.D["hardware_contention"]["levels"])
        conts = {}
        p_nohw, m_nohw, _ = self._load_lock("abl_no_hw")
        p_hwo, m_hwo, _ = self._load_lock("hardware_only")
        variants = {"qroute_hw": (m_main, Q.QParams(**{**p_main.__dict__, "deadline": self.deadline})),
                    "qroute_no_hw": (m_nohw, Q.QParams(**{**p_nohw.__dict__, "deadline": self.deadline})),
                    "hardware_only": (m_hwo, Q.QParams(**{**p_hwo.__dict__, "deadline": self.deadline}))}
        with self._contention() as g:
            for name, (mdl, pp) in list(variants.items()) + [("fixed_fp16_640", (None, None)), ("fixed_int8_640", (None, None))]:
                stp = FixedStepper(self.backends[name[6:]], frames) if pp is None else \
                    QRouteStepper(self.backends, mdl, pp, self.ff, frames, starts, area, self.tau_c)
                rows = []
                for lv in levels:
                    g.set_level(lv)
                    t_end = time.perf_counter() + phase
                    while time.perf_counter() < t_end:
                        r = stp.step()
                        r["contention"] = lv
                        rows.append(r)
                g.set_level(0.0)
                firsts = {}
                for r in rows:
                    firsts.setdefault(fids[r["frame"]], r)
                ev = CocoEvaluator(self.test_gt, self.cfg["eval"]["max_dets"])
                mp = ev.evaluate(dets_to_coco_results({i: firsts[i]["dets"] for i in firsts}), list(firsts))["map50_95"]
                tot = np.array([r["total_ms"] for r in rows])
                conts[name] = {"frames": len(rows), "deadline_ms": self.deadline, "miss_rate": float(np.mean(tot > self.deadline)),
                               "latency_median_ms": float(np.median(tot)), "latency_p95_ms": float(np.percentile(tot, 95)),
                               "by_level": {str(lv): {"miss_rate": float(np.mean([r["total_ms"] > self.deadline for r in rows if r["contention"] == lv] or [0])),
                                                      "latency_median_ms": float(np.median([r["total_ms"] for r in rows if r["contention"] == lv] or [0]))}
                                            for lv in levels},
                               "allocation": {c: float(np.mean([r.get("level", name[6:]) == c for r in rows])) for c in self.configs},
                               "map50_95_first_pass": mp, "n_first_pass_frames": len(firsts), "kind": "measured (live)"}
        val = {"agreement_main_hw_adaptive": agree, "agreement_no_hw": agree_nohw, "main_hw_adaptive": lv_main, "no_hw": lv_nohw,
               "clips": pick,
               "note": "no_hw must match replay exactly (implementation check); main differs only through measured latency"}
        self.live = {"protocol": prot, "memory": mem, "live_vs_replay": val, "contention": conts, "clips": pick}
        self._write("live", "measurement", "final_test", {"energy.json": prot, "memory.json": mem, "contention_live.json": conts,
                                                          "live_validation.json": val},
                    extra={"mode": "live", "energy_method": prot["method"]})
        s = prot["summary"]
        q4 = {"qroute_latency_ms": s["qroute_main_B_int8"]["latency_total_ms_median"]["median"],
              "fixed_int8_640_latency_ms": s["fixed_int8_640"]["latency_total_ms_median"]["median"]}
        q4["supported"] = bool(q4["qroute_latency_ms"] <= 1.05 * q4["fixed_int8_640_latency_ms"])
        q5 = {"miss_rate_hw": conts["qroute_hw"]["miss_rate"], "miss_rate_no_hw": conts["qroute_no_hw"]["miss_rate"],
              "map_diff_pts": 100 * (conts["qroute_hw"]["map50_95_first_pass"] - conts["qroute_no_hw"]["map50_95_first_pass"])}
        q5["supported"] = bool(q5["miss_rate_hw"] < q5["miss_rate_no_hw"] and q5["map_diff_pts"] > -0.3)
        self.live_hyp = {"Q4_live_cost": q4, "Q5_live": q5}
        self._write("live_hypotheses", "hypotheses", "final_test", {"hypotheses_live.json": self.live_hyp}, extra={"mode": "live"})
        checks = [{"check": f"measured {k}", "pass": s[k]["throughput_fps"] is not None} for k in steppers]
        soft = [{"check": "live vs replay agreement, no-hw variant (implementation check) >= 0.99", "pass": agree_nohw >= 0.99,
                 "detail": agree_nohw},
                {"check": "live vs replay agreement, hw-adaptive main >= 0.9", "pass": agree >= 0.9, "detail": agree}]
        self.gate("S16_live", checks, soft)
        return self.live_hyp

    # ------------------------------------------------------------------ S19 completion report
    def s19_summary(self) -> dict:
        exp = ["S3_data", "S4_models", "S5_caches", "S7_headroom", "S8_S9_features_predictor", "S10_controller_check", "S11_hardware",
               "S12_controller_locks", "S15_final_eval", "S16_live"]
        hyp = {**getattr(self, "final", {}).get("hypotheses", {}), **getattr(self, "live_hyp", {})}
        rep = {"stages_passed": {k: self.stages.get(k, {}).get("pass") for k in exp},
               "incomplete": [k for k in exp if not self.stages.get(k, {}).get("pass")],
               "hypotheses": {k: v.get("supported") for k, v in hyp.items()},
               "measured_vs_estimated": {"measured (live)": ["S4 profile", "S11 contention profile", "S16 energy/latency/contention"],
                                         "estimated (replay on cached detections)": ["S15 mAP of policies (exact on cached detections)",
                                                                                      "S15 mean cost", "S15 contention replay"]},
               "skipped_variants": getattr(self, "skipped_variants", {}),
               "visdrone": "validated only if attached; not part of the pre-registered pipeline",
               "int4": "unsupported in this toolchain (see r06_models/int4.json)",
               "prereg_qroute_sha256": self.prereg_sha}
        self._write("experiment_summary", "summary", "all", {"experiment_summary.json": rep})
        return rep
