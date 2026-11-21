"""Notebook 04: everything that is left, in one gated run (user decision 2026-11-20).

  1 engines + calib cache  (rebuild FP32/FP16/INT8, microbench, switch penalty, calib cache, Step 2 checks)
    GATE: binding Step 2 sanity passes; calib mAP within tol of the committed Step 2 values
  2 calib controller dev   (Step 3 logic on this run's calib cache)      GATE: Step 3 sanity passes
  3 calibrate + lock       (PLAN §3.5 grid incl. weight simplex; Pareto deltas; ablations)
    GATE: locked files written and hash-verified
  4 test cache             (ONLY after a verified lock; test + spliced-stream cues and detections)
    GATE: complete cache, test/calib disjoint
  5 timing/energy/memory   (PLAN §7 protocol; live adaptive sequential + pipelined; live validation)
    GATE: every config measured (live-vs-replay gap is reported, soft check)
  6 test evaluation        (locked configs only; slices; spliced scene-change; bootstrap CIs; H1-H4)
    GATE: locked replay reproduces exactly via fast replay; all policies evaluated

Each stage records its gate in results/<prefix>stages/stages.json. On a failed gate the
notebook zips what exists and stops (no later stage runs on unchecked inputs).
All experiment ids start with full.exp_prefix ("r04_").
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .. import baselines
from .. import calibrate as CAL
from .. import hypotheses as HYP
from ..controller import ControllerParams
from ..data import bdd_mot
from ..energy import protocol as PROTO
from ..energy.nvml_meter import NvmlMeter
from ..eval.coco_gt import build_coco_gt
from ..eval.fast_ap import FastAP, cluster_bootstrap
from ..eval.map_eval import CocoEvaluator, ComposedEvaluator, dets_to_coco_results
from ..eval.perframe import gt_arrays_from_coco, perframe_f1
from ..features import FrameFeaturizer
from ..live import AdaptiveStepper, FixedStepper, run_live_sequences
from ..replay import replay
from ..results import ExperimentWriter
from .controller_dev import Step3
from .eval_fixed import SLICE_COLS, Step2, _read_csv

log = logging.getLogger("adapres.full")
LEVELS = ("int8", "fp16")
ALL_LEVELS = ("fp32", "fp16", "int8")


class GateError(RuntimeError):
    pass


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, ControllerParams):
        return CAL.params_dict(o)
    return o


class FullRun:
    def __init__(self, cfg: dict, smoke: bool, repo_dir: str | Path, search_root: str | Path | None = None,
                 exporter: Callable | None = None, backend_factory: Callable | None = None,
                 image_reader: Callable | None = None, featurizer_factory: Callable | None = None,
                 meter_factory: Callable | None = None, release_fn: Callable | None = None):
        cfg = json.loads(json.dumps(cfg))
        cfg["step2"]["pt_reference"] = False          # PyTorch reference was Step 2's job
        cfg["step2"]["levels"] = list(ALL_LEVELS)
        self.cfg, self.smoke, self.repo = cfg, smoke, Path(repo_dir)
        self.F = cfg["full"]
        self.prefix = self.F["exp_prefix"]
        pre = yaml.safe_load((self.repo / "configs" / "preregistration.yaml").read_text())
        if abs(pre["delta_map_points"] - self.F["delta_pts"]) > 1e-12:
            raise GateError(f"full.delta_pts {self.F['delta_pts']} != pre-registered {pre['delta_map_points']}")
        self.s2 = Step2(cfg, smoke, repo_dir, search_root=search_root, exporter=exporter,
                        backend_factory=backend_factory, image_reader=image_reader, exp_prefix=self.prefix)
        f = cfg["features"]
        self.ff = featurizer_factory or (lambda: FrameFeaturizer(f["width"], f["canny_low"], f["canny_high"],
                                                                  f["hist_bins"]))
        self.read = self.s2.image_reader
        self.meter_factory = meter_factory
        self.release_fn = release_fn or self._release_cuda
        self.results_root = self.s2.results_root
        self.work = self.s2.work.parent / "work04"
        self.lock_dir = self.results_root / "locked"
        self.calib_cfg = self.F["smoke_calib"] if smoke else self.F["calib"]
        self.timing_cfg = self.F["smoke_timing"] if smoke else self.F["timing"]
        self.tau_c, self.u_empty = cfg["model"]["conf_cue"], cfg["controller"]["u_empty"]
        self.stages: dict = {}
        self.backends: dict = {}

    # ------------------------------------------------------------------ infra
    @staticmethod
    def _release_cuda():
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _writer(self, exp_id: str, policy: str, split: str, dataset: str = "bdd100k_mot", extra: dict | None = None):
        meta = self.s2._meta(self.prefix + exp_id, policy, {"split": split, "mode": "replay", **(extra or {})},
                             dataset=dataset)
        meta["notes"] = "notebook 04 (combined run, rebuilt engines)"
        return ExperimentWriter(self.results_root, meta)

    def gate(self, stage: str, checks: list[dict], soft: list[dict] | None = None) -> dict:
        ok = all(c["pass"] for c in checks)
        rec = {"stage": stage, "pass": ok, "checks": checks, "soft_checks": soft or [],
               "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.stages[stage] = rec
        d = self.results_root / f"{self.prefix}stages"
        d.mkdir(parents=True, exist_ok=True)
        (d / "stages.json").write_text(json.dumps(_jsonable(self.stages), indent=2))
        for c in checks + (soft or []):
            log.info("[%s] %s %s", "PASS" if c["pass"] else ("FAIL" if c in checks else "WARN"), c["check"],
                     c.get("detail", ""))
        if not ok:
            raise GateError(f"gate '{stage}' failed: " + "; ".join(c["check"] for c in checks if not c["pass"]))
        return rec

    # ------------------------------------------------------------------ 1. engines + calib cache
    def stage1_engines(self, weights: str | Path, names: dict[int, str]) -> dict:
        s2 = self.s2
        s2.coco_names = names
        info = s2.prepare()
        y = s2.int8_dataset(names)
        s2.export(weights, y)
        self.backends = s2.backends()
        mb = s2.microbench(self.backends)
        self.engine_ms_all = {lv: mb["per_backend"][lv]["infer_ms"]["median"] for lv in ALL_LEVELS}
        calib_frames = [self.read(s2.paths[i]) for i in s2.image_ids[: self.timing_cfg["switch_penalty_frames"]]]
        self.switch = PROTO.switch_penalty(self.backends, calib_frames, self.timing_cfg["switch_penalty_frames"],
                                           repeats=1 if self.smoke else 3)
        w = self._writer("switch_penalty", "switch_penalty", "calib", dataset="microbench")
        w.write_json("switch_penalty.json", self.switch)
        w.register(policy="switch_penalty", mode="microbench", replace=True)
        s2.cache_runs(self.backends)
        self.calib_metrics = s2.evaluate()
        san = s2.sanity(self.calib_metrics)
        checks = [{"check": f"Step 2 binding sanity: {c['check']}", "pass": c["pass"], "detail": c["value_pts"]}
                  for c in san["checks"] if c.get("binding", True)]
        ref = self.repo / "results" / "bdd100k_mot"
        tol = self.F["gate_calib_map_tol_pts"]
        if not self.smoke:  # committed Step 2 covers all 20 calib clips
            for lv in ALL_LEVELS:
                old = json.loads((ref / f"calib_fixed_{lv}" / "metrics.json").read_text())
                d = {sl: 100 * abs(self.calib_metrics[lv][sl]["map50_95"] - old[sl]["map50_95"])
                     for sl in old if sl in self.calib_metrics[lv]}
                checks.append({"check": f"rebuilt {lv} calib mAP50-95 within {tol} pts of committed Step 2 (all slices)",
                               "pass": max(d.values()) <= tol, "detail": d})
        self.gate("1_engines_calib_cache", checks)
        return {"prepare": info, "microbench": mb, "switch_penalty": self.switch["penalty_ms"]}

    # ------------------------------------------------------------------ 2. calib controller dev
    def stage2_calib_dev(self) -> dict:
        st = Step3(self.cfg, self.smoke, self.repo, image_reader=self.read, featurizer_factory=self.ff,
                   exp_prefix=self.prefix, ref_results=self.results_root, step2=self.s2)
        st.prepare()
        st.locate_cache(self.s2.cache_dir)
        st.verify_cache()
        st.features()
        st.fit()
        st.policies()
        san = st.sanity()
        self.st = st
        self.gate("2_calib_dev", [{"check": c["check"], "pass": c["pass"], "detail": c["detail"]} for c in san["checks"]])
        return san

    # ------------------------------------------------------------------ 3. calibrate + lock
    def _map_fn(self, arrays, ev, ids):
        cache = {}

        def f(p: ControllerParams) -> float:
            key = json.dumps(CAL.params_dict(p), sort_keys=True)
            if key not in cache:
                ch = CAL.fast_replay(arrays, p)
                cache[key] = ev.evaluate({int(i): LEVELS[c] for i, c in zip(arrays.image_ids, ch, strict=True)},
                                         {"all": ids})["all"]["map50_95"]
            return cache[key]
        return f

    def stage3_calibrate(self) -> dict:
        st, C = self.st, self.calib_cfg
        cc = self.cfg["controller"]
        arrays = CAL.build_arrays(st.seq_frames, st.cues, st.dets, st.f1, self.tau_c, self.u_empty)
        self.engine_ms = st.engine_ms
        self.ctrl_overhead_ms = (st.feature_timing["featurize_ms"]["median"] +
                                 st.results[f"adaptive_K{cc['K']}"]["all"]["ctrl_step_ms"]["median"])
        sp = self.switch["penalty_ms"]
        mapf = self._map_fn(arrays, st.ev, st.ids)
        fp16_map = st.results["fixed_fp16"]["all"]["map50_95"]
        base = {"start_level": cc["start_level"], "H": cc["H"], "scene_override": cc["scene_override"], "w_h": cc["w_h"]}
        weights = CAL.simplex_weights(C["simplex_step"])
        t0 = time.time()
        rows = CAL.grid_search(arrays, base, st.norm, weights, C["theta_quantiles"], C["Ks"], C["alphas"],
                               C["theta_scs"], self.engine_ms, sp, self.ctrl_overhead_ms)
        log.info("stage A: %d configs in %.0fs", len(rows), time.time() - t0)
        sel = CAL.select(rows, mapf, fp16_map, self.F["delta_pts"], C["stage_a_f1_slack"], C["stage_b_top"],
                         C["stage_b_max"])
        locks = {}
        common = {"dataset": "bdd100k_mot", "fit_split": "calib", "fp16_calib_map50_95": fp16_map,
                  "engine_sha256": {k: v.get("sha256") for k, v in self.s2.engine_info.items()},
                  "git_commit": self.s2.env["git"]["commit"], "tau_c": self.tau_c, "u_empty": self.u_empty}

        def do_lock(name, s, extra):
            b = s["best"]
            h = CAL.lock(self.lock_dir / f"locked_bdd_{name}.yaml", b["_params"],
                         {**common, **extra, "status": s["status"], "calib": {k: b[k] for k in (
                             "map50_95", "saving", "lat_ms", "switch_rate_per_100", "frac_int8", "mean_f1",
                             "theta_quantile") if k in b}})
            locks[name] = {"hash": h, "status": s["status"], "map50_95": b.get("map50_95"), "saving": b["saving"],
                           "frac_int8": b["frac_int8"], "switch_rate_per_100": b["switch_rate_per_100"],
                           "params": CAL.params_dict(b["_params"])}

        do_lock("main", sel, {"delta_pts": self.F["delta_pts"], "role": "main"})
        main = sel["best"]["_params"]
        for dlt in C.get("pareto_deltas", []):
            s = CAL.select(rows, mapf, fp16_map, dlt, C["stage_a_f1_slack"], C["stage_b_top"], C["stage_b_max"])
            do_lock(f"delta{dlt}", s, {"delta_pts": dlt, "role": "pareto"})
        # ---- ablations: same recalibration procedure, restricted grid, alpha/theta_sc from main
        abl_specs = {}
        for cue in "mcnus":
            w = dict(main.weights)
            w[cue] = 0.0
            tot = sum(w.values())
            if tot > 0:
                abl_specs[f"A-{cue}"] = {"weights": [{k: v / tot for k, v in w.items()}]}
            abl_specs[f"S-{cue}"] = {"weights": [{k: (1.0 if k == cue else 0.0) for k in "mcnus"}]}
        abl_specs["A-hyst"] = {"weights": [main.weights], "Ks": [1]}
        abl_specs["A-sc-override"] = {"weights": [main.weights], "base": {**base, "scene_override": False}}
        abl_specs["A-asym"] = {"weights": [main.weights], "Ks": [k for k in C["Ks"] if k >= 2] or [2], "K_mode": "asym"}
        abl_specs["A-ema"] = {"weights": [main.weights], "alphas": [1.0]}
        abl_info = {}
        for name, spec in abl_specs.items():
            r2 = CAL.grid_search(arrays, spec.get("base", base), st.norm, spec["weights"], C["theta_quantiles"],
                                 spec.get("Ks", C["Ks"]), spec.get("alphas", [main.alpha]), [main.theta_sc],
                                 self.engine_ms, sp, self.ctrl_overhead_ms, K_mode=spec.get("K_mode", "sym"))
            s = CAL.select(r2, mapf, fp16_map, self.F["delta_pts"], C["stage_a_f1_slack"], C["ablation_stage_b_top"],
                           C["ablation_stage_b_max"])
            do_lock(f"abl_{name}", s, {"delta_pts": self.F["delta_pts"], "role": "ablation", "ablation": name})
            abl_info[name] = locks[f"abl_{name}"]
        for K in C["K_sweep"]:  # theta fixed at the locked value (no recalibration), PLAN §1.3
            p = ControllerParams(**{**{k: getattr(main, k) for k in main.__dataclass_fields__}, "K_up": K, "K_down": K})
            h = CAL.lock(self.lock_dir / f"locked_bdd_ksweep{K}.yaml", p, {**common, "role": "K-sweep", "K": K,
                                                                            "delta_pts": self.F["delta_pts"]})
            locks[f"ksweep{K}"] = {"hash": h, "params": CAL.params_dict(p)}
        # ---- write grid + summary
        w = self._writer("calibration", "calibration", "calib", extra={"locks": {k: v["hash"] for k, v in locks.items()}})
        flat = [{**{f"w_{k}": r["weights"][k] for k in "mcnus"}, **{k: r[k] for k in (
            "alpha", "theta_sc", "theta_quantile", "theta", "K_up", "K_down", "mean_f1", "f1_drop_vs_fp16", "saving",
            "lat_ms", "switch_rate_per_100", "frac_int8")}, "map50_95_stageB": r.get("map50_95")} for r in rows]
        w.write_csv("grid.csv", flat, list(flat[0].keys()))
        w.write_json("calibration.json", {"main": _jsonable({k: v for k, v in sel.items() if k != "evaluated"}),
                                          "stage_b_evaluated": [_jsonable({k: v for k, v in r.items() if k != "_params"})
                                                                for r in sel["evaluated"]],
                                          "locks": locks, "fp16_calib_map50_95": fp16_map,
                                          "engine_ms": self.engine_ms, "switch_penalty_ms": sp,
                                          "ctrl_overhead_ms_sequential": self.ctrl_overhead_ms,
                                          "n_stage_a_configs": len(rows)})
        w.register(policy="calibration", mode="calibrate", replace=True)
        self.locks = locks
        checks = []
        for name in locks:
            try:
                CAL.load_locked(self.lock_dir / f"locked_bdd_{name}.yaml")
                checks.append({"check": f"lock {name} written and hash-verified", "pass": True})
            except CAL.LockError as e:
                checks.append({"check": f"lock {name}", "pass": False, "detail": str(e)})
        soft = [{"check": "main config feasible on calib (mAP >= FP16 - delta)", "pass": sel["status"] == "feasible",
                 "detail": sel["status"]}]
        self.gate("3_calibrate_lock", checks, soft)
        return {"main": locks["main"], "n_configs": len(rows), "status": sel["status"]}

    # ------------------------------------------------------------------ 4. test data + cache
    def _require_lock(self) -> tuple[ControllerParams, dict, str]:
        return CAL.load_locked(self.lock_dir / "locked_bdd_main.yaml")

    def stage4_test_cache(self) -> dict:
        p_main, _, h_main = self._require_lock()     # test is never read before a verified lock
        sub = self.repo / "results" / "bdd100k_mot" / "bdd_subsets"
        sel = json.loads((sub / "selection.json").read_text())
        rows = _read_csv(sub / "frames_test.csv")
        seqs = sorted({r["seq"] for r in rows})
        if self.smoke:
            rng = np.random.default_rng(self.s2.seed)
            seqs = sorted(seqs[i] for i in rng.choice(len(seqs), self.F["test"]["smoke_n_sequences"], replace=False))
        rows = sorted([r for r in rows if r["seq"] in set(seqs)], key=lambda r: (r["seq"], int(r["frame_idx"])))
        lay = self.s2.layout
        frames_by_key = {}
        for s in seqs:
            for fr in bdd_mot.load_video_labels(lay.label_dir / f"{s}.json"):
                frames_by_key[(s, fr.frame_idx)] = fr
        self.t_rows = rows
        self.t_ids = [int(r["image_id"]) for r in rows]
        self.t_seq_of = {int(r["image_id"]): r["seq"] for r in rows}
        self.t_paths = {int(r["image_id"]): lay.image_root / r["rel_path"] for r in rows}
        self.t_seq_frames = {}
        for i in self.t_ids:
            self.t_seq_frames.setdefault(self.t_seq_of[i], []).append(i)
        self.t_gt = build_coco_gt([(int(r["image_id"]), frames_by_key[(r["seq"], int(r["frame_idx"]))], int(r["W"]),
                                    int(r["H"])) for r in rows], self.cfg["classmap"]["ignore_mode"], "test")
        self.t_slices = {"all": self.t_ids, **{n: [int(r["image_id"]) for r in rows if r[c] == "1"]
                                               for n, c in SLICE_COLS.items()}}
        # cache: decode each frame once, cues + every level's detections
        cdir = self.work / "test_cache"
        cdir.mkdir(parents=True, exist_ok=True)
        self.t_dets = {lv: {} for lv in ALL_LEVELS}
        self.t_cues = {}
        for s, ids in sorted(self.t_seq_frames.items()):
            f = cdir / f"{s}.npz"
            if not f.exists():
                fz = self.ff()
                cues, dets, tims = [], {lv: [] for lv in ALL_LEVELS}, []
                for iid in ids:
                    img = self.read(self.t_paths[iid])
                    cues.append(fz(img))
                    for lv in ALL_LEVELS:
                        d, t = self.backends[lv].predict(img)
                        d = np.asarray(d, np.float32).reshape(-1, 6)
                        dets[lv].append(np.column_stack([np.full(len(d), iid), d]))
                        tims.append([iid, ALL_LEVELS.index(lv), t["infer_ms"], t["wall_ms"]])
                tmp = f.with_name(f.stem + ".tmp.npz")
                np.savez_compressed(tmp, ids=np.asarray(ids), m=[c["m"] for c in cues], c=[c["c"] for c in cues],
                                    d=[c["d"] for c in cues], timings=np.asarray(tims, float),
                                    **{f"dets_{lv}": np.concatenate(dets[lv]) if dets[lv] else np.zeros((0, 7))
                                       for lv in ALL_LEVELS})
                tmp.replace(f)
            z = np.load(f)
            for iid, m, c, d in zip(z["ids"], z["m"], z["c"], z["d"], strict=True):
                self.t_cues[int(iid)] = {"m": float(m), "c": float(c), "d": float(d)}
            for lv in ALL_LEVELS:
                per = {i: [] for i in ids}
                for row in z[f"dets_{lv}"]:
                    per[int(row[0])].append(row[1:])
                for i in ids:
                    self.t_dets[lv][i] = np.asarray(per[i], float).reshape(-1, 6)
        # spliced streams (scene-change slice): only streams whose sources are all in this subset
        streams = _read_csv(sub / "splice_streams.csv")
        by_stream = {}
        for r in streams:
            by_stream.setdefault(r["stream_id"], []).append(r)
        src_ok = {sid: rs for sid, rs in by_stream.items() if all(int(r["src_image_id"]) in self.t_seq_of for r in rs)}
        self.sp_rows = [r for sid in sorted(src_ok) for r in sorted(src_ok[sid], key=lambda x: int(x["pos"]))]
        self.sp_src = {int(r["image_id"]): int(r["src_image_id"]) for r in self.sp_rows}
        self.sp_seq_frames = {}
        for r in self.sp_rows:
            self.sp_seq_frames.setdefault(r["stream_id"], []).append(int(r["image_id"]))
        size = {int(r["image_id"]): (int(r["W"]), int(r["H"])) for r in rows}
        fr_of = {int(r["image_id"]): frames_by_key[(r["seq"], int(r["frame_idx"]))] for r in rows}
        self.sp_gt = build_coco_gt([(sid, fr_of[src], *size[src]) for sid, src in self.sp_src.items()],
                                   self.cfg["classmap"]["ignore_mode"], "splice") if self.sp_rows else None
        self.sp_cues = {}
        for ids in self.sp_seq_frames.values():
            fz = self.ff()
            for i in ids:
                self.sp_cues[i] = fz(self.read(self.t_paths[self.sp_src[i]]))
        self.sp_slices = {"scene_change": [int(r["image_id"]) for r in self.sp_rows if r["is_splice_window"] == "1"],
                          "streams_all": [int(r["image_id"]) for r in self.sp_rows]}
        calib_seqs = set(sel["splits"]["calib"])
        checks = [
            {"check": "test cache complete (cues + 3 levels for every test frame)",
             "pass": all(i in self.t_cues and all(i in self.t_dets[lv] for lv in ALL_LEVELS) for i in self.t_ids),
             "detail": len(self.t_ids)},
            {"check": "test and calib clip sets are disjoint", "pass": not (set(seqs) & calib_seqs)},
            {"check": "locked main config hash verified before reading test", "pass": True, "detail": h_main},
        ]
        soft = [{"check": "spliced streams available for the scene-change slice", "pass": bool(self.sp_rows),
                 "detail": len(src_ok)}]
        self.gate("4_test_cache", checks, soft)
        return {"n_test_frames": len(self.t_ids), "n_sequences": len(seqs), "n_streams": len(src_ok)}

    # ------------------------------------------------------------------ 5. timing / energy / memory / live
    def stage5_timing(self) -> dict:
        p_main, _, h_main = self._require_lock()
        T = self.timing_cfg
        rng = np.random.default_rng(self.s2.seed)
        seqs = sorted(self.t_seq_frames)
        pick = sorted(seqs[i] for i in rng.choice(len(seqs), min(T["n_measure_clips"], len(seqs)), replace=False))
        frames, starts = [], []
        for s in pick:
            starts.append(len(frames))
            frames += [self.read(self.t_paths[i]) for i in self.t_seq_frames[s][: T["frames_per_clip"]]]
        meter = self.meter_factory() if self.meter_factory else NvmlMeter()
        # memory first, with no engines resident
        engine_paths = {lv: self.s2.engine_info[lv]["path"] for lv in ALL_LEVELS}
        self.backends = {}
        self.release_fn()

        def load(lv):
            b = self.s2.backend_factory(lv, engine_paths[lv])
            b.predict(frames[0])
            return b

        mem = PROTO.memory_profile(meter, load, release_fn=self.release_fn)
        mem["engine_file_MB"] = {lv: self.s2.engine_info[lv]["bytes"] / 1e6 for lv in ALL_LEVELS}
        try:
            import resource

            mem["host_maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:  # noqa: BLE001
            mem["host_maxrss_MB"] = None
        self.backends = {lv: load(lv) for lv in ALL_LEVELS}
        steppers = {f"fixed_{lv}": FixedStepper(self.backends[lv], frames) for lv in ALL_LEVELS}
        for mode in ("sequential", "pipelined"):
            steppers[f"adaptive_{mode}"] = AdaptiveStepper(self.backends, p_main, self.ff, frames, starts, self.tau_c,
                                                           self.u_empty, pipelined=(mode == "pipelined"))
        prot = PROTO.run_protocol(steppers, meter, T["warmup_s"], T["settle_s"], T["idle_s"], T["measure_s"],
                                  T["repeats"], self.s2.seed, log=log.info)
        for s_ in steppers.values():
            s_.close()
        self.energy = prot
        # live validation: locked adaptive, sequential, on a seeded subset of test clips vs replay on the cache
        LV = self.F["live_validation"]
        n_clips = LV["smoke_n_clips"] if self.smoke else max(1, round(LV["frac_clips"] * len(seqs)))
        vclips = sorted(seqs[i] for i in rng.choice(len(seqs), min(n_clips, len(seqs)), replace=False))
        vframes, vstarts, vids = [], [], []
        for s in vclips:
            vstarts.append(len(vframes))
            vids += self.t_seq_frames[s]
            vframes += [self.read(self.t_paths[i]) for i in self.t_seq_frames[s]]
        live = run_live_sequences(self.backends, p_main, self.ff, vframes, vstarts, self.tau_c, self.u_empty,
                                  pipelined=False)
        rep = replay(p_main, {s: self.t_seq_frames[s] for s in vclips}, self.t_cues,
                     {lv: self.t_dets[lv] for lv in LEVELS}, self.tau_c, self.u_empty)
        agree = float(np.mean([a["level"] == b["level"] for a, b in zip(live, rep, strict=True)]))
        ev = CocoEvaluator(self.t_gt, self.cfg["eval"]["max_dets"])
        live_map = ev.evaluate(dets_to_coco_results({i: r["dets"] for i, r in zip(vids, live, strict=True)}), vids)
        rep_map = ev.evaluate(dets_to_coco_results({r["image_id"]: self.t_dets[r["level"]][r["image_id"]] for r in rep}),
                              vids)
        self.live_val = {"clips": vclips, "n_frames": len(vids), "level_agreement": agree,
                         "live": {k: live_map[k] for k in ("map50_95", "map50")},
                         "replay": {k: rep_map[k] for k in ("map50_95", "map50")},
                         "abs_diff_pts": {k: 100 * abs(live_map[k] - rep_map[k]) for k in ("map50_95", "map50")},
                         "live_frac_int8": float(np.mean([r["level"] == "int8" for r in live])),
                         "replay_frac_int8": float(np.mean([r["level"] == "int8" for r in rep])),
                         "live_latency_total_ms_median": float(np.median([r["total_ms"] for r in live]))}
        w = self._writer("test_energy_latency", "measurement", "test", dataset="bdd100k_mot",
                         extra={"locked_config_sha256": h_main, "energy_method": prot["method"],
                                "measurement_clips": pick})
        w.write_json("energy.json", prot)
        w.write_json("memory.json", mem)
        w.write_json("live_validation.json", self.live_val)
        w.register(policy="measurement", mode="live", locked_sha=h_main, replace=True)
        checks = [{"check": f"protocol measured {k}", "pass": prot["summary"][k]["throughput_fps"] is not None}
                  for k in steppers]
        checks.append({"check": "energy method recorded", "pass": prot["method"] in ("nvml_energy_counter",
                                                                                     "nvml_power_sampling"),
                       "detail": prot["method"]})
        soft = [{"check": f"live vs replay level agreement >= {LV['soft_min_level_agreement']}",
                 "pass": agree >= LV["soft_min_level_agreement"], "detail": agree},
                {"check": "live vs replay |mAP50-95 diff| <= 0.5 pts", "pass": self.live_val["abs_diff_pts"]["map50_95"] <= 0.5,
                 "detail": self.live_val["abs_diff_pts"]}]
        self.gate("5_timing_energy", checks, soft)
        return {"summary": prot["summary"], "memory": mem, "live_validation": self.live_val}

    # ------------------------------------------------------------------ 6. test evaluation
    def stage6_test_eval(self) -> dict:
        p_main, body_main, h_main = self._require_lock()
        t0 = time.time()
        ev = ComposedEvaluator(self.t_gt, {lv: self.t_dets[lv] for lv in ALL_LEVELS}, self.cfg["eval"]["max_dets"])
        fast = FastAP(ev)
        pf = {lv: perframe_f1(self.t_dets[lv], gt_arrays_from_coco(self.t_gt), self.tau_c,
                              self.cfg["eval"]["perframe_iou"], self.cfg["eval"]["ignore_ioa"]) for lv in ALL_LEVELS}
        f1 = {lv: {i: pf[lv][i]["f1"] for i in self.t_ids} for lv in ALL_LEVELS}
        arrays = CAL.build_arrays(self.t_seq_frames, self.t_cues, self.t_dets, f1, self.tau_c, self.u_empty)
        sp_ev = sp_arrays = None
        if self.sp_rows:
            sp_dets = {lv: {i: self.t_dets[lv][src] for i, src in self.sp_src.items()} for lv in ALL_LEVELS}
            sp_ev = ComposedEvaluator(self.sp_gt, sp_dets, self.cfg["eval"]["max_dets"])
            sp_f1 = {lv: {i: f1[lv][src] for i, src in self.sp_src.items()} for lv in ALL_LEVELS}
            sp_arrays = CAL.build_arrays(self.sp_seq_frames, self.sp_cues, sp_dets, sp_f1, self.tau_c, self.u_empty)
        log.info("test matching done in %.0fs", time.time() - t0)
        summ = self.energy["summary"]
        e_dyn = {lv: summ[f"fixed_{lv}"]["E_dyn_mJ_per_frame"]["median"] if summ[f"fixed_{lv}"]["E_dyn_mJ_per_frame"] else None
                 for lv in ALL_LEVELS}
        sp_ms = self.switch["penalty_ms"]

        def choice_dict(arr, ch, levels=LEVELS):
            return {int(i): levels[int(c)] for i, c in zip(arr.image_ids, ch, strict=True)}

        def metrics_for(choice: dict[int, str], arr, is_adaptive: bool) -> dict:
            m = ev.evaluate(choice, self.t_slices)
            order = [int(i) for i in arr.image_ids]
            lv_arr = np.asarray([0 if choice[i] == "int8" else 1 if choice[i] == "fp16" else 2 for i in order])
            for name, ids in self.t_slices.items():
                if not ids:
                    continue
                lvl = [choice[i] for i in ids]
                m[name]["frac_level"] = {lv: float(np.mean([x == lv for x in lvl])) for lv in ALL_LEVELS}
                eng = [self.engine_ms_all[x] for x in lvl]
                m[name]["lat_mean_ms"] = float(np.mean(eng)) + (self.ctrl_overhead_ms if is_adaptive else 0.0)
                m[name]["energy_dyn_mJ_per_frame"] = (float(np.mean([e_dyn[x] for x in lvl]))
                                                      if all(e_dyn[x] is not None for x in lvl) else None)
                m[name]["mean_f1_05"] = float(np.mean([f1[x][i] for x, i in zip(lvl, ids, strict=True)]))
            same = np.ones(len(order), bool)
            same[arr.seq_starts] = False
            sw = int(np.sum((lv_arr[1:] != lv_arr[:-1]) & same[1:]))
            m["all"]["switch_rate_per_100"] = 100.0 * sw / len(order)
            m["all"]["lat_mean_ms"] += sp_ms * sw / len(order)
            m["all"]["latency_kind"] = "estimated (replay: microbench engine medians + controller overhead + switch penalty)"
            m["all"]["energy_kind"] = "estimated (per-level measured E_dyn/frame x chosen levels)"
            return m

        def stream_metrics(choice_sp: dict[int, str]) -> dict:
            if sp_ev is None:
                return {}
            r = sp_ev.evaluate(choice_sp, self.sp_slices)
            for name, ids in self.sp_slices.items():
                r[name]["frac_level"] = {lv: float(np.mean([choice_sp[i] == lv for i in ids])) for lv in ALL_LEVELS}
            return r

        policies, stream_choice, logs = {}, {}, {}
        for lv in ALL_LEVELS:
            policies[f"fixed_{lv}"] = {i: lv for i in self.t_ids}
            if sp_arrays is not None:
                stream_choice[f"fixed_{lv}"] = {int(i): lv for i in sp_arrays.image_ids}
        # locked main: reference replay (full log) must equal fast replay exactly
        rows_main = replay(p_main, self.t_seq_frames, self.t_cues, {lv: self.t_dets[lv] for lv in LEVELS},
                           self.tau_c, self.u_empty)
        fast_main = CAL.fast_replay(arrays, p_main)
        same_main = [0 if r["level"] == "int8" else 1 for r in rows_main] == fast_main.tolist()
        policies["adaptive_main"] = choice_dict(arrays, fast_main)
        logs["adaptive_main"] = rows_main
        locked = {"adaptive_main": (p_main, h_main)}
        for name in self.locks:
            if name == "main":
                continue
            p, _, h = CAL.load_locked(self.lock_dir / f"locked_bdd_{name}.yaml")
            locked[f"adaptive_{name}"] = (p, h)
            policies[f"adaptive_{name}"] = choice_dict(arrays, CAL.fast_replay(arrays, p))
        if sp_arrays is not None:
            for name, (p, _) in locked.items():
                stream_choice[name] = choice_dict(sp_arrays, CAL.fast_replay(sp_arrays, p))
        # oracle
        f1a = {lv: [f1[lv][i] for i in arrays.image_ids] for lv in LEVELS}
        for kind, fn in (("cheap", lambda: baselines.oracle_cheap(f1a, LEVELS, "fp16")),
                         ("best", lambda: baselines.oracle_best(f1a, LEVELS))):
            policies[f"oracle_{kind}"] = choice_dict(arrays, fn())
            if sp_arrays is not None:
                f1s = {lv: [sp_arrays.f1[lv][k] for k in range(len(sp_arrays.image_ids))] for lv in LEVELS}
                fs = baselines.oracle_cheap(f1s, LEVELS, "fp16") if kind == "cheap" else baselines.oracle_best(f1s, LEVELS)
                stream_choice[f"oracle_{kind}"] = choice_dict(sp_arrays, fs)
        # random baselines (Markov from the locked main policy's TEST log, PLAN §6.1)
        by_seq = {}
        for r in rows_main:
            by_seq.setdefault(self.t_seq_of[r["image_id"]], []).append(0 if r["level"] == "int8" else 1)
        mk = baselines.estimate_markov(by_seq, 2)
        n_seeds = self.F["test"]["random_seeds"]
        groups = {"random_matched": [], "random_iid": []}
        lens = {s: len(v) for s, v in self.t_seq_frames.items()}
        sp_lens = {s: len(v) for s, v in self.sp_seq_frames.items()}
        for k in range(n_seeds):
            seed = self.s2.seed + k
            for kind in groups:
                gen = (lambda L, sd: baselines.markov_random(L, mk["P"], mk["init"], sd)) if kind == "random_matched" \
                    else (lambda L, sd: baselines.iid_random(L, self.cfg["step3"]["iid_p_int8"], sd))
                ch = gen(lens, seed)
                name = f"{kind}_s{seed}"
                policies[name] = {iid: LEVELS[c] for s in self.t_seq_frames for iid, c in zip(self.t_seq_frames[s], ch[s], strict=True)}
                groups[kind].append(name)
                if sp_arrays is not None:
                    chs = gen(sp_lens, seed)
                    stream_choice[name] = {iid: LEVELS[c] for s in self.sp_seq_frames
                                           for iid, c in zip(self.sp_seq_frames[s], chs[s], strict=True)}
        # ---- evaluate + write
        out = {}
        for name, ch in policies.items():
            is_ad = name.startswith("adaptive")
            m = metrics_for(ch, arrays, is_ad)
            if name in stream_choice:
                sm = stream_metrics(stream_choice[name])
                m["scene_change"] = sm.get("scene_change", {})
                m["streams_all"] = sm.get("streams_all", {})
            out[name] = m
        agg = {}
        for g, members in groups.items():
            a = {}
            for sl in list(self.t_slices) + (["scene_change", "streams_all"] if sp_arrays is not None else []):
                vals = [out[mm][sl] for mm in members if out[mm].get(sl)]
                if not vals or "map50_95" not in vals[0]:
                    continue
                a[sl] = {"n_images": vals[0]["n_images"],
                         **{k: float(np.mean([v[k] for v in vals])) for k in ("map50_95", "map50")},
                         **{f"{k}_sd": float(np.std([v[k] for v in vals], ddof=1)) if len(vals) > 1 else 0.0
                            for k in ("map50_95", "map50")},
                         "frac_level": {lv: float(np.mean([v["frac_level"][lv] for v in vals])) for lv in ALL_LEVELS}}
                for k in ("lat_mean_ms", "energy_dyn_mJ_per_frame", "switch_rate_per_100"):
                    xs = [v.get(k) for v in vals]
                    if xs and all(x is not None for x in xs):
                        a[sl][k] = float(np.mean(xs))
            a["all"]["seeds"] = [self.s2.seed + k for k in range(n_seeds)]
            if g == "random_matched":
                a["all"]["markov"] = {"P": mk["P"].tolist(), "init": mk["init"].tolist(), "from": "adaptive_main test log"}
            agg[g] = a
        # bootstrap CIs (paired, by sequence)
        n_b = self.F["test"]["smoke_bootstrap_resamples"] if self.smoke else self.F["test"]["bootstrap_resamples"]
        boot_pol = {k: policies[k] for k in ("adaptive_main", "fixed_fp16", "fixed_int8", "fixed_fp32",
                                              "adaptive_ksweep1", "oracle_cheap") if k in policies}
        for g in groups.values():
            for mm in g:
                boot_pol[mm] = policies[mm]
        pairs = [("adaptive_main", "fixed_fp16"), ("adaptive_main", "fixed_int8"), ("adaptive_main", "random_matched"),
                 ("adaptive_main", "random_iid"), ("oracle_cheap", "fixed_fp16")]
        if "adaptive_ksweep1" in boot_pol:
            pairs.append(("adaptive_main", "adaptive_ksweep1"))
        t1 = time.time()
        boot = cluster_bootstrap(fast, {s: v for s, v in self.t_seq_frames.items()}, boot_pol, pairs, n_b,
                                 self.s2.seed, random_groups=groups)
        log.info("bootstrap %d resamples in %.0fs", n_b, time.time() - t1)
        # write experiments
        for name, m in list(out.items()) + list(agg.items()):
            if name.startswith(("random_matched_s", "random_iid_s")):
                continue  # per-seed values are inside the aggregate
            lk = locked.get(name)
            extra = {"locked_config_sha256": lk[1] if lk else None, "locked_config": f"results/locked/locked_bdd_{name[9:]}.yaml" if lk else None}
            w = self._writer(f"test_{name}", name, "test", extra=extra)
            if name in boot["per_policy"]:
                for k in ("map50_95", "map50"):
                    m["all"][f"{k}_ci"] = boot["per_policy"][name][k]["ci95"]
            w.write_json("metrics.json", _jsonable(m))
            if name == "adaptive_main":
                row_of = {int(r["image_id"]): r for r in self.t_rows}
                pfr = [{"seq": self.t_seq_of[r["image_id"]],
                        "frame_idx": row_of[r["image_id"]]["frame_idx"], "image_id": r["image_id"], "level": r["level"],
                        "switched": r["switched"], "score_S": round(r["score_S"], 6), "score_hat": round(r["score_hat"], 6),
                        "m": r["m"], "c": r["c"], "n": r["n"], "u": r["u"], "s": r["s"], "d": r["d"],
                        "proposal": r["proposal"], "override": r["override"],
                        "latency_engine_est_ms": self.engine_ms_all[r["level"]], "energy_est_mJ": e_dyn[r["level"]],
                        "n_dets_025": int((self.t_dets[r["level"]][r["image_id"]][:, 4] >= self.tau_c).sum()),
                        "f1_05": round(f1[r["level"]][r["image_id"]], 6),
                        **{c: row_of[r["image_id"]][c] for c in SLICE_COLS.values()}} for r in rows_main]
                w.write_csv("per_frame.csv", pfr, list(pfr[0].keys()))
            w.register(policy=name, mode="replay", locked_sha=lk[1] if lk else "", replace=True)
        w = self._writer("test_bootstrap", "bootstrap", "test", extra={"locked_config_sha256": h_main})
        w.write_json("bootstrap.json", boot)
        w.register(policy="bootstrap", mode="replay", locked_sha=h_main, replace=True)
        hyp = self._hypotheses(out, agg, boot)
        w = self._writer("test_hypotheses", "hypotheses", "test", extra={"locked_config_sha256": h_main})
        w.write_json("hypotheses.json", hyp)
        w.register(policy="hypotheses", mode="replay", locked_sha=h_main, replace=True)
        checks = [{"check": "locked main: reference replay == fast replay (identical decisions)", "pass": same_main},
                  {"check": "every policy evaluated on all test frames",
                   "pass": all(out[n]["all"]["n_images"] == len(self.t_ids) for n in out)},
                  {"check": f"bootstrap ran ({n_b} resamples)", "pass": boot["n_resamples"] == n_b}]
        self.gate("6_test_eval", checks)
        return {"main": out["adaptive_main"]["all"], "hypotheses": hyp, "aggregates": agg}

    def _hypotheses(self, out, agg, boot) -> dict:
        """Pre-registered H1-H4, computed by adapres.hypotheses (revision 2: E_total primary,
        H3 on the K-sweep as a sensitivity analysis, H4 engine-latency-only saving share).
        The original r04 run used revision 1 (superseded, see DEVIATIONS.md)."""
        ks = {int(k[len("adaptive_ksweep"):]): v["all"] for k, v in out.items() if k.startswith("adaptive_ksweep")}
        return HYP.compute(out, agg, boot, self.energy["summary"], ks, self.engine_ms_all, self.F["delta_pts"],
                           int(self.locks["main"]["params"]["K_down"]))
