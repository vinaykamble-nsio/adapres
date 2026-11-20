"""Step 3 / notebook 03: complexity cues, controller with hysteresis, matched-random,
i.i.d.-random and oracle baselines, replayed on notebook 02's detection cache.
CALIB SPLIT ONLY (development); calibration + locking is notebook 04b, test comes later.

Stages: prepare -> locate_cache -> verify_cache -> features -> fit -> policies -> sanity
Integrity checks built in:
  * the attached cache must reproduce the committed per_frame.csv detection counts exactly;
  * fixed FP16/INT8 and oracle-cheap re-evaluated through ComposedEvaluator must equal the
    committed notebook 02 metrics (every slice) to 1e-9.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .. import baselines
from ..config import config_sha256
from ..controller import params_from_config
from ..eval.map_eval import ComposedEvaluator
from ..eval.perframe import perframe_f1
from ..features import FrameFeaturizer
from ..replay import fit_norm, open_loop_scores, replay
from ..results import ExperimentWriter
from .eval_fixed import Step2

log = logging.getLogger("adapres.controller_dev")
LEVELS = ("int8", "fp16")


class Step3Error(RuntimeError):
    pass


def find_cache_root(input_root: str | Path, model: str, max_depth: int = 7,
                    skip_names: tuple[str, ...] = ("track_images_val", "images", "labels", "box_track_labels_trainval")
                    ) -> Path:
    """Locate <...>/cache/<model>/ (with fp16/ and int8/ .npz) under the attached inputs,
    without descending into large image/label folders."""
    root = Path(input_root)
    base = len(root.parts)
    hits = []
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if len(p.parts) - base >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in skip_names]
        if p.name == model and p.parent.name == "cache" and (p / "fp16").is_dir() and (p / "int8").is_dir():
            hits.append(p)
            dirnames[:] = []
    if not hits:
        raise Step3Error(f"no cache/{model}/{{fp16,int8}} under {root}: attach notebook 02's cache "
                         "(the private 'adapres-cache-02' dataset, or notebook 02's output)")
    if len(hits) > 1:
        raise Step3Error(f"several caches found, attach only one notebook 02 cache source: {hits}")
    return hits[0]


def _stats(a) -> dict:
    a = np.asarray(a, float)
    return {"median": float(np.median(a)), "mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
            "n": int(a.size)} if a.size else {"n": 0}


class Step3:
    def __init__(self, cfg: dict, smoke: bool, repo_dir: str | Path, search_root: str | Path | None = None,
                 image_reader: Callable | None = None, featurizer_factory: Callable | None = None):
        cfg = json.loads(json.dumps(cfg))
        cfg["step2"]["smoke_n_sequences"] = cfg["step3"]["smoke_n_sequences"]
        self.cfg, self.smoke, self.repo = cfg, smoke, Path(repo_dir)
        self.s2 = Step2(cfg, smoke, repo_dir, search_root=search_root)
        self.image_reader = image_reader or Step2._cv2_read
        f = cfg["features"]
        self.featurizer_factory = featurizer_factory or (lambda: FrameFeaturizer(f["width"], f["canny_low"],
                                                                                 f["canny_high"], f["hist_bins"]))
        self.work = self.s2.work.parent / "work03"
        self.results_root = self.s2.results_root
        self.c = cfg["controller"]
        self.tau_c = cfg["model"]["conf_cue"]

    # ------------------------------------------------------------------ helpers
    def _meta(self, exp_id: str, policy: str, extra: dict | None = None, dataset: str = "bdd100k_mot") -> dict:
        m = self.s2._meta(exp_id, policy, {"mode": "replay", "notes": "calib split only (Step 3 development); "
                                           "uses notebook 02 cache; not a locked config", **(extra or {})},
                          dataset=dataset)
        m["config_sha256"] = config_sha256(self.cfg)
        m["config"] = self.cfg
        m["cache_commit"] = getattr(self, "cache_commit", None)
        return m

    def _committed_ids(self) -> set[str]:
        idx = self.repo / "results" / "index.csv"
        return {r["exp_id"] for r in csv.DictReader(open(idx, newline=""))} if idx.exists() else set()

    def _write(self, exp_id: str, policy: str, metrics: dict, per_frame: list[dict] | None = None,
               extra_meta: dict | None = None, extra_files: dict | None = None) -> None:
        if exp_id in self._committed_ids():
            raise Step3Error(f"exp_id {exp_id!r} already exists in the committed results/index.csv "
                             "(written by another notebook); refusing to overwrite it")
        w = ExperimentWriter(self.results_root, self._meta(exp_id, policy, extra_meta))
        w.write_json("metrics.json", metrics)
        if per_frame:
            w.write_csv("per_frame.csv", per_frame, list(per_frame[0].keys()))
        for name, obj in (extra_files or {}).items():
            w.write_json(name, obj)
        w.register(policy=policy, mode="replay", replace=True)

    # ------------------------------------------------------------------ stages
    def prepare(self) -> dict:
        info = self.s2.prepare()
        self.ids = self.s2.image_ids
        self.seq_frames = {}
        for iid in self.ids:
            self.seq_frames.setdefault(self.s2.seq_of[iid], []).append(iid)
        self.row_of = {int(r["image_id"]): r for r in self.s2.rows}
        return info

    def locate_cache(self, cache_root: str | Path | None = None) -> Path:
        root = Path(cache_root) if cache_root else find_cache_root(self.cfg["paths"]["input_root"], self.s2.model_name)
        self.s2.cache_dir = root
        log.info("cache root: %s", root)
        return root

    def verify_cache(self) -> dict:
        """Cache must reproduce the committed per_frame.csv (n_dets@tau_c per frame) exactly."""
        res = self.repo / "results" / "bdd100k_mot"
        self.dets = {}
        report = {}
        for lv in LEVELS:
            dets, _ = self.s2.load_cache(lv)
            self.dets[lv] = dets
            committed = {int(r["image_id"]): int(r["n_dets_025"])
                         for r in csv.DictReader(open(res / f"calib_fixed_{lv}" / "per_frame.csv", newline=""))}
            meta = json.loads((res / f"calib_fixed_{lv}" / "meta.json").read_text())
            self.cache_commit = meta["git_commit"]
            mism = [i for i in self.ids
                    if int((dets[i][:, 4] >= self.tau_c).sum()) != committed.get(i, -1)]
            report[lv] = {"frames": len(self.ids), "mismatches": len(mism)}
            if mism:
                raise Step3Error(f"attached {lv} cache does not match committed results on {len(mism)} frames "
                                 f"(e.g. {mism[:3]}): wrong notebook output attached?")
        report["cache_commit"] = self.cache_commit
        return report

    def features(self, workers: int = 1) -> dict:
        """Image-only cues per frame, checkpointed per sequence; timing recorded (CPU)."""
        out_dir = self.work / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.cues: dict[int, dict] = {}
        t_feat, t_read = [], []
        for seq, ids in sorted(self.seq_frames.items()):
            f = out_dir / f"{seq}.npz"
            if f.exists():
                z = np.load(f)
                if z["ids"].tolist() == ids:
                    for i, m, c, d in zip(z["ids"], z["m"], z["c"], z["d"], strict=True):
                        self.cues[int(i)] = {"m": float(m), "c": float(c), "d": float(d)}
                    t_feat += z["t_feat"].tolist()
                    t_read += z["t_read"].tolist()
                    continue
            fz = self.featurizer_factory()
            vals, tf, tr = [], [], []
            for iid in ids:
                t0 = time.perf_counter()
                img = self.image_reader(self.s2.paths[iid])
                t1 = time.perf_counter()
                v = fz(img)
                t2 = time.perf_counter()
                vals.append(v)
                tr.append((t1 - t0) * 1000)
                tf.append((t2 - t1) * 1000)
            for iid, v in zip(ids, vals, strict=True):
                self.cues[iid] = v
            tmp = f.with_name(f.stem + ".tmp.npz")
            np.savez(tmp, ids=np.asarray(ids), m=[v["m"] for v in vals], c=[v["c"] for v in vals],
                     d=[v["d"] for v in vals], t_feat=tf, t_read=tr)
            tmp.replace(f)
            t_feat += tf
            t_read += tr
        self.feature_timing = {"featurize_ms": _stats(t_feat), "image_read_ms": _stats(t_read),
                               "note": "CPU, Kaggle notebook; features only (decode reported separately)"}
        return self.feature_timing

    def fit(self) -> dict:
        """Normalisation percentiles and dev theta, both from calib (this split)."""
        from ..features import detection_cues

        n_vals, u_vals = [], []
        for ids in self.seq_frames.values():
            for t in range(len(ids)):
                prev = self.dets["fp16"][ids[t - 1]] if t > 0 else None
                n, u = detection_cues(prev, self.tau_c, self.c["u_empty"])
                n_vals.append(n)
                u_vals.append(u)
        vals = {"m": [self.cues[i]["m"] for i in self.ids], "c": [self.cues[i]["c"] for i in self.ids],
                "n": n_vals, "u": u_vals}
        self.norm = fit_norm(vals, self.c["norm_percentiles"])
        p0 = params_from_config(self.c, theta=0.0, norm=self.norm)
        s_hat = open_loop_scores(p0, self.seq_frames, self.cues, self.dets["fp16"], self.tau_c, self.c["u_empty"])
        self.theta = float(np.quantile(s_hat, self.c["theta_quantile"]))
        d = np.asarray([self.cues[i]["d"] for i in self.ids])
        self.fit_info = {"norm": {k: list(v) for k, v in self.norm.items()}, "theta": self.theta,
                         "theta_quantile": self.c["theta_quantile"], "open_loop_S_hat": _stats(s_hat),
                         "frac_frames_scene_cue": float(np.mean(d >= self.c["theta_sc"])),
                         "cue_percentiles": {k: {q: float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95)}
                                             for k, v in {**vals, "d": d}.items()},
                         "fit_on": "calib (this split)"}
        return self.fit_info

    def _policy_metrics(self, ev: ComposedEvaluator, choice: dict[int, str]) -> dict:
        m = ev.evaluate(choice, self.s2.slices)
        lv_idx = {lv: i for i, lv in enumerate(LEVELS)}
        for name, ids in self.s2.slices.items():
            ch = [lv_idx[choice[i]] for i in ids]
            m[name]["frac_level"] = baselines.level_fractions(ch, LEVELS) if ids else {}
            m[name]["mean_f1_05"] = float(np.mean([self.f1[choice[i]][i] for i in ids])) if ids else None
            m[name]["est_engine_ms_mean"] = (float(np.mean([self.engine_ms[choice[i]] for i in ids]))
                                             if ids and self.engine_ms else None)
        seq_order = [self.s2.seq_of[i] for i in self.ids]
        m["all"]["switch_rate_per_100"] = baselines.switch_rate_per_100([lv_idx[choice[i]] for i in self.ids], seq_order)
        m["all"]["est_note"] = ("est_engine_ms_mean = mean of microbench median engine latency of the chosen level; "
                                "excludes controller overhead and switch penalty (Step 4 measures those)")
        return m

    def policies(self) -> dict:
        t0 = time.time()
        self.ev = ComposedEvaluator(self.s2.gt, {lv: self.dets[lv] for lv in LEVELS}, self.cfg["eval"]["max_dets"])
        log.info("per-level matching done in %.0fs", time.time() - t0)
        pf = {lv: perframe_f1(self.dets[lv], self.s2.gt_index, self.tau_c, self.cfg["eval"]["perframe_iou"],
                              self.cfg["eval"]["ignore_ioa"]) for lv in LEVELS}
        self.f1 = {lv: {i: pf[lv][i]["f1"] for i in self.ids} for lv in LEVELS}
        mb = self.repo / "results" / "microbench" / f"microbench_{self.s2.model_name}" / "microbench.json"
        self.engine_ms = ({lv: json.loads(mb.read_text())["per_backend"][lv]["infer_ms"]["median"] for lv in LEVELS}
                          if mb.exists() else {})
        out = {}
        # fixed + oracle through the composed evaluator (cross-checked against notebook 02 in sanity())
        for lv in LEVELS:
            out[f"fixed_{lv}"] = self._policy_metrics(self.ev, {i: lv for i in self.ids})
        f1arr = {lv: [self.f1[lv][i] for i in self.ids] for lv in LEVELS}
        oc = baselines.oracle_cheap(f1arr, LEVELS, "fp16")
        out["oracle_cheap"] = self._policy_metrics(self.ev, {i: LEVELS[c] for i, c in zip(self.ids, oc, strict=True)})
        # adaptive (dev defaults) + K sweep at the same theta
        self.adaptive_logs = {}
        Ks = sorted(set([self.c["K"]] + list(self.cfg["step3"]["K_sweep"])))
        for K in Ks:
            c = {**self.c, "K": K, "K_up": None, "K_down": None}
            p = params_from_config(c, self.theta, self.norm)
            rows = replay(p, self.seq_frames, self.cues, self.dets, self.tau_c, self.c["u_empty"])
            choice = {r["image_id"]: r["level"] for r in rows}
            name = f"adaptive_K{K}"
            out[name] = self._policy_metrics(self.ev, choice)
            out[name]["all"]["controller_params"] = p.to_dict()
            out[name]["all"]["ctrl_step_ms"] = _stats([r["latency_ctrl_step_ms"] for r in rows])
            self.adaptive_logs[name] = rows
        main = f"adaptive_K{self.c['K']}"
        # matched random (Markov from the main adaptive log) and i.i.d. random, 10 seeds each
        by_seq = {}
        for r in self.adaptive_logs[main]:
            by_seq.setdefault(r["seq"], []).append(LEVELS.index(r["level"]))
        mk = baselines.estimate_markov(by_seq, len(LEVELS))
        lens = {s: len(v) for s, v in self.seq_frames.items()}
        for kind in ("random_matched", "random_iid"):
            seeds = []
            for k in range(self.cfg["step3"]["random_seeds"]):
                seed = self.s2.seed + k
                ch = (baselines.markov_random(lens, mk["P"], mk["init"], seed) if kind == "random_matched"
                      else baselines.iid_random(lens, self.cfg["step3"]["iid_p_int8"], seed))
                choice = {iid: LEVELS[c] for s in self.seq_frames for iid, c in zip(self.seq_frames[s], ch[s], strict=True)}
                seeds.append(self._policy_metrics(self.ev, choice))
            agg = {}
            for sl in self.s2.slices:
                vals = {k: [s[sl][k] for s in seeds] for k in ("map50_95", "map50")}
                agg[sl] = {"n_images": seeds[0][sl]["n_images"],
                           **{k: float(np.mean(v)) for k, v in vals.items()},
                           **{f"{k}_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in vals.items()},
                           "frac_level": {lv: float(np.mean([s[sl]["frac_level"].get(lv, 0) for s in seeds])) for lv in LEVELS},
                           "mean_f1_05": float(np.mean([s[sl]["mean_f1_05"] for s in seeds])) if seeds[0][sl]["mean_f1_05"] is not None else None}
            agg["all"]["switch_rate_per_100"] = float(np.mean([s["all"]["switch_rate_per_100"] for s in seeds]))
            agg["all"]["switch_rate_per_100_sd"] = float(np.std([s["all"]["switch_rate_per_100"] for s in seeds], ddof=1))
            agg["all"]["n_seeds"] = len(seeds)
            agg["all"]["seeds"] = [self.s2.seed + k for k in range(len(seeds))]
            agg["all"]["per_seed_map50_95"] = [s["all"]["map50_95"] for s in seeds]
            if kind == "random_matched":
                agg["all"]["markov"] = {"P": mk["P"].tolist(), "init": mk["init"].tolist(), "from": main}
            out[kind] = agg
        self.results = out
        self._write_all()
        return out

    def _write_all(self) -> None:
        common = {"theta": self.theta, "norm": {k: list(v) for k, v in self.norm.items()}}
        feats = [{"seq": self.s2.seq_of[i], "frame_idx": self.row_of[i]["frame_idx"], "image_id": i,
                  **{k: round(self.cues[i][k], 6) for k in ("m", "c", "d")}} for i in self.ids]
        w = ExperimentWriter(self.results_root, self._meta(f"calib_features_{self.s2.model_name}", "features", common))
        w.write_csv("features.csv", feats, list(feats[0].keys()))
        w.write_json("fit.json", self.fit_info)
        w.write_json("feature_timing.json", self.feature_timing)
        w.register(policy="features", mode="replay", replace=True)
        for name, m in self.results.items():
            per_frame = None
            if name.startswith("adaptive_"):
                per_frame = []
                for r in self.adaptive_logs[name]:
                    row = self.row_of[r["image_id"]]
                    per_frame.append({"seq": r["seq"], "frame_idx": row["frame_idx"], "image_id": r["image_id"],
                                      "level": r["level"], "switched": r["switched"], "proposal": r["proposal"],
                                      "override": r["override"], "score_S": round(r["score_S"], 6),
                                      "score_hat": round(r["score_hat"], 6), "m": r["m"], "c": r["c"], "n": r["n"],
                                      "u": r["u"], "d": r["d"], "s": r["s"],
                                      "latency_ctrl_step_ms": round(r["latency_ctrl_step_ms"], 5),
                                      "f1_05": round(self.f1[r["level"]][r["image_id"]], 6),
                                      "slice_night": row["slice_night"], "slice_crowded": row["slice_crowded"],
                                      "slice_scene_change_natural": row["slice_scene_change_natural"],
                                      "slice_easy": row["slice_easy"]})
            # fixed/oracle are re-evaluations of notebook 02 policies: distinct ids, never notebook 02's
            exp_id = f"calib_{name}_composed_dev" if name.startswith(("fixed_", "oracle_")) else f"calib_{name}_dev"
            self._write(exp_id, name, m, per_frame, common)

    def sanity(self) -> dict:
        res = self.repo / "results" / "bdd100k_mot"
        checks = []

        def add(name, ok, detail):
            checks.append({"check": name, "pass": bool(ok), "detail": detail})

        if not self.smoke:  # committed notebook 02 metrics cover all 20 calib clips
            for name, exp in (("fixed_fp16", "calib_fixed_fp16"), ("fixed_int8", "calib_fixed_int8"),
                              ("oracle_cheap", "calib_oracle_cheap_dev")):
                ref = json.loads((res / exp / "metrics.json").read_text())
                diffs = {sl: abs(self.results[name][sl]["map50_95"] - ref[sl]["map50_95"]) +
                         abs(self.results[name][sl]["map50"] - ref[sl]["map50"]) for sl in self.s2.slices}
                add(f"{name}: composed eval == notebook 02 (all slices)", max(diffs.values()) < 1e-9,
                    {k: float(v) for k, v in diffs.items()})
        main = f"adaptive_K{self.c['K']}"
        a, r = self.results[main]["all"], self.results["random_matched"]["all"]
        add("random_matched: INT8 fraction matches adaptive (|diff| <= 0.05)",
            abs(a["frac_level"]["int8"] - r["frac_level"]["int8"]) <= 0.05,
            {"adaptive": a["frac_level"]["int8"], "random_matched": r["frac_level"]["int8"]})
        add("random_matched: switch rate matches adaptive (within 25% relative or 1/100)",
            abs(a["switch_rate_per_100"] - r["switch_rate_per_100"]) <= max(1.0, 0.25 * a["switch_rate_per_100"]),
            {"adaptive": a["switch_rate_per_100"], "random_matched": r["switch_rate_per_100"]})
        sw = {k: self.results[k]["all"]["switch_rate_per_100"] for k in self.results if k.startswith("adaptive_K")}
        ks = sorted(int(k[len("adaptive_K"):]) for k in sw)
        nonincreasing = all(sw[f"adaptive_K{a}"] >= sw[f"adaptive_K{b}"] - 1e-12 for a, b in zip(ks, ks[1:], strict=False))
        out = {"checks": checks, "all_pass": all(c["pass"] for c in checks),
               "informational": {"switch_rate_by_K": sw, "switch_rate_nonincreasing_in_K": bool(nonincreasing)}}
        w = ExperimentWriter(self.results_root, self._meta(f"sanity_03_{self.s2.model_name}", "sanity",
                                                           dataset="sanity"))
        w.write_json("sanity.json", out)
        w.register(policy="sanity", mode="check", replace=True)
        return out
