"""Notebook 05: controller v2 follow-up (configs/prereg_addendum.yaml), one gated run.

  1 engines + calib cache            (FullRun.stage1, prefix r05_)
  2 calib cues + dev                 (FullRun.stage2)
  3 v2 fit + lock (calib only)       addendum hash verified; GroupKFold model selection; cross-fitted
                                     closed-loop budget thresholds; K per budget by calib mAP; v1@rho and
                                     lagged-oracle thresholds; every lock hashed.  GATE: locks verify
  4 test cache                       (FullRun.stage4; refuses without the verified v2@0.50 lock)
  5 energy on mixed switching        6 seeded test clips: FP16, INT8, v2@0.50 pipelined + sequential,
                                     forced alternation; live-vs-replay decision check. GATE: all measured
  6 test evaluation                  per budget: v2, matched random (10 seeds), v1, oracle, lagged oracle;
                                     slices + spliced scene changes; paired cluster bootstrap; P1-P5.
                                     GATE: replay == step-by-step controller; all evaluated
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

from .. import baselines
from .. import calibrate as CAL
from .. import risk as R
from ..controller import ControllerParams
from ..energy import protocol as PROTO
from ..energy.nvml_meter import NvmlMeter
from ..eval.fast_ap import FastAP, cluster_bootstrap
from ..eval.map_eval import ComposedEvaluator
from ..eval.perframe import gt_arrays_from_coco, perframe_f1
from ..live import AlternatingStepper, FixedStepper, V2Stepper, run_live_sequences  # noqa: F401
from .full import ALL_LEVELS, FullRun, GateError, _jsonable

log = logging.getLogger("adapres.v2")
LV = R.LEVELS


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_lock(path: Path, body: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(json.loads(json.dumps(body, default=float)), sort_keys=True)
    path.write_text(text, encoding="utf-8")
    h = _sha(text.encode("utf-8"))
    path.with_suffix(path.suffix + ".sha256").write_text(h + "\n")
    return h


def read_lock(path: Path) -> tuple[dict, str]:
    hp = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not hp.exists():
        raise CAL.LockError(f"locked config or hash missing: {path}")
    text = path.read_text(encoding="utf-8")
    h = _sha(text.encode("utf-8"))
    if h != hp.read_text().strip():
        raise CAL.LockError(f"hash mismatch for {path}: file was modified after locking")
    return yaml.safe_load(text), h


class V2Run(FullRun):
    def __init__(self, cfg: dict, smoke: bool, repo_dir, **kw):
        cfg = json.loads(json.dumps(cfg))
        cfg["full"]["exp_prefix"] = cfg["v2"]["exp_prefix"]
        super().__init__(cfg, smoke, repo_dir, **kw)
        self.V = cfg["v2"]
        ap = self.repo / self.V["addendum"]
        hp = ap.with_suffix(ap.suffix + ".sha256")
        if not hp.exists():
            raise GateError("prereg addendum hash file missing: the v2 design must be frozen before this run")
        self.addendum_sha = _sha(ap.read_bytes())
        if self.addendum_sha != hp.read_text().strip():
            raise GateError("configs/prereg_addendum.yaml changed after it was frozen (hash mismatch)")
        self.A = yaml.safe_load(ap.read_text(encoding="utf-8"))
        self.budgets = list(self.A["decision"]["budgets_rho"])
        self.Kgrid = self.V["smoke"]["K_grid"] if smoke else list(self.A["decision"]["persistence_K_grid"])
        self.grid = self.V["smoke"]["grid"] if smoke else dict(self.A["risk_estimator"]["hyperparameter_grid"])
        self.n_folds = self.V["smoke"]["n_folds"] if smoke else 5
        self.theta_sc, self.H = 0.5, 3
        self.n_random = self.V["smoke"]["random_seeds"] if smoke else self.V["random_seeds"]

    # the test split is unlocked by the v2@0.50 lock (plus the frozen addendum)
    def _require_lock(self):
        body, h = read_lock(self.lock_dir / "locked_v2_rho0.5.yaml")
        if body.get("addendum_sha256") != self.addendum_sha:
            raise CAL.LockError("v2 lock was made under a different addendum")
        return None, body, h

    def _model(self):
        body, _ = read_lock(self.lock_dir / "locked_v2_rho0.5.yaml")
        mp = self.lock_dir / "v2_model.pkl"
        if _sha(mp.read_bytes()) != body["model_sha256"]:
            raise CAL.LockError("v2_model.pkl does not match the locked model hash")
        return pickle.loads(mp.read_bytes())

    def _area(self, rows) -> dict[int, float]:
        return {int(r["image_id"]): float(r["W"]) * float(r["H"]) for r in rows}

    # ------------------------------------------------------------------ 3. fit + lock on calib
    def stage3_v2(self) -> dict:
        st = self.st
        area = self._area(self.s2.rows)
        sd = R.build_seqdata(st.seq_frames, st.cues, st.dets, area, st.f1, self.tau_c)
        X, y, grp = R.training_rows(sd)
        sel = R.select_and_fit(X, y, grp, self.grid, self.s2.seed, self.n_folds)
        model_bytes = pickle.dumps(sel["model"])
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        (self.lock_dir / "v2_model.pkl").write_bytes(model_bytes)
        model_sha = _sha(model_bytes)
        cf = {si: sel["fold_models"][sel["fold_of_group"][si]] for si in range(len(sd.seq_ids))}
        rt = R.risk_table(None, sd, cf)
        dseq = [np.array([st.cues[i]["d"] for i in ids]) for ids in sd.image_ids]
        ids_all = [i for ids in sd.image_ids for i in ids]
        fp16_map = st.results["fixed_fp16"]["all"]["map50_95"]
        # v1 (r04 hand-weighted) re-thresholded per budget on calib
        v1p, _, v1h = CAL.load_locked(self.repo / self.V["v1_lock"])
        arrays = CAL.build_arrays(st.seq_frames, st.cues, st.dets, st.f1, self.tau_c, self.u_empty)
        g_seq = [np.array([sd.g[i] for i in ids]) for ids in sd.image_ids]
        locks, per_rho = {}, {}
        common = {"addendum_sha256": self.addendum_sha, "model_sha256": model_sha, "fit_split": "calib",
                  "model_params": sel["params"], "cv_best_spearman": sel["best_spearman"],
                  "engine_sha256": {k: v.get("sha256") for k, v in self.s2.engine_info.items()},
                  "git_commit": self.s2.env["git"]["commit"], "theta_sc": self.theta_sc, "H": self.H,
                  "tau_c": self.tau_c}
        for rho in self.budgets:
            cands = []
            for K in self.Kgrid:
                tau, frac = R.tau_for_budget(rt, dseq, K, rho, self.theta_sc, self.H)
                ch = np.concatenate(R.fast_replay_v2(rt, dseq, R.V2Params(tau, K, self.theta_sc, self.H)))
                m = st.ev.evaluate({i: LV[c] for i, c in zip(ids_all, ch, strict=True)}, {"all": ids_all})["all"]["map50_95"]
                cands.append({"K": K, "tau": tau, "calib_frac_int8": frac, "calib_map50_95": m})
            best = sorted(cands, key=lambda c: (-c["calib_map50_95"], -c["K"]))[0]
            locks[f"v2_rho{rho}"] = write_lock(self.lock_dir / f"locked_v2_rho{rho}.yaml",
                                               {**common, "rho": rho, "tau": best["tau"], "K": best["K"],
                                                "calib": best, "candidates": cands})
            # v1@rho: bisection on theta with the r04 weights/alpha/theta_sc/K
            lo, hi, bestv = -1.0, 2.0, None
            for _ in range(40):
                mid = (lo + hi) / 2
                p = ControllerParams(**{**{k: getattr(v1p, k) for k in v1p.__dataclass_fields__}, "theta": mid})
                f = float(np.mean(CAL.fast_replay(arrays, p) == 0))
                if bestv is None or abs(f - rho) < bestv[1]:
                    bestv = (mid, abs(f - rho), f)
                lo, hi = (mid, hi) if f < rho else (lo, mid)
            locks[f"v1_rho{rho}"] = write_lock(self.lock_dir / f"locked_v1_rho{rho}.yaml",
                                               {"addendum_sha256": self.addendum_sha, "rho": rho, "theta": bestv[0],
                                                "calib_frac_int8": bestv[2], "v1_source_lock_sha256": v1h,
                                                "controller": CAL.params_dict(v1p)})
            lt = R.lagged_tau_for_budget(g_seq, rho)
            locks[f"lagged_rho{rho}"] = write_lock(self.lock_dir / f"locked_lagged_rho{rho}.yaml",
                                                   {"addendum_sha256": self.addendum_sha, "rho": rho, "tau": lt,
                                                    "note": "REFERENCE: uses GT one frame late; not deployable"})
            per_rho[rho] = {"v2": best, "v2_candidates": cands, "v1_theta": bestv[0], "v1_calib_frac": bestv[2],
                            "lagged_tau": lt}
        w = self._writer("v2_calibration", "v2_calibration", "calib",
                         extra={"addendum_sha256": self.addendum_sha, "locks": locks})
        w.write_json("v2_calibration.json", _jsonable({"cv": sel["cv"], "params": sel["params"],
                                                       "best_spearman": sel["best_spearman"], "per_rho": per_rho,
                                                       "fp16_calib_map50_95": fp16_map, "n_train_rows": len(y),
                                                       "features": list(R.FEATURES), "locks": locks}))
        w.register(policy="v2_calibration", mode="calibrate", replace=True)
        self.v2_locks = locks
        checks = [{"check": "prereg addendum hash verified", "pass": True, "detail": self.addendum_sha[:16]}]
        for name in locks:
            try:
                read_lock(self.lock_dir / f"locked_{name}.yaml")
                checks.append({"check": f"lock {name} written and hash-verified", "pass": True})
            except CAL.LockError as e:
                checks.append({"check": f"lock {name}", "pass": False, "detail": str(e)})
        soft = [{"check": "held-out Spearman(pred, g) > 0 on calib", "pass": sel["best_spearman"] > 0,
                 "detail": sel["best_spearman"]}]
        soft += [{"check": f"calib INT8 fraction within 0.01 of rho={rho} (addendum)", "pass": abs(v["v2"]["calib_frac_int8"] - rho) <= 0.01,
                  "detail": v["v2"]["calib_frac_int8"]} for rho, v in per_rho.items()]
        self.gate("3_v2_fit_lock", checks, soft)
        return {"params": sel["params"], "spearman": sel["best_spearman"], "per_rho": per_rho}

    # ------------------------------------------------------------------ 5. energy on mixed switching
    def stage5_v2_energy(self) -> dict:
        _, body, h = self._require_lock()
        model = self._model()
        p = R.V2Params(body["tau"], body["K"], body["theta_sc"], body["H"])
        n_clips = self.V["smoke"]["energy_clips"] if self.smoke else 6   # addendum energy_protocol: 6 clips
        n_fr = self.V["smoke"]["energy_frames_per_clip"] if self.smoke else 150
        rng = np.random.default_rng(self.s2.seed)
        seqs = sorted(self.t_seq_frames)
        pick = sorted(seqs[i] for i in rng.choice(len(seqs), min(n_clips, len(seqs)), replace=False))
        frames, starts, fids = [], [], []
        for s in pick:
            starts.append(len(frames))
            ids = self.t_seq_frames[s][:n_fr]
            fids += ids
            frames += [self.read(self.t_paths[i]) for i in ids]
        area = float(np.median([float(r["W"]) * float(r["H"]) for r in self.t_rows]))
        meter = self.meter_factory() if self.meter_factory else NvmlMeter()
        steppers = {"fixed_fp16": FixedStepper(self.backends["fp16"], frames),
                    "fixed_int8": FixedStepper(self.backends["int8"], frames),
                    "adaptive_v2_pipelined": V2Stepper(self.backends, model, p, self.ff, frames, starts, area, self.tau_c, True),
                    "adaptive_v2_sequential": V2Stepper(self.backends, model, p, self.ff, frames, starts, area, self.tau_c, False),
                    "forced_alternation": AlternatingStepper(self.backends, frames)}
        T = self.timing_cfg
        prot = PROTO.run_protocol(steppers, meter, T["warmup_s"], T["settle_s"], T["idle_s"], T["measure_s"],
                                  T["repeats"], self.s2.seed, log=log.info)
        for s_ in steppers.values():
            s_.close()
        # live (sequential, one pass) vs replay on the cached test detections of the same frames
        live = []
        stp = V2Stepper(self.backends, model, p, self.ff, frames, starts, area, self.tau_c, False)
        for _ in range(len(frames)):
            live.append(stp.step()["level"])
        stp.close()
        sub = {s: self.t_seq_frames[s][:n_fr] for s in pick}
        sd = R.build_seqdata(sub, self.t_cues, self.t_dets, self._area(self.t_rows), None, self.tau_c)
        rt = R.risk_table(model, sd)
        rep = np.concatenate(R.fast_replay_v2(rt, [np.array([self.t_cues[i]["d"] for i in ids]) for ids in sd.image_ids], p))
        agree = float(np.mean([LV[c] == lv for c, lv in zip(rep, live, strict=True)]))
        self.energy = prot
        self.v2_live = {"clips": pick, "n_frames": len(frames), "level_agreement_live_vs_replay": agree,
                        "live_frac_int8": float(np.mean([x == "int8" for x in live]))}
        w = self._writer("test_v2_energy_latency", "measurement", "test",
                         extra={"locked_config_sha256": h, "energy_method": prot["method"], "measurement_clips": pick,
                                "addendum_sha256": self.addendum_sha})
        w.write_json("energy.json", prot)
        w.write_json("live_validation.json", self.v2_live)
        w.register(policy="measurement", mode="live", locked_sha=h, replace=True)
        checks = [{"check": f"protocol measured {k}", "pass": prot["summary"][k]["throughput_fps"] is not None}
                  for k in steppers]
        soft = [{"check": "v2 live vs replay decision agreement >= 0.9", "pass": agree >= 0.9, "detail": agree},
                {"check": "measurement set has mixed switching (0.2 <= INT8 frac <= 0.8)",
                 "pass": 0.2 <= self.v2_live["live_frac_int8"] <= 0.8, "detail": self.v2_live["live_frac_int8"]}]
        self.gate("5_v2_energy", checks, soft)
        return {"summary": prot["summary"], "live": self.v2_live}

    # ------------------------------------------------------------------ 6. test evaluation
    def stage6_v2_eval(self) -> dict:
        _, body05, h05 = self._require_lock()
        model = self._model()
        t0 = time.time()
        ev = ComposedEvaluator(self.t_gt, {lv: self.t_dets[lv] for lv in ALL_LEVELS}, self.cfg["eval"]["max_dets"])
        fast = FastAP(ev)
        pf = {lv: perframe_f1(self.t_dets[lv], gt_arrays_from_coco(self.t_gt), self.tau_c,
                              self.cfg["eval"]["perframe_iou"], self.cfg["eval"]["ignore_ioa"]) for lv in ALL_LEVELS}
        f1 = {lv: {i: pf[lv][i]["f1"] for i in self.t_ids} for lv in ALL_LEVELS}
        area = self._area(self.t_rows)
        sd = R.build_seqdata(self.t_seq_frames, self.t_cues, self.t_dets, area, f1, self.tau_c)
        rt = R.risk_table(model, sd)
        dseq = [np.array([self.t_cues[i]["d"] for i in ids]) for ids in sd.image_ids]
        order = [i for ids in sd.image_ids for i in ids]
        g_all = np.array([sd.g[i] for i in order])
        g_seq = [np.array([sd.g[i] for i in ids]) for ids in sd.image_ids]
        arrays = CAL.build_arrays(self.t_seq_frames, self.t_cues, self.t_dets, f1, self.tau_c, self.u_empty)
        log.info("test matching + risk tables in %.0fs", time.time() - t0)
        e_sum = self.energy["summary"]

        def e_lv(lv, key):
            s = e_sum.get(f"fixed_{lv}", {}).get(key)
            return s["median"] if s else None

        def metrics(choice):
            m = ev.evaluate(choice, self.t_slices)
            for name, ids in self.t_slices.items():
                if not ids:
                    continue
                lvl = [choice[i] for i in ids]
                m[name]["frac_level"] = {lv: float(np.mean([x == lv for x in lvl])) for lv in ALL_LEVELS}
                m[name]["est_engine_ms"] = float(np.mean([self.engine_ms_all[x] for x in lvl]))
                for key, tag in (("E_total_mJ_per_frame", "est_E_total_mJ_per_frame"),
                                 ("E_dyn_mJ_per_frame", "est_E_dyn_mJ_per_frame")):
                    vals = [e_lv(x, key) for x in lvl]
                    m[name][tag] = float(np.mean(vals)) if all(v is not None for v in vals) else None
            ch = np.array([0 if choice[i] == "int8" else 1 if choice[i] == "fp16" else 2 for i in order])
            same = np.ones(len(order), bool)
            same[arrays.seq_starts] = False
            m["all"]["switch_rate_per_100"] = 100.0 * float(np.sum((ch[1:] != ch[:-1]) & same[1:])) / len(order)
            m["all"]["energy_kind"] = "estimated from measured fixed-level E/frame (E_total primary)"
            return m

        pol, out, groups, logs = {}, {}, {}, {}
        for lv in ALL_LEVELS:
            pol[f"fixed_{lv}"] = {i: lv for i in self.t_ids}
        same_ref = True
        for rho in self.budgets:
            b, h = read_lock(self.lock_dir / f"locked_v2_rho{rho}.yaml")
            p = R.V2Params(b["tau"], b["K"], b["theta_sc"], b["H"])
            ch = R.fast_replay_v2(rt, dseq, p)
            if rho == 0.5:  # reference: step-by-step controller on the cache
                for si, ids in enumerate(sd.image_ids):
                    ctl = R.RiskController(model, p, self.tau_c)
                    ref = []
                    for iid in ids:
                        o = ctl.step(self.t_cues[iid])
                        ref.append(LV.index(o["level"]))
                        ctl.observe(self.t_dets[o["level"]][iid], area[iid])
                    same_ref &= ref == ch[si].tolist()
            pol[f"v2_rho{rho}"] = {i: LV[c] for i, c in zip(order, np.concatenate(ch), strict=True)}
            logs[rho] = {s: c for s, c in zip(sd.seq_ids, ch, strict=True)}
            vb, _ = read_lock(self.lock_dir / f"locked_v1_rho{rho}.yaml")
            c = vb["controller"]
            vp = ControllerParams(weights=c["weights"], theta=vb["theta"], norm={k: tuple(v) for k, v in c["norm"].items()},
                                  levels=tuple(c["levels"]), start_level=c["start_level"], alpha=c["alpha"],
                                  K_up=c["K_up"], K_down=c["K_down"], H=c["H"], theta_sc=c["theta_sc"],
                                  scene_override=c["scene_override"], w_h=c["w_h"])
            pol[f"v1_rho{rho}"] = {int(i): LV[c_] for i, c_ in zip(arrays.image_ids, CAL.fast_replay(arrays, vp), strict=True)}
            pol[f"oracle_rho{rho}"] = {i: LV[c_] for i, c_ in zip(order, R.oracle_at_budget(g_all, rho), strict=True)}
            lb, _ = read_lock(self.lock_dir / f"locked_lagged_rho{rho}.yaml")
            pol[f"lagged_oracle_rho{rho}"] = {i: LV[c_] for i, c_ in zip(order, np.concatenate(R.lagged_oracle(g_seq, lb["tau"])), strict=True)}
            mk = baselines.estimate_markov({s: c_.tolist() for s, c_ in logs[rho].items()}, 2)
            lens = {s: len(v) for s, v in self.t_seq_frames.items()}
            groups[f"random_rho{rho}"] = []
            for k in range(self.n_random):
                rc = baselines.markov_random(lens, mk["P"], mk["init"], self.s2.seed + k)
                name = f"random_rho{rho}_s{self.s2.seed + k}"
                pol[name] = {iid: LV[c_] for s in self.t_seq_frames for iid, c_ in zip(self.t_seq_frames[s], rc[s], strict=True)}
                groups[f"random_rho{rho}"].append(name)
        for name, ch in pol.items():
            out[name] = metrics(ch)
        agg = {}
        for gname, members in groups.items():
            a = {}
            for sl in self.t_slices:
                vals = [out[m][sl] for m in members if out[m].get(sl, {}).get("n_images")]
                if not vals:
                    continue
                a[sl] = {"n_images": vals[0]["n_images"], **{k: float(np.mean([v[k] for v in vals])) for k in ("map50_95", "map50")},
                         **{f"{k}_sd": float(np.std([v[k] for v in vals], ddof=1)) if len(vals) > 1 else 0.0 for k in ("map50_95", "map50")},
                         "frac_level": {lv: float(np.mean([v["frac_level"][lv] for v in vals])) for lv in ALL_LEVELS}}
            a["all"]["switch_rate_per_100"] = float(np.mean([out[m]["all"]["switch_rate_per_100"] for m in members]))
            agg[gname] = a
        # bootstrap (addendum): v2@rho - random@rho for every rho; v2@0.5 - v1@0.5; v2@0.5 vs fixed
        n_b = self.V["smoke"]["bootstrap_resamples"] if self.smoke else self.V["bootstrap_resamples"]
        boot_pol = {k: pol[k] for k in ["fixed_fp16", "fixed_int8"] + [f"v2_rho{r}" for r in self.budgets] + ["v1_rho0.5"]}
        for members in groups.values():
            for m in members:
                boot_pol[m] = pol[m]
        pairs = [(f"v2_rho{r}", f"random_rho{r}") for r in self.budgets] + [("v2_rho0.5", "v1_rho0.5"),
                                                                             ("v2_rho0.5", "fixed_fp16"),
                                                                             ("v2_rho0.5", "fixed_int8")]
        t1 = time.time()
        boot = cluster_bootstrap(fast, dict(self.t_seq_frames), boot_pol, pairs, n_b, self.s2.seed, random_groups=groups)
        log.info("bootstrap %d resamples in %.0fs", n_b, time.time() - t1)
        # headroom captured (descriptive, P4)
        head = {}
        for r in self.budgets:
            v, rnd, orc = out[f"v2_rho{r}"]["all"]["map50_95"], agg[f"random_rho{r}"]["all"]["map50_95"], out[f"oracle_rho{r}"]["all"]["map50_95"]
            head[r] = {"v2": v, "random": rnd, "oracle": orc, "lagged_oracle": out[f"lagged_oracle_rho{r}"]["all"]["map50_95"],
                       "v1": out[f"v1_rho{r}"]["all"]["map50_95"],
                       "captured": (v - rnd) / (orc - rnd) if orc != rnd else None}
        hyp = self._v2_hypotheses(boot, head)
        # write
        for name, m in list(out.items()) + list(agg.items()):
            if "_s" in name and name.startswith("random_rho"):
                continue
            if name in boot["per_policy"]:
                for k in ("map50_95", "map50"):
                    m["all"][f"{k}_ci"] = boot["per_policy"][name][k]["ci95"]
            w = self._writer(f"test_{name}", name, "test", extra={"locked_config_sha256": h05,
                                                                  "addendum_sha256": self.addendum_sha})
            w.write_json("metrics.json", _jsonable(m))
            w.register(policy=name, mode="replay", locked_sha=h05, replace=True)
        w = self._writer("test_v2_bootstrap", "bootstrap", "test", extra={"locked_config_sha256": h05})
        w.write_json("bootstrap.json", boot)
        w.register(policy="bootstrap", mode="replay", locked_sha=h05, replace=True)
        w = self._writer("test_v2_hypotheses", "hypotheses", "test", extra={"locked_config_sha256": h05,
                                                                            "addendum_sha256": self.addendum_sha})
        w.write_json("hypotheses.json", _jsonable(hyp))
        w.write_json("headroom.json", _jsonable(head))
        w.register(policy="hypotheses", mode="replay", locked_sha=h05, replace=True)
        checks = [{"check": "v2@0.50: fast replay == step-by-step RiskController on test", "pass": bool(same_ref)},
                  {"check": "every policy evaluated on all test frames",
                   "pass": all(out[n]["all"]["n_images"] == len(self.t_ids) for n in out)},
                  {"check": f"bootstrap ran ({n_b} resamples)", "pass": boot["n_resamples"] == n_b}]
        self.gate("6_v2_test_eval", checks)
        return {"hypotheses": hyp, "headroom": head}

    def _v2_hypotheses(self, boot, head) -> dict:
        def pair(a, b):
            v = boot["pairs"][f"{a} - {b}"]["map50_95"]
            return {"mean_pts": 100 * v["mean"], "ci95_pts": [100 * x for x in v["ci95"]],
                    "above_zero": bool(v["ci95"][0] > 0)}

        P1 = {**pair("v2_rho0.5", "random_rho0.5")}
        P1["supported"] = P1["above_zero"]
        P2 = {str(r): {**pair(f"v2_rho{r}", f"random_rho{r}")} for r in self.budgets if r != 0.5}
        for v in P2.values():
            v["supported"] = v["above_zero"]
        P3 = {**pair("v2_rho0.5", "v1_rho0.5")}
        P3["supported"] = P3["above_zero"]
        s = self.energy["summary"]

        def med(k, m):
            d = s.get(k, {}).get(m)
            return d["median"] if d else None

        P5 = {"latency_ms": {"v2_pipelined": med("adaptive_v2_pipelined", "latency_total_ms_median"),
                             "fixed_fp16": med("fixed_fp16", "latency_total_ms_median")},
              "E_total_mJ_per_frame": {"v2_pipelined": med("adaptive_v2_pipelined", "E_total_mJ_per_frame"),
                                       "fixed_fp16": med("fixed_fp16", "E_total_mJ_per_frame")},
              "E_dyn_mJ_per_frame_secondary": {"v2_pipelined": med("adaptive_v2_pipelined", "E_dyn_mJ_per_frame"),
                                               "fixed_fp16": med("fixed_fp16", "E_dyn_mJ_per_frame")},
              "measured_frac_int8": s.get("adaptive_v2_pipelined", {}).get("frac_int8"),
              "measured_switch_rate_per_100": s.get("adaptive_v2_pipelined", {}).get("switch_rate_per_100")}
        P5["supported"] = bool(P5["latency_ms"]["v2_pipelined"] is not None and
                               P5["latency_ms"]["v2_pipelined"] < P5["latency_ms"]["fixed_fp16"] and
                               P5["E_total_mJ_per_frame"]["v2_pipelined"] < P5["E_total_mJ_per_frame"]["fixed_fp16"])
        return {"P1_primary": P1, "P2": P2, "P3": P3, "P4_headroom": head, "P5_energy": P5,
                "note": "prereg addendum " + self.addendum_sha[:16] + "; r04 remains the pre-registered primary result"}
