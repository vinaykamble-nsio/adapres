"""Calibration on the calib split and config locking (PLAN §3.5).

fast_replay() reproduces Controller/replay exactly (tests/test_calibrate.py) on precomputed
arrays, so the full grid (weights x theta x K x alpha x theta_sc) runs in minutes.

Objective (PLAN §3.5): maximise replay latency saving 1 - lat/lat_FP16 subject to
mAP@0.5:0.95(policy) >= mAP@0.5:0.95(FP16) - delta on calib. Ties: lower switch rate, then larger K.
Two stages: A ranks every config by saving among those whose mean per-frame F1 is within
stage_a_f1_slack of FP16 (proxy); B computes true mAP (composed pycocotools) for the top
stage_b_top configs (more in batches while none is feasible, up to stage_b_max).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from .controller import ControllerParams
from .features import detection_cues

LEVELS = ("int8", "fp16")
CUE_KEYS = ("m", "c", "n", "u", "s")


def simplex_weights(step: float = 0.25, include_equal: bool = True) -> list[dict[str, float]]:
    """All non-negative weight vectors over (m,c,n,u,s) on a grid of `step` that sum to 1
    (70 for step 0.25), plus the equal-weight vector."""
    n = round(1 / step)
    out = []
    for comb in itertools.product(range(n + 1), repeat=5):
        if sum(comb) == n:
            out.append({k: c * step for k, c in zip(CUE_KEYS, comb, strict=True)})
    if include_equal:
        out.insert(0, {k: 0.2 for k in CUE_KEYS})
    return out


@dataclass
class ReplayArrays:
    """Per-frame arrays in a fixed global order (sequences sorted, frames in order)."""
    seq_starts: np.ndarray            # index of each sequence's first frame
    image_ids: np.ndarray
    m: np.ndarray
    c: np.ndarray
    d: np.ndarray
    n_after: dict[str, np.ndarray]    # level -> n cue computed from that level's dets AT frame t (used at t+1)
    u_after: dict[str, np.ndarray]
    f1: dict[str, np.ndarray]
    u_empty: float


def build_arrays(seq_frames: Mapping[str, Sequence[int]], cues: Mapping[int, dict],
                 dets: Mapping[str, Mapping[int, np.ndarray]], f1: Mapping[str, Mapping[int, float]],
                 tau_c: float, u_empty: float) -> ReplayArrays:
    ids, starts = [], []
    for s in sorted(seq_frames):
        starts.append(len(ids))
        ids += list(seq_frames[s])
    n_after, u_after = {}, {}
    for lv in LEVELS:
        nu = [detection_cues(dets[lv].get(i, np.zeros((0, 6))), tau_c, u_empty) for i in ids]
        n_after[lv] = np.asarray([x[0] for x in nu])
        u_after[lv] = np.asarray([x[1] for x in nu])
    return ReplayArrays(np.asarray(starts), np.asarray(ids), np.asarray([cues[i]["m"] for i in ids]),
                        np.asarray([cues[i]["c"] for i in ids]), np.asarray([cues[i]["d"] for i in ids]),
                        n_after, u_after, {lv: np.asarray([f1[lv][i] for i in ids]) for lv in LEVELS},
                        float(u_empty))


def _norm(x, lohi):
    lo, hi = lohi
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def fast_replay(a: ReplayArrays, p: ControllerParams) -> np.ndarray:
    """Chosen level index per frame (0 = int8, 1 = fp16), identical to replay.replay()."""
    if p.levels != LEVELS:
        raise ValueError("fast_replay supports levels ('int8', 'fp16')")
    w = p.weights
    s_bin = (a.d >= p.theta_sc).astype(np.int8)
    base = w["m"] * _norm(a.m, p.norm["m"]) + w["c"] * _norm(a.c, p.norm["c"]) + w["s"] * s_bin
    nn = [w["n"] * _norm(a.n_after[lv], p.norm["n"]) for lv in LEVELS]
    uu = [w["u"] * _norm(a.u_after[lv], p.norm["u"]) for lv in LEVELS]
    first = w["n"] * float(_norm(np.array([0.0]), p.norm["n"])[0]) + w["u"] * float(_norm(np.array([a.u_empty]), p.norm["u"])[0])
    base_l, s_l = base.tolist(), s_bin.tolist()
    nn_l = [x.tolist() for x in nn]
    uu_l = [x.tolist() for x in uu]
    N = len(base_l)
    out = [0] * N
    ends = list(a.seq_starts[1:]) + [N]
    alpha, theta, K_up, K_down, H = p.alpha, p.theta, p.K_up, p.K_down, p.H
    override, w_h = p.scene_override, p.w_h
    start_level = LEVELS.index(p.start_level)
    for s0, s1 in zip(a.seq_starts.tolist(), ends, strict=True):
        L, cand, k, hold, sh = start_level, -1, 0, 0, None
        prev = None
        for t in range(s0, s1):
            S = base_l[t] + (first if prev is None else nn_l[prev][t - 1] + uu_l[prev][t - 1])
            sh = S if sh is None else alpha * S + (1 - alpha) * sh
            prop = 0 if (sh - w_h * 0.0) < theta else 1
            if override and s_l[t] == 1:
                L, cand, k, hold = 1, -1, 0, H
            elif hold > 0:
                hold -= 1
                cand, k = -1, 0
            else:
                if prop == L:
                    k, cand = 0, -1
                elif prop == cand:
                    k += 1
                else:
                    cand, k = prop, 1
                if cand != -1 and k >= (K_up if cand > L else K_down):
                    L, cand, k = cand, -1, 0
            out[t] = L
            prev = L
    return np.asarray(out, np.int8)


def open_loop_shat(a: ReplayArrays, weights: dict, alpha: float, theta_sc: float, norm: dict) -> np.ndarray:
    """S_hat with detection cues from FP16 (open loop), for theta quantiles."""
    s_bin = (a.d >= theta_sc).astype(float)
    base = weights["m"] * _norm(a.m, norm["m"]) + weights["c"] * _norm(a.c, norm["c"]) + weights["s"] * s_bin
    n_prev = np.concatenate(([0.0], a.n_after["fp16"][:-1]))
    u_prev = np.concatenate(([a.u_empty], a.u_after["fp16"][:-1]))
    starts = a.seq_starts
    n_prev[starts] = 0.0
    u_prev[starts] = a.u_empty
    S = base + weights["n"] * _norm(n_prev, norm["n"]) + weights["u"] * _norm(u_prev, norm["u"])
    out = np.empty_like(S)
    st = set(starts.tolist())
    sh = 0.0
    for t, v in enumerate(S.tolist()):
        sh = v if t in st else alpha * v + (1 - alpha) * sh
        out[t] = sh
    return out


def policy_cost(choice: np.ndarray, a: ReplayArrays, engine_ms: Mapping[str, float], switch_penalty_ms: float,
                ctrl_overhead_ms: float) -> dict:
    """Replay latency (PLAN §2.3): engine latency of the chosen level + controller overhead
    + switch penalty x switches (switches counted within sequences only)."""
    ms = np.where(choice == 0, engine_ms["int8"], engine_ms["fp16"])
    same = np.ones(len(choice), bool)
    same[a.seq_starts] = False
    switches = int(np.sum((choice[1:] != choice[:-1]) & same[1:]))
    lat = (ms.sum() + ctrl_overhead_ms * len(choice) + switch_penalty_ms * switches) / len(choice)
    return {"lat_ms": float(lat), "switches": switches, "switch_rate_per_100": 100.0 * switches / len(choice),
            "frac_int8": float(np.mean(choice == 0))}


def params_dict(p: ControllerParams) -> dict:
    d = p.to_dict()
    d["norm"] = {k: [float(x) for x in v] for k, v in p.norm.items()}
    return d


def grid_search(a: ReplayArrays, base: dict, norm: dict, weights_list: list[dict], theta_quantiles: Sequence[float],
                Ks: Sequence[int], alphas: Sequence[float], theta_scs: Sequence[float], engine_ms: Mapping[str, float],
                switch_penalty_ms: float, ctrl_overhead_ms: float, K_mode: str = "sym",
                fixed_theta: float | None = None, progress: Callable | None = None) -> list[dict]:
    """Stage A over the full grid. base: controller config dict (levels, start, H, override, u_empty...).
    K_mode 'sym' -> K_up = K_down = K; 'asym' -> K_up = 1, K_down = K."""
    rows = []
    fp16_f1 = float(a.f1["fp16"].mean())
    ref_lat = engine_ms["fp16"]
    total = len(weights_list) * len(alphas) * len(theta_scs)
    done = 0
    for w in weights_list:
        for alpha in alphas:
            for tsc in theta_scs:
                if fixed_theta is None:
                    sh = open_loop_shat(a, w, alpha, tsc, norm)
                    thetas = [(q, float(np.quantile(sh, q))) for q in theta_quantiles]
                else:
                    thetas = [(None, float(fixed_theta))]
                for (q, th), K in itertools.product(thetas, Ks):
                    p = ControllerParams(weights=w, theta=th, norm=norm, levels=LEVELS, start_level=base["start_level"],
                                         alpha=alpha, K_up=1 if K_mode == "asym" else K, K_down=K, H=base["H"],
                                         theta_sc=tsc, scene_override=base["scene_override"], w_h=base.get("w_h", 0.0))
                    ch = fast_replay(a, p)
                    cost = policy_cost(ch, a, engine_ms, switch_penalty_ms, ctrl_overhead_ms)
                    f1 = float(np.where(ch == 0, a.f1["int8"], a.f1["fp16"]).mean())
                    rows.append({"weights": w, "alpha": alpha, "theta_sc": tsc, "theta_quantile": q, "theta": th,
                                 "K_up": p.K_up, "K_down": p.K_down, "mean_f1": f1, "f1_drop_vs_fp16": fp16_f1 - f1,
                                 "saving": 1.0 - cost["lat_ms"] / ref_lat, **cost, "_params": p})
                done += 1
                if progress:
                    progress(done, total)
    return rows


def select(rows: list[dict], map_fn: Callable[[ControllerParams], float], fp16_map: float, delta_pts: float,
           f1_slack: float, top: int, max_eval: int) -> dict:
    """Stage B: evaluate true mAP for configs in order of saving (proxy-feasible first) and
    pick the feasible config with max saving (ties: lower switch rate, larger K_down)."""
    cand = sorted([r for r in rows if r["f1_drop_vs_fp16"] <= f1_slack],
                  key=lambda r: (-r["saving"], r["switch_rate_per_100"], -r["K_down"]))
    if not cand:
        cand = sorted(rows, key=lambda r: (r["f1_drop_vs_fp16"], -r["saving"]))
    evaluated = []
    limit = fp16_map - delta_pts / 100.0
    for start in range(0, min(len(cand), max_eval), top):
        for r in cand[start:start + top]:
            r["map50_95"] = float(map_fn(r["_params"]))
            r["feasible"] = r["map50_95"] >= limit
            evaluated.append(r)
        if any(r["feasible"] for r in evaluated):
            break
    feas = [r for r in evaluated if r["feasible"]]
    if feas:
        best = sorted(feas, key=lambda r: (-r["saving"], r["switch_rate_per_100"], -r["K_down"]))[0]
        status = "feasible"
    else:
        best = sorted(evaluated, key=lambda r: (-r["map50_95"], -r["saving"]))[0]
        status = "infeasible_on_calib (locked the highest-mAP evaluated config)"
    return {"best": best, "evaluated": evaluated, "status": status, "limit_map50_95": limit,
            "n_candidates_stage_a": len(cand), "n_evaluated_stage_b": len(evaluated)}


# ----------------------------------------------------------------------------- locking

def lock(path: str | Path, p: ControllerParams, info: dict) -> str:
    """Write a locked controller config + its .sha256. Returns the hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"controller": params_dict(p), **info}
    text = yaml.safe_dump(json.loads(json.dumps(body, default=float)), sort_keys=True)
    path.write_text(text, encoding="utf-8")
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(h + "\n")
    return h


class LockError(RuntimeError):
    pass


def load_locked(path: str | Path) -> tuple[ControllerParams, dict, str]:
    """Load a locked config and verify its hash; raises LockError if missing or modified."""
    path = Path(path)
    hp = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not hp.exists():
        raise LockError(f"locked config or hash missing: {path}")
    text = path.read_text(encoding="utf-8")
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if h != hp.read_text().strip():
        raise LockError(f"hash mismatch for {path}: file was modified after locking")
    body = yaml.safe_load(text)
    c = body["controller"]
    p = ControllerParams(weights=c["weights"], theta=c["theta"], norm={k: tuple(v) for k, v in c["norm"].items()},
                         levels=tuple(c["levels"]), start_level=c["start_level"], alpha=c["alpha"], K_up=c["K_up"],
                         K_down=c["K_down"], H=c["H"], theta_sc=c["theta_sc"], scene_override=c["scene_override"],
                         w_h=c["w_h"])
    return p, body, h
