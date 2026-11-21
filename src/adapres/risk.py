"""Controller v2 (configs/prereg_addendum.yaml): a learned per-frame risk estimator under a fixed
INT8 budget.

risk_t = predicted g_t = F1(FP16, t) - F1(INT8, t), from CAUSAL features: image cues of frames
(t, t-1) and statistics of the detections of the levels actually used at t-1 and t-2. Decision:
FP16 if risk_t >= tau, else INT8, with K-frame persistence and the scene-change override.

Because risk_t depends on the previous two chosen levels only through the detection features,
the replay precomputes risk for every (L_{t-1}, L_{t-2}) combination in one batched predict;
fast_replay_v2() then reproduces RiskController step by step (tests/test_risk.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .data.classmap import COCO_TO_EVAL

LEVELS = ("int8", "fp16")
FEATURES = ("m", "c", "d", "n", "mean_score", "min_score", "frac_low_conf", "n_near_threshold", "frac_small",
            "mean_log_rel_area", "class_entropy", "delta_n")
EMPTY_LOG_AREA = math.log(1e-6)


def det_stats(dets: np.ndarray | None, img_area: float, tau_c: float = 0.25) -> dict:
    """Statistics of one frame's detections (evaluated classes). Detections are canonicalised to
    float32 first (the precision of the detection cache), so live detections and cached ones give
    bit-identical features and therefore identical decisions."""
    d = np.zeros((0, 6)) if dets is None else np.asarray(dets, np.float32).astype(np.float64).reshape(-1, 6)
    ev = np.array([COCO_TO_EVAL.get(int(c)) is not None for c in d[:, 5]], bool) if len(d) else np.zeros(0, bool)
    d = d[ev]
    near = int(((d[:, 4] >= 0.10) & (d[:, 4] < tau_c)).sum()) if len(d) else 0
    k = d[d[:, 4] >= tau_c] if len(d) else d
    if not len(k):
        return {"n": 0.0, "mean_score": 0.0, "min_score": 0.0, "frac_low_conf": 0.0, "n_near_threshold": float(near),
                "frac_small": 0.0, "mean_log_rel_area": EMPTY_LOG_AREA, "class_entropy": 0.0}
    area = np.clip((k[:, 2] - k[:, 0]) * (k[:, 3] - k[:, 1]), 1e-9, None) / img_area
    cls = np.array([COCO_TO_EVAL[int(c)] for c in k[:, 5]])
    _, cnt = np.unique(cls, return_counts=True)
    p = cnt / cnt.sum()
    return {"n": float(len(k)), "mean_score": float(k[:, 4].mean()), "min_score": float(k[:, 4].min()),
            "frac_low_conf": float(np.mean(k[:, 4] < 0.5)), "n_near_threshold": float(near),
            "frac_small": float(np.mean(area < 1e-3)), "mean_log_rel_area": float(np.mean(np.log(area))),
            "class_entropy": float(-(p * np.log(p)).sum())}


EMPTY_STATS = det_stats(None, 1.0)


def feature_vector(img_cues: Mapping[str, float], prev: dict | None, prevprev_n: float | None) -> list[float]:
    """prev: det_stats of the detections used at t-1 (None at a sequence start).
    prevprev_n: count at t-2 (None if unavailable -> delta_n = 0)."""
    s = prev if prev is not None else EMPTY_STATS
    dn = 0.0 if (prev is None or prevprev_n is None) else s["n"] - prevprev_n
    return [img_cues["m"], img_cues["c"], img_cues["d"], s["n"], s["mean_score"], s["min_score"], s["frac_low_conf"],
            s["n_near_threshold"], s["frac_small"], s["mean_log_rel_area"], s["class_entropy"], dn]


# ----------------------------------------------------------------------------- data
@dataclass
class SeqData:
    """Per-sequence arrays (frames in order) needed to train and replay v2."""
    seq_ids: list[str]
    image_ids: list[list[int]]
    cues: Mapping[int, dict]
    stats: dict[str, dict[int, dict]]          # level -> image_id -> det_stats
    g: dict[int, float] = field(default_factory=dict)   # FP16 - INT8 per-frame F1 (only where known)


def build_seqdata(seq_frames: Mapping[str, Sequence[int]], cues: Mapping[int, dict],
                  dets: Mapping[str, Mapping[int, np.ndarray]], img_area: Mapping[int, float],
                  f1: Mapping[str, Mapping[int, float]] | None, tau_c: float) -> SeqData:
    seqs = sorted(seq_frames)
    stats = {lv: {i: det_stats(dets[lv].get(i), img_area[i], tau_c) for s in seqs for i in seq_frames[s]}
             for lv in LEVELS}
    g = {i: f1["fp16"][i] - f1["int8"][i] for s in seqs for i in seq_frames[s]} if f1 else {}
    return SeqData(seqs, [list(seq_frames[s]) for s in seqs], cues, stats, g)


def training_rows(sd: SeqData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, y, group(sequence index). Each frame twice: previous detections from INT8 and from FP16."""
    X, y, grp = [], [], []
    for si, ids in enumerate(sd.image_ids):
        for t, iid in enumerate(ids):
            for lv in LEVELS:
                prev = sd.stats[lv][ids[t - 1]] if t >= 1 else None
                pp = sd.stats[lv][ids[t - 2]]["n"] if t >= 2 else None
                X.append(feature_vector(sd.cues[iid], prev, pp))
                y.append(sd.g[iid])
                grp.append(si)
    return np.asarray(X, float), np.asarray(y, float), np.asarray(grp)


def risk_table(model, sd: SeqData, models_by_seq: Mapping[int, object] | None = None) -> list[np.ndarray]:
    """Per sequence an array r[t, a, b] = risk at t if level a was used at t-1 and b at t-2
    (a, b in {0: int8, 1: fp16}). At t = 0 all entries are the no-history risk; at t = 1 the
    t-2 level is irrelevant. models_by_seq: cross-fitted model per sequence index (else `model`)."""
    out = []
    for si, ids in enumerate(sd.image_ids):
        rows, idx = [], []
        for t, iid in enumerate(ids):
            for a in range(2):
                for b in range(2):
                    prev = sd.stats[LEVELS[a]][ids[t - 1]] if t >= 1 else None
                    pp = sd.stats[LEVELS[b]][ids[t - 2]]["n"] if t >= 2 else None
                    rows.append(feature_vector(sd.cues[iid], prev, pp))
                    idx.append((t, a, b))
        m = models_by_seq[si] if models_by_seq is not None else model
        pred = m.predict(np.asarray(rows, float)) if rows else np.zeros(0)
        r = np.zeros((len(ids), 2, 2))
        for (t, a, b), v in zip(idx, pred, strict=True):
            r[t, a, b] = v
        out.append(r)
    return out


# ----------------------------------------------------------------------------- model selection
def select_and_fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, grid: Mapping[str, Sequence], seed: int,
                   n_folds: int = 5) -> dict:
    """GroupKFold model selection by mean held-out Spearman; ties -> simpler. Returns the chosen
    params, CV table, the final model (all calib) and cross-fitted fold models."""
    from itertools import product

    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    k = min(n_folds, len(np.unique(groups)))
    folds = list(GroupKFold(n_splits=k).split(X, y, groups))
    keys = sorted(grid)
    table = []
    for combo in product(*[grid[kk] for kk in keys]):
        params = dict(zip(keys, combo, strict=True))
        rhos = []
        for tr, te in folds:
            m = HistGradientBoostingRegressor(random_state=seed, **params).fit(X[tr], y[tr])
            r = spearmanr(m.predict(X[te]), y[te]).statistic
            rhos.append(0.0 if not np.isfinite(r) else float(r))
        table.append({**params, "spearman_mean": float(np.mean(rhos)), "spearman_folds": rhos})
    best = sorted(table, key=lambda r: (-round(r["spearman_mean"], 12), r.get("max_depth", 0), r.get("max_iter", 0),
                                        -r.get("learning_rate", 0)))[0]
    params = {kk: best[kk] for kk in keys}
    final = HistGradientBoostingRegressor(random_state=seed, **params).fit(X, y)
    fold_models, fold_of_group = [], {}
    for fi, (tr, te) in enumerate(folds):
        fold_models.append(HistGradientBoostingRegressor(random_state=seed, **params).fit(X[tr], y[tr]))
        for gidx in np.unique(groups[te]):
            fold_of_group[int(gidx)] = fi
    return {"params": params, "cv": table, "best_spearman": best["spearman_mean"], "model": final,
            "fold_models": fold_models, "fold_of_group": fold_of_group}


# ----------------------------------------------------------------------------- controller
@dataclass
class V2Params:
    tau: float
    K: int
    theta_sc: float = 0.5
    H: int = 3
    start_level: str = "fp16"


class RiskController:
    """Causal, stateful (reset per sequence). step(img_cues, prev_dets, img_area) -> log dict."""

    def __init__(self, model, p: V2Params, tau_c: float = 0.25):
        self.model, self.p, self.tau_c = model, p, tau_c
        self.reset()

    def reset(self):
        self.L, self.cand, self.k, self.hold = LEVELS.index(self.p.start_level), -1, 0, 0
        self.prev_stats, self.prev_n = None, None

    def observe(self, dets, img_area):
        """Call after running the detector on the frame: stores the detections used."""
        st = det_stats(dets, img_area, self.tau_c)
        self.prev_n = self.prev_stats["n"] if self.prev_stats is not None else None
        self.prev_stats = st

    def step(self, img_cues: Mapping[str, float]) -> dict:
        x = feature_vector(img_cues, self.prev_stats, self.prev_n)
        r = float(self.model.predict(np.asarray([x], float))[0])
        prop = 1 if r >= self.p.tau else 0
        prevL = self.L
        over = False
        if img_cues["d"] >= self.p.theta_sc:
            self.L, self.cand, self.k, self.hold = 1, -1, 0, self.p.H
            over = True
        elif self.hold > 0:
            self.hold -= 1
            self.cand, self.k = -1, 0
        else:
            if prop == self.L:
                self.k, self.cand = 0, -1
            elif prop == self.cand:
                self.k += 1
            else:
                self.cand, self.k = prop, 1
            if self.cand != -1 and self.k >= self.p.K:
                self.L, self.cand, self.k = self.cand, -1, 0
        return {"level": LEVELS[self.L], "risk": r, "proposal": LEVELS[prop], "override": int(over),
                "switched": int(self.L != prevL)}


def fast_replay_v2(rt: Sequence[np.ndarray], d: Sequence[np.ndarray], p: V2Params) -> list[np.ndarray]:
    """Closed-loop replay using precomputed risk tables; identical decisions to RiskController."""
    out = []
    for r, dd in zip(rt, d, strict=True):
        n = len(r)
        L, cand, k, hold = LEVELS.index(p.start_level), -1, 0, 0
        ch = np.zeros(n, np.int8)
        prev1 = prev2 = None
        for t in range(n):
            a = prev1 if prev1 is not None else 0
            b = prev2 if prev2 is not None else a
            risk = r[t, a, b]
            prop = 1 if risk >= p.tau else 0
            if dd[t] >= p.theta_sc:
                L, cand, k, hold = 1, -1, 0, p.H
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
                if cand != -1 and k >= p.K:
                    L, cand, k = cand, -1, 0
            ch[t] = L
            prev2, prev1 = prev1, L
        out.append(ch)
    return out


def tau_for_budget(rt, d, K: int, rho: float, theta_sc: float, H: int, iters: int = 40) -> tuple[float, float]:
    """Bisection on tau so the realised closed-loop INT8 fraction is closest to rho."""
    allr = np.concatenate([r.ravel() for r in rt])
    lo, hi = float(allr.min()) - 1e-6, float(allr.max()) + 1e-6
    best = (None, 9.0, None)

    def frac(tau):
        ch = fast_replay_v2(rt, d, V2Params(tau, K, theta_sc, H))
        return float(np.mean(np.concatenate(ch) == 0))

    for _ in range(iters):
        mid = (lo + hi) / 2
        f = frac(mid)
        if abs(f - rho) < best[1]:
            best = (mid, abs(f - rho), f)
        if f < rho:      # too few INT8 frames -> raise tau
            lo = mid
        else:
            hi = mid
    return best[0], best[2]


# ----------------------------------------------------------------------------- references
def oracle_at_budget(g: Sequence[float], rho: float) -> np.ndarray:
    """UPPER BOUND (uses GT): INT8 (0) on the floor(rho*N) frames with the smallest g."""
    g = np.asarray(g, float)
    n8 = int(math.floor(rho * len(g)))
    order = np.argsort(g, kind="stable")
    ch = np.ones(len(g), np.int8)
    ch[order[:n8]] = 0
    return ch


def lagged_oracle(g_seq: Sequence[np.ndarray], tau: float) -> list[np.ndarray]:
    """REFERENCE (GT one frame late, not deployable): INT8 iff g_{t-1} < tau; FP16 on the first frame."""
    out = []
    for g in g_seq:
        ch = np.ones(len(g), np.int8)
        if len(g) > 1:
            ch[1:] = (np.asarray(g[:-1]) >= tau).astype(np.int8)
        out.append(ch)
    return out


def lagged_tau_for_budget(g_seq: Sequence[np.ndarray], rho: float) -> float:
    prev = np.concatenate([np.asarray(g[:-1]) for g in g_seq if len(g) > 1])
    return float(np.quantile(prev, rho)) + 1e-12
