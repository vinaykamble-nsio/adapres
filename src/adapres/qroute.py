"""Q-ROUTE: quantization-regret-aware precision x resolution routing (configs/prereg_qroute.yaml).

Per frame t the controller predicts, for every action c (precision x input resolution), the regret
r_c(t) = F1(reference, t) - F1(c, t) from CAUSAL features of the detections actually used at t-1 and
t-2 (near-free) and optionally cheap image cues, then picks
    argmin_c  r_hat_c(t) + lambda * cost_c(t) / cost_reference,
    cost_c(t) = isolated_latency_c * slowdown(t) + controller_overhead,
subject to a deadline (if set), with K-frame persistence and a reference fallback on the first frame.
slowdown(t) is an EMA of observed/isolated latency of the engine used at t-1 (hardware awareness).

fast_replay_q() reproduces QRouteController exactly (tests/test_qroute.py) from a precomputed table of
predictions for every (config at t-1, config at t-2) history, so lambda can be tuned by bisection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .risk import det_stats

DEFAULT_CONFIGS = ("fp16_640", "int8_640", "fp16_480", "int8_480", "fp16_320", "int8_320")


def parse_config(name: str) -> tuple[str, int]:
    lv, res = name.split("_")
    return lv, int(res)


LIGHT_FEATURES = ("n", "mean_score", "min_score", "frac_low_conf", "n_near_threshold", "frac_small",
                  "mean_log_rel_area", "class_entropy", "delta_n", "delta_mean_score", "prev_res_480",
                  "prev_res_320", "prev_int8")
TEMPORAL_FEATURES = ("delta_n", "delta_mean_score")
IMAGE_FEATURES = ("m", "c", "d")


def feature_names(use_img: bool, use_temporal: bool = True) -> list[str]:
    f = [x for x in LIGHT_FEATURES if use_temporal or x not in TEMPORAL_FEATURES]
    return f + (list(IMAGE_FEATURES) if use_img else [])


def feature_vector(prev: dict, prevprev: dict | None, prev_cfg: str, img_cues: Mapping[str, float] | None,
                   use_img: bool, use_temporal: bool = True) -> list[float]:
    lv, res = parse_config(prev_cfg)
    v = [prev["n"], prev["mean_score"], prev["min_score"], prev["frac_low_conf"], prev["n_near_threshold"],
         prev["frac_small"], prev["mean_log_rel_area"], prev["class_entropy"]]
    if use_temporal:
        v += [0.0 if prevprev is None else prev["n"] - prevprev["n"],
              0.0 if prevprev is None else prev["mean_score"] - prevprev["mean_score"]]
    v += [float(res == 480), float(res == 320), float(lv == "int8")]
    if use_img:
        v += [img_cues["m"], img_cues["c"], img_cues["d"]]
    return v


# ----------------------------------------------------------------------------- data
@dataclass
class QData:
    seq_ids: list[str]
    image_ids: list[list[int]]
    configs: list[str]
    stats: dict[str, dict[int, dict]]                  # config -> image_id -> det_stats
    cues: Mapping[int, dict]
    f1: dict[str, dict[int, float]] = field(default_factory=dict)


def build_qdata(seq_frames: Mapping[str, Sequence[int]], configs: Sequence[str],
                dets: Mapping[str, Mapping[int, np.ndarray]], area: Mapping[int, float], cues: Mapping[int, dict],
                f1: Mapping[str, Mapping[int, float]] | None, tau_c: float) -> QData:
    seqs = sorted(seq_frames)
    stats = {c: {i: det_stats(dets[c].get(i), area[i], tau_c) for s in seqs for i in seq_frames[s]} for c in configs}
    return QData(seqs, [list(seq_frames[s]) for s in seqs], list(configs), stats, cues,
                 {c: dict(f1[c]) for c in configs} if f1 else {})


def regret_matrix(qd: QData, ref: str, ids: Sequence[int]) -> np.ndarray:
    """(N, C) true regret F1(ref) - F1(c) for the given frames."""
    return np.array([[qd.f1[ref][i] - qd.f1[c][i] for c in qd.configs] for i in ids])


def training_rows(qd: QData, ref: str, use_img: bool, use_temporal: bool = True):
    """X, Y (regret per config), groups. Each frame t >= 1 once per possible previous config p
    (history t-1 and t-2 both from p)."""
    X, Y, G = [], [], []
    for si, ids in enumerate(qd.image_ids):
        for t in range(1, len(ids)):
            y = [qd.f1[ref][ids[t]] - qd.f1[c][ids[t]] for c in qd.configs]
            for p in qd.configs:
                pp = qd.stats[p][ids[t - 2]] if t >= 2 else None
                X.append(feature_vector(qd.stats[p][ids[t - 1]], pp, p, qd.cues.get(ids[t]), use_img, use_temporal))
                Y.append(y)
                G.append(si)
    return np.asarray(X, float), np.asarray(Y, float), np.asarray(G)


# ----------------------------------------------------------------------------- predictors
class RidgeMulti:
    """Multi-output ridge regression with standardisation; numpy only (microsecond prediction)."""

    kind = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.ym = Y.mean(0)
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.W = np.linalg.solve(A, Z.T @ (Y - self.ym))
        return self

    def predict(self, X):
        return ((np.asarray(X, float) - self.mu) / self.sd) @ self.W + self.ym

    def predict_one(self, x):
        return ((np.asarray(x, float) - self.mu) / self.sd) @ self.W + self.ym


class GBTMulti:
    """One sklearn HistGradientBoostingRegressor per output, exported to numpy arrays so single-row
    prediction avoids sklearn's per-call overhead. validate() checks the export against sklearn."""

    kind = "gbt"

    def __init__(self, max_depth=3, max_iter=100, learning_rate=0.1, seed=2027):
        self.kw = dict(max_depth=max_depth, max_iter=max_iter, learning_rate=learning_rate, random_state=seed)

    def fit(self, X, Y):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.models = [HistGradientBoostingRegressor(**self.kw).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        self._export()
        return self

    def _export(self):
        self.trees = []
        for m in self.models:
            base = float(np.ravel(m._baseline_prediction)[0])
            ts = []
            for it in m._predictors:
                nd = it[0].nodes
                ts.append((nd["feature_idx"].astype(np.int64), nd["num_threshold"].astype(float),
                           nd["left"].astype(np.int64), nd["right"].astype(np.int64), nd["is_leaf"].astype(bool),
                           nd["value"].astype(float), nd["missing_go_to_left"].astype(bool)))
            self.trees.append((base, ts))

    def predict(self, X):
        return np.column_stack([m.predict(np.asarray(X, float)) for m in self.models])

    def predict_one(self, x):
        x = np.asarray(x, float)
        out = np.empty(len(self.trees))
        for j, (base, ts) in enumerate(self.trees):
            s = base
            for f, thr, left, right, leaf, val, mleft in ts:
                node = 0
                while not leaf[node]:
                    xv = x[f[node]]
                    node = (left[node] if mleft[node] else right[node]) if np.isnan(xv) else \
                        (left[node] if xv <= thr[node] else right[node])
                s += val[node]
            out[j] = s
        return out

    def validate(self, X, atol: float = 1e-6) -> float:
        a = self.predict(X)
        b = np.array([self.predict_one(x) for x in X])
        return float(np.max(np.abs(a - b)))


def make_predictor(spec: Mapping, seed: int):
    if spec["kind"] == "ridge":
        return RidgeMulti(alpha=spec["alpha"])
    return GBTMulti(max_depth=spec["max_depth"], max_iter=spec["max_iter"], learning_rate=spec["learning_rate"], seed=seed)


def select_predictor(X, Y, groups, candidates: Sequence[Mapping], ref_idx: int, seed: int, n_folds: int = 5) -> dict:
    """Clip-grouped CV; metric = mean held-out Spearman over non-reference outputs; ties within 0.01
    -> the cheaper predictor (ridge before gbt, then fewer parameters). Returns final + fold models."""
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    k = min(n_folds, len(np.unique(groups)))
    folds = list(GroupKFold(n_splits=k).split(X, Y, groups))
    cols = [j for j in range(Y.shape[1]) if j != ref_idx]
    table = []
    for spec in candidates:
        rhos, mae = [], []
        for tr, te in folds:
            m = make_predictor(spec, seed).fit(X[tr], Y[tr])
            P = m.predict(X[te])
            for j in cols:
                r = spearmanr(P[:, j], Y[te, j]).statistic
                rhos.append(0.0 if not np.isfinite(r) else float(r))
            mae.append(float(np.mean(np.abs(P[:, cols] - Y[te][:, cols]))))
        table.append({**spec, "spearman_mean": float(np.mean(rhos)), "mae_mean": float(np.mean(mae))})
    best_val = max(t["spearman_mean"] for t in table)
    cheap_order = {"ridge": 0, "gbt": 1}
    near = [t for t in table if t["spearman_mean"] >= best_val - 0.01]
    best = sorted(near, key=lambda t: (cheap_order[t["kind"]], t.get("max_depth", 0), t.get("max_iter", 0),
                                       -t.get("alpha", 0)))[0]
    spec = {k: best[k] for k in ("kind", "alpha", "max_depth", "max_iter", "learning_rate") if k in best}
    final = make_predictor(spec, seed).fit(X, Y)
    fold_models, fold_of_group = [], {}
    for fi, (tr, te) in enumerate(folds):
        fold_models.append(make_predictor(spec, seed).fit(X[tr], Y[tr]))
        for g in np.unique(groups[te]):
            fold_of_group[int(g)] = fi
    return {"spec": spec, "table": table, "best_spearman": best["spearman_mean"], "model": final,
            "fold_models": fold_models, "fold_of_group": fold_of_group, "folds": folds}


class ConstantPredictor:
    """'No regret prediction' / hardware-only: the development mean regret per config."""

    kind = "constant"

    def __init__(self, mean: np.ndarray):
        self.mean = np.asarray(mean, float)

    def predict(self, X):
        return np.tile(self.mean, (len(X), 1))

    def predict_one(self, x):
        return self.mean


# ----------------------------------------------------------------------------- controller
@dataclass
class QParams:
    configs: list[str]
    ref: str
    lam: float
    K: int
    cost_iso: dict[str, float]                 # isolated per-config latency (ms) or energy (mJ)
    overhead: float = 0.0                      # controller cost per frame (same unit), constant
    deadline: float | None = None
    use_hw: bool = True
    beta: float = 0.3
    use_img: bool = False
    use_temporal: bool = True


class QRouteController:
    def __init__(self, predictor, p: QParams, tau_c: float = 0.25):
        self.pred, self.p, self.tau_c = predictor, p, tau_c
        self.ref_i = p.configs.index(p.ref)
        self.reset()

    def reset(self):
        self.cur = self.p.ref
        self.cand, self.k, self.h = None, 0, 1.0
        self.prev_stats = self.prevprev_stats = None
        self.prev_cfg = None
        self.t = 0

    def _costs(self) -> np.ndarray:
        h = self.h if self.p.use_hw else 1.0
        return np.array([self.p.cost_iso[c] * h + self.p.overhead for c in self.p.configs])

    def decide(self, rhat: np.ndarray | None) -> dict:
        p = self.p
        prev = self.cur
        if rhat is None:                              # safe fallback: first frame of a clip
            self.cur, self.cand, self.k = p.ref, None, 0
            return {"config": self.cur, "proposal": p.ref, "switched": int(prev != self.cur) if self.t else 0,
                    "fallback": 1}
        rhat = np.array(rhat, float)
        rhat[self.ref_i] = 0.0
        cost = self._costs()
        feas = np.ones(len(cost), bool) if p.deadline is None else cost <= p.deadline
        if not feas.any():
            feas = cost == cost.min()
        J = rhat + p.lam * cost / p.cost_iso[p.ref]
        J[~feas] = np.inf
        prop = p.configs[int(np.argmin(J))]
        cur_i = p.configs.index(self.cur)
        if prop == self.cur:
            self.k, self.cand = 0, None
        elif not feas[cur_i]:                          # deadline violated: switch immediately
            self.cur, self.cand, self.k = prop, None, 0
        else:
            if prop == self.cand:
                self.k += 1
            else:
                self.cand, self.k = prop, 1
            if self.k >= p.K:
                self.cur, self.cand, self.k = prop, None, 0
        return {"config": self.cur, "proposal": prop, "switched": int(prev != self.cur), "fallback": 0}

    def step(self, img_cues: Mapping[str, float] | None = None) -> dict:
        rhat = None
        if self.prev_stats is not None:
            x = feature_vector(self.prev_stats, self.prevprev_stats, self.prev_cfg, img_cues, self.p.use_img,
                               self.p.use_temporal)
            rhat = self.pred.predict_one(x)
        out = self.decide(rhat)
        out["slowdown"] = self.h
        self.t += 1
        return out

    def observe(self, dets, img_area: float, observed_cost: float | None = None):
        self.prevprev_stats = self.prev_stats
        self.prev_stats = det_stats(dets, img_area, self.tau_c)
        self.prev_cfg = self.cur
        if observed_cost is not None and self.p.use_hw:
            ratio = observed_cost / self.p.cost_iso[self.cur]
            self.h = self.p.beta * ratio + (1 - self.p.beta) * self.h


def prediction_table(qd: QData, model_for_seq: Callable[[int], object], use_img: bool, use_temporal: bool = True
                     ) -> list[np.ndarray]:
    """Per sequence an array P[t, a, b, :] = predicted regrets at t if config a was used at t-1 and b at
    t-2 (index into qd.configs). t = 0 is unused (fallback); at t = 1, b is irrelevant."""
    C = len(qd.configs)
    out = []
    for si, ids in enumerate(qd.image_ids):
        T = len(ids)
        rows, idx = [], []
        for t in range(1, T):
            for a, ca in enumerate(qd.configs):
                for b, cb in enumerate(qd.configs):
                    if t == 1 and b > 0:
                        continue
                    pp = qd.stats[cb][ids[t - 2]] if t >= 2 else None
                    rows.append(feature_vector(qd.stats[ca][ids[t - 1]], pp, ca, qd.cues.get(ids[t]), use_img, use_temporal))
                    idx.append((t, a, b))
        P = np.zeros((T, C, C, C))
        if rows:
            pred = model_for_seq(si).predict(np.asarray(rows, float))
            for (t, a, b), v in zip(idx, pred, strict=True):
                if t == 1:
                    P[t, a, :, :] = v
                else:
                    P[t, a, b, :] = v
        out.append(P)
    return out


def fast_replay_q(tables: Sequence[np.ndarray], p: QParams, slowdown: Sequence[np.ndarray] | None = None
                  ) -> list[np.ndarray]:
    """Closed-loop replay; identical decisions to QRouteController. slowdown[si][t, c] is the
    observed/isolated latency ratio of config c at frame t (None -> 1.0, no contention)."""
    C = len(p.configs)
    ref_i = p.configs.index(p.ref)
    iso = np.array([p.cost_iso[c] for c in p.configs])
    out = []
    for si, P in enumerate(tables):
        T = P.shape[0]
        ch = np.zeros(T, np.int64)
        cur, cand, k, h = ref_i, -1, 0, 1.0
        prev1 = prev2 = None
        for t in range(T):
            if t == 0:
                cur, cand, k = ref_i, -1, 0
            else:
                a = prev1
                b = prev2 if prev2 is not None else 0
                rhat = P[t, a, b].copy()
                rhat[ref_i] = 0.0
                hh = h if p.use_hw else 1.0
                cost = iso * hh + p.overhead
                feas = np.ones(C, bool) if p.deadline is None else cost <= p.deadline
                if not feas.any():
                    feas = cost == cost.min()
                J = rhat + p.lam * cost / p.cost_iso[p.ref]
                J[~feas] = np.inf
                prop = int(np.argmin(J))
                if prop == cur:
                    k, cand = 0, -1
                elif not feas[cur]:
                    cur, cand, k = prop, -1, 0
                else:
                    if prop == cand:
                        k += 1
                    else:
                        cand, k = prop, 1
                    if k >= p.K:
                        cur, cand, k = prop, -1, 0
            ch[t] = cur
            # observe: latency of the engine used at t
            if p.use_hw:
                ratio = 1.0 if slowdown is None else float(slowdown[si][t, cur])
                h = p.beta * ratio + (1 - p.beta) * h
            prev2, prev1 = prev1, cur
        out.append(ch)
    return out


def mean_cost(choices: Sequence[np.ndarray], p: QParams, slowdown: Sequence[np.ndarray] | None = None) -> float:
    iso = np.array([p.cost_iso[c] for c in p.configs])
    tot, n = 0.0, 0
    for si, ch in enumerate(choices):
        sl = 1.0 if slowdown is None else slowdown[si][np.arange(len(ch)), ch]
        tot += float(np.sum(iso[ch] * sl)) + p.overhead * len(ch)
        n += len(ch)
    return tot / n


def tune_lambda(tables, p: QParams, budget: float, slowdown=None, iters: int = 40, lo: float = 0.0, hi: float = 50.0
                ) -> tuple[float, float]:
    """Bisection on lambda so the mean (isolated or slowed) cost is closest to the budget."""
    best = (None, math.inf, None)
    for _ in range(iters):
        mid = (lo + hi) / 2
        q = QParams(**{**p.__dict__, "lam": mid})
        c = mean_cost(fast_replay_q(tables, q, slowdown), q, slowdown)
        if abs(c - budget) < best[1]:
            best = (mid, abs(c - budget), c)
        if c > budget:
            lo = mid
        else:
            hi = mid
    return best[0], best[2]


def oracle_choices(regret: np.ndarray, cost: np.ndarray, budget: float, iters: int = 60) -> tuple[np.ndarray, float]:
    """UPPER BOUND (uses GT): per-frame argmin of true regret + lambda * cost (no persistence), lambda set
    by bisection so the mean cost meets the budget. regret (N, C), cost (C,)."""
    lo, hi = 0.0, 1e3
    best = None
    for _ in range(iters):
        lam = (lo + hi) / 2
        ch = np.argmin(regret + lam * cost[None, :] / cost.max(), axis=1)
        c = float(cost[ch].mean())
        if best is None or abs(c - budget) < best[1]:
            best = (ch, abs(c - budget), c)
        if c > budget:
            lo = lam
        else:
            hi = lam
    return best[0], best[2]


def contention_slowdown(seq_lens: Sequence[int], configs: Sequence[str], ratios: Mapping[float, Mapping[str, float]],
                        schedule: Sequence[float], frames_per_phase: int) -> list[np.ndarray]:
    """Per sequence array [t, c] of slowdown ratios following a repeating contention schedule
    (levels from `schedule`, each lasting frames_per_phase frames); ratios[level][config] measured."""
    out = []
    g = 0
    for T in seq_lens:
        arr = np.ones((T, len(configs)))
        for t in range(T):
            lvl = schedule[((g + t) // frames_per_phase) % len(schedule)]
            arr[t] = [ratios[lvl][c] for c in configs]
        out.append(arr)
        g += T
    return out
