"""Fast COCO AP for multisets of images (cluster bootstrap) and per-frame level choices.

Re-implements pycocotools COCOeval.accumulate() for area 'all' and maxDets[-1] on the
per-image matching results of ComposedEvaluator (one COCOeval.evaluate() per level).

Weighted formulation: for each category all detections of all levels are concatenated in
image-id order (levels in a fixed order within an image) and stably sorted by score ONCE.
A query (image multiset + level per image) only sets a weight per detection,
  w_det = multiplicity(image) * [level(det) == chosen level(image)],
and accumulates weighted TP/FP counts. This equals pycocotools exactly for any selection
without duplicated images (same stable order), and for duplicated images too, because the
copies of one detection are all-TP or all-FP and the interpolated precision is unchanged
(tests/test_fast_ap.py checks both against pycocotools).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class FastAP:
    def __init__(self, composed):
        """composed: eval.map_eval.ComposedEvaluator (evaluate() done per level)."""
        ref = composed.ev[composed.levels[0]]
        p = ref._paramsEval
        self.levels = list(composed.levels)
        self.img_ids = list(p.imgIds)
        self.pos = {iid: i for i, iid in enumerate(self.img_ids)}
        self.cat_ids = list(p.catIds)
        self.iou_thrs = np.asarray(p.iouThrs)
        self.rec_thrs = np.asarray(p.recThrs)
        I0, A0 = len(self.img_ids), len(p.areaRng)
        a_all = [tuple(x) for x in p.areaRng].index(tuple(ref.params.areaRng[ref.params.areaRngLbl.index("all")]))
        T = len(self.iou_thrs)
        self.t50 = int(np.flatnonzero(np.isclose(self.iou_thrs, 0.5))[0])
        self.cat = []
        for ki in range(len(self.cat_ids)):
            sc, mt, ig, im, lv = [], [], [], [], []
            npig = np.zeros(I0)
            for i in range(I0):
                for li, lvn in enumerate(self.levels):
                    e = composed.ev[lvn].evalImgs[ki * A0 * I0 + a_all * I0 + i]
                    if e is None:
                        continue
                    npig[i] = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))  # GT property, same for all levels
                    d = np.asarray(e["dtScores"], float)
                    if not len(d):
                        continue
                    sc.append(d)
                    mt.append(np.asarray(e["dtMatches"]).reshape(T, -1) > 0)
                    ig.append(np.asarray(e["dtIgnore"]).reshape(T, -1).astype(bool))
                    im.append(np.full(len(d), i, np.int64))
                    lv.append(np.full(len(d), li, np.int8))
            if sc:
                s = np.concatenate(sc)
                order = np.argsort(-s, kind="mergesort")
                m = np.concatenate(mt, axis=1)[:, order]
                g = np.concatenate(ig, axis=1)[:, order]
                self.cat.append({"tp": m & ~g, "fp": ~m & ~g, "img": np.concatenate(im)[order],
                                 "lvl": np.concatenate(lv)[order], "npig": npig})
            else:
                self.cat.append({"tp": np.zeros((T, 0), bool), "fp": np.zeros((T, 0), bool),
                                 "img": np.zeros(0, np.int64), "lvl": np.zeros(0, np.int8), "npig": npig})

    def ap_weighted(self, weights: np.ndarray, choice: np.ndarray) -> dict:
        """weights: multiplicity per image position (len = #images); choice: level index per position."""
        T, R = len(self.iou_thrs), len(self.rec_thrs)
        prec = -np.ones((T, R, len(self.cat)))
        for ki, c in enumerate(self.cat):
            npig = float(np.dot(weights, c["npig"]))
            if npig == 0:
                continue
            wd = weights[c["img"]] * (c["lvl"] == choice[c["img"]])
            act = wd > 0
            q = np.zeros((T, R))
            if act.any():
                wa = wd[act].astype(float)
                tp = np.cumsum(c["tp"][:, act] * wa, axis=1)
                fp = np.cumsum(c["fp"][:, act] * wa, axis=1)
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                nd = pr.shape[1]
                for t in range(T):
                    inds = np.searchsorted(rc[t], self.rec_thrs, side="left")
                    ok = inds < nd
                    q[t, ok] = pr[t, inds[ok]]
            prec[:, :, ki] = q
        valid = prec[prec > -1]
        v50 = prec[self.t50][prec[self.t50] > -1]
        return {"map50_95": float(valid.mean()) if valid.size else -1.0,
                "map50": float(v50.mean()) if v50.size else -1.0}

    def encode(self, img_ids: Sequence[int], levels: Sequence[str] | str) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray([self.pos[int(i)] for i in img_ids], np.int64)
        w = np.bincount(pos, minlength=len(self.img_ids)).astype(float)
        choice = np.zeros(len(self.img_ids), np.int8)
        if isinstance(levels, str):
            choice[:] = self.levels.index(levels)
        else:
            li = np.asarray([self.levels.index(x) for x in levels], np.int8)
            choice[pos] = li
        return w, choice

    def ap(self, img_ids: Sequence[int], levels: Sequence[str] | str) -> dict:
        return self.ap_weighted(*self.encode(img_ids, levels))


def cluster_bootstrap(fast: FastAP, seq_images: dict[str, list[int]], policies: dict[str, dict[int, str]],
                      pairs: list[tuple[str, str]], n_resamples: int, seed: int,
                      random_groups: dict[str, list[str]] | None = None) -> dict:
    """Paired cluster (sequence) bootstrap of mAP@0.5:0.95 and mAP@0.5.

    policies: name -> {image_id: level}. random_groups: name -> list of policy names whose
    per-resample mAP is averaged (e.g. the 10 matched-random seeds). pairs: (a, b) -> CI of a - b.
    Every policy is evaluated on the SAME resampled sequences in each resample."""
    rng = np.random.default_rng(seed)
    seqs = sorted(seq_images)
    seq_pos = [np.asarray([fast.pos[int(i)] for i in seq_images[s]], np.int64) for s in seqs]
    choices = {}
    for n, ch in policies.items():
        c = np.zeros(len(fast.img_ids), np.int8)
        for iid, lv in ch.items():
            if int(iid) in fast.pos:
                c[fast.pos[int(iid)]] = fast.levels.index(lv)
        choices[n] = c
    names = list(policies) + list(random_groups or {})
    vals = {n: {"map50_95": [], "map50": []} for n in names}
    for _ in range(n_resamples):
        pick = rng.integers(0, len(seqs), len(seqs))
        w = np.bincount(np.concatenate([seq_pos[j] for j in pick]), minlength=len(fast.img_ids)).astype(float)
        cache = {}
        for n in policies:
            cache[n] = fast.ap_weighted(w, choices[n])
            for k in ("map50_95", "map50"):
                vals[n][k].append(cache[n][k])
        for g, members in (random_groups or {}).items():
            for k in ("map50_95", "map50"):
                vals[g][k].append(float(np.mean([cache[m][k] for m in members])))
    out = {"n_resamples": n_resamples, "seed": seed, "unit": "sequence", "per_policy": {}, "pairs": {}}
    for n in names:
        out["per_policy"][n] = {k: {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                    "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0} for k, v in vals[n].items()}
    for a, b in pairs:
        out["pairs"][f"{a} - {b}"] = {}
        for k in ("map50_95", "map50"):
            d = np.asarray(vals[a][k]) - np.asarray(vals[b][k])
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            out["pairs"][f"{a} - {b}"][k] = {"ci95": [lo, hi], "mean": float(d.mean()),
                                             "excludes_zero": bool(lo > 0 or hi < 0)}
    return out
