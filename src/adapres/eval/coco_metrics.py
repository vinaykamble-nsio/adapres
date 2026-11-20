"""Version-robust AP extraction from a pycocotools COCOeval.

pycocotools 2.0.11's summarize() hardcodes maxDets=100 for stats[0] (AP@[.5:.95]),
so with params.maxDets=[1,10,300] stats[0] is -1. We therefore read AP directly
from ev.eval['precision'] (shape [T, R, K, A, M]) using our own maxDets setting.
"""

from __future__ import annotations

import numpy as np


def ap_from_eval(ev, iou: float | None = None, area: str = "all", max_det: int | None = None,
                 cat_ids: list[int] | None = None) -> float:
    """COCO-style AP: mean precision over recall thresholds, IoU thresholds (or one IoU)
    and categories, ignoring -1 entries. Returns -1.0 if nothing is valid (no GT)."""
    p = ev.params
    prec = ev.eval["precision"]  # T x R x K x A x M
    a = p.areaRngLbl.index(area)
    m = p.maxDets.index(max_det if max_det is not None else p.maxDets[-1])
    s = prec[:, :, :, a, m]
    if iou is not None:
        t = np.where(np.isclose(p.iouThrs, iou))[0]
        if t.size == 0:
            raise ValueError(f"IoU {iou} not in params.iouThrs")
        s = s[t]
    if cat_ids is not None:
        k = [p.catIds.index(c) for c in cat_ids]
        s = s[:, :, k]
    valid = s[s > -1]
    return float(valid.mean()) if valid.size else -1.0


def per_class_ap(ev, max_det: int | None = None) -> dict[int, float]:
    return {c: ap_from_eval(ev, max_det=max_det, cat_ids=[c]) for c in ev.params.catIds}
