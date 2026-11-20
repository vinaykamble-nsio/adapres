"""Per-frame raw cues (PLAN §3.1), causal.

Image-only cues (depend only on frames t and t-1, not on the policy) are computed once
per frame and cached:  m_t (motion), c_t (clutter), d_t (HSV Bhattacharyya distance; the
binary scene-change cue is s_t = 1[d_t >= theta_sc], applied by the controller).
Detection cues come from the detections of the level actually used at t-1 and are
therefore computed during replay/live:  n_t (count), u_t (uncertainty).
"""

from __future__ import annotations

import numpy as np

from .data.classmap import COCO_TO_EVAL


class FrameFeaturizer:
    """Stateful over one sequence: call reset() at each sequence start."""

    def __init__(self, width: int = 320, canny_low: int = 100, canny_high: int = 200, hist_bins: int = 32):
        import cv2  # lazy: Kaggle image or local venv

        self.cv2 = cv2
        self.width, self.lo, self.hi, self.bins = width, canny_low, canny_high, hist_bins
        self.reset()

    def reset(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.prev_hist: np.ndarray | None = None

    def __call__(self, bgr: np.ndarray) -> dict:
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (self.width, max(1, round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.lo, self.hi)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [self.bins, self.bins], [0, 180, 0, 256])
        hist = hist / max(float(hist.sum()), 1.0)
        if self.prev_gray is None:
            m, d = 0.0, 0.0  # first frame of a sequence (PLAN §3.1)
        else:
            m = float(np.mean(np.abs(gray.astype(np.int16) - self.prev_gray.astype(np.int16))) / 255.0)
            d = float(cv2.compareHist(hist.astype(np.float32), self.prev_hist.astype(np.float32),
                                      cv2.HISTCMP_BHATTACHARYYA))
        self.prev_gray, self.prev_hist = gray, hist
        return {"m": m, "c": float(np.count_nonzero(edges)) / edges.size, "d": d}


def detection_cues(dets: np.ndarray | None, tau_c: float, u_empty: float) -> tuple[float, float]:
    """(n, u) from the previous frame's detections of the level actually used:
    n = number of evaluated-class detections with score >= tau_c; u = 1 - mean score (u_empty if none).
    None (no previous frame) -> (0, u_empty)."""
    if dets is None or len(dets) == 0:
        return 0.0, float(u_empty)
    d = np.asarray(dets, float).reshape(-1, 6)
    keep = (d[:, 4] >= tau_c) & np.array([COCO_TO_EVAL.get(int(c)) is not None for c in d[:, 5]], bool)
    if not keep.any():
        return 0.0, float(u_empty)
    return float(keep.sum()), float(1.0 - d[keep, 4].mean())
