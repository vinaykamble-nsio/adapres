"""Objective frame-level slice labels (PLAN §4.3). All thresholds are fit on calib
frames only and then applied unchanged to test."""

from __future__ import annotations

import numpy as np


def luma_bt601(rgb: np.ndarray) -> float:
    """Mean BT.601 luma of an HxWx3 uint8 RGB image."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {rgb.shape}")
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    return float((0.299 * r + 0.587 * g + 0.114 * b).mean())


def otsu_threshold(values: np.ndarray, bins: int = 256, value_range: tuple[float, float] = (0.0, 255.0)) -> float:
    """Otsu threshold on a 1-D sample; returns the bin edge maximising between-class variance."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("need at least 2 values for Otsu")
    hist, edges = np.histogram(values, bins=bins, range=value_range)
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1
    # With well-separated modes every threshold in the empty gap gives the same
    # partition (a plateau of maxima). Return the middle of the first plateau
    # instead of its left edge, so the threshold sits in the valley.
    best = between.max()
    k0 = int(np.argmax(between >= best - 1e-12 * max(abs(best), 1.0)))
    k1 = k0
    while k1 + 1 < len(between) and between[k1 + 1] >= best - 1e-12 * max(abs(best), 1.0):
        k1 += 1
    return float((edges[k0 + 1] + edges[k1 + 1]) / 2)


def fit_night_threshold(calib_luma: np.ndarray, calib_is_night: np.ndarray | None) -> dict:
    """If time-of-day labels exist, pick the luma threshold maximising agreement with
    `night` on calib frames; otherwise Otsu on calib luma. Returns the threshold, the
    method and (if labels exist) the calib agreement rate."""
    calib_luma = np.asarray(calib_luma, dtype=np.float64)
    if calib_is_night is not None:
        lab = np.asarray(calib_is_night, dtype=bool)
        if lab.shape != calib_luma.shape:
            raise ValueError("calib_is_night shape mismatch")
        if lab.any() and (~lab).any():
            cands = np.unique(np.round(calib_luma, 2))
            # threshold t: night iff luma < t; test midpoints plus extremes
            mids = np.concatenate([[cands[0]], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-6]])
            acc = np.array([np.mean((calib_luma < t) == lab) for t in mids])
            best = int(np.argmax(acc))  # first max -> lowest threshold on ties
            return {"tau_night": float(mids[best]), "method": "max_agreement_timeofday",
                    "calib_agreement": float(acc[best])}
    t = otsu_threshold(calib_luma)
    out = {"tau_night": t, "method": "otsu_calib_luma", "calib_agreement": None}
    if calib_is_night is not None:
        lab = np.asarray(calib_is_night, dtype=bool)
        out["calib_agreement"] = float(np.mean((calib_luma < t) == lab))
    return out


def fit_count_thresholds(calib_counts: np.ndarray, crowded_percentile: float) -> dict:
    c = np.asarray(calib_counts, dtype=np.float64)
    if c.size == 0:
        raise ValueError("no calib frames")
    return {"n_crowd": float(np.percentile(c, crowded_percentile)),
            "easy_max_count": float(np.median(c)),
            "crowded_percentile": crowded_percentile}


def natural_scene_change(track_ids_per_frame: list[set], new_ratio: float, min_tracks: int) -> np.ndarray:
    """Within one sequence: 1 if |IDs_t \\ IDs_{t-1}| / max(|IDs_t|,1) >= new_ratio and
    |IDs_t| >= min_tracks. First frame is 0."""
    out = np.zeros(len(track_ids_per_frame), dtype=bool)
    for t in range(1, len(track_ids_per_frame)):
        cur, prev = track_ids_per_frame[t], track_ids_per_frame[t - 1]
        if len(cur) >= min_tracks and len(cur - prev) / max(len(cur), 1) >= new_ratio:
            out[t] = True
    return out


def label_frames(luma: np.ndarray, gt_count: np.ndarray, scene_change: np.ndarray,
                 natural_sc: np.ndarray, thr: dict, easy_excludes_natural: bool) -> dict[str, np.ndarray]:
    luma = np.asarray(luma, dtype=np.float64)
    cnt = np.asarray(gt_count, dtype=np.float64)
    sc = np.asarray(scene_change, dtype=bool)
    nsc = np.asarray(natural_sc, dtype=bool)
    night = luma < thr["tau_night"]
    crowded = cnt >= thr["n_crowd"]
    easy = ~night & ~crowded & ~sc & (cnt <= thr["easy_max_count"])
    if easy_excludes_natural:
        easy &= ~nsc
    return {"slice_night": night, "slice_crowded": crowded, "slice_scene_change": sc,
            "slice_scene_change_natural": nsc, "slice_easy": easy}
