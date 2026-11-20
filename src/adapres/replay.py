"""Causal replay of the controller on cached per-level detections (PLAN §2.1, §2.3).

At frame t the controller sees the image cues of frame t (m, c, d: computed from frames t
and t-1, available before inference) and the detection cues (n, u) of the detections the
*chosen* level produced at t-1, taken from that level's cache. Exactly the same
Controller.step() is used live.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from .controller import Controller, ControllerParams, score
from .features import detection_cues


def replay_sequence(params: ControllerParams, image_ids: Sequence[int], image_cues: Mapping[int, dict],
                    dets: Mapping[str, Mapping[int, np.ndarray]], tau_c: float, u_empty: float) -> list[dict]:
    """Returns one log row per frame (in order) with the chosen level and all cue/score values."""
    ctl = Controller(params)
    rows = []
    prev_dets = None
    for t, iid in enumerate(image_ids):
        n, u = detection_cues(prev_dets, tau_c, u_empty) if t > 0 else (0.0, float(u_empty))
        cues = {**{k: image_cues[iid][k] for k in ("m", "c", "d")}, "n": n, "u": u}
        t0 = time.perf_counter()
        out = ctl.step(cues)
        ctrl_ms = (time.perf_counter() - t0) * 1000
        rows.append({"image_id": int(iid), "t": t, **cues, **out, "latency_ctrl_step_ms": ctrl_ms})
        prev_dets = dets[out["level"]].get(int(iid), np.zeros((0, 6)))
    return rows


def replay(params: ControllerParams, seq_frames: Mapping[str, Sequence[int]], image_cues: Mapping[int, dict],
           dets: Mapping[str, Mapping[int, np.ndarray]], tau_c: float, u_empty: float) -> list[dict]:
    """All sequences (controller state reset per sequence), rows tagged with 'seq'."""
    rows = []
    for seq in sorted(seq_frames):
        for r in replay_sequence(params, seq_frames[seq], image_cues, dets, tau_c, u_empty):
            rows.append({"seq": seq, **r})
    return rows


def open_loop_scores(params: ControllerParams, seq_frames: Mapping[str, Sequence[int]], image_cues: Mapping[int, dict],
                     ref_dets: Mapping[int, np.ndarray], tau_c: float, u_empty: float) -> np.ndarray:
    """S_hat per frame with detection cues from a fixed reference level (FP16), used only to
    place theta at a quantile (PLAN §3.5 'theta in quantiles of S_hat on calib')."""
    out = []
    for seq in sorted(seq_frames):
        s_hat = None
        prev = None
        for t, iid in enumerate(seq_frames[seq]):
            n, u = detection_cues(prev, tau_c, u_empty) if t > 0 else (0.0, float(u_empty))
            S, _ = score({**image_cues[iid], "n": n, "u": u}, params)
            s_hat = S if s_hat is None else params.alpha * S + (1 - params.alpha) * s_hat
            out.append(s_hat)
            prev = ref_dets.get(int(iid), np.zeros((0, 6)))
    return np.asarray(out)


def fit_norm(values: Mapping[str, Sequence[float]], percentiles: Sequence[float]) -> dict[str, tuple[float, float]]:
    """(p_lo, p_hi) per cue from calib values. Raises if degenerate (p_hi <= p_lo)."""
    lo_q, hi_q = percentiles
    out = {}
    for k, v in values.items():
        a = np.asarray(v, float)
        lo, hi = float(np.percentile(a, lo_q)), float(np.percentile(a, hi_q))
        if not hi > lo:
            raise ValueError(f"degenerate calib distribution for cue {k}: p{lo_q}={lo}, p{hi_q}={hi}")
        out[k] = (lo, hi)
    return out
