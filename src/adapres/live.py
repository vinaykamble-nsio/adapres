"""Live (online) adaptive inference: the same Controller.step() as replay, driving real engines.

Two modes (user decision 2026-11-20, report both):
  sequential  - frame t: compute image cues, controller step, run the chosen engine.
  pipelined   - a CPU worker computes the image cues of frame t+1 while the GPU runs frame t.
                The decision for frame t still uses only frames <= t and detections of t-1
                (identical decisions to sequential; only the timing differs).
Image decoding is outside both (frames are pre-decoded), as in PLAN §7.3.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Mapping, Sequence

import numpy as np

from .controller import Controller, ControllerParams
from .features import detection_cues


class AdaptiveStepper:
    """Steps through a list of pre-decoded frames grouped in sequences; one call = one frame.
    Controller and featurizer state are reset at each sequence start."""

    def __init__(self, backends: Mapping[str, object], params: ControllerParams, featurizer_factory: Callable,
                 frames: Sequence[np.ndarray], seq_starts: Sequence[int], tau_c: float, u_empty: float,
                 pipelined: bool):
        self.be, self.p, self.ff = backends, params, featurizer_factory
        self.frames, self.starts = frames, set(int(s) for s in seq_starts)
        self.tau_c, self.u_empty, self.pipelined = tau_c, u_empty, pipelined
        self.i = 0
        self.ex = ThreadPoolExecutor(max_workers=1) if pipelined else None
        self._fut = None
        self._fz = None

    def close(self):
        if self.ex:
            self.ex.shutdown(wait=True)

    def _start_seq(self):
        self.ctl = Controller(self.p)
        self._fz = self.ff()
        self.prev = None

    def step(self) -> dict:
        n = len(self.frames)
        i = self.i % n
        t0 = time.perf_counter()
        if i in self.starts or self._fz is None:
            if self._fut is not None:
                self._fut.result()  # drain a pending look-ahead of the previous sequence
                self._fut = None
            self._start_seq()
        if self.pipelined:
            if self._fut is None:
                self._fut = self.ex.submit(self._fz, self.frames[i])
            img_cues = self._fut.result()
            nxt = (i + 1) % n
            # look ahead only within the same sequence (the featurizer is stateful)
            self._fut = self.ex.submit(self._fz, self.frames[nxt]) if nxt not in self.starts else None
        else:
            img_cues = self._fz(self.frames[i])
        t1 = time.perf_counter()
        nn, uu = detection_cues(self.prev, self.tau_c, self.u_empty) if self.prev is not None else (0.0, self.u_empty)
        out = self.ctl.step({**img_cues, "n": nn, "u": uu})
        t2 = time.perf_counter()
        dets, tim = self.be[out["level"]].predict(self.frames[i])
        t3 = time.perf_counter()
        self.prev = dets
        self.i += 1
        return {"frame": i, "level": out["level"], "switched": out["switched"], "dets": dets,
                "feat_wait_ms": (t1 - t0) * 1000, "ctrl_ms": (t2 - t1) * 1000, "infer_wall_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - t0) * 1000, "engine_infer_ms": tim.get("infer_ms")}


class V2Stepper:
    """Live controller v2 (learned risk, fixed budget). Same step contract as AdaptiveStepper.
    pipelined: image cues of frame t+1 are computed on a CPU worker while the GPU runs frame t."""

    def __init__(self, backends: Mapping[str, object], model, params, featurizer_factory: Callable,
                 frames: Sequence[np.ndarray], seq_starts: Sequence[int], img_area: float, tau_c: float,
                 pipelined: bool):
        from .risk import RiskController

        self.be, self.ff, self.frames = backends, featurizer_factory, frames
        self.starts, self.area, self.pipelined = set(int(s) for s in seq_starts), img_area, pipelined
        self.ctl = RiskController(model, params, tau_c)
        self.ex = ThreadPoolExecutor(max_workers=1) if pipelined else None
        self.i, self._fut, self._fz = 0, None, None

    def close(self):
        if self.ex:
            self.ex.shutdown(wait=True)

    def step(self) -> dict:
        n = len(self.frames)
        i = self.i % n
        t0 = time.perf_counter()
        if i in self.starts or self._fz is None:
            if self._fut is not None:
                self._fut.result()
                self._fut = None
            self.ctl.reset()
            self._fz = self.ff()
        if self.pipelined:
            if self._fut is None:
                self._fut = self.ex.submit(self._fz, self.frames[i])
            img_cues = self._fut.result()
            nxt = (i + 1) % n
            self._fut = self.ex.submit(self._fz, self.frames[nxt]) if nxt not in self.starts else None
        else:
            img_cues = self._fz(self.frames[i])
        t1 = time.perf_counter()
        out = self.ctl.step(img_cues)
        t2 = time.perf_counter()
        dets, tim = self.be[out["level"]].predict(self.frames[i])
        t3 = time.perf_counter()
        self.ctl.observe(dets, self.area)
        self.i += 1
        return {"frame": i, "level": out["level"], "switched": out["switched"], "dets": dets, "risk": out["risk"],
                "feat_wait_ms": (t1 - t0) * 1000, "ctrl_ms": (t2 - t1) * 1000, "infer_wall_ms": (t3 - t2) * 1000,
                "total_ms": (time.perf_counter() - t0) * 1000, "engine_infer_ms": tim.get("infer_ms")}


class QRouteStepper:
    """Live Q-ROUTE: features of the previous detections (+ image cues if gated in), predicted regret,
    hardware-aware choice; the measured wall time of each inference feeds the slowdown EMA."""

    def __init__(self, backends: Mapping[str, object], model, params, featurizer_factory: Callable | None,
                 frames: Sequence[np.ndarray], seq_starts: Sequence[int], img_area: float, tau_c: float):
        from .qroute import QRouteController

        self.be, self.frames, self.area = backends, frames, img_area
        self.starts = set(int(s) for s in seq_starts)
        self.ff = featurizer_factory if params.use_img else None
        self.ctl = QRouteController(model, params, tau_c)
        self.i, self._fz = 0, None

    def close(self):
        pass

    def step(self) -> dict:
        n = len(self.frames)
        i = self.i % n
        t0 = time.perf_counter()
        if i in self.starts or self.i == 0:
            self.ctl.reset()
            self._fz = self.ff() if self.ff else None
        cues = self._fz(self.frames[i]) if self._fz else None
        t1 = time.perf_counter()
        out = self.ctl.step(cues)
        t2 = time.perf_counter()
        dets, tim = self.be[out["config"]].predict(self.frames[i])
        t3 = time.perf_counter()
        wall = (t3 - t2) * 1000
        self.ctl.observe(dets, self.area, observed_cost=wall)
        self.i += 1
        return {"frame": i, "level": out["config"], "switched": out["switched"], "dets": dets,
                "feat_wait_ms": (t1 - t0) * 1000, "ctrl_ms": (t2 - t1) * 1000, "infer_wall_ms": wall,
                "total_ms": (time.perf_counter() - t0) * 1000, "engine_infer_ms": tim.get("infer_ms"),
                "slowdown": out["slowdown"]}


class AlternatingStepper:
    """Forced switching every frame (INT8, FP16, INT8, ...): the worst-case switching cost."""

    def __init__(self, backends: Mapping[str, object], frames: Sequence[np.ndarray]):
        self.be, self.frames, self.i = backends, frames, 0

    def close(self):
        pass

    def step(self) -> dict:
        i = self.i % len(self.frames)
        lv = "int8" if self.i % 2 == 0 else "fp16"
        t0 = time.perf_counter()
        dets, tim = self.be[lv].predict(self.frames[i])
        self.i += 1
        return {"frame": i, "level": lv, "switched": 1, "dets": dets, "total_ms": (time.perf_counter() - t0) * 1000,
                "engine_infer_ms": tim.get("infer_ms")}


class FixedStepper:
    def __init__(self, backend, frames: Sequence[np.ndarray]):
        self.be, self.frames, self.i = backend, frames, 0

    def close(self):
        pass

    def step(self) -> dict:
        i = self.i % len(self.frames)
        t0 = time.perf_counter()
        dets, tim = self.be.predict(self.frames[i])
        self.i += 1
        return {"frame": i, "dets": dets, "total_ms": (time.perf_counter() - t0) * 1000,
                "engine_infer_ms": tim.get("infer_ms")}


def run_live_sequences(backends, params, featurizer_factory, frames, seq_starts, tau_c, u_empty,
                       pipelined: bool) -> list[dict]:
    """One pass over all frames (live validation): returns one row per frame."""
    st = AdaptiveStepper(backends, params, featurizer_factory, frames, seq_starts, tau_c, u_empty, pipelined)
    try:
        return [st.step() for _ in range(len(frames))]
    finally:
        st.close()
