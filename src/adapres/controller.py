"""Content-aware precision controller (PLAN §3.2-3.4). Pure and causal: the same
Controller.step() is used by replay.py (on cached detections) and by live runs.

Per frame t (cues are raw, see features.py):
  x~ = clip((x - p5_x) / (p95_x - p5_x), 0, 1)              for x in m, c, n, u   (norm fit on calib)
  s  = 1[d >= theta_sc]
  S  = w_m m~ + w_c c~ + w_n n~ + w_u u~ + w_s s
  S^ = alpha S + (1 - alpha) S^_{t-1}      (S^_0 = S_0)
  S' = S^ - w_h h
  proposal L* = cheaper level if S' < theta else FP16       (two levels; ordered cheap -> expensive)
Persistence (state L_cur, cand, k):
  L* == L_cur              -> k = 0, cand = None
  L* == cand               -> k += 1
  else                     -> cand = L*, k = 1
  switch to cand when k >= K_up (cand more expensive) or k >= K_down (cand cheaper); then k = 0.
Scene-change override (if enabled): s == 1 -> L_cur = most expensive level on this frame,
  k = 0, cand = None, and the next H frames stay there regardless of proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CUES = ("m", "c", "n", "u")


@dataclass
class ControllerParams:
    weights: dict[str, float]
    theta: float
    norm: dict[str, tuple[float, float]]             # cue -> (p_lo, p_hi), fit on calib
    levels: tuple[str, ...] = ("int8", "fp16")
    start_level: str = "fp16"
    alpha: float = 0.6
    K_up: int = 2
    K_down: int = 2
    H: int = 3
    theta_sc: float = 0.4
    scene_override: bool = True
    w_h: float = 0.0

    def __post_init__(self):
        if len(self.levels) != 2:
            raise ValueError("two-level controller (INT8/FP16) is the must-have; got levels=%r" % (self.levels,))
        if self.start_level not in self.levels:
            raise ValueError("start_level not in levels")
        if set(self.weights) != {"m", "c", "n", "u", "s"} or any(v < 0 for v in self.weights.values()):
            raise ValueError(f"bad weights {self.weights}")
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1")
        if not 0 < self.alpha <= 1 or self.K_up < 1 or self.K_down < 1 or self.H < 0:
            raise ValueError("alpha in (0,1], K_up/K_down >= 1, H >= 0")
        for k in CUES:
            lo, hi = self.norm[k]
            if not hi > lo:
                raise ValueError(f"degenerate normalisation for {k}: {lo}, {hi}")

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def normalise(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def score(cues: dict, p: ControllerParams) -> tuple[float, int]:
    """Instantaneous S_t and the binary scene-change cue s_t."""
    s = int(cues["d"] >= p.theta_sc)
    val = sum(p.weights[k] * normalise(cues[k], *p.norm[k]) for k in CUES) + p.weights["s"] * s
    return float(val), s


@dataclass
class Controller:
    p: ControllerParams
    state: dict = field(default_factory=dict)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.state = {"L": self.p.start_level, "cand": None, "k": 0, "S_hat": None, "hold": 0}

    def step(self, cues: dict, h: float = 0.0) -> dict:
        """cues: {'m','c','d','n','u'} raw. Returns a log dict with the chosen 'level'."""
        p, st = self.p, self.state
        lv_cheap, lv_top = p.levels[0], p.levels[-1]
        S, s = score(cues, p)
        st["S_hat"] = S if st["S_hat"] is None else p.alpha * S + (1 - p.alpha) * st["S_hat"]
        S_prime = st["S_hat"] - p.w_h * h
        proposal = lv_cheap if S_prime < p.theta else lv_top
        prev = st["L"]
        overridden = False
        if p.scene_override and s == 1:
            st.update(L=lv_top, cand=None, k=0, hold=p.H)
            overridden = True
        elif st["hold"] > 0:
            st["hold"] -= 1
            st.update(cand=None, k=0)  # held at FP16; proposals do not accumulate
        else:
            if proposal == st["L"]:
                st.update(k=0, cand=None)
            elif proposal == st["cand"]:
                st["k"] += 1
            else:
                st.update(cand=proposal, k=1)
            if st["cand"] is not None:
                up = p.levels.index(st["cand"]) > p.levels.index(st["L"])
                if st["k"] >= (p.K_up if up else p.K_down):
                    st.update(L=st["cand"], cand=None, k=0)
        return {"level": st["L"], "switched": int(st["L"] != prev), "proposal": proposal, "score_S": S,
                "score_hat": st["S_hat"], "s": s, "override": int(overridden)}


def params_from_config(c: dict, theta: float, norm: dict[str, tuple[float, float]]) -> ControllerParams:
    K = c["K"]
    return ControllerParams(weights=dict(c["weights"]), theta=float(theta), norm=norm, levels=tuple(c["levels"]),
                            start_level=c["start_level"], alpha=float(c["alpha"]),
                            K_up=int(c.get("K_up") or K), K_down=int(c.get("K_down") or K), H=int(c["H"]),
                            theta_sc=float(c["theta_sc"]), scene_override=bool(c["scene_override"]),
                            w_h=float(c["w_h"]))
