"""Baseline policies on cached per-level outputs (PLAN §6). Step 2 implements the
fixed and oracle policies; the matched-random Markov baseline comes in Step 3.

Levels are ordered cheapest -> most expensive, e.g. ("int8", "fp16").
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _stack(f1: Mapping[str, Sequence[float]], levels: Sequence[str]) -> np.ndarray:
    arrs = [np.asarray(f1[lv], float) for lv in levels]
    n = {a.shape for a in arrs}
    if len(n) != 1 or arrs[0].ndim != 1:
        raise ValueError(f"per-level F1 arrays must be 1-D and equal length, got {n}")
    if np.isnan(np.stack(arrs)).any():
        raise ValueError("NaN in per-frame F1")
    return np.stack(arrs)  # L x T


def oracle_cheap(f1: Mapping[str, Sequence[float]], levels: Sequence[str], reference: str = "fp16",
                 eps: float = 0.0) -> np.ndarray:
    """Per frame, the cheapest level whose F1 >= F1_reference - eps (PLAN §6.2).
    Returns level indices into `levels`. The reference level always qualifies."""
    if reference not in levels:
        raise ValueError(f"reference {reference!r} not in levels {levels}")
    m = _stack(f1, levels)
    ref = m[list(levels).index(reference)]
    ok = m >= ref[None, :] - eps
    return np.argmax(ok, axis=0)  # first True = cheapest qualifying


def oracle_best(f1: Mapping[str, Sequence[float]], levels: Sequence[str]) -> np.ndarray:
    """Per frame argmax F1; ties go to the cheapest level (PLAN §6.3)."""
    m = _stack(f1, levels)
    return np.argmax(m == m.max(axis=0, keepdims=True), axis=0)


def fixed(level: str, levels: Sequence[str], n: int) -> np.ndarray:
    return np.full(n, list(levels).index(level), dtype=int)


def select_dets(choice: Sequence[int], image_ids: Sequence[int], levels: Sequence[str],
                dets: Mapping[str, Mapping[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """Assemble one detection set: frame t takes the detections of levels[choice[t]]."""
    if len(choice) != len(image_ids):
        raise ValueError("choice and image_ids length differ")
    out = {}
    for c, iid in zip(choice, image_ids, strict=True):
        out[int(iid)] = dets[levels[int(c)]].get(int(iid), np.zeros((0, 6)))
    return out


def level_fractions(choice: Sequence[int], levels: Sequence[str]) -> dict[str, float]:
    c = np.asarray(choice)
    return {lv: float(np.mean(c == i)) if len(c) else 0.0 for i, lv in enumerate(levels)}


def switch_rate_per_100(choice: Sequence[int], seq_ids: Sequence[str]) -> float:
    """Switches per 100 frames, never counting across sequence boundaries."""
    c = np.asarray(choice)
    s = np.asarray(seq_ids)
    if len(c) < 2:
        return 0.0
    same_seq = s[1:] == s[:-1]
    switches = int(np.sum((c[1:] != c[:-1]) & same_seq))
    return 100.0 * switches / len(c)
