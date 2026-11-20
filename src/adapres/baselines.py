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


def estimate_markov(choices_by_seq: Mapping[str, Sequence[int]], n_levels: int) -> dict:
    """Transition matrix and initial distribution from a policy log (PLAN §6.1). Transitions
    are counted within sequences only. Rows with no observations fall back to 'stay'."""
    counts = np.zeros((n_levels, n_levels))
    init = np.zeros(n_levels)
    for c in choices_by_seq.values():
        c = np.asarray(c, int)
        if len(c) == 0:
            continue
        init[c[0]] += 1
        np.add.at(counts, (c[:-1], c[1:]), 1)
    P = np.eye(n_levels)
    rows = counts.sum(1) > 0
    P[rows] = counts[rows] / counts[rows].sum(1, keepdims=True)
    return {"P": P, "init": init / max(init.sum(), 1), "counts": counts}


def markov_random(seq_lengths: Mapping[str, int], P: np.ndarray, init: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Matched random baseline: a Markov chain with the adaptive policy's transition matrix and
    initial distribution, so the precision mix and switch rate match in expectation and only
    the placement of switches differs. Deterministic given seed (sequences in sorted order)."""
    rng = np.random.default_rng(seed)
    out = {}
    for seq in sorted(seq_lengths):
        n = int(seq_lengths[seq])
        c = np.empty(n, int)
        if n:
            c[0] = rng.choice(len(init), p=init)
            for t in range(1, n):
                c[t] = rng.choice(P.shape[1], p=P[c[t - 1]])
        out[seq] = c
    return out


def iid_random(seq_lengths: Mapping[str, int], p_cheap: float, seed: int) -> dict[str, np.ndarray]:
    """Naive i.i.d. baseline: each frame is level 0 (cheapest) with probability p_cheap, else level 1."""
    rng = np.random.default_rng(seed)
    return {seq: (rng.random(int(seq_lengths[seq])) >= p_cheap).astype(int) for seq in sorted(seq_lengths)}


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
