"""Deterministic, seeded, stratified, disjoint calib/test clip selection (PLAN §4.1,
deviation D1: both drawn from MOT val)."""

from __future__ import annotations

import hashlib
import json

import numpy as np

STRATA_ORDER = ("daytime", "night", "dawn_dusk")


class SubsetError(ValueError):
    pass


def select_disjoint_splits(seq_strata: dict[str, str], targets: dict[str, dict[str, int]],
                           seed: int, split_order: tuple[str, ...] = ("test", "calib")) -> dict:
    """Pick disjoint clip sets per split.

    seq_strata: sequence id -> stratum in STRATA_ORDER.
    targets: split -> {stratum: count}.
    Splits are filled in `split_order` (test first, so test gets its target mix when
    a stratum is short). If a stratum runs out, the shortfall is filled from the
    remaining pool (all strata, seeded order) and recorded in `shortfalls`.
    """
    bad = {s for s in seq_strata.values() if s not in STRATA_ORDER}
    if bad:
        raise SubsetError(f"unknown strata in input: {sorted(bad)}")
    need = sum(sum(t.values()) for t in targets.values())
    if need > len(seq_strata):
        raise SubsetError(f"need {need} sequences but only {len(seq_strata)} available")

    rng = np.random.default_rng(seed)
    pools: dict[str, list[str]] = {}
    for st in STRATA_ORDER:
        seqs = sorted(s for s, v in seq_strata.items() if v == st)
        pools[st] = [seqs[i] for i in rng.permutation(len(seqs))]

    chosen: dict[str, list[str]] = {sp: [] for sp in targets}
    shortfalls: dict[str, dict[str, int]] = {sp: {} for sp in targets}
    for sp in split_order:
        for st in STRATA_ORDER:
            want = int(targets[sp].get(st, 0))
            take = pools[st][:want]
            pools[st] = pools[st][want:]
            chosen[sp].extend(take)
            if len(take) < want:
                shortfalls[sp][st] = want - len(take)
    # Fill shortfalls from what is left, in a seeded order over all strata.
    for sp in split_order:
        missing = sum(shortfalls[sp].values())
        if missing:
            rest = sorted(s for st in STRATA_ORDER for s in pools[st])
            rest = [rest[i] for i in rng.permutation(len(rest))]
            fill = rest[:missing]
            for s in fill:
                pools[seq_strata[s]].remove(s)
            chosen[sp].extend(fill)

    for sp in chosen:
        chosen[sp] = sorted(chosen[sp])
    all_sel = [s for sp in chosen for s in chosen[sp]]
    if len(all_sel) != len(set(all_sel)):
        raise SubsetError("internal error: splits overlap")

    counts = {sp: {st: sum(seq_strata[s] == st for s in chosen[sp]) for st in STRATA_ORDER} for sp in chosen}
    return {"seed": seed, "splits": chosen, "strata_counts": counts,
            "targets": targets, "shortfalls": shortfalls}


def sample_frames(frame_keys: list[tuple[str, int]], n: int, seed: int) -> list[tuple[str, int]]:
    """Seeded uniform sample without replacement (e.g. INT8 calibration frames)."""
    keys = sorted(frame_keys)
    if n > len(keys):
        raise SubsetError(f"asked for {n} frames, only {len(keys)} available")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(keys), size=n, replace=False))
    return [keys[i] for i in idx]


def selection_sha256(selection: dict) -> str:
    return hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
