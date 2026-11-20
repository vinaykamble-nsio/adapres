"""Spliced scene-change streams built from test sequences (PLAN §4.4)."""

from __future__ import annotations

import numpy as np


class SpliceError(ValueError):
    pass


def build_streams(seq_frames: dict[str, list[int]], seq_tod: dict[str, str], n_streams: int,
                  segment_len: int, min_segments: int, max_segments: int, window: int,
                  seed: int) -> list[dict]:
    """Each stream concatenates 2-3 segments of `segment_len` consecutive frames from
    distinct test sequences, preferring a different time-of-day from the previous
    segment so day<->night cuts occur. Returns rows:
    {stream_id, pos, seq, frame_idx, segment, is_splice_window}.
    The first `window` frames of every segment after the first are scene-change frames.
    """
    eligible = sorted(s for s, fr in seq_frames.items() if len(fr) >= segment_len)
    if len(eligible) < max_segments:
        raise SpliceError(f"need >= {max_segments} sequences with >= {segment_len} frames, have {len(eligible)}")
    if window > segment_len:
        raise SpliceError("window longer than segment")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for sid in range(n_streams):
        n_seg = int(rng.integers(min_segments, max_segments + 1))
        used: list[str] = []
        pos = 0
        for k in range(n_seg):
            avail = [s for s in eligible if s not in used]
            if used:
                diff = [s for s in avail if seq_tod.get(s) != seq_tod.get(used[-1])]
                if diff:
                    avail = diff
            seq = avail[int(rng.integers(len(avail)))]
            used.append(seq)
            frames = sorted(seq_frames[seq])
            start = int(rng.integers(0, len(frames) - segment_len + 1))
            for j, fidx in enumerate(frames[start:start + segment_len]):
                rows.append({"stream_id": f"splice_{sid:03d}", "pos": pos, "seq": seq,
                             "frame_idx": fidx, "segment": k,
                             "is_splice_window": int(k > 0 and j < window)})
                pos += 1
    return rows
