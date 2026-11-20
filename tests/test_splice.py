import pytest

from adapres.data.splice import SpliceError, build_streams

SEQ = {f"s{i}": list(range(60)) for i in range(6)}
TOD = {"s0": "daytime", "s1": "night", "s2": "daytime", "s3": "night", "s4": "daytime", "s5": "dawn_dusk"}
KW = dict(n_streams=5, segment_len=40, min_segments=2, max_segments=3, window=5, seed=11)


def test_deterministic():
    assert build_streams(SEQ, TOD, **KW) == build_streams(SEQ, TOD, **KW)


def test_structure():
    rows = build_streams(SEQ, TOD, **KW)
    streams = {}
    for r in rows:
        streams.setdefault(r["stream_id"], []).append(r)
    assert len(streams) == 5
    for rs in streams.values():
        segs = sorted({r["segment"] for r in rs})
        assert 2 <= len(segs) <= 3
        assert [r["pos"] for r in rs] == list(range(len(rs)))
        assert len(rs) == 40 * len(segs)
        seqs = [next(r["seq"] for r in rs if r["segment"] == k) for k in segs]
        assert len(set(seqs)) == len(seqs)  # distinct source sequences
        # adjacent segments differ in time of day when possible (always possible here)
        assert all(TOD[a] != TOD[b] for a, b in zip(seqs, seqs[1:], strict=False))
        assert sum(r["is_splice_window"] for r in rs) == 5 * (len(segs) - 1)
        for k in segs:
            seg = [r for r in rs if r["segment"] == k]
            fidx = [r["frame_idx"] for r in seg]
            assert fidx == list(range(fidx[0], fidx[0] + 40))  # consecutive frames
            assert all(r["is_splice_window"] == int(k > 0 and j < 5) for j, r in enumerate(seg))


def test_errors():
    with pytest.raises(SpliceError):
        build_streams({"a": list(range(10))}, {}, **KW)
    with pytest.raises(SpliceError):
        build_streams(SEQ, TOD, **{**KW, "window": 41})
