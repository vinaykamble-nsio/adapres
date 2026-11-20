"""Features on tiny synthetic images, detection cues, causal replay, normalisation fit."""

import numpy as np
import pytest

from adapres.controller import ControllerParams
from adapres.features import FrameFeaturizer, detection_cues
from adapres.replay import fit_norm, open_loop_scores, replay, replay_sequence

CAR, DOG = 2, 16


def test_featurizer_synthetic():
    f = FrameFeaturizer(width=32)
    flat = np.full((18, 64, 3), 100, np.uint8)
    out0 = f(flat)
    assert out0 == {"m": 0.0, "c": 0.0, "d": 0.0}           # first frame; flat image has no edges
    out1 = f(flat.copy())
    assert out1["m"] == 0.0 and out1["d"] == pytest.approx(0.0, abs=1e-6)
    checker = np.zeros((18, 64, 3), np.uint8)
    checker[:, ::4] = 255
    out2 = f(checker)
    assert out2["m"] > 0 and out2["c"] > 0
    red = np.zeros((18, 64, 3), np.uint8)
    red[..., 2] = 255
    out3 = f(red)
    assert out3["d"] > 0.5                                   # strong colour change -> large distance
    f.reset()
    assert f(red)["m"] == 0.0


def test_detection_cues():
    d = np.array([[0, 0, 1, 1, 0.9, CAR], [0, 0, 1, 1, 0.5, CAR], [0, 0, 1, 1, 0.1, CAR], [0, 0, 1, 1, 0.99, DOG]])
    n, u = detection_cues(d, 0.25, 0.3)
    assert n == 2 and u == pytest.approx(1 - 0.7)
    assert detection_cues(None, 0.25, 0.3) == (0.0, 0.3)
    assert detection_cues(np.zeros((0, 6)), 0.25, 0.3) == (0.0, 0.3)


NORM = {"m": (0.0, 1.0), "c": (0.0, 1.0), "n": (0.0, 10.0), "u": (0.0, 1.0)}


def params(**kw):
    base = dict(weights={"m": 0, "c": 0, "n": 1.0, "u": 0, "s": 0}, theta=0.25, norm=NORM, alpha=1.0, K_up=1,
                K_down=1, H=0)
    base.update(kw)
    return ControllerParams(**base)


def _dets(k):
    return np.array([[0, 0, 1, 1, 0.9, CAR]] * k).reshape(-1, 6)


def test_replay_is_causal_and_uses_chosen_level_dets():
    """Count-only controller: fp16 sees 5 objects/frame, int8 sees 1. Starting at fp16 the
    controller sees n=5 (> 2.5) -> stays fp16. If replay wrongly used int8 dets it would see n=1."""
    ids = list(range(1, 7))
    cues = {i: {"m": 0, "c": 0, "d": 0} for i in ids}
    dets = {"fp16": {i: _dets(5) for i in ids}, "int8": {i: _dets(1) for i in ids}}
    rows = replay_sequence(params(), ids, cues, dets, 0.25, 0.0)
    assert rows[0]["n"] == 0 and rows[0]["level"] == "int8"        # frame 0: n=0 (no previous) -> int8
    assert rows[1]["n"] == 1                                         # uses int8 dets of frame 0
    assert [r["level"] for r in rows] == ["int8"] * 6
    # now int8 is the busy one: frame 1 sees n=5 from int8 -> fp16; then fp16 dets (n=1) -> int8 ...
    dets2 = {"fp16": {i: _dets(1) for i in ids}, "int8": {i: _dets(5) for i in ids}}
    lv = [r["level"] for r in replay_sequence(params(), ids, cues, dets2, 0.25, 0.0)]
    assert lv == ["int8", "fp16", "int8", "fp16", "int8", "fp16"]


def test_replay_resets_per_sequence():
    cues = {i: {"m": 0, "c": 0, "d": 0} for i in range(1, 5)}
    dets = {"fp16": {i: _dets(5) for i in cues}, "int8": {i: _dets(5) for i in cues}}
    rows = replay(params(), {"a": [1, 2], "b": [3, 4]}, cues, dets, 0.25, 0.0)
    assert [(r["seq"], r["t"], r["n"]) for r in rows] == [("a", 0, 0), ("a", 1, 5), ("b", 0, 0), ("b", 1, 5)]


def test_open_loop_scores_and_fit_norm():
    cues = {i: {"m": 0, "c": 0, "d": 0} for i in range(1, 4)}
    s = open_loop_scores(params(), {"a": [1, 2, 3]}, cues, {i: _dets(5) for i in cues}, 0.25, 0.0)
    assert s.tolist() == [0.0, 0.5, 0.5]
    nm = fit_norm({"x": np.arange(101)}, (5, 95))
    assert nm["x"] == (5.0, 95.0)
    with pytest.raises(ValueError):
        fit_norm({"x": [1, 1, 1]}, (5, 95))
