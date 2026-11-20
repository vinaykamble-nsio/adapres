"""Oracle + fixed-policy unit tests on synthetic per-frame F1 (CLAUDE.md: required)."""

import numpy as np
import pytest

from adapres import baselines as B

LV = ("int8", "fp16")


def test_oracle_cheap():
    f1 = {"int8": [1.0, 0.5, 0.8, 0.0, 0.7], "fp16": [1.0, 0.6, 0.8, 0.0, 0.6]}
    # int8 qualifies when f1_int8 >= f1_fp16 (eps=0): frames 0, 2, 3, 4
    assert B.oracle_cheap(f1, LV).tolist() == [0, 1, 0, 0, 0]
    # with eps=0.1 frame 1 also qualifies
    assert B.oracle_cheap(f1, LV, eps=0.1).tolist() == [0, 0, 0, 0, 0]


def test_oracle_best_ties_go_cheap():
    f1 = {"int8": [1.0, 0.5, 0.9], "fp16": [1.0, 0.6, 0.8]}
    assert B.oracle_best(f1, LV).tolist() == [0, 1, 0]


def test_oracle_three_levels():
    lv = ("int4", "int8", "fp16")
    f1 = {"int4": [0.9, 0.2, 0.5], "int8": [0.9, 0.7, 0.4], "fp16": [0.9, 0.7, 0.6]}
    assert B.oracle_cheap(f1, lv).tolist() == [0, 1, 2]
    assert B.oracle_best(f1, lv).tolist() == [0, 1, 2]


def test_oracle_cheap_never_worse_than_reference_per_frame():
    rng = np.random.default_rng(0)
    f1 = {"int8": rng.random(500), "fp16": rng.random(500)}
    c = B.oracle_cheap(f1, LV)
    chosen = np.where(c == 0, f1["int8"], f1["fp16"])
    assert np.all(chosen >= f1["fp16"])
    best = B.oracle_best(f1, LV)
    assert np.all(np.where(best == 0, f1["int8"], f1["fp16"]) >= chosen)


def test_oracle_input_validation():
    with pytest.raises(ValueError):
        B.oracle_cheap({"int8": [1.0], "fp16": [1.0, 0.5]}, LV)
    with pytest.raises(ValueError):
        B.oracle_cheap({"int8": [np.nan], "fp16": [1.0]}, LV)
    with pytest.raises(ValueError):
        B.oracle_cheap({"int8": [1.0]}, ("int8",), reference="fp16")


def test_select_dets_and_stats():
    dets = {"int8": {1: np.ones((1, 6)), 2: np.ones((2, 6))}, "fp16": {1: np.zeros((3, 6)), 2: np.zeros((4, 6))}}
    out = B.select_dets([1, 0], [1, 2], LV, dets)
    assert out[1].shape == (3, 6) and out[2].shape == (2, 6)
    assert B.level_fractions([0, 0, 1, 1], LV) == {"int8": 0.5, "fp16": 0.5}
    # switches inside a sequence count, across a boundary do not
    assert B.switch_rate_per_100([0, 1, 1, 0], ["a", "a", "b", "b"]) == pytest.approx(100 * 2 / 4)
    assert B.switch_rate_per_100([0, 1], ["a", "b"]) == 0.0
    assert B.fixed("fp16", LV, 3).tolist() == [1, 1, 1]
