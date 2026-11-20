import numpy as np
import pytest

from adapres.data import slices as SL


def test_luma_bt601():
    white = np.full((4, 4, 3), 255, np.uint8)
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 0] = 255
    assert SL.luma_bt601(white) == pytest.approx(255.0)
    assert SL.luma_bt601(red) == pytest.approx(0.299 * 255)
    with pytest.raises(ValueError):
        SL.luma_bt601(np.zeros((4, 4)))


def test_otsu_bimodal():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(40, 5, 500), rng.normal(160, 10, 500)])
    t = SL.otsu_threshold(v)
    assert 60 < t < 130


def test_night_threshold_from_labels():
    luma = np.array([20, 30, 40, 120, 130, 140], float)
    lab = np.array([1, 1, 1, 0, 0, 0], bool)
    out = SL.fit_night_threshold(luma, lab)
    assert out["method"] == "max_agreement_timeofday"
    assert out["calib_agreement"] == 1.0
    assert 40 < out["tau_night"] <= 120


def test_night_threshold_otsu_fallback():
    out = SL.fit_night_threshold(np.array([20, 25, 30, 150, 160, 170.0]), None)
    assert out["method"] == "otsu_calib_luma" and out["calib_agreement"] is None
    assert 30 < out["tau_night"] < 150


def test_count_thresholds():
    out = SL.fit_count_thresholds(np.arange(101), 90)
    assert out["n_crowd"] == 90 and out["easy_max_count"] == 50


def test_natural_scene_change():
    ids = [{1, 2, 3}, {1, 2, 3}, {7, 8, 9, 1}, {7, 8}]
    out = SL.natural_scene_change(ids, 0.5, 3)
    assert out.tolist() == [False, False, True, False]  # last frame: only 2 tracks


def test_label_frames_easy_is_exclusive():
    thr = {"tau_night": 50, "n_crowd": 10, "easy_max_count": 4}
    luma = np.array([100, 20, 100, 100, 100, 100])
    cnt = np.array([2, 2, 12, 2, 6, 2])
    sc = np.array([0, 0, 0, 1, 0, 0], bool)
    nsc = np.array([0, 0, 0, 0, 0, 1], bool)
    lab = SL.label_frames(luma, cnt, sc, nsc, thr, easy_excludes_natural=True)
    assert lab["slice_easy"].tolist() == [True, False, False, False, False, False]
    assert lab["slice_night"].tolist() == [False, True, False, False, False, False]
    assert lab["slice_crowded"].tolist() == [False, False, True, False, False, False]
    for k in ("slice_night", "slice_crowded", "slice_scene_change"):
        assert not (lab["slice_easy"] & lab[k]).any()
    lab2 = SL.label_frames(luma, cnt, sc, nsc, thr, easy_excludes_natural=False)
    assert lab2["slice_easy"][5]
