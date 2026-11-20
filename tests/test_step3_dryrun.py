"""DRY-RUN of notebook 03 on top of the fake Step 2 run (synthetic images, real OpenCV
features, fake detections). No GPU, no real data."""

import csv
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest
from test_step2_dryrun import _cfg, _run_all, _step, repo  # noqa: F401

from adapres.pipeline import controller_dev as CD


def fake_image(path):
    """Deterministic synthetic BGR frame: per-sequence base colour + per-frame texture."""
    p = Path(str(path))
    rng = np.random.default_rng(zlib.crc32(p.name.encode()))
    base = zlib.crc32(p.parent.name.encode()) % 200
    img = np.full((36, 64, 3), base, np.uint8)
    img[:, : int(rng.integers(8, 56))] = 255 - base
    for _ in range(int(rng.integers(0, 6))):       # varying number of blocks -> varying clutter
        y, x = int(rng.integers(0, 30)), int(rng.integers(0, 58))
        img[y:y + 6, x:x + 6] = int(rng.integers(0, 256))
    return img


@pytest.fixture
def step3_env(tmp_path, repo):  # noqa: F811
    s2 = _step(tmp_path, repo, smoke=False)
    _run_all(s2, tmp_path)
    res = tmp_path / "work" / "results"
    for d in ("calib_fixed_fp16", "calib_fixed_int8", "calib_oracle_cheap_dev"):
        shutil.copytree(res / "bdd100k_mot" / d, repo / "results" / "bdd100k_mot" / d)
    shutil.copytree(res / "microbench", repo / "results" / "microbench")
    cache = tmp_path / "work" / "work02" / "cache" / "yolo11s"
    return tmp_path, repo, cache


def _make(tmp_path, repo_dir, smoke=False):
    cfg = _cfg(tmp_path)
    cfg["features"]["width"] = 32
    cfg["paths"]["work_root"] = str(tmp_path / "work3")
    return CD.Step3(cfg, smoke, repo_dir, image_reader=fake_image)


def _run(st, cache):
    info = st.prepare()
    st.locate_cache(cache)
    vc = st.verify_cache()
    st.features()
    fit = st.fit()
    res = st.policies()
    san = st.sanity()
    return info, vc, fit, res, san


def test_step3_full_dryrun(step3_env):
    tmp_path, repo_dir, cache = step3_env
    st = _make(tmp_path, repo_dir)
    info, vc, fit, res, san = _run(st, cache)
    assert vc["fp16"]["mismatches"] == 0 and vc["int8"]["mismatches"] == 0
    assert san["all_pass"], san["checks"]
    names = {c["check"] for c in san["checks"]}
    assert any("composed eval == notebook 02" in n for n in names)          # cross-check really ran
    assert {"adaptive_K1", "adaptive_K2", "adaptive_K3", "adaptive_K5", "random_matched", "random_iid",
            "oracle_cheap", "fixed_fp16", "fixed_int8"} <= set(res)
    assert res["random_matched"]["all"]["n_seeds"] == 10
    sw = san["informational"]["switch_rate_by_K"]
    assert sw["adaptive_K1"] >= sw["adaptive_K5"]
    assert set(fit["norm"]) == {"m", "c", "n", "u"} and np.isfinite(fit["theta"])
    out = tmp_path / "work3" / "results"
    pf = list(csv.DictReader(open(out / "bdd100k_mot" / "calib_adaptive_K2_dev" / "per_frame.csv")))
    assert len(pf) == info["n_frames"] and {"score_hat", "override", "proposal", "f1_05"} <= set(pf[0])
    idx = {r["exp_id"] for r in csv.DictReader(open(out / "index.csv"))}
    assert {"calib_features_yolo11s", "calib_adaptive_K2_dev", "calib_random_matched_dev", "sanity_03_yolo11s",
            "calib_fixed_fp16_composed_dev", "calib_fixed_int8_composed_dev", "calib_oracle_cheap_composed_dev"} <= idx
    # notebook 02's experiment ids are never written by notebook 03
    assert not {"calib_oracle_cheap_dev", "calib_fixed_fp16", "calib_fixed_int8"} & idx - _committed(repo_dir)
    assert not (out / "bdd100k_mot" / "calib_oracle_cheap_dev").exists()
    meta = json.loads((out / "bdd100k_mot" / "calib_adaptive_K2_dev" / "meta.json").read_text())
    assert meta["split"] == "calib" and meta["mode"] == "replay" and meta["cache_commit"] == "dryrun0002"
    # no test-split frames anywhere
    test_ids = {r["image_id"] for r in csv.DictReader(open(repo_dir / "results/bdd100k_mot/bdd_subsets/frames_test.csv"))}
    assert not test_ids & {r["image_id"] for r in pf}


def _committed(repo_dir):
    return {r["exp_id"] for r in csv.DictReader(open(repo_dir / "results" / "index.csv"))}


def test_refuses_to_overwrite_committed_experiment(step3_env):
    tmp_path, repo_dir, cache = step3_env
    idx = repo_dir / "results" / "index.csv"
    rows = list(csv.DictReader(open(idx)))
    rows.append({**rows[0], "exp_id": "calib_adaptive_K2_dev"})
    with open(idx, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    st = _make(tmp_path, repo_dir)
    st.prepare()
    st.locate_cache(cache)
    st.verify_cache()
    st.features()
    st.fit()
    with pytest.raises(CD.Step3Error, match="refusing to overwrite"):
        st.policies()


def test_features_checkpoint_resume(step3_env):
    tmp_path, repo_dir, cache = step3_env
    st = _make(tmp_path, repo_dir)
    st.prepare()
    st.features()
    first = dict(st.cues)
    calls = []
    st.image_reader = lambda p: calls.append(p) or fake_image(p)
    st.features()
    assert calls == [] and st.cues == first


def test_wrong_cache_is_rejected(step3_env):
    tmp_path, repo_dir, cache = step3_env
    bad = tmp_path / "badcache" / "cache" / "yolo11s"
    shutil.copytree(cache, bad)
    f = sorted((bad / "fp16").glob("*.npz"))[0]
    z = dict(np.load(f))
    z["dets"] = z["dets"][: len(z["dets"]) // 2]       # drop detections -> counts differ
    np.savez_compressed(f, **z)
    st = _make(tmp_path, repo_dir)
    st.prepare()
    st.locate_cache(bad)
    with pytest.raises(CD.Step3Error, match="does not match committed"):
        st.verify_cache()


def test_find_cache_root(tmp_path):
    (tmp_path / "inp" / "nb02out" / "work02" / "cache" / "yolo11s" / "fp16").mkdir(parents=True)
    (tmp_path / "inp" / "nb02out" / "work02" / "cache" / "yolo11s" / "int8").mkdir(parents=True)
    (tmp_path / "inp" / "ds" / "track_images_val" / "cache" / "yolo11s" / "fp16").mkdir(parents=True)  # pruned
    assert CD.find_cache_root(tmp_path / "inp", "yolo11s") == tmp_path / "inp" / "nb02out" / "work02" / "cache" / "yolo11s"
    with pytest.raises(CD.Step3Error, match="attach notebook 02"):
        CD.find_cache_root(tmp_path / "inp", "yolo11m")
