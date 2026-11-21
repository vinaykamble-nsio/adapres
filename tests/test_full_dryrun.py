"""DRY-RUN of notebook 04 (all six stages) with fake engines, fake NVML, synthetic frames.
No GPU, no real data. Checks wiring, gates, the test-split guard, and output records."""

import csv
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest
from test_energy_live import FakeNvml
from test_step2_dryrun import FakeBackend, _cfg, _run_all, _step, repo  # noqa: F401

from adapres import calibrate as CAL
from adapres.data import bdd_mot
from adapres.data.classmap import EVAL_CATEGORIES
from adapres.energy.nvml_meter import NvmlMeter
from adapres.pipeline import full as FULL

pytestmark = pytest.mark.slow  # end-to-end dry-run; run with: pytest -m slow

REG: dict[str, int] = {}
NAMES: list[str] = []


def tag_reader(path):
    """Synthetic BGR frame with the file identity encoded in two pixels (for the fake engines)."""
    p = Path(str(path))
    if p.name not in REG:
        REG[p.name] = len(NAMES)
        NAMES.append(p.name)
    k = REG[p.name]
    rng = np.random.default_rng(zlib.crc32(p.name.encode()))
    base = zlib.crc32(p.parent.name.encode()) % 200
    img = np.full((36, 64, 3), base, np.uint8)
    img[:, : int(rng.integers(8, 56))] = 255 - base
    for _ in range(int(rng.integers(0, 6))):
        y, x = int(rng.integers(1, 30)), int(rng.integers(2, 58))
        img[y:y + 6, x:x + 6] = int(rng.integers(0, 256))
    img[0, 0, 0], img[0, 1, 0] = k % 256, k // 256
    return img


class ArrayBackend(FakeBackend):
    """Fake engine on paths or tagged arrays: returns the GT boxes; with drop_every, misses
    roughly 1/drop_every of boxes (keyed by file name + box index, not always the first box)."""

    def predict(self, src):
        self.calls += 1
        name = NAMES[int(src[0, 0, 0]) + 256 * int(src[0, 1, 0])] if isinstance(src, np.ndarray) else Path(str(src)).name
        dets = []
        for k, (x, y, w, h, cid) in enumerate(self.gt.get(name, [])):
            if self.drop and (zlib.crc32(name.encode()) + 7 * k) % self.drop == 0:
                continue
            dets.append([x, y, x + w, y + h, 0.9 - 0.001 * k, EVAL_CATEGORIES[cid][1]])
        t = {"pre_ms": 1.0, "infer_ms": {"int8": 2.0, "fp16": 3.0}.get(self.name, 5.0), "post_ms": 0.5, "wall_ms": 9.0}
        return np.asarray(dets, float).reshape(-1, 6), t


def gt_by_name(input_root):
    lay = bdd_mot.discover_layout(input_root, "track_images_val", "box_track_labels_trainval", "val")
    out = {}
    for f in lay.label_dir.glob("*.json"):
        for fr in bdd_mot.load_video_labels(f):
            out[fr.name] = [(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1, b.eval_id) for b in fr.boxes
                            if b.kind == "eval" and not b.crowd]
    return out


INT8_DROP = 40   # fake INT8 misses every 40th box: a mild drop, within "slightly lower"


def _factory(gtn):
    def factory(name, path):
        return ArrayBackend(name, dict(gtn), drop_every=INT8_DROP if name == "int8" else 0)
    return factory


@pytest.fixture
def env(tmp_path, repo):  # noqa: F811
    gtn = gt_by_name(tmp_path / "input")
    s2 = _step(tmp_path, repo, smoke=False)          # reference "committed" Step 2 run, same fake engines
    s2.backend_factory = _factory(gtn)
    s2.image_reader = tag_reader
    _run_all(s2, tmp_path)
    res = tmp_path / "work" / "results"
    for d in ("calib_fixed_fp32", "calib_fixed_fp16", "calib_fixed_int8", "calib_oracle_cheap_dev"):
        shutil.copytree(res / "bdd100k_mot" / d, repo / "results" / "bdd100k_mot" / d)
    shutil.copytree(REPO_CONFIGS, repo / "configs")
    return tmp_path, repo, gtn


REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _full_cfg(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["paths"]["work_root"] = str(tmp_path / "w4")
    cfg["features"]["width"] = 32
    cfg["full"]["calib"].update(theta_quantiles=[0.3, 0.6], Ks=[1, 2], alphas=[1.0], theta_scs=[0.4],
                                simplex_step=0.5, stage_b_top=3, stage_b_max=6, pareto_deltas=[0.5, 2.0],
                                ablation_stage_b_top=2, ablation_stage_b_max=3, K_sweep=[1, 2])
    cfg["full"]["timing"].update(n_measure_clips=1, frames_per_clip=12, warmup_s=0.02, settle_s=0, idle_s=0.03,
                                 measure_s=0.06, repeats=1, switch_penalty_frames=6)
    cfg["full"]["test"].update(random_seeds=3, bootstrap_resamples=15)
    cfg["full"]["live_validation"].update(frac_clips=0.34)
    return cfg


def _make(tmp_path, repo_dir, gtn):
    factory = _factory(gtn)

    def exporter(weights, level, imgsz, out_dir, int8_data=None, batch=1):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"fake_{level}.engine"
        p.write_bytes(level.encode())
        return {"level": level, "path": str(p), "sha256": "fake-" + level, "bytes": 1000, "build_s": 0.0}

    nv = FakeNvml()
    return FULL.FullRun(_full_cfg(tmp_path), False, repo_dir, exporter=exporter, backend_factory=factory,
                        image_reader=tag_reader, meter_factory=lambda: NvmlMeter(nvml=nv, sample_hz=200),
                        release_fn=lambda: None)


def _run_to(fr, tmp_path, upto=6):
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    out = {1: fr.stage1_engines(w, {i: f"c{i}" for i in range(80)})}
    for k, fn in ((2, fr.stage2_calib_dev), (3, fr.stage3_calibrate), (4, fr.stage4_test_cache),
                  (5, fr.stage5_timing), (6, fr.stage6_test_eval)):
        if k > upto:
            break
        out[k] = fn()
    return out


def test_full_pipeline_dryrun(env):
    tmp_path, repo_dir, gtn = env
    fr = _make(tmp_path, repo_dir, gtn)
    out = _run_to(fr, tmp_path)
    assert all(s["pass"] for s in fr.stages.values()) and len(fr.stages) == 6
    res = tmp_path / "w4" / "results"
    idx = {r["exp_id"]: r for r in csv.DictReader(open(res / "index.csv"))}
    for e in ("r04_calib_fixed_fp16", "r04_calibration", "r04_test_adaptive_main", "r04_test_fixed_fp16",
              "r04_test_random_matched", "r04_test_oracle_cheap", "r04_test_bootstrap", "r04_test_hypotheses",
              "r04_test_energy_latency", "r04_test_adaptive_abl_A-m", "r04_test_adaptive_ksweep1"):
        assert e in idx, e
    assert "bdd_subsets" in idx                      # committed rows carried over
    assert not any(k.startswith("calib_") and k in ("calib_fixed_fp16",) and idx[k]["git_commit"] != "dryrun0002"
                   for k in idx)
    main = json.loads((res / "bdd100k_mot" / "r04_test_adaptive_main" / "meta.json").read_text())
    p, _, h = CAL.load_locked(res / "locked" / "locked_bdd_main.yaml")
    assert main["locked_config_sha256"] == h and main["split"] == "test"
    m = json.loads((res / "bdd100k_mot" / "r04_test_adaptive_main" / "metrics.json").read_text())
    assert {"all", "easy", "crowded", "night", "scene_change"} <= set(m)
    assert "map50_95_ci" in m["all"] and m["all"]["energy_kind"].startswith("estimated")
    hyp = json.loads((res / "bdd100k_mot" / "r04_test_hypotheses" / "hypotheses.json").read_text())
    assert set(hyp) >= {"H1", "H2", "H3", "H4"}
    en = json.loads((res / "bdd100k_mot" / "r04_test_energy_latency" / "energy.json").read_text())
    assert set(en["summary"]) == {"fixed_fp32", "fixed_fp16", "fixed_int8", "adaptive_sequential", "adaptive_pipelined"}
    lv = json.loads((res / "bdd100k_mot" / "r04_test_energy_latency" / "live_validation.json").read_text())
    assert lv["level_agreement"] == 1.0          # fake engines are deterministic -> live == replay
    stg = json.loads((res / "r04_stages" / "stages.json").read_text())
    assert list(stg) == ["1_engines_calib_cache", "2_calib_dev", "3_calibrate_lock", "4_test_cache",
                         "5_timing_energy", "6_test_eval"]
    # committed Step 2 records in the repo are untouched by the combined run
    assert json.loads((repo_dir / "results/bdd100k_mot/calib_fixed_fp16/meta.json").read_text())["git_commit"] == "dryrun0002"
    assert out[6]["main"]["n_images"] > 0


def test_test_split_refused_without_valid_lock(env):
    tmp_path, repo_dir, gtn = env
    fr = _make(tmp_path, repo_dir, gtn)
    _run_to(fr, tmp_path, upto=3)
    f = fr.lock_dir / "locked_bdd_main.yaml"
    f.write_text(f.read_text() + "\n# edited\n")                # tamper after locking
    with pytest.raises(CAL.LockError, match="modified"):
        fr.stage4_test_cache()


def test_gate_failure_stops(env):
    tmp_path, repo_dir, gtn = env
    ref = repo_dir / "results/bdd100k_mot/calib_fixed_fp16/metrics.json"
    d = json.loads(ref.read_text())
    d["all"]["map50_95"] -= 0.05                                   # committed value 5 pts away
    ref.write_text(json.dumps(d))
    fr = _make(tmp_path, repo_dir, gtn)
    with pytest.raises(FULL.GateError, match="1_engines_calib_cache"):
        _run_to(fr, tmp_path, upto=1)
    assert fr.stages["1_engines_calib_cache"]["pass"] is False
