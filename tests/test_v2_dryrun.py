"""DRY-RUN of notebook 05 (controller v2) with fake engines / NVML / frames. No GPU, no data."""

import csv
import hashlib
import json

import pytest
from test_energy_live import FakeNvml
from test_full_dryrun import _factory, _full_cfg, env, tag_reader  # noqa: F401
from test_step2_dryrun import repo  # noqa: F401

from adapres import calibrate as CAL
from adapres.controller import ControllerParams
from adapres.energy.nvml_meter import NvmlMeter
from adapres.pipeline import full as FULL
from adapres.pipeline import v2 as V2

pytestmark = pytest.mark.slow  # end-to-end dry-run; run with: pytest -m slow


def _freeze_addendum(repo_dir):
    a = repo_dir / "configs" / "prereg_addendum.yaml"
    (a.with_suffix(".yaml.sha256")).write_text(hashlib.sha256(a.read_bytes()).hexdigest() + "\n")


def _v1_lock(repo_dir):
    p = ControllerParams(weights={"m": 0.25, "c": 0.0, "n": 0.5, "u": 0.0, "s": 0.25}, theta=0.5,
                         norm={"m": (0.0, 0.3), "c": (0.0, 0.5), "n": (0.0, 10.0), "u": (0.0, 0.8)}, K_up=1, K_down=1)
    CAL.lock(repo_dir / "results" / "locked" / "locked_bdd_main.yaml", p, {"role": "main"})


def _make(tmp_path, repo_dir, gtn):
    cfg = _full_cfg(tmp_path)
    cfg["paths"]["work_root"] = str(tmp_path / "w5")
    cfg["v2"]["smoke"].update(K_grid=[1, 2], n_folds=2, bootstrap_resamples=10, random_seeds=2)

    def exporter(weights, level, imgsz, out_dir, int8_data=None, batch=1):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"fake_{level}.engine"
        p.write_bytes(level.encode())
        return {"level": level, "path": str(p), "sha256": "fake-" + level, "bytes": 1000, "build_s": 0.0}

    nv = FakeNvml()
    return V2.V2Run(cfg, False, repo_dir, exporter=exporter, backend_factory=_factory(gtn), image_reader=tag_reader,
                    meter_factory=lambda: NvmlMeter(nvml=nv, sample_hz=200), release_fn=lambda: None)


def _run(fr, tmp_path, upto=6):
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    out = {1: fr.stage1_engines(w, {i: f"c{i}" for i in range(80)})}
    for k, fn in ((2, fr.stage2_calib_dev), (3, fr.stage3_v2), (4, fr.stage4_test_cache), (5, fr.stage5_v2_energy),
                  (6, fr.stage6_v2_eval)):
        if k > upto:
            break
        out[k] = fn()
    return out


def test_v2_requires_frozen_addendum(env):  # noqa: F811
    tmp_path, repo_dir, gtn = env
    with pytest.raises(FULL.GateError, match="frozen"):
        _make(tmp_path, repo_dir, gtn)
    _freeze_addendum(repo_dir)
    a = repo_dir / "configs" / "prereg_addendum.yaml"
    a.write_text(a.read_text() + "\n# edited after freezing\n")
    with pytest.raises(FULL.GateError, match="changed after it was frozen"):
        _make(tmp_path, repo_dir, gtn)


def test_v2_pipeline_dryrun(env):  # noqa: F811
    tmp_path, repo_dir, gtn = env
    _freeze_addendum(repo_dir)
    _v1_lock(repo_dir)
    fr = _make(tmp_path, repo_dir, gtn)
    out = _run(fr, tmp_path)
    assert list(fr.stages) == ["1_engines_calib_cache", "2_calib_dev", "3_v2_fit_lock", "4_test_cache",
                               "5_v2_energy", "6_v2_test_eval"] and all(s["pass"] for s in fr.stages.values())
    res = tmp_path / "w5" / "results"
    idx = {r["exp_id"] for r in csv.DictReader(open(res / "index.csv"))}
    for e in ("r05_v2_calibration", "r05_test_v2_rho0.5", "r05_test_random_rho0.5", "r05_test_v1_rho0.5",
              "r05_test_oracle_rho0.5", "r05_test_lagged_oracle_rho0.5", "r05_test_v2_bootstrap",
              "r05_test_v2_hypotheses", "r05_test_v2_energy_latency", "r05_test_fixed_fp16"):
        assert e in idx, e
    hyp = json.loads((res / "bdd100k_mot" / "r05_test_v2_hypotheses" / "hypotheses.json").read_text())
    assert {"P1_primary", "P2", "P3", "P4_headroom", "P5_energy"} <= set(hyp)
    en = json.loads((res / "bdd100k_mot" / "r05_test_v2_energy_latency" / "energy.json").read_text())
    assert {"fixed_fp16", "fixed_int8", "adaptive_v2_pipelined", "adaptive_v2_sequential",
            "forced_alternation"} == set(en["summary"])
    assert en["summary"]["forced_alternation"]["switch_rate_per_100"] == pytest.approx(100.0)
    lv = json.loads((res / "bdd100k_mot" / "r05_test_v2_energy_latency" / "live_validation.json").read_text())
    assert lv["level_agreement_live_vs_replay"] == 1.0
    m = json.loads((res / "bdd100k_mot" / "r05_test_v2_rho0.5" / "metrics.json").read_text())
    assert 0.3 <= m["all"]["frac_level"]["int8"] <= 0.7 and "map50_95_ci" in m["all"]
    # the v2 lock records the frozen addendum hash; tampering blocks the test stages
    f = res / "locked" / "locked_v2_rho0.5.yaml"
    f.write_text(f.read_text() + "\n# edited\n")
    with pytest.raises(CAL.LockError, match="modified"):
        fr.stage4_test_cache()
    assert out[3]["per_rho"]
