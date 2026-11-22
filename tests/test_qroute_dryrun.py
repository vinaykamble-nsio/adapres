"""DRY-RUN of QROUTE_COMPLETE_EXPERIMENT.ipynb (every stage) with fake multi-resolution engines, fake NVML,
a fake contention generator and synthetic frames. No GPU, no real data."""

import csv
import hashlib
import json
import zlib

import numpy as np
import pytest
from test_energy_live import FakeNvml
from test_full_dryrun import NAMES, _full_cfg, env, tag_reader  # noqa: F401
from test_step2_dryrun import repo  # noqa: F401
from test_v2_dryrun import _v1_lock

from adapres import calibrate as CAL
from adapres.data.classmap import EVAL_CATEGORIES
from adapres.energy.nvml_meter import NvmlMeter
from adapres.pipeline import full as FULL
from adapres.pipeline import qroute_run as QR

pytestmark = pytest.mark.slow  # end-to-end dry-run; run with: pytest -m slow

WALL = {"fp32_640": 9.0, "fp16_640": 6.0, "int8_640": 5.0, "fp16_480": 4.2, "int8_480": 3.6, "fp16_320": 3.0, "int8_320": 2.6}
DROP = {640: 0, 480: 9, 320: 4}          # lower resolution misses more boxes (keyed by name + box index)


class MultiResBackend:
    def __init__(self, name, gtn):
        self.name, self.gt, self.calls = name, gtn, 0
        lv, res = name.split("_")
        self.drop_res, self.int8 = DROP[int(res)], lv == "int8"

    def predict(self, src):
        self.calls += 1
        name = NAMES[int(src[0, 0, 0]) + 256 * int(src[0, 1, 0])]
        dets = []
        for k, (x, y, w, h, cid) in enumerate(self.gt.get(name, [])):
            hsh = zlib.crc32(name.encode()) + 7 * k
            if (self.drop_res and hsh % self.drop_res == 0) or (self.int8 and hsh % 23 == 0):
                continue
            dets.append([x, y, x + w, y + h, 0.9 - 0.001 * k, EVAL_CATEGORIES[cid][1]])
        t = {"pre_ms": 0.5, "infer_ms": WALL[self.name] - 1.0, "post_ms": 0.5, "wall_ms": WALL[self.name]}
        return np.asarray(dets, float).reshape(-1, 6), t


class FakeContention:
    def __init__(self):
        self.levels = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def set_level(self, d):
        if not 0 <= d <= 1:
            raise ValueError(d)
        self.levels.append(d)


def _freeze(repo_dir):
    p = repo_dir / "configs" / "prereg_qroute.yaml"
    p.with_suffix(".yaml.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "\n")


def _make(tmp_path, repo_dir, gtn):
    cfg = _full_cfg(tmp_path)
    cfg["paths"]["work_root"] = str(tmp_path / "w6")
    cfg["qroute"].update(new_test_strata={"daytime": 1, "night": 1, "dawn_dusk": 0}, random_seeds=2, bootstrap_resamples=10,
                         live_clips=1, live_frames_per_clip=10, contention_phase_s=0.05, contention_iters=3,
                         energy_probe_s=0.05, frames_per_phase=5)
    exported = []

    def exporter(weights, level, imgsz, out_dir, int8_data=None, batch=1, tag=None):
        assert (level == "int8") == (int8_data is not None)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"fake_{tag}.engine"
        p.write_bytes(tag.encode())
        exported.append(tag)
        return {"level": level, "path": str(p), "sha256": "fake-" + tag, "bytes": 1000, "build_s": 0.0}

    nv = FakeNvml()
    run = QR.QRun(cfg, False, repo_dir, exporter=exporter, backend_factory=lambda name, path, imgsz: MultiResBackend(name, gtn),
                  image_reader=tag_reader, meter_factory=lambda: NvmlMeter(nvml=nv, sample_hz=200),
                  contention_factory=FakeContention)
    run._exported = exported
    return run


def _run(q, tmp_path, upto="s19"):
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    out = {"s3": q.s3_data(), "s4": q.s4_models(w, {i: f"c{i}" for i in range(80)})}
    for k, fn in (("s5", q.s5_caches), ("fixed", q.fixed_dev_table), ("regret", q.regret_summary), ("s7", q.s7_headroom),
                  ("s8", q.s8_s9_predictor), ("s10", q.s10_controller_check), ("s11", q.s11_hardware),
                  ("s12", q.s12_controller), ("s15", q.s15_final_eval), ("s16", q.s16_live), ("s19", q.s19_summary)):
        out[k] = fn()
        if k == upto:
            break
    return out


def test_qroute_requires_frozen_prereg(env):  # noqa: F811
    tmp_path, repo_dir, gtn = env
    (repo_dir / "configs" / "prereg_qroute.yaml.sha256").unlink(missing_ok=True)  # the copied repo configs carry the real hash
    with pytest.raises(FULL.GateError, match="frozen"):
        _make(tmp_path, repo_dir, gtn)
    _freeze(repo_dir)
    p = repo_dir / "configs" / "prereg_qroute.yaml"
    p.write_text(p.read_text() + "\n# edited after freezing\n")
    with pytest.raises(FULL.GateError, match="changed after it was frozen"):
        _make(tmp_path, repo_dir, gtn)


def test_qroute_pipeline_dryrun(env):  # noqa: F811
    tmp_path, repo_dir, gtn = env
    _freeze(repo_dir)
    _v1_lock(repo_dir)
    q = _make(tmp_path, repo_dir, gtn)
    out = _run(q, tmp_path)
    assert all(s["pass"] for s in q.stages.values())
    assert list(q.stages) == ["S3_data", "S4_models", "S5_caches", "S7_headroom", "S8_S9_features_predictor",
                              "S10_controller_check", "S11_hardware", "S12_controller_locks", "S15_final_eval", "S16_live"]
    assert sorted(q._exported) == sorted(WALL)
    # new final-test clips: never used before, ids from the offset, disjoint from dev
    sel = json.loads((repo_dir / "results/bdd100k_mot/bdd_subsets/selection.json").read_text())
    used = set(sel["splits"]["calib"]) | set(sel["splits"]["test"])
    assert len(q.test_seqs) == 2 and not set(q.test_seqs) & used
    assert min(q.test_ids) >= 2_000_000 and not set(q.test_ids) & set(q.dev_ids)
    # cost profile follows the fake engines; budgets as pre-registered
    assert q.cost_iso["int8_640"] == pytest.approx(5.0) and q.budgets["B_int8"] == pytest.approx(5.0)
    assert q.budgets["B_mid"] == pytest.approx(5.5)
    res = tmp_path / "w6" / "results"
    idx = {r["exp_id"] for r in csv.DictReader(open(res / "index.csv"))}
    for e in ("r06_split", "r06_models", "r06_dev_headroom", "r06_dev_predictor", "r06_dev_regret", "r06_hardware_profile",
              "r06_controller_locks", "r06_final_qroute_main_B_int8", "r06_final_qroute_main_B_mid",
              "r06_final_best_fixed_B_int8", "r06_final_fixed_fp32_640", "r06_final_oracle_B_int8",
              "r06_final_v1_r04", "r06_final_random_B_int8", "r06_final_qroute_precision_only",
              "r06_final_qroute_resolution_only", "r06_final_qroute_hardware_only", "r06_final_qroute_abl_no_temporal",
              "r06_final_qroute_abl_no_persistence", "r06_final_qroute_abl_energy_cost",
              "r06_final_contention_qroute_main_B_int8", "r06_final_bootstrap", "r06_final_hypotheses", "r06_live",
              "r06_live_hypotheses", "r06_experiment_summary"):
        assert e in idx, e
    hyp = out["s15"]["hypotheses"]
    assert {"Q1_primary", "Q2", "Q3", "Q5_replay"} <= set(hyp)
    assert {"Q4_live_cost", "Q5_live"} == set(out["s16"])
    lv = json.loads((res / "bdd100k_mot" / "r06_live" / "live_validation.json").read_text())
    assert lv["agreement_no_hw"] == 1.0          # deterministic fake engines: live == replay without hw feedback
    assert lv["no_hw"]["map_diff_pts"] == pytest.approx(0.0, abs=1e-9)
    assert all(isinstance(b["budget_met_on_dev"], bool) for b in q.locks.values())
    m = json.loads((res / "bdd100k_mot" / "r06_final_qroute_main_B_int8" / "metrics.json").read_text())
    assert m["all"]["n_images"] == len(q.test_ids) and abs(sum(m["all"]["allocation"].values()) - 1) < 1e-9
    # every lock verifies and records the frozen design hash; tampering is refused
    for n in q.locks:
        assert q._load_lock(n)[2]["prereg_qroute_sha256"] == q.prereg_sha
    f = q.lock_dir / "locked_qroute_main_B_int8.yaml"
    f.write_text(f.read_text() + "\n# edited\n")
    with pytest.raises(CAL.LockError, match="modified"):
        q._load_lock("main_B_int8")
    summ = json.loads((res / "bdd100k_mot" / "r06_experiment_summary" / "experiment_summary.json").read_text())
    assert summ["incomplete"] == [] and summ["hypotheses"] == out["s19"]["hypotheses"]


def test_qroute_figures_from_dryrun_results(env):  # noqa: F811
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
    import make_qroute

    tmp_path, repo_dir, gtn = env
    _freeze(repo_dir)
    _v1_lock(repo_dir)
    q = _make(tmp_path, repo_dir, gtn)
    _run(q, tmp_path)
    made = make_qroute.build(tmp_path / "w6" / "results", tmp_path / "figs", tmp_path / "tables")
    assert set(made["figures"]) == {"qroute_pareto_latency", "qroute_pareto_energy", "qroute_oracle_headroom",
                                    "qroute_pred_vs_actual", "qroute_per_scene", "qroute_overhead", "qroute_allocation",
                                    "qroute_contention", "qroute_ablations", "qroute_bootstrap"}
    for f in made["figures"]:
        assert (tmp_path / "figs" / f"{f}.pdf").stat().st_size > 1000
    assert (tmp_path / "tables" / "qroute_final_policies.csv").exists()
