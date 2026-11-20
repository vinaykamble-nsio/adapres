"""DRY-RUN of notebook 02's pipeline with fake exporter/backends (no GPU, no real data).
Fake detectors return the GT boxes (FP32/FP16/pt) or GT minus every third box (INT8),
so the expected mAP ordering and the sanity checks are known in advance."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO, fake_reader, make_fake_bdd

from adapres import config as C
from adapres.data.classmap import EVAL_CATEGORIES
from adapres.pipeline import build_bdd_subsets, eval_fixed

STRATA = {**{f"day{i:02d}": "daytime" for i in range(6)}, **{f"nit{i:02d}": "night" for i in range(4)}}


def _cfg(tmp, **step2):
    over = {"paths": {"input_root": str(tmp / "input"), "work_root": str(tmp / "work")},
            "bdd": {"subsets": {"calib": {"n_sequences": 3, "strata_targets": {"daytime": 2, "night": 1, "dawn_dusk": 0}},
                                "test": {"n_sequences": 3, "strata_targets": {"daytime": 2, "night": 1, "dawn_dusk": 0}},
                                "int8_calib_frames": 20}},
            "splice": {"n_streams": 2},
            "step2": {"n_sequences": 3, "smoke_n_sequences": 2, "warmup_frames": 3, "microbench_iters": 5,
                      "microbench_warmup": 1, "val_check_frames": 30, "smoke_val_check_frames": 10, **step2}}
    return C.load(REPO / "configs" / "default.yaml", over)


@pytest.fixture
def repo(tmp_path):
    """A fake repo whose results/ holds a real (non-smoke) Step 1 build of the fake dataset."""
    make_fake_bdd(tmp_path / "input", STRATA)
    r = tmp_path / "repo"
    r.mkdir()
    (r / "GIT_COMMIT").write_text("dryrun0002\n")
    cfg = _cfg(tmp_path)
    out = build_bdd_subsets.run(cfg, smoke=False, repo_dir=r, read_rgb=fake_reader(
        {k: ("night" if v == "night" else "day") for k, v in STRATA.items()}), workers=2)
    dst = r / "results" / "bdd100k_mot" / "bdd_subsets"
    shutil.copytree(out["results_dir"], dst)
    shutil.copy(Path(out["results_dir"]).parents[1] / "index.csv", r / "results" / "index.csv")
    shutil.rmtree(tmp_path / "work")  # notebook 02 starts from a clean working dir
    return r


class FakeBackend:
    def __init__(self, name, gt_by_name, drop_every=0):
        self.name, self.gt, self.drop, self.calls = name, gt_by_name, drop_every, 0

    def predict(self, src):
        self.calls += 1
        boxes = self.gt.get(Path(str(src)).name, []) if not isinstance(src, np.ndarray) else []
        dets = []
        for k, (x, y, w, h, cid) in enumerate(boxes):
            if self.drop and k % self.drop == 0:
                continue
            dets.append([x, y, x + w, y + h, 0.9 - 0.001 * k, EVAL_CATEGORIES[cid][1]])
        t = {"pre_ms": 1.0, "infer_ms": {"int8": 2.0, "fp16": 3.0}.get(self.name, 5.0), "post_ms": 0.5, "wall_ms": 9.0}
        return np.asarray(dets, float).reshape(-1, 6), t


def fake_exporter(weights, level, imgsz, out_dir, int8_data=None, batch=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"fake_{level}.engine"
    p.write_bytes(level.encode())
    if level == "int8":
        assert int8_data is not None and Path(int8_data).exists()
    return {"level": level, "path": str(p), "sha256": "fake-" + level, "bytes": p.stat().st_size, "build_s": 0.0}


def _step(tmp_path, repo, smoke, **kw):
    cfg = _cfg(tmp_path, **kw.pop("step2", {}))
    holder = {}

    def factory(name, path):
        s = holder["step"]
        by_name = {}
        for a in s.gt["annotations"]:
            if a["iscrowd"]:
                continue
            fname = Path(str(s.paths[a["image_id"]])).name
            by_name.setdefault(fname, []).append((*a["bbox"], a["category_id"]))
        return FakeBackend(name, by_name, drop_every=3 if name == "int8" else 0)

    step = eval_fixed.Step2(cfg, smoke=smoke, repo_dir=repo, exporter=fake_exporter, backend_factory=factory,
                            image_reader=lambda p: np.zeros((4, 4, 3), np.uint8),
                            val_fn=kw.pop("val_fn", lambda y: {"map50_95": 1.0, "map50": 1.0}))
    holder["step"] = step
    return step


def _run_all(step, tmp_path):
    info = step.prepare()
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    y = step.int8_dataset({i: f"c{i}" for i in range(80)})
    step.export(weights, y)
    be = step.backends()
    mb = step.microbench(be)
    step.cache_runs(be)
    metrics = step.evaluate()
    vc = step.val_check()
    san = step.sanity(metrics)
    return info, mb, metrics, vc, san, be


def test_full_dryrun(tmp_path, repo):
    step = _step(tmp_path, repo, smoke=False)
    info, mb, metrics, vc, san, be = _run_all(step, tmp_path)
    assert info["matches_step1_report"] and len(info["sequences"]) == 3
    for k in ("pt_fp32", "fp32", "fp16"):
        assert metrics[k]["all"]["map50_95"] == pytest.approx(1.0)
    assert metrics["int8"]["all"]["map50_95"] < 1.0
    # oracle-cheap keeps FP16 quality per frame, so its mAP is >= the INT8 one
    assert metrics["oracle_cheap_dev"]["all"]["mean_f1_05"] == pytest.approx(metrics["fp16"]["all"]["mean_f1_05"])
    assert metrics["oracle_cheap_dev"]["all"]["map50_95"] >= metrics["int8"]["all"]["map50_95"]
    assert mb["size_rule"]["int8_speedup_vs_fp16"] == pytest.approx(1 - 2 / 3)
    assert mb["size_rule"]["triggered"] is False
    # INT8 drop in this fake is large -> sanity must flag it, everything else passes
    failed = [c["check"] for c in san["checks"] if not c["pass"]]
    assert failed == ["INT8 - FP16"]
    res = tmp_path / "work" / "results"
    idx = list(csv.DictReader(open(res / "index.csv")))
    ids = {r["exp_id"] for r in idx}
    assert {"bdd_subsets", "engines_yolo11s", "microbench_yolo11s", "calib_fixed_fp16", "calib_fixed_int8",
            "calib_fixed_pt_fp32", "calib_oracle_cheap_dev", "sanity_02_yolo11s"} <= ids
    m = json.loads((res / "bdd100k_mot" / "calib_fixed_fp16" / "meta.json").read_text())
    assert m["split"] == "calib" and m["git_commit"] == "dryrun0002" and m["engine_sha256"]["fp16"] == "fake-fp16"
    pf = list(csv.DictReader(open(res / "bdd100k_mot" / "calib_fixed_int8" / "per_frame.csv")))
    assert len(pf) == info["n_frames"] and {r["level"] for r in pf} == {"int8"}
    assert (res / "sanity" / "sanity_02_yolo11s" / "val_check.json").exists()
    # no test-split frames were touched
    test_ids = {r["image_id"] for r in csv.DictReader(open(repo / "results/bdd100k_mot/bdd_subsets/frames_test.csv"))}
    assert not test_ids & {r["image_id"] for r in pf}


def test_cache_resume_skips_done_sequences(tmp_path, repo):
    step = _step(tmp_path, repo, smoke=True)
    step.prepare()
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    step.export(w, step.int8_dataset({i: str(i) for i in range(80)}))
    be = step.backends()
    step.cache_runs(be)
    calls = {k: b.calls for k, b in be.items()}
    step.cache_runs(be)  # second call: everything cached
    assert {k: b.calls for k, b in be.items()} == calls
    assert len(step.sequences) == 2


def test_val_check_mismatch_flagged(tmp_path, repo):
    step = _step(tmp_path, repo, smoke=True, val_fn=lambda y: {"map50_95": 0.9, "map50": 0.95})
    *_, san, _ = _run_all(step, tmp_path)
    by = {c["check"]: c for c in san["checks"]}
    for label in ("mAP50-95", "mAP50"):  # both metrics are checked and reported ...
        c = by[f"pipeline vs model.val() |diff| {label} (end-to-end)"]
        assert c["pass"] is False and c["binding"] is False  # ... but informational under option 1 (D7)
    binding_fail = {c["check"] for c in san["checks"] if c["binding"] and not c["pass"]}
    assert binding_fail == {"INT8 - FP16"}  # the only binding failure is the planted INT8 drop


def test_refuses_smoke_selection(tmp_path, repo):
    sel = repo / "results/bdd100k_mot/bdd_subsets/selection.json"
    d = json.loads(sel.read_text())
    d["smoke"] = True
    sel.write_text(json.dumps(d))
    with pytest.raises(eval_fixed.Step2Error, match="SMOKE"):
        _step(tmp_path, repo, smoke=False).prepare()


def test_gt_mismatch_with_step1_fails(tmp_path, repo):
    rep = repo / "results/bdd100k_mot/bdd_subsets/report.json"
    d = json.loads(rep.read_text())
    d["coco_gt"]["calib"]["n_annotations"] += 1
    rep.write_text(json.dumps(d))
    with pytest.raises(eval_fixed.Step2Error, match="Step 1 report"):
        _step(tmp_path, repo, smoke=False).prepare()


def test_config_rejects_test_split(tmp_path):
    with pytest.raises(C.ConfigError, match="test stays untouched"):
        _cfg(tmp_path, split="test")
