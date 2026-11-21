"""DRY-RUN of the subset build on a synthetic mini dataset with a mocked image reader.
No real data, no GPU; checks wiring, determinism, disjointness and calib-only fitting."""

import csv
import json

import pytest
from conftest import REPO, fake_reader, make_fake_bdd

from adapres import config as C
from adapres.pipeline import build_bdd_subsets as B

pytestmark = pytest.mark.slow  # end-to-end dry-run; run with: pytest -m slow

STRATA = {**{f"day{i:02d}": "daytime" for i in range(8)},
          **{f"nit{i:02d}": "night" for i in range(5)},
          **{f"dd{i:02d}": "dawn/dusk" for i in range(2)}}


def _cfg(tmp_path, work):
    return C.load(REPO / "configs" / "default.yaml",
                  {"paths": {"input_root": str(tmp_path / "input"), "work_root": str(work)},
                   "bdd": {"smoke": {"calib": {"n_sequences": 4, "strata_targets": {"daytime": 2, "night": 1, "dawn_dusk": 1}},
                                     "test": {"n_sequences": 5, "strata_targets": {"daytime": 3, "night": 1, "dawn_dusk": 1}},
                                     "int8_calib_frames": 20}},
                   "splice": {"smoke_n_streams": 3}})


@pytest.fixture
def repo_dir(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    (d / "GIT_COMMIT").write_text("dryrun0000\n")
    return d


def _run(tmp_path, work, repo_dir, strata=STRATA, with_tod=True):
    if not (tmp_path / "input").exists():
        make_fake_bdd(tmp_path / "input", strata, with_tod=with_tod)
    cfg = _cfg(tmp_path, work)
    reader = fake_reader({k: ("night" if v == "night" else "day") for k, v in strata.items()})
    return B.run(cfg, smoke=True, repo_dir=repo_dir, read_rgb=reader, workers=2)


def test_dryrun_outputs(tmp_path, repo_dir):
    out = _run(tmp_path, tmp_path / "work", repo_dir)
    d = tmp_path / "work" / "smoke" / "results" / "bdd100k_mot" / "bdd_subsets"
    assert out["results_dir"] == str(d)
    for f in ("meta.json", "selection.json", "frames_calib.csv", "frames_test.csv", "splice_streams.csv",
              "int8_calib_frames.csv", "slice_thresholds.json", "report.json"):
        assert (d / f).exists(), f
    meta = json.loads((d / "meta.json").read_text())
    assert meta["git_commit"] == "dryrun0000" and meta["seed"] == 2027 and meta["smoke"] is True
    sel = json.loads((d / "selection.json").read_text())
    assert not set(sel["splits"]["calib"]) & set(sel["splits"]["test"])
    assert sel["timeofday_source"] == "label_attribute"
    assert sel["strata_counts"]["test"] == {"daytime": 3, "night": 1, "dawn_dusk": 1}
    rep = out["report"]
    assert all(v["gt_as_det_map50_95"] == pytest.approx(1.0) for v in rep["coco_gt"].values())
    assert rep["frames"]["calib"]["n_frames"] == 4 * 45
    assert rep["frames"]["test"]["slice_night"] == 45  # the one night test sequence, all frames dark
    assert rep["frames"]["splice"]["slice_scene_change"] > 0
    assert rep["frames"]["test"]["slice_scene_change_natural"] > 0
    int8 = list(csv.DictReader(open(d / "int8_calib_frames.csv")))
    assert len(int8) == 20 and {r["seq"] for r in int8} <= set(sel["splits"]["calib"])
    # COCO GT goes to data/, not results/ (licence)
    assert (tmp_path / "work" / "smoke" / "data" / "bdd" / "coco_gt_test.json").exists()
    assert not list(d.glob("coco_gt*"))
    # derived image paths are relative (no hardcoded absolute paths)
    rows = list(csv.DictReader(open(d / "frames_test.csv")))
    assert all(not r["rel_path"].startswith(("/", "\\")) and ":" not in r["rel_path"] for r in rows)


def test_dryrun_deterministic_and_resumable(tmp_path, repo_dir):
    a = _run(tmp_path, tmp_path / "w", repo_dir)
    # second run in the same work dir reuses luma checkpoints and gives identical outputs
    b = _run(tmp_path, tmp_path / "w", repo_dir)
    for f in ("selection.json", "frames_test.csv", "splice_streams.csv", "slice_thresholds.json"):
        assert (open(a["results_dir"] + "/" + f).read() == open(b["results_dir"] + "/" + f).read())


def test_thresholds_depend_on_calib_only(tmp_path, repo_dir):
    a = _run(tmp_path, tmp_path / "w1", repo_dir)
    sel = json.load(open(a["results_dir"] + "/selection.json"))
    test_seqs = set(sel["splits"]["test"])
    # brighten every TEST image: calib-fitted thresholds must not move
    strata = {k: ("night" if v == "night" else "day") for k, v in STRATA.items()}
    base = fake_reader(strata)

    def reader(p):
        img = base(p)
        return img // 2 + 120 if p.parent.name in test_seqs else img

    b = B.run(_cfg(tmp_path, tmp_path / "w2"), smoke=True, repo_dir=repo_dir, read_rgb=reader, workers=2)
    ta = json.load(open(a["results_dir"] + "/slice_thresholds.json"))
    tb = json.load(open(b["results_dir"] + "/slice_thresholds.json"))
    assert ta == tb


def test_luma_fallback_without_timeofday(tmp_path, repo_dir):
    out = _run(tmp_path, tmp_path / "w", repo_dir, with_tod=False)
    sel = json.load(open(out["results_dir"] + "/selection.json"))
    assert sel["timeofday_source"] == "median_frame_luma_otsu"
    # dawn/dusk cannot be detected from luma: its target is folded into daytime
    assert sel["strata_counts"]["test"]["dawn_dusk"] == 0
    assert out["report"]["luma_fallback"]["n_sequences"] == len(STRATA)


def test_missing_images_fail_loudly(tmp_path, repo_dir):
    make_fake_bdd(tmp_path / "input", STRATA)
    img_root = next((tmp_path / "input").rglob("track_images_val")) / "bdd100k" / "images" / "track" / "val"
    for seq in ("day00", "day01"):
        next((img_root / seq).glob("*.jpg")).unlink()
    with pytest.raises(B.BuildError, match="missing images"):
        _run(tmp_path, tmp_path / "w", repo_dir)


def test_missing_dataset_fails_loudly(tmp_path, repo_dir):
    (tmp_path / "input" / "other").mkdir(parents=True)
    from adapres.data.bdd_mot import LayoutError

    with pytest.raises(LayoutError, match="Tree seen"):
        B.run(_cfg(tmp_path, tmp_path / "w"), smoke=True, repo_dir=repo_dir, read_rgb=lambda p: None)
