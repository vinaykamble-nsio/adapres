"""Notebook 02b diagnostic: parsing, prediction comparison, decomposition and decision rule."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_step2_dryrun import _run_all, _step, repo  # noqa: F401  (fixture reuse)

from adapres.data.classmap import EVAL_CATEGORIES
from adapres.pipeline import val_diag as V


def test_parse_jdict_by_file_name_and_numeric_id():
    stem_to_id = {"seqA__seqA-0000001": 10, "42": 11}
    jd = [{"image_id": "seqA__seqA-0000001", "file_name": "seqA__seqA-0000001.jpg", "category_id": 3,
           "bbox": [10, 20, 30, 40], "score": 0.9},
          {"image_id": 42, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.5}]
    out = V.parse_ultralytics_jdict(jd, stem_to_id)
    assert out[10].tolist() == [[10, 20, 40, 60, 0.9, 2]]  # category_id 3 -> COCO-80 index 2 (car)
    assert out[11].tolist() == [[0, 0, 5, 5, 0.5, 0]]
    with pytest.raises(V.DiagError):
        V.parse_ultralytics_jdict([{"image_id": "nope", "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1}], stem_to_id)


def test_compare_predictions():
    a = {1: np.array([[0, 0, 10, 10, 0.9, 2], [20, 20, 30, 30, 0.8, 0], [0, 0, 5, 5, 0.1, 2]])}
    same = V.compare_predictions(a, {1: a[1].copy()})
    assert same["frac_ours_matched"] == 1.0 and same["n_ours"] == 2 and same["median_abs_score_diff"] == 0.0
    shifted = {1: a[1] + np.array([3, 0, 3, 0, 0, 0])}
    assert V.compare_predictions(a, shifted)["frac_ours_matched"] == 0.0


def test_multilabel_signature():
    single = {1: np.array([[0, 0, 10, 10, 0.9, 2], [20, 20, 30, 30, 0.5, 0]])}
    multi = {1: np.array([[0, 0, 10, 10, 0.9, 2], [0, 0, 10, 10, 0.01, 7], [20, 20, 30, 30, 0.5, 0]])}
    assert V.multilabel_signature(single, 300)["frac_boxes_with_same_box_other_class"] == 0.0
    s = V.multilabel_signature(multi, 3)
    assert s["frac_boxes_with_same_box_other_class"] == pytest.approx(2 / 3)
    assert s["frac_images_at_max_det"] == 1.0 and s["mean_boxes_per_image"] == 3


def _abc(a, b, c):
    return ({"map50_95": a[0], "map50": a[1]}, {"map50_95": b[0], "map50": b[1]}, {"map50_95": c[0], "map50": c[1]})


def test_decompose_metric_cause_switches_binding_check():
    # notebook-02-like: A=(.2684,.4003), predictions identical (B=A), metric adds +0.3/+1.1 pts
    out = V.decompose(*_abc((.2684, .4003), (.2684, .4003), (.2713, .4113)), tol_pts=0.5)
    assert out["components"]["map50"]["cause"] == "metric_implementation"
    assert out["components"]["map50"]["metric_component_pts"] == pytest.approx(1.10)
    assert out["end_to_end_failing_metrics"] == ["map50"]
    assert out["binding_check"] == "identical_predictions"
    assert out["binding_pass"] is False  # |C-B| = 1.1 > 0.5 on identical predictions


def test_decompose_prediction_cause_keeps_end_to_end():
    out = V.decompose(*_abc((.2684, .4003), (.2713, .4113), (.2714, .4115)), tol_pts=0.5)
    assert out["components"]["map50"]["cause"] == "predictions"
    assert out["binding_check"] == "end_to_end" and out["binding_pass"] is False


def test_decompose_all_within_tolerance():
    out = V.decompose(*_abc((.27, .41), (.271, .412), (.272, .413)), tol_pts=0.5)
    assert out["end_to_end_failing_metrics"] == [] and out["binding_check"] == "end_to_end" and out["binding_pass"]


def test_run_dryrun(tmp_path, repo):  # noqa: F811
    step = _step(tmp_path, repo, smoke=False)
    _run_all(step, tmp_path)  # prepares, caches, writes Step 2 results
    gt_boxes = {}
    for a in step.gt["annotations"]:
        if not a["iscrowd"]:
            gt_boxes.setdefault(a["image_id"], []).append((*a["bbox"], a["category_id"]))
    path_to_id = {str(p): i for i, p in step.paths.items()}

    def predict(path):
        return np.array([[x, y, x + w, y + h, 0.9, EVAL_CATEGORIES[c][1]] for x, y, w, h, c in gt_boxes.get(path_to_id[path], [])]).reshape(-1, 6)

    def val_fn(yaml_path):
        stems = {Path(p).stem for p in (Path(yaml_path).parent / "images").iterdir()}
        vs = step.val_check_set("val_mini_02b")
        jd = []
        for stem, iid in vs["stem_to_id"].items():
            assert stem in stems
            for x, y, w, h, c in gt_boxes.get(iid, []):
                jd.append({"image_id": stem, "file_name": stem + ".jpg", "category_id": EVAL_CATEGORIES[c][1] + 1,
                           "bbox": [x, y, w, h], "score": 0.9})
        return {"map50_95": 0.999, "map50": 0.999, "jdict": jd, "args": {"conf": 0.001}}

    committed = json.loads((Path(step.results_root) / "sanity/sanity_02_yolo11s/val_check.json").read_text())
    out = V.run(step, predict, val_fn, 0.5, committed)
    c = out["components"]["map50_95"]
    assert c["A_ours_pred_ours_metric"] == pytest.approx(100) and c["B_their_pred_ours_metric"] == pytest.approx(100)
    assert c["metric_component_pts"] == pytest.approx(-0.1)
    assert out["binding_check"] == "end_to_end" and out["binding_pass"]
    assert out["prediction_similarity"]["frac_ours_matched"] == 1.0
    assert "reproduces_notebook02" in out
    assert (Path(step.results_root) / "sanity/val_diag_02b_yolo11s/diag.json").exists()
