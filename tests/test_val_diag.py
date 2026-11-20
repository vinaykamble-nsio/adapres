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


THR = {"ours_dup_other_class_max": 0.01, "theirs_dup_other_class_min": 0.10, "high_conf_match_min": 0.95}


def _expl(ours_dup=0.0005, theirs_dup=0.4047, mo=1.0, mt=0.9829):
    sim = {"frac_ours_matched": mo, "frac_theirs_matched": mt}
    sig = {"ours": {"frac_boxes_with_same_box_other_class": ours_dup},
           "theirs": {"frac_boxes_with_same_box_other_class": theirs_dup}}
    return V.nms_explanation(sim, sig, THR)


def test_decompose_metric_cause():
    out = V.decompose(*_abc((.2684, .4003), (.2684, .4003), (.2713, .4113)), tol_pts=0.5)
    assert out["components"]["map50"]["cause"] == "metric_implementation"
    assert out["components"]["map50"]["metric_component_pts"] == pytest.approx(1.10)
    assert out["end_to_end_failing_metrics"] == ["map50"]
    assert out["checks"]["identical_predictions"] == {"map50_95": True, "map50": False}
    # a metric-implementation gap > tol fails the binding check even if NMS explains predictions
    assert V.binding_verdict(out, _expl())["binding_pass"] is False


def test_smoke_like_prediction_cause_passes_option1():
    # 02b smoke numbers: predictions cause the gap, metric agrees within 0.11 pts
    dec = V.decompose(*_abc((.22652, .4786), (.23479, .49441), (.2337, .4937)), tol_pts=0.5)
    assert {k: v["cause"] for k, v in dec["components"].items()} == {"map50_95": "predictions", "map50": "predictions"}
    assert dec["end_to_end_failing_metrics"] == ["map50_95", "map50"]
    v = V.binding_verdict(dec, _expl())
    assert v["identical_predictions_pass"] and v["nms_explained"] and v["binding_pass"]


@pytest.mark.parametrize("kw,part", [({"ours_dup": 0.05}, "ours_single_label"),
                                     ({"theirs_dup": 0.02}, "theirs_multi_label"),
                                     ({"mt": 0.90}, "high_conf_boxes_agree")])
def test_unexplained_gap_fails(kw, part):
    e = _expl(**kw)
    assert e["parts"][part] is False and e["explained"] is False
    dec = V.decompose(*_abc((.2265, .4786), (.2348, .4944), (.2337, .4937)), tol_pts=0.5)
    assert V.binding_verdict(dec, e)["binding_pass"] is False


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
                coco = EVAL_CATEGORIES[c][1]
                jd.append({"image_id": stem, "file_name": stem + ".jpg", "category_id": coco + 1,
                           "bbox": [x, y, w, h], "score": 0.9})
                # multi-label NMS emulation: same box re-emitted under another class at low score
                other = 7 if coco != 7 else 2
                jd.append({"image_id": stem, "file_name": stem + ".jpg", "category_id": other + 1,
                           "bbox": [x, y, w, h], "score": 0.01})
        return {"map50_95": 0.999, "map50": 0.999, "jdict": jd, "args": {"conf": 0.001}}

    committed = json.loads((Path(step.results_root) / "sanity/sanity_02_yolo11s/val_check.json").read_text())
    out = V.run(step, predict, val_fn, 0.5, committed)
    c = out["components"]["map50_95"]
    assert c["A_ours_pred_ours_metric"] == pytest.approx(100) and c["B_their_pred_ours_metric"] == pytest.approx(100)
    assert c["metric_component_pts"] == pytest.approx(-0.1)
    assert out["nms_explanation"]["parts"] == {"ours_single_label": True, "theirs_multi_label": True,
                                               "high_conf_boxes_agree": True}
    assert out["identical_predictions_pass"] and out["binding_pass"]
    assert out["multilabel_signature"]["theirs"]["frac_boxes_with_same_box_other_class"] == pytest.approx(1.0)
    assert out["prediction_similarity"]["frac_ours_matched"] == 1.0
    assert "reproduces_notebook02" in out
    assert (Path(step.results_root) / "sanity/val_diag_02b_yolo11s/diag.json").exists()
