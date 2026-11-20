"""COCO GT converter + mAP plumbing on synthetic boxes (pycocotools, CPU, ms)."""

import contextlib
import io
import json

import pytest
from conftest import box, make_frame

from adapres.data.bdd_mot import parse_video_labels
from adapres.eval.coco_gt import build_coco_gt, gt_counts
from adapres.eval.coco_metrics import ap_from_eval
from adapres.pipeline.build_bdd_subsets import BuildError, validate_coco_gt


def frames():
    return parse_video_labels([
        make_frame("v", 0, [box("a", "car", 10, 10, 30, 30), box("b", "pedestrian", 40, 5, 50, 30),
                            box("i", "other vehicle", 60, 60, 90, 90)], "daytime"),
        make_frame("v", 1, [box("a", "car", 12, 10, 32, 30), box("c", "bus", 0, 0, 200, 100),
                            box("k", "car", 70, 0, 80, 10, crowd=True)], "daytime"),
    ])


def gt():
    fr = frames()
    return build_coco_gt([(1, fr[0], 100, 100), (2, fr[1], 100, 100)], "all_eval_classes")


def test_structure():
    g = gt()
    assert [i["id"] for i in g["images"]] == [1, 2]
    ign = [a for a in g["annotations"] if a.get("ignore_region")]
    assert len(ign) == 7 and all(a["iscrowd"] == 1 for a in ign)
    assert {a["category_id"] for a in ign} == set(range(1, 8))
    crowd = [a for a in g["annotations"] if a["iscrowd"] and not a.get("ignore_region")]
    assert len(crowd) == 1 and crowd[0]["category_id"] == 2
    bus = next(a for a in g["annotations"] if a["bdd_category"] == "bus")
    assert bus["bbox"] == [0.0, 0.0, 100.0, 100.0]  # clipped to image
    assert bus["area"] == 100.0 * 100.0
    assert [gt_counts(f) for f in frames()] == [2, 2]  # crowd + ignore not counted


def test_ignore_mode_none():
    fr = frames()
    g = build_coco_gt([(1, fr[0], 100, 100)], "none")
    assert all(a["bdd_category"] != "other vehicle" for a in g["annotations"])


def test_duplicate_image_id():
    fr = frames()
    with pytest.raises(ValueError):
        build_coco_gt([(1, fr[0], 100, 100), (1, fr[1], 100, 100)])


def test_gt_as_detections_gives_perfect_map():
    out = validate_coco_gt(gt())
    assert out["gt_as_det_map50_95"] == pytest.approx(1.0)
    assert out["gt_as_det_map50"] == pytest.approx(1.0)


def test_validate_catches_broken_gt():
    g = gt()
    for a in g["annotations"]:
        a["iscrowd"] = 1
    with pytest.raises(BuildError):
        validate_coco_gt(g)


def _map(g, dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO()
    coco.dataset = json.loads(json.dumps(g))
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(dets), "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
    return ap_from_eval(ev)


def test_detection_on_ignore_region_is_not_a_false_positive():
    g = gt()
    tps = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.5}
           for a in g["annotations"] if not a["iscrowd"]]
    # a high-scoring car detection exactly on the 'other vehicle' ignore region
    on_ignore = {"image_id": 1, "category_id": 2, "bbox": [60, 60, 30, 30], "score": 0.99}
    # the same detection on empty background
    on_bg = {"image_id": 1, "category_id": 2, "bbox": [60, 30, 30, 20], "score": 0.99}
    assert _map(g, tps + [on_ignore]) == pytest.approx(1.0)
    assert _map(g, tps + [on_bg]) < 0.999
