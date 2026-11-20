"""mAP + class-mapping unit tests on synthetic boxes (CLAUDE.md: required before big runs)."""

import numpy as np
import pytest
from conftest import box, make_frame

from adapres.data.bdd_mot import parse_video_labels
from adapres.eval.coco_gt import build_coco_gt
from adapres.eval.map_eval import CocoEvaluator, dets_to_coco_results

# Ultralytics COCO-80 indices
PERSON, BICYCLE, CAR, MOTO, BUS, TRAIN, TRUCK, DOG = 0, 1, 2, 3, 5, 6, 7, 16


def _gt(frames_spec, ignore_mode="all_eval_classes"):
    frames = []
    for i, boxes in enumerate(frames_spec):
        frames.append(parse_video_labels([make_frame(f"v{i}", 0, boxes, None)])[0])
    return build_coco_gt([(i + 1, f, 200, 200) for i, f in enumerate(frames)], ignore_mode)


def _det(x1, y1, x2, y2, s, c):
    return [x1, y1, x2, y2, s, c]


def test_dets_to_coco_results_maps_and_drops():
    r = dets_to_coco_results({7: np.array([_det(10, 20, 30, 60, 0.9, TRUCK), _det(0, 0, 5, 5, 0.8, DOG),
                                           _det(5, 5, 5, 9, 0.7, CAR)])})
    assert r == [{"image_id": 7, "category_id": 3, "bbox": [10.0, 20.0, 20.0, 40.0], "score": pytest.approx(0.9)}]


@pytest.mark.parametrize("bdd_cat,coco_cls", [("pedestrian", PERSON), ("rider", PERSON), ("car", CAR),
                                              ("truck", TRUCK), ("bus", BUS), ("train", TRAIN),
                                              ("motorcycle", MOTO), ("bicycle", BICYCLE)])
def test_every_class_round_trips_to_perfect_ap(bdd_cat, coco_cls):
    """A COCO detection of the mapped class on each BDD box gives AP 1.0; the wrong class gives 0."""
    gt = _gt([[box("a", bdd_cat, 10, 10, 60, 90)]])
    ev = CocoEvaluator(gt)
    good = ev.evaluate(dets_to_coco_results({1: np.array([_det(10, 10, 60, 90, 0.9, coco_cls)])}))
    assert good["map50_95"] == pytest.approx(1.0)
    wrong = (coco_cls + 1) % 8 if coco_cls != CAR else TRUCK
    bad = ev.evaluate(dets_to_coco_results({1: np.array([_det(10, 10, 60, 90, 0.9, wrong)])}))
    assert bad["map50_95"] == pytest.approx(0.0)


def test_hand_computed_ap():
    """2 GT cars; dets TP(0.9), FP(0.8), TP(0.7). COCO 101-point interpolated precision:
    recall<=0.5 -> 1.0 (51 points), 0.5<recall<=1.0 -> 2/3 (50 points)."""
    gt = _gt([[box("a", "car", 10, 10, 50, 50), box("b", "car", 100, 100, 150, 150)]])
    dets = np.array([_det(10, 10, 50, 50, 0.9, CAR), _det(160, 10, 190, 40, 0.8, CAR),
                     _det(100, 100, 150, 150, 0.7, CAR)])
    m = CocoEvaluator(gt).evaluate(dets_to_coco_results({1: dets}))
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert m["map50"] == pytest.approx(expected, abs=1e-6)
    assert m["map50_95"] == pytest.approx(expected, abs=1e-6)  # exact boxes: same at every IoU
    assert m["per_class_ap50_95"]["car"] == pytest.approx(expected, abs=1e-6)
    assert m["per_class_ap50_95"]["bus"] == -1.0  # no GT -> excluded, not 0


def test_iou_dependence():
    """Box shifted by 23 px: IoU = 77/123 = 0.626 -> TP at IoU thresholds 0.50, 0.55, 0.60 and FP
    from 0.65 up (kept away from a threshold boundary) -> AP50 = 1, AP50:95 = 3/10."""
    gt = _gt([[box("a", "car", 0, 0, 100, 100)]])
    m = CocoEvaluator(gt).evaluate(dets_to_coco_results({1: np.array([_det(23, 0, 123, 100, 0.9, CAR)])}))
    assert m["map50"] == pytest.approx(1.0)
    assert m["map50_95"] == pytest.approx(0.3)


def test_unmapped_classes_do_not_change_map():
    gt = _gt([[box("a", "car", 10, 10, 50, 50)]])
    base = np.array([_det(10, 10, 50, 50, 0.5, CAR)])
    noise = np.array([_det(60, 60, 90, 90, 0.99, DOG), _det(10, 10, 50, 50, 0.99, 9)])  # dog, traffic light
    ev = CocoEvaluator(gt)
    a = ev.evaluate(dets_to_coco_results({1: base}))
    b = ev.evaluate(dets_to_coco_results({1: np.vstack([base, noise])}))
    assert a["map50_95"] == b["map50_95"] == pytest.approx(1.0)


def test_missed_categories_count_as_zero_and_empty_results():
    gt = _gt([[box("a", "car", 10, 10, 50, 50), box("b", "bus", 100, 100, 190, 190)]])
    ev = CocoEvaluator(gt)
    only_car = ev.evaluate(dets_to_coco_results({1: np.array([_det(10, 10, 50, 50, 0.9, CAR)])}))
    assert only_car["per_class_ap50_95"]["bus"] == pytest.approx(0.0)
    assert only_car["map50_95"] == pytest.approx(0.5)  # mean over car(1) and bus(0)
    empty = ev.evaluate([])
    assert empty["map50_95"] == pytest.approx(0.0) and empty["n_dets"] == 0


def test_slices_use_img_id_filter():
    gt = _gt([[box("a", "car", 10, 10, 50, 50)], [box("b", "car", 10, 10, 50, 50)]])
    dets = {1: np.array([_det(10, 10, 50, 50, 0.9, CAR)]), 2: np.zeros((0, 6))}
    ev = CocoEvaluator(gt)
    s = ev.evaluate_slices(dets_to_coco_results(dets), {"all": [1, 2], "hit": [1], "miss": [2], "none": []})
    assert s["hit"]["map50_95"] == pytest.approx(1.0)
    assert s["miss"]["map50_95"] == pytest.approx(0.0)
    assert 0.0 < s["all"]["map50_95"] < 1.0
    assert s["none"]["n_images"] == 0 and s["none"]["map50_95"] == -1.0
    with pytest.raises(ValueError):
        ev.evaluate([], img_ids=[99])


def test_ignore_region_and_crowd_in_map():
    gt = _gt([[box("a", "car", 10, 10, 50, 50), box("i", "trailer", 100, 100, 180, 180),
               box("k", "pedestrian", 60, 10, 90, 80, crowd=True)]])
    ev = CocoEvaluator(gt)
    tp = [_det(10, 10, 50, 50, 0.5, CAR)]
    on_ignore = [_det(100, 100, 180, 180, 0.99, TRUCK)]        # truck on trailer ignore region
    on_crowd = [_det(60, 10, 90, 80, 0.99, PERSON)]            # person on crowd person box
    assert ev.evaluate(dets_to_coco_results({1: np.array(tp + on_ignore + on_crowd)}))["map50_95"] == pytest.approx(1.0)
