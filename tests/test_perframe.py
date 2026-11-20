import numpy as np
import pytest

from adapres.eval.perframe import f1_from_counts, frame_counts, gt_arrays_from_coco, iou_matrix, perframe_f1

CAR, PERSON, DOG = 2, 0, 16
E = np.zeros((0, 4))
EI = np.zeros((0,), int)


def d(x1, y1, x2, y2, s, c):
    return [x1, y1, x2, y2, s, c]


def test_iou_matrix():
    m = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert m[0].tolist() == pytest.approx([1.0, 50 / 150, 0.0])


def test_empty_frame_rules():
    assert f1_from_counts(*frame_counts(np.zeros((0, 6)), E, EI, E, EI, 0.25)) == 1.0
    assert f1_from_counts(*frame_counts(np.array([d(0, 0, 5, 5, 0.9, CAR)]), E, EI, E, EI, 0.25)) == 0.0
    assert f1_from_counts(*frame_counts(np.zeros((0, 6)), [[0, 0, 5, 5]], [2], E, EI, 0.25)) == 0.0


def test_counts_with_class_threshold_and_ignore():
    gt_b = [[0, 0, 10, 10], [20, 0, 30, 10]]
    gt_c = [2, 1]  # car, person (eval ids)
    ign_b, ign_c = [[50, 50, 80, 80]], [2]
    dets = np.array([
        d(0, 0, 10, 10, 0.9, CAR),        # TP car
        d(20, 0, 30, 10, 0.8, CAR),       # wrong class on the person -> FP
        d(52, 52, 78, 78, 0.7, CAR),      # inside car ignore region -> discarded
        d(52, 52, 78, 78, 0.7, PERSON),   # ignore box is car-only here -> FP
        d(0, 0, 10, 10, 0.6, CAR),        # duplicate of matched GT -> FP
        d(20, 0, 30, 10, 0.1, PERSON),    # below tau_c -> dropped
        d(90, 90, 99, 99, 0.95, DOG),     # unmapped class -> dropped
    ])
    tp, fp, fn = frame_counts(dets, gt_b, gt_c, ign_b, ign_c, tau_c=0.25)
    assert (tp, fp, fn) == (1, 3, 1)
    assert f1_from_counts(tp, fp, fn) == pytest.approx(2 / (2 + 3 + 1))


def test_greedy_highest_score_first():
    gt_b, gt_c = [[0, 0, 10, 10]], [2]
    dets = np.array([d(1, 0, 11, 10, 0.5, CAR), d(0, 0, 10, 10, 0.9, CAR)])
    assert frame_counts(dets, gt_b, gt_c, E, EI, 0.25) == (1, 1, 0)


def test_perframe_f1_from_coco_gt():
    gt = {"images": [{"id": 1}, {"id": 2}],
          "annotations": [{"image_id": 1, "bbox": [0, 0, 10, 10], "category_id": 2, "iscrowd": 0},
                          {"image_id": 1, "bbox": [40, 40, 10, 10], "category_id": 2, "iscrowd": 1}]}
    idx = gt_arrays_from_coco(gt)
    assert idx[1]["gt_boxes"].tolist() == [[0, 0, 10, 10]] and idx[1]["ign_boxes"].tolist() == [[40, 40, 50, 50]]
    out = perframe_f1({1: np.array([d(0, 0, 10, 10, 0.9, CAR), d(40, 40, 50, 50, 0.9, CAR)])}, idx, 0.25)
    assert out[1]["f1"] == 1.0 and out[2]["f1"] == 1.0  # image 2: no GT, no dets
