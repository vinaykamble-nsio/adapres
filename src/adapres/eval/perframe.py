"""Per-frame F1@0.5 (PLAN §6): the oracle's and the calibration proxy's quality signal.

Rules
- Detections with score >= tau_c only; classes outside the eval set are dropped.
- Class-aware greedy matching in descending score order: a detection matches the
  unmatched non-ignore GT of the same class with the highest IoU, if IoU >= iou_thr.
- An unmatched detection that lies on an ignore box of the same class (intersection /
  detection area >= ioa_thr, the COCO crowd rule) is discarded: neither TP nor FP.
- F1 = 2TP / (2TP + FP + FN). No GT and no (kept) detections -> 1.0; no GT but some
  detections -> 0.0.
"""

from __future__ import annotations

import numpy as np

from ..data.classmap import COCO_TO_EVAL


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a (N,4) and b (M,4)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def ioa_det(dets: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """intersection / detection area, (N,M)."""
    dets = np.asarray(dets, float).reshape(-1, 4)
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    ix1 = np.maximum(dets[:, None, 0], boxes[None, :, 0])
    iy1 = np.maximum(dets[:, None, 1], boxes[None, :, 1])
    ix2 = np.minimum(dets[:, None, 2], boxes[None, :, 2])
    iy2 = np.minimum(dets[:, None, 3], boxes[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area[:, None] > 0, inter / area[:, None], 0.0)


def frame_counts(dets: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray, ign_boxes: np.ndarray,
                 ign_cls: np.ndarray, tau_c: float, iou_thr: float = 0.5, ioa_thr: float = 0.5) -> tuple[int, int, int]:
    """Return (TP, FP, FN) for one frame.

    dets: (N,6) x1,y1,x2,y2,score,coco_cls. gt_*: eval-class GT (xyxy, eval id).
    ign_*: ignore/crowd boxes (xyxy, eval id)."""
    d = np.asarray(dets, float).reshape(-1, 6)
    d = d[d[:, 4] >= tau_c]
    eid = np.array([COCO_TO_EVAL.get(int(c), -1) for c in d[:, 5]], dtype=int)
    keep = eid >= 0
    d, eid = d[keep], eid[keep]
    d_order = np.argsort(-d[:, 4], kind="stable")
    d, eid = d[d_order], eid[d_order]

    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    gt_cls = np.asarray(gt_cls, int).reshape(-1)
    ign_boxes = np.asarray(ign_boxes, float).reshape(-1, 4)
    ign_cls = np.asarray(ign_cls, int).reshape(-1)

    ious = iou_matrix(d[:, :4], gt_boxes) if len(d) and len(gt_boxes) else np.zeros((len(d), len(gt_boxes)))
    ioas = ioa_det(d[:, :4], ign_boxes) if len(d) and len(ign_boxes) else np.zeros((len(d), len(ign_boxes)))
    matched = np.zeros(len(gt_boxes), bool)
    tp = fp = 0
    for i in range(len(d)):
        cand = (gt_cls == eid[i]) & ~matched
        if cand.any():
            row = np.where(cand, ious[i], -1.0)
            j = int(np.argmax(row))
            if row[j] >= iou_thr:
                matched[j] = True
                tp += 1
                continue
        if len(ign_boxes) and np.any((ign_cls == eid[i]) & (ioas[i] >= ioa_thr)):
            continue  # on an ignore region: discarded
        fp += 1
    fn = int((~matched).sum())
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp + fp + fn == 0:
        return 1.0  # no GT and no kept detections
    return 2 * tp / (2 * tp + fp + fn)


def gt_arrays_from_coco(gt: dict) -> dict[int, dict[str, np.ndarray]]:
    """Index a COCO GT dict by image: eval boxes/classes and ignore boxes/classes (xyxy)."""
    idx: dict[int, dict[str, list]] = {im["id"]: {"gt_boxes": [], "gt_cls": [], "ign_boxes": [], "ign_cls": []}
                                       for im in gt["images"]}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        key = "ign" if a["iscrowd"] else "gt"
        idx[a["image_id"]][f"{key}_boxes"].append([x, y, x + w, y + h])
        idx[a["image_id"]][f"{key}_cls"].append(a["category_id"])
    return {k: {kk: np.asarray(vv, float if "boxes" in kk else int).reshape((-1, 4) if "boxes" in kk else (-1,))
                for kk, vv in v.items()} for k, v in idx.items()}


def perframe_f1(dets_by_image: dict[int, np.ndarray], gt_index: dict[int, dict[str, np.ndarray]],
                tau_c: float, iou_thr: float = 0.5, ioa_thr: float = 0.5) -> dict[int, dict]:
    out = {}
    for image_id, g in gt_index.items():
        d = dets_by_image.get(image_id, np.zeros((0, 6)))
        tp, fp, fn = frame_counts(d, g["gt_boxes"], g["gt_cls"], g["ign_boxes"], g["ign_cls"], tau_c, iou_thr, ioa_thr)
        out[image_id] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1_from_counts(tp, fp, fn)}
    return out
