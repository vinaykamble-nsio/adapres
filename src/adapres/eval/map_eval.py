"""Detections -> COCO results -> mAP (overall, per slice, per class). PLAN §5.3.

Detections use the PLAN §2.1 layout: rows of (x1, y1, x2, y2, score, coco_cls) in
original-image pixel coordinates, with coco_cls the Ultralytics COCO-80 index.
Classes outside the evaluated set are dropped before evaluation (classmap.py).
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import Iterable, Mapping

import numpy as np

from ..data.classmap import COCO_TO_EVAL, EVAL_CATEGORIES
from .coco_metrics import ap_from_eval

DET_COLS = ("x1", "y1", "x2", "y2", "score", "coco_cls")


def dets_to_coco_results(dets_by_image: Mapping[int, np.ndarray]) -> list[dict]:
    """{image_id: (N,6) array} -> list of COCO result dicts (eval category ids, xywh)."""
    out: list[dict] = []
    for image_id, d in dets_by_image.items():
        d = np.asarray(d, dtype=np.float64).reshape(-1, 6)
        for x1, y1, x2, y2, s, c in d:
            eid = COCO_TO_EVAL.get(int(c))
            if eid is None:
                continue
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            out.append({"image_id": int(image_id), "category_id": eid,
                        "bbox": [float(x1), float(y1), float(w), float(h)], "score": float(s)})
    return out


class CocoEvaluator:
    """Holds one GT (loaded once) and evaluates result lists, optionally on image subsets."""

    def __init__(self, gt: dict, max_dets: int = 300):
        from pycocotools.coco import COCO

        self.max_dets = max_dets
        self.gt = COCO()
        self.gt.dataset = json.loads(json.dumps(gt))
        with contextlib.redirect_stdout(io.StringIO()):
            self.gt.createIndex()
        self.all_img_ids = sorted(self.gt.getImgIds())

    def evaluate(self, results: list[dict], img_ids: Iterable[int] | None = None) -> dict:
        from pycocotools.cocoeval import COCOeval

        ids = sorted(int(i) for i in (img_ids if img_ids is not None else self.all_img_ids))
        unknown = set(ids) - set(self.all_img_ids)
        if unknown:
            raise ValueError(f"{len(unknown)} img_ids not in GT, e.g. {sorted(unknown)[:5]}")
        out = {"n_images": len(ids), "n_dets": 0, "map50_95": -1.0, "map50": -1.0,
               "per_class_ap50_95": {}, "per_class_ap50": {}}
        if not ids:
            return out
        idset = set(ids)
        res = [r for r in results if r["image_id"] in idset]
        out["n_dets"] = len(res)
        with contextlib.redirect_stdout(io.StringIO()):
            if res:
                dt = self.gt.loadRes(res)
            else:
                # pycocotools.loadRes([]) raises; build a genuinely empty result set instead.
                # COCOeval then scores precision 0 for every category with GT.
                from pycocotools.coco import COCO

                dt = COCO()
                dt.dataset = {"images": self.gt.dataset["images"], "categories": self.gt.dataset["categories"],
                              "annotations": []}
                dt.createIndex()
            ev = COCOeval(self.gt, dt, iouType="bbox")
            ev.params.imgIds = ids
            ev.params.maxDets = [1, 10, self.max_dets]
            ev.evaluate()
            ev.accumulate()
        out["map50_95"] = ap_from_eval(ev)
        out["map50"] = ap_from_eval(ev, iou=0.5)
        for cid, (name, _) in EVAL_CATEGORIES.items():
            out["per_class_ap50_95"][name] = ap_from_eval(ev, cat_ids=[cid])
            out["per_class_ap50"][name] = ap_from_eval(ev, iou=0.5, cat_ids=[cid])
        return out

    def evaluate_slices(self, results: list[dict], slices: Mapping[str, Iterable[int]]) -> dict:
        return {name: self.evaluate(results, ids) for name, ids in slices.items()}

    def matched(self, results: list[dict]):
        """COCOeval after evaluate() (per-image matching) on all images, before accumulate()."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with contextlib.redirect_stdout(io.StringIO()):
            if results:
                dt = self.gt.loadRes(results)
            else:
                dt = COCO()
                dt.dataset = {"images": self.gt.dataset["images"], "categories": self.gt.dataset["categories"],
                              "annotations": []}
                dt.createIndex()
            ev = COCOeval(self.gt, dt, iouType="bbox")
            ev.params.imgIds = list(self.all_img_ids)
            ev.params.maxDets = [1, 10, self.max_dets]
            ev.evaluate()
        return ev


def _accumulate_masked(ev, eval_imgs: list, keep: set[int] | None) -> dict:
    """accumulate() over a subset of images by masking other images' per-image results to None
    (COCOeval.accumulate skips None). Equivalent to evaluating the subset on its own, because
    matching in COCOeval is per image and the image order is unchanged."""
    saved = ev.evalImgs
    ev.evalImgs = eval_imgs if keep is None else [e if (e is not None and e["image_id"] in keep) else None
                                                  for e in eval_imgs]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ev.accumulate()
        out = {"map50_95": ap_from_eval(ev), "map50": ap_from_eval(ev, iou=0.5), "per_class_ap50_95": {},
               "per_class_ap50": {}}
        for cid, (name, _) in EVAL_CATEGORIES.items():
            out["per_class_ap50_95"][name] = ap_from_eval(ev, cat_ids=[cid])
            out["per_class_ap50"][name] = ap_from_eval(ev, iou=0.5, cat_ids=[cid])
        return out
    finally:
        ev.evalImgs = saved


class ComposedEvaluator:
    """mAP of any per-frame selection among precomputed levels without re-matching.

    COCOeval.evaluate() matches detections to GT independently per image, so the per-image
    results of a policy that takes frame i from level L(i) are exactly level L(i)'s per-image
    results. evaluate() runs once per level; each policy then only re-runs accumulate()."""

    def __init__(self, gt: dict, dets_by_level: Mapping[str, Mapping[int, np.ndarray]], max_dets: int = 300):
        self.base = CocoEvaluator(gt, max_dets)
        self.levels = list(dets_by_level)
        self.ev = {lv: self.base.matched(dets_to_coco_results(d)) for lv, d in dets_by_level.items()}
        ref = self.ev[self.levels[0]]
        self.order = list(ref._paramsEval.imgIds)
        self.I0 = len(self.order)
        for lv, ev in self.ev.items():
            if list(ev._paramsEval.imgIds) != self.order or list(ev._paramsEval.catIds) != list(ref._paramsEval.catIds) \
                    or len(ev.evalImgs) != len(ref.evalImgs):
                raise ValueError(f"level {lv} was evaluated with different params")
        self.n_dets = {lv: sum(len(np.asarray(v).reshape(-1, 6)) for v in d.values()) for lv, d in dets_by_level.items()}

    def compose(self, choice: Mapping[int, str]) -> list:
        missing = [i for i in self.order if i not in choice]
        if missing:
            raise ValueError(f"choice lacks {len(missing)} images, e.g. {missing[:3]}")
        pos_level = [choice[i] for i in self.order]
        bad = set(pos_level) - set(self.levels)
        if bad:
            raise ValueError(f"unknown levels {bad}")
        lists = {lv: self.ev[lv].evalImgs for lv in self.levels}
        return [lists[pos_level[k % self.I0]][k] for k in range(len(lists[self.levels[0]]))]

    def evaluate(self, choice: Mapping[int, str], slices: Mapping[str, Iterable[int]]) -> dict:
        composed = self.compose(choice)
        ref = self.ev[self.levels[0]]
        out = {}
        for name, ids in slices.items():
            ids = set(int(i) for i in ids)
            if not ids:
                out[name] = {"n_images": 0, "map50_95": -1.0, "map50": -1.0, "per_class_ap50_95": {},
                             "per_class_ap50": {}}
                continue
            out[name] = {"n_images": len(ids), **_accumulate_masked(ref, composed, ids)}
        return out
