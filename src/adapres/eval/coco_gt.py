"""Dataset GT -> COCO-format JSON (PLAN §5.3). Ignore classes and crowd boxes become
iscrowd=1 annotations so detections on them are neither TP nor FP."""

from __future__ import annotations

from typing import Iterable

from ..data.bdd_mot import Frame
from ..data.classmap import EVAL_CATEGORIES, coco_categories


def build_coco_gt(items: Iterable[tuple[int, Frame, int, int]], ignore_mode: str = "all_eval_classes",
                  description: str = "") -> dict:
    """items: (image_id, frame, width, height). Returns a COCO dict."""
    if ignore_mode not in ("all_eval_classes", "none"):
        raise ValueError(f"bad ignore_mode {ignore_mode!r}")
    images, anns = [], []
    seen_ids: set[int] = set()
    ann_id = 1
    for image_id, fr, w, h in items:
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id {image_id}")
        seen_ids.add(image_id)
        images.append({"id": int(image_id), "file_name": fr.name, "width": int(w), "height": int(h),
                       "seq": fr.seq, "frame_idx": fr.frame_idx})
        for b in fr.boxes:
            # clip to image bounds
            x1, y1 = max(0.0, b.x1), max(0.0, b.y1)
            x2, y2 = min(float(w), b.x2), min(float(h), b.y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            base = {"image_id": int(image_id), "bbox": [x1, y1, bw, bh], "area": bw * bh,
                    "track_id": b.track_id, "bdd_category": b.category}
            if b.kind == "eval":
                anns.append({**base, "id": ann_id, "category_id": b.eval_id, "iscrowd": int(b.crowd)})
                ann_id += 1
            elif ignore_mode == "all_eval_classes":
                for cid in EVAL_CATEGORIES:
                    anns.append({**base, "id": ann_id, "category_id": cid, "iscrowd": 1, "ignore_region": 1})
                    ann_id += 1
    return {"info": {"description": description, "version": "1"}, "licenses": [],
            "images": images, "annotations": anns, "categories": coco_categories()}


def gt_counts(frame: Frame) -> int:
    """Number of evaluated (non-ignore, non-crowd) GT objects in a frame."""
    return sum(1 for b in frame.boxes if b.kind == "eval" and not b.crowd)
