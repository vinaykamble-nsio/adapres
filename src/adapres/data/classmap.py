"""COCO -> BDD100K MOT class mapping (PLAN §5.1).

Evaluation categories use our own ids 1..7. Pedestrian and rider are merged into
person because COCO has no rider class. 'other person', 'other vehicle' and
'trailer' are ignore regions (iscrowd=1).
"""

from __future__ import annotations

# Our evaluated categories: eval_id -> (name, Ultralytics COCO-80 index)
EVAL_CATEGORIES: dict[int, tuple[str, int]] = {
    1: ("person", 0),
    2: ("car", 2),
    3: ("truck", 7),
    4: ("bus", 5),
    5: ("train", 6),
    6: ("motorcycle", 3),
    7: ("bicycle", 1),
}

# BDD MOT category -> eval_id
BDD_TO_EVAL: dict[str, int] = {
    "pedestrian": 1,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
}

BDD_IGNORE: frozenset[str] = frozenset({"other person", "other vehicle", "trailer"})

COCO_TO_EVAL: dict[int, int] = {coco: eid for eid, (_, coco) in EVAL_CATEGORIES.items()}


class UnknownCategoryError(KeyError):
    pass


def bdd_category(cat: str) -> tuple[str, int | None]:
    """Return ('eval', eval_id) or ('ignore', None). Unknown categories raise, so a
    label-format surprise is caught instead of silently dropping boxes."""
    if cat in BDD_TO_EVAL:
        return "eval", BDD_TO_EVAL[cat]
    if cat in BDD_IGNORE:
        return "ignore", None
    raise UnknownCategoryError(f"unknown BDD MOT category: {cat!r}")


def coco_to_eval(coco_cls: int) -> int | None:
    """Map a COCO-80 class index from the detector to an eval id, or None (dropped)."""
    return COCO_TO_EVAL.get(int(coco_cls))


def coco_categories() -> list[dict]:
    return [{"id": eid, "name": name, "supercategory": "object"} for eid, (name, _) in EVAL_CATEGORIES.items()]
