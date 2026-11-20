import pytest

from adapres.data import classmap as cm


@pytest.mark.parametrize("bdd,eid", [("pedestrian", 1), ("rider", 1), ("car", 2), ("truck", 3),
                                     ("bus", 4), ("train", 5), ("motorcycle", 6), ("bicycle", 7)])
def test_bdd_eval_classes(bdd, eid):
    assert cm.bdd_category(bdd) == ("eval", eid)


@pytest.mark.parametrize("bdd", ["other person", "other vehicle", "trailer"])
def test_bdd_ignore_classes(bdd):
    assert cm.bdd_category(bdd) == ("ignore", None)


def test_unknown_bdd_category_raises():
    with pytest.raises(cm.UnknownCategoryError):
        cm.bdd_category("traffic light")


@pytest.mark.parametrize("coco,eid", [(0, 1), (2, 2), (7, 3), (5, 4), (6, 5), (3, 6), (1, 7)])
def test_coco_to_eval(coco, eid):
    # Ultralytics COCO-80 indices: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 6 train, 7 truck
    assert cm.coco_to_eval(coco) == eid


@pytest.mark.parametrize("coco", [4, 8, 9, 15, 79])  # airplane, boat, traffic light, cat, toothbrush
def test_unmapped_coco_dropped(coco):
    assert cm.coco_to_eval(coco) is None


def test_mapping_is_bijective_on_eval_set():
    cocos = [c for _, c in cm.EVAL_CATEGORIES.values()]
    assert len(set(cocos)) == len(cocos) == 7
    assert {e["id"] for e in cm.coco_categories()} == set(range(1, 8))
    # every BDD eval class maps to a COCO class that maps back to the same eval id
    for eid in cm.BDD_TO_EVAL.values():
        assert cm.coco_to_eval(cm.EVAL_CATEGORIES[eid][1]) == eid
