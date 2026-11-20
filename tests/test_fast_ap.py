"""FastAP must equal pycocotools exactly: mixed per-frame levels, subsets, and duplicated
images (the cluster-bootstrap case, checked against a GT with physically duplicated images)."""

import numpy as np
import pytest
from test_composed_eval import synth

from adapres.eval.fast_ap import FastAP, cluster_bootstrap
from adapres.eval.map_eval import CocoEvaluator, ComposedEvaluator, dets_to_coco_results


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fastap_equals_pycocotools_mixed_levels(seed):
    gt, dets = synth(n_img=15, seed=seed)
    comp = ComposedEvaluator(gt, dets)
    fa = FastAP(comp)
    ids = sorted(dets["fp16"])
    rng = np.random.default_rng(seed)
    for _ in range(4):
        choice = {i: ("int8" if rng.random() < 0.5 else "fp16") for i in ids}
        sub = sorted(rng.choice(ids, size=9, replace=False).tolist())
        for sel in (ids, sub):
            exp = comp.evaluate(choice, {"s": sel})["s"]
            got = fa.ap(sel, [choice[i] for i in sel])
            assert got["map50_95"] == pytest.approx(exp["map50_95"], abs=1e-12)
            assert got["map50"] == pytest.approx(exp["map50"], abs=1e-12)


def test_fastap_with_duplicated_images_equals_physical_duplication():
    gt, dets = synth(n_img=10, seed=5)
    comp = ComposedEvaluator(gt, dets)
    fa = FastAP(comp)
    ids = sorted(dets["fp16"])
    multiset = ids + [ids[2], ids[2], ids[7]]           # bootstrap-style repetition, originals first
    # physically duplicate: new image ids after all originals, same boxes, same detections
    gt2 = {k: (list(v) if isinstance(v, list) else v) for k, v in gt.items()}
    gt2["images"] = list(gt["images"])
    gt2["annotations"] = list(gt["annotations"])
    dets2 = dict(dets["fp16"])
    next_id, next_ann = 1000, 10_000
    for src in multiset[len(ids):]:
        im = next(x for x in gt["images"] if x["id"] == src)
        gt2["images"].append({**im, "id": next_id})
        for a in gt["annotations"]:
            if a["image_id"] == src:
                gt2["annotations"].append({**a, "id": next_ann, "image_id": next_id})
                next_ann += 1
        dets2[next_id] = dets["fp16"][src]
        next_id += 1
    exp = CocoEvaluator(gt2).evaluate(dets_to_coco_results(dets2))
    got = fa.ap(multiset, "fp16")
    assert got["map50_95"] == pytest.approx(exp["map50_95"], abs=1e-12)
    assert got["map50"] == pytest.approx(exp["map50"], abs=1e-12)


def test_cluster_bootstrap_paired_and_deterministic():
    gt, dets = synth(n_img=16, seed=7)
    comp = ComposedEvaluator(gt, dets)
    fa = FastAP(comp)
    ids = sorted(dets["fp16"])
    seq_images = {f"s{k}": ids[4 * k:4 * k + 4] for k in range(4)}
    pol = {"a": {i: "fp16" for i in ids}, "b": {i: "int8" for i in ids}, "a2": {i: "fp16" for i in ids}}
    out = cluster_bootstrap(fa, seq_images, pol, [("a", "b"), ("a", "a2")], 50, seed=1,
                            random_groups={"ab_mean": ["a", "b"]})
    assert out == cluster_bootstrap(fa, seq_images, pol, [("a", "b"), ("a", "a2")], 50, seed=1,
                                    random_groups={"ab_mean": ["a", "b"]})
    same = out["pairs"]["a - a2"]["map50_95"]
    assert same["ci95"] == [0.0, 0.0] and same["excludes_zero"] is False   # paired: identical policies -> 0
    lo, hi = out["per_policy"]["a"]["map50_95"]["ci95"]
    full = fa.ap(ids, "fp16")["map50_95"]
    assert lo <= full <= hi
    assert "ab_mean" in out["per_policy"]
