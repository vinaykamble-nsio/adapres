"""ComposedEvaluator must equal a direct pycocotools evaluation of the selected detections,
overall and on slices, for arbitrary per-frame choices. Also random baselines."""

import numpy as np
import pytest
from conftest import box, make_frame

from adapres import baselines as B
from adapres.data.bdd_mot import parse_video_labels
from adapres.eval.coco_gt import build_coco_gt
from adapres.eval.map_eval import CocoEvaluator, ComposedEvaluator, dets_to_coco_results

COCO_OF = {"car": 2, "pedestrian": 0, "truck": 7, "bus": 5}


def synth(n_img=12, seed=0):
    rng = np.random.default_rng(seed)
    frames, dets = [], {"int8": {}, "fp16": {}}
    for i in range(n_img):
        boxes, gtb = [], []
        for j in range(int(rng.integers(0, 5))):
            cat = ["car", "pedestrian", "truck", "bus"][int(rng.integers(4))]
            x, y = rng.uniform(0, 150, 2)
            w, h = rng.uniform(10, 40, 2)
            boxes.append(box(f"{i}_{j}", cat, x, y, x + w, y + h))
            gtb.append((x, y, x + w, y + h, COCO_OF[cat]))
        if i % 4 == 0:
            boxes.append(box(f"{i}_ign", "other vehicle", 160, 160, 190, 190))
        frames.append(parse_video_labels([make_frame(f"v{i}", 0, boxes, None)])[0])
        for lv, noise, drop in (("fp16", 2.0, 0.1), ("int8", 5.0, 0.3)):
            d = []
            for (x1, y1, x2, y2, c) in gtb:
                if rng.random() < drop:
                    continue
                j = rng.normal(0, noise, 4)
                d.append([x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3], rng.uniform(0.3, 1.0), c])
            for _ in range(int(rng.integers(0, 4))):  # false positives, some low score
                x, y = rng.uniform(0, 170, 2)
                d.append([x, y, x + 20, y + 20, rng.uniform(0.001, 0.6), [0, 2, 7][int(rng.integers(3))]])
            dets[lv][i + 1] = np.asarray(d, float).reshape(-1, 6)
    gt = build_coco_gt([(i + 1, f, 200, 200) for i, f in enumerate(frames)])
    return gt, dets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_composed_equals_direct(seed):
    gt, dets = synth(seed=seed)
    ids = sorted(dets["fp16"])
    comp = ComposedEvaluator(gt, dets)
    direct = CocoEvaluator(gt)
    rng = np.random.default_rng(seed + 10)
    slices = {"all": ids, "odd": [i for i in ids if i % 2], "few": ids[:3]}
    for _ in range(3):
        choice = {i: ("int8" if rng.random() < 0.5 else "fp16") for i in ids}
        sel = {i: dets[choice[i]][i] for i in ids}
        got = comp.evaluate(choice, slices)
        for name, s in slices.items():
            exp = direct.evaluate(dets_to_coco_results(sel), s)
            assert got[name]["map50_95"] == pytest.approx(exp["map50_95"], abs=1e-12), name
            assert got[name]["map50"] == pytest.approx(exp["map50"], abs=1e-12), name
            assert got[name]["per_class_ap50_95"] == pytest.approx(exp["per_class_ap50_95"], abs=1e-12)
    # fixed policies reproduce the per-level evaluation
    for lv in ("int8", "fp16"):
        assert comp.evaluate({i: lv for i in ids}, {"all": ids})["all"]["map50_95"] == pytest.approx(
            direct.evaluate(dets_to_coco_results(dets[lv]))["map50_95"], abs=1e-12)


def test_composed_validation():
    gt, dets = synth()
    comp = ComposedEvaluator(gt, dets)
    with pytest.raises(ValueError, match="lacks"):
        comp.evaluate({1: "fp16"}, {"all": [1]})
    with pytest.raises(ValueError, match="unknown levels"):
        comp.evaluate({i: "fp32" for i in dets["fp16"]}, {"all": [1]})


def test_markov_estimate_and_matched_random():
    choices = {"a": [1, 1, 0, 0, 0, 1, 1, 1], "b": [1, 0, 0, 0]}
    m = B.estimate_markov(choices, 2)
    # transitions: a: 1-1,1-0,0-0,0-0,0-1,1-1,1-1 ; b: 1-0,0-0,0-0
    assert m["counts"].tolist() == [[4, 1], [2, 3]]
    assert m["init"].tolist() == [0.0, 1.0]
    lens = {f"s{i}": 400 for i in range(25)}
    r1 = B.markov_random(lens, m["P"], m["init"], seed=3)
    assert {k: v.tolist() for k, v in r1.items()} == {k: v.tolist() for k, v in B.markov_random(lens, m["P"], m["init"], 3).items()}
    est = B.estimate_markov(r1, 2)
    assert est["P"] == pytest.approx(m["P"], abs=0.03)       # matched transitions (law of large numbers)
    assert all(v[0] == 1 for v in r1.values())               # matched initial level (fp16)
    stationary = 1 / 3  # pi_0 = P10 / (P01 + P10) = 0.4 / (0.2 + 0.4)
    allc = np.concatenate(list(r1.values()))
    assert np.mean(allc == 0) == pytest.approx(0.4 / 0.6, abs=0.03)
    assert stationary < 0.4  # (sanity of the arithmetic above)


def test_iid_random():
    r = B.iid_random({"a": 10000}, 0.5, seed=1)
    assert np.mean(r["a"] == 0) == pytest.approx(0.5, abs=0.02)
    assert r["a"].tolist() == B.iid_random({"a": 10000}, 0.5, seed=1)["a"].tolist()
