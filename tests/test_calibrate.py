"""Calibration: fast_replay == reference replay(); grid/selection logic; lock + hash."""

import numpy as np
import pytest

from adapres import calibrate as CAL
from adapres.controller import ControllerParams
from adapres.replay import replay

CAR = 2


def synthetic(seed=0, n_seq=4, T=40):
    rng = np.random.default_rng(seed)
    seq_frames, cues, dets, f1 = {}, {}, {"int8": {}, "fp16": {}}, {"int8": {}, "fp16": {}}
    iid = 1
    for s in range(n_seq):
        ids = []
        for _t in range(T):
            cues[iid] = {"m": float(rng.random() * 0.1), "c": float(rng.random() * 0.2), "d": float(rng.random() ** 3)}
            for lv in ("int8", "fp16"):
                k = int(rng.integers(0, 12))
                dets[lv][iid] = np.column_stack([np.zeros((k, 4)), rng.random(k), np.full(k, CAR)]) if k else np.zeros((0, 6))
                f1[lv][iid] = float(rng.random())
            ids.append(iid)
            iid += 1
        seq_frames[f"s{s}"] = ids
    return seq_frames, cues, dets, f1


NORM = {"m": (0.005, 0.095), "c": (0.01, 0.19), "n": (1.0, 10.0), "u": (0.1, 0.7)}


@pytest.mark.parametrize("seed", range(6))
def test_fast_replay_equals_reference(seed):
    sf, cues, dets, f1 = synthetic(seed)
    a = CAL.build_arrays(sf, cues, dets, f1, tau_c=0.25, u_empty=0.3)
    rng = np.random.default_rng(100 + seed)
    for w in CAL.simplex_weights()[:: 7]:
        p = ControllerParams(weights=w, theta=float(rng.uniform(0.1, 0.6)), norm=NORM, alpha=float(rng.choice([0.6, 1.0])),
                             K_up=int(rng.integers(1, 4)), K_down=int(rng.integers(1, 6)), H=int(rng.integers(0, 4)),
                             theta_sc=float(rng.choice([0.3, 0.5])), scene_override=bool(rng.integers(2)))
        ref = [0 if r["level"] == "int8" else 1 for r in replay(p, sf, cues, dets, 0.25, 0.3)]
        assert CAL.fast_replay(a, p).tolist() == ref


def test_simplex_weights():
    ws = CAL.simplex_weights()
    assert len(ws) == 71 and ws[0] == {k: 0.2 for k in "mcnus"}
    assert all(abs(sum(w.values()) - 1) < 1e-12 and min(w.values()) >= 0 for w in ws)
    assert len({tuple(sorted(w.items())) for w in ws}) == 71


def test_policy_cost_and_open_loop():
    sf, cues, dets, f1 = synthetic(1)
    a = CAL.build_arrays(sf, cues, dets, f1, 0.25, 0.3)
    ch = np.zeros(len(a.image_ids), np.int8)
    ch[5:10] = 1
    c = CAL.policy_cost(ch, a, {"int8": 2.0, "fp16": 3.0}, switch_penalty_ms=1.0, ctrl_overhead_ms=0.5)
    assert c["switches"] == 2 and c["lat_ms"] == pytest.approx((2.0 * (len(ch) - 5) + 15 + 0.5 * len(ch) + 2) / len(ch))
    sh = CAL.open_loop_shat(a, {k: 0.2 for k in "mcnus"}, 1.0, 0.4, NORM)
    assert sh.shape == ch.shape and np.isfinite(sh).all()


def test_grid_and_select():
    sf, cues, dets, f1 = synthetic(2)
    a = CAL.build_arrays(sf, cues, dets, f1, 0.25, 0.3)
    base = {"start_level": "fp16", "H": 2, "scene_override": True}
    rows = CAL.grid_search(a, base, NORM, CAL.simplex_weights()[:3], [0.2, 0.5, 0.8], [1, 3], [1.0], [0.4],
                           {"int8": 2.0, "fp16": 3.0}, 0.1, 0.0)
    assert len(rows) == 3 * 3 * 2
    # fake mAP: decreasing in INT8 fraction; FP16 = 0.30, delta 0.5 pt -> limit 0.295
    mapf = lambda p: 0.30 - 0.02 * float(np.mean(CAL.fast_replay(a, p) == 0))  # noqa: E731
    sel = CAL.select(rows, mapf, fp16_map=0.30, delta_pts=0.5, f1_slack=1.0, top=4, max_eval=18)
    assert sel["status"] == "feasible" and sel["best"]["map50_95"] >= 0.295
    feas = [r for r in sel["evaluated"] if r["feasible"]]
    assert sel["best"]["saving"] == max(r["saving"] for r in feas)
    sel2 = CAL.select(rows, lambda p: 0.0, 0.30, 0.5, 1.0, 4, 8)
    assert sel2["status"].startswith("infeasible") and sel2["n_evaluated_stage_b"] == 8


def test_lock_roundtrip_and_tamper(tmp_path):
    p = ControllerParams(weights={k: 0.2 for k in "mcnus"}, theta=0.3, norm=NORM)
    h = CAL.lock(tmp_path / "locked.yaml", p, {"dataset": "x", "delta_pts": 0.5})
    p2, body, h2 = CAL.load_locked(tmp_path / "locked.yaml")
    assert h == h2 and p2.theta == 0.3 and p2.norm["n"] == (1.0, 10.0) and body["delta_pts"] == 0.5
    f = tmp_path / "locked.yaml"
    f.write_text(f.read_text().replace("theta: 0.3", "theta: 0.31"))
    with pytest.raises(CAL.LockError, match="modified"):
        CAL.load_locked(f)
    with pytest.raises(CAL.LockError, match="missing"):
        CAL.load_locked(tmp_path / "nope.yaml")
