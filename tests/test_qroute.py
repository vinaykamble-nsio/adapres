"""Q-ROUTE core: predictors (incl. numpy tree export), controller == fast replay (persistence, deadline,
hardware slowdown), lambda tuning, oracle, contention schedule, VisDrone parser, contention process."""

import numpy as np
import pytest

from adapres import qroute as Q
from adapres.data import visdrone as VD
from adapres.hw import contention as HC

CFG = list(Q.DEFAULT_CONFIGS)
CAR, PERSON = 2, 0
ISO = {"fp16_640": 2.7, "int8_640": 2.0, "fp16_480": 1.8, "int8_480": 1.4, "fp16_320": 1.1, "int8_320": 0.9}


def synth(n_seq=6, T=50, seed=0):
    """Frames with persistent 'small object' periods where low resolution hurts (learnable from the
    previous frame's box sizes), INT8 adds small noise."""
    rng = np.random.default_rng(seed)
    seq_frames, cues, dets, f1, area = {}, {}, {c: {} for c in CFG}, {c: {} for c in CFG}, {}
    iid = 1
    for s in range(n_seq):
        ids, small = [], False
        for _t in range(T):
            small = (rng.random() < 0.9) if small else (rng.random() < 0.15)
            cues[iid] = {"m": float(rng.random()), "c": float(rng.random()), "d": float(rng.random() ** 8)}
            base = float(rng.uniform(0.5, 0.9))
            for c in CFG:
                lv, res = Q.parse_config(c)
                k = int(rng.integers(2, 12))
                w = rng.uniform(5, 15, k) if small else rng.uniform(40, 120, k)
                sc = rng.uniform(0.3, 1.0, k) * (0.9 if lv == "int8" else 1.0) * (0.7 if (small and res < 640) else 1.0)
                x = rng.uniform(0, 1100, k)
                y = rng.uniform(0, 600, k)
                dets[c][iid] = np.column_stack([x, y, x + w, y + w, sc, rng.choice([CAR, PERSON], k)])
                pen = (0.3 if res == 320 else 0.15 if res == 480 else 0.0) * (1.0 if small else 0.1)
                f1[c][iid] = base - pen - (0.02 if lv == "int8" else 0.0) + float(rng.normal(0, 0.01))
            area[iid] = 1280 * 720.0
            ids.append(iid)
            iid += 1
        seq_frames[f"s{s}"] = ids
    return seq_frames, cues, dets, f1, area


@pytest.fixture(scope="module")
def data():
    sf, cues, dets, f1, area = synth()
    qd = Q.build_qdata(sf, CFG, dets, area, cues, f1, 0.25)
    return qd, dets, area, cues


def test_training_rows_shapes(data):
    qd, *_ = data
    X, Y, G = Q.training_rows(qd, "fp16_640", use_img=False)
    assert X.shape == (6 * 49 * 6, len(Q.feature_names(False))) and Y.shape == (X.shape[0], 6)
    assert np.allclose(Y[:, 0], 0.0)                     # reference regret is zero by definition
    X2, _, _ = Q.training_rows(qd, "fp16_640", use_img=True, use_temporal=False)
    assert X2.shape[1] == len(Q.feature_names(True, False))


def test_gbt_numpy_export_matches_sklearn(data):
    qd, *_ = data
    X, Y, _ = Q.training_rows(qd, "fp16_640", use_img=True)
    m = Q.GBTMulti(max_depth=3, max_iter=30).fit(X[:1200], Y[:1200])
    assert m.validate(X[1200:1500]) < 1e-9
    r = Q.RidgeMulti(1.0).fit(X, Y)
    assert np.allclose(r.predict(X[:5]), np.array([r.predict_one(x) for x in X[:5]]))


def test_select_predictor_finds_signal(data):
    qd, *_ = data
    X, Y, G = Q.training_rows(qd, "fp16_640", use_img=False)
    cands = [{"kind": "ridge", "alpha": 1.0}, {"kind": "gbt", "max_depth": 2, "max_iter": 30, "learning_rate": 0.1}]
    sel = Q.select_predictor(X, Y, G, cands, ref_idx=0, seed=2027, n_folds=3)
    assert sel["best_spearman"] > 0.3 and set(sel["fold_of_group"]) == set(range(6))
    assert {t["kind"] for t in sel["table"]} == {"ridge", "gbt"}


def _params(**kw):
    base = dict(configs=CFG, ref="fp16_640", lam=0.5, K=2, cost_iso=ISO, overhead=0.05)
    base.update(kw)
    return Q.QParams(**base)


@pytest.mark.parametrize("kw", [dict(), dict(K=1), dict(K=3, lam=1.5), dict(deadline=1.6), dict(use_hw=False),
                                dict(lam=0.2, deadline=2.2, K=2)])
def test_controller_equals_fast_replay(data, kw):
    qd, dets, area, cues = data
    X, Y, G = Q.training_rows(qd, "fp16_640", use_img=False)
    model = Q.RidgeMulti(1.0).fit(X, Y)
    tables = Q.prediction_table(qd, lambda si: model, use_img=False)
    sched = Q.contention_slowdown([len(i) for i in qd.image_ids], CFG,
                                  {0.0: {c: 1.0 for c in CFG}, 0.9: {c: 1.8 for c in CFG}}, [0.0, 0.9], 15)
    p = _params(**kw)
    fast = Q.fast_replay_q(tables, p, sched)
    for si, ids in enumerate(qd.image_ids):
        ctl = Q.QRouteController(model, p)
        got = []
        for t, iid in enumerate(ids):
            out = ctl.step(cues[iid])
            got.append(CFG.index(out["config"]))
            ctl.observe(dets[out["config"]][iid], area[iid], observed_cost=ISO[out["config"]] * sched[si][t, CFG.index(out["config"])])
        assert got == fast[si].tolist(), (kw, si)


def test_tune_lambda_meets_budget_and_oracle(data):
    qd, *_ = data
    X, Y, G = Q.training_rows(qd, "fp16_640", use_img=False)
    model = Q.RidgeMulti(1.0).fit(X, Y)
    tables = Q.prediction_table(qd, lambda si: model, use_img=False)
    p = _params(overhead=0.0)
    for budget in (1.5, 2.0):
        lam, c = Q.tune_lambda(tables, p, budget)
        assert abs(c - budget) / budget < 0.05, (budget, c)
    ids = [i for s in qd.image_ids for i in s]
    R = Q.regret_matrix(qd, "fp16_640", ids)
    cost = np.array([ISO[c] for c in CFG])
    ch, c = Q.oracle_choices(R, cost, 1.5)
    assert abs(c - 1.5) < 0.05
    # oracle never has more total regret than any fixed config with lower or equal mean cost
    for j, cc in enumerate(cost):
        if cc <= c + 1e-9:
            assert R[np.arange(len(ch)), ch].sum() <= R[:, j].sum() + 1e-9


def test_constant_predictor_is_content_blind(data):
    qd, *_ = data
    pred = Q.ConstantPredictor(np.array([0, .1, .1, .2, .3, .4]))
    tables = Q.prediction_table(qd, lambda si: pred, use_img=False)
    ch = Q.fast_replay_q(tables, _params(K=1))
    assert all(len(set(c[1:].tolist())) == 1 for c in ch)   # same choice on every non-first frame


def test_visdrone_parser(tmp_path):
    f = tmp_path / "uav0001.txt"
    f.write_text("1,1,10,20,30,40,1,4,0,0\n1,2,0,0,5,5,0,1,0,0\n2,1,12,20,30,40,1,7,0,0\n2,3,50,50,10,10,1,10,0,0\n")
    fr = VD.parse_sequence(f)
    assert [x.frame_idx for x in fr] == [1, 2]
    assert fr[0].boxes[0].eval_id == 2 and fr[0].boxes[1].kind == "ignore"          # car; score 0 -> ignore
    assert fr[1].boxes[0].kind == "ignore" and fr[1].boxes[1].eval_id == 6         # tricycle ignore; motor
    bad = tmp_path / "bad.txt"
    bad.write_text("1,1,10,20\n")
    with pytest.raises(VD.LabelFormatError, match="expected >= 8"):
        VD.parse_sequence(bad)


def test_contention_process_lifecycle():
    with HC.ContentionGenerator(period_s=0.01, worker=HC._sleep_worker, device="cpu") as g:
        g.set_level(0.5)
        assert g.proc.is_alive()
        with pytest.raises(ValueError):
            g.set_level(1.5)
    assert not g.proc.is_alive()
