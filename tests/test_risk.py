"""Controller v2: features, model selection, cross-fitting, budget threshold, fast replay == controller."""

import numpy as np
import pytest

from adapres import risk as R

CAR, PERSON, DOG = 2, 0, 16


def synth(n_seq=6, T=60, seed=0):
    rng = np.random.default_rng(seed)
    seq_frames, cues, dets, f1, area = {}, {}, {"int8": {}, "fp16": {}}, {"int8": {}, "fp16": {}}, {}
    iid = 1
    for s in range(n_seq):
        ids = []
        hard = np.zeros(T, bool)     # difficulty persists over time (Markov), as in real video
        for t in range(1, T):
            hard[t] = (rng.random() < 0.9) if hard[t - 1] else (rng.random() < 0.1)
        for t in range(T):
            cues[iid] = {"m": float(rng.random() * 0.1), "c": float(rng.random() * 0.2 + (0.15 if hard[t] else 0.0)),
                         "d": float(rng.random() ** 6)}
            for lv in R.LEVELS:
                k = int(rng.integers(0, 15)) + (6 if hard[t] else 0)
                sc = rng.uniform(0.1, 1.0, k) * (0.7 if (hard[t] and lv == "int8") else 1.0)
                x = rng.uniform(0, 1200, k)
                y = rng.uniform(0, 700, k)
                w = rng.uniform(5, 80, k)
                dets[lv][iid] = np.column_stack([x, y, x + w, y + w, sc, rng.choice([CAR, PERSON, DOG], k)]) if k else np.zeros((0, 6))
            f1["fp16"][iid] = float(rng.uniform(0.4, 0.9))
            f1["int8"][iid] = f1["fp16"][iid] - (0.2 if hard[t] else 0.0) + float(rng.normal(0, 0.03))
            area[iid] = 1280 * 720.0
            ids.append(iid)
            iid += 1
        seq_frames[f"s{s}"] = ids
    return seq_frames, cues, dets, f1, area


def test_det_stats():
    d = np.array([[0, 0, 10, 10, 0.9, CAR], [0, 0, 100, 100, 0.4, PERSON], [0, 0, 5, 5, 0.2, CAR],
                  [0, 0, 5, 5, 0.95, DOG]])
    s = R.det_stats(d, 1e6)
    assert s["n"] == 2 and s["n_near_threshold"] == 1 and s["frac_low_conf"] == 0.5
    assert s["mean_score"] == pytest.approx(0.65) and s["class_entropy"] == pytest.approx(np.log(2))
    assert s["frac_small"] == 0.5  # 10x10 / 1e6 = 1e-4 (small); 100x100 / 1e6 = 1e-2 (not small)
    e = R.det_stats(None, 1.0)
    assert e["n"] == 0 and e["mean_log_rel_area"] == R.EMPTY_LOG_AREA


def _fit(seed=0):
    sf, cues, dets, f1, area = synth(seed=seed)
    sd = R.build_seqdata(sf, cues, dets, area, f1, 0.25)
    X, y, g = R.training_rows(sd)
    sel = R.select_and_fit(X, y, g, {"max_depth": [2, 3], "learning_rate": [0.1], "max_iter": [50]}, seed=2027, n_folds=3)
    return sd, sel, dets, area


def test_training_rows_and_selection_deterministic():
    sd, sel, _, _ = _fit()
    X, y, g = R.training_rows(sd)
    assert X.shape == (2 * sum(len(i) for i in sd.image_ids), len(R.FEATURES)) and len(set(g)) == 6
    sd2, sel2, _, _ = _fit()
    assert sel["params"] == sel2["params"] and sel["best_spearman"] == sel2["best_spearman"]
    assert sel["best_spearman"] > 0.2            # the synthetic hard frames are learnable
    assert set(sel["fold_of_group"]) == set(range(6))


def test_fast_replay_equals_controller():
    sd, sel, dets, area = _fit()
    rt = R.risk_table(sel["model"], sd)
    dseq = [np.array([sd.cues[i]["d"] for i in ids]) for ids in sd.image_ids]
    allr = np.concatenate([r.ravel() for r in rt])
    for tau in np.quantile(allr, [0.2, 0.5, 0.8]):
        for K in (1, 2, 3):
            p = R.V2Params(float(tau), K, theta_sc=0.3, H=2)
            fast = R.fast_replay_v2(rt, dseq, p)
            for si, ids in enumerate(sd.image_ids):
                ctl = R.RiskController(sel["model"], p)
                lv = []
                for iid in ids:
                    out = ctl.step(sd.cues[iid])
                    lv.append(R.LEVELS.index(out["level"]))
                    ctl.observe(dets[out["level"]][iid], area[iid])
                assert lv == fast[si].tolist(), (tau, K, si)


def test_budget_threshold_hits_target():
    sd, sel, _, _ = _fit()
    rt = R.risk_table(None, sd, {si: sel["fold_models"][sel["fold_of_group"][si]] for si in range(6)})
    dseq = [np.array([sd.cues[i]["d"] for i in ids]) for ids in sd.image_ids]
    for rho in (0.25, 0.5, 0.75):
        tau, frac = R.tau_for_budget(rt, dseq, K=2, rho=rho, theta_sc=0.9, H=3)
        assert abs(frac - rho) <= 0.02, (rho, frac)


def test_learned_beats_random_on_synthetic_gain():
    """Sanity of the whole idea on synthetic data: at a 50% INT8 budget the learned controller
    keeps more per-frame F1 than random placement with the same budget."""
    sd, sel, _, _ = _fit(seed=1)
    rt = R.risk_table(None, sd, {si: sel["fold_models"][sel["fold_of_group"][si]] for si in range(6)})
    dseq = [np.array([sd.cues[i]["d"] for i in ids]) for ids in sd.image_ids]
    tau, frac = R.tau_for_budget(rt, dseq, K=1, rho=0.5, theta_sc=2.0, H=0)
    ch = np.concatenate(R.fast_replay_v2(rt, dseq, R.V2Params(tau, 1, 2.0, 0)))
    g = np.array([sd.g[i] for ids in sd.image_ids for i in ids])
    loss_learned = g[ch == 0].sum()
    rng = np.random.default_rng(0)
    loss_random = np.mean([g[rng.permutation(len(g))[: int((ch == 0).sum())]].sum() for _ in range(200)])
    assert loss_learned < loss_random


def test_oracle_and_lagged():
    g = np.array([0.3, -0.1, 0.0, 0.5, 0.2, 0.0])
    ch = R.oracle_at_budget(g, 0.5)
    assert ch.tolist().count(0) == 3 and set(np.flatnonzero(ch == 0)) == {1, 2, 5}
    lag = R.lagged_oracle([g], tau=0.1)
    assert lag[0].tolist() == [1, 1, 0, 0, 1, 1]
    assert R.lagged_tau_for_budget([g], 0.5) == pytest.approx(np.quantile(g[:-1], 0.5) + 1e-12)
