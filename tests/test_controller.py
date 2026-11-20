"""Controller unit tests on synthetic sequences (PLAN §3.4 required tests + scoring/EMA)."""

import pytest

from adapres.controller import Controller, ControllerParams, normalise, params_from_config, score

NORM = {k: (0.0, 1.0) for k in ("m", "c", "n", "u")}
W = {"m": 0.2, "c": 0.2, "n": 0.2, "u": 0.2, "s": 0.2}


def P(**kw):
    base = dict(weights=W, theta=0.5, norm=NORM, alpha=1.0, K_up=3, K_down=3, H=2, theta_sc=0.5)
    base.update(kw)
    return ControllerParams(**base)


def cues(x, d=0.0):
    """All four continuous cues = x (so S = 0.8 x when s = 0)."""
    return {"m": x, "c": x, "n": x, "u": x, "d": d}


LOW, HIGH = cues(0.0), cues(1.0)   # S = 0 -> propose int8; S = 0.8 -> propose fp16


def run(ctl, seq):
    return [ctl.step(c)["level"] for c in seq]


def test_score_and_normalisation():
    assert normalise(5, 0, 10) == 0.5 and normalise(-1, 0, 10) == 0.0 and normalise(11, 0, 10) == 1.0
    p = P(norm={"m": (0, 2), "c": (0, 1), "n": (0, 10), "u": (0, 1)})
    S, s = score({"m": 1, "c": 0.5, "n": 20, "u": 0, "d": 0.6}, p)
    assert s == 1 and S == pytest.approx(0.2 * (0.5 + 0.5 + 1.0 + 0.0) + 0.2)


def test_constant_proposal_switches_after_exactly_K_frames():
    for K in (1, 2, 3, 5):
        ctl = Controller(P(K_up=K, K_down=K))
        levels = run(ctl, [LOW] * 10)
        first_int8 = levels.index("int8")
        assert first_int8 == K - 1  # K consecutive int8 proposals: frames 0..K-1, switch on frame K-1
        assert levels[:K - 1] == ["fp16"] * (K - 1) and set(levels[K - 1:]) == {"int8"}


def test_oscillating_proposals_never_switch_with_K_gt_1():
    for K in (2, 3, 5):
        ctl = Controller(P(K_up=K, K_down=K))
        assert set(run(ctl, [LOW, HIGH] * 20)) == {"fp16"}
    # with K = 1 the same input flips every frame
    ctl = Controller(P(K_up=1, K_down=1))
    assert run(ctl, [LOW, HIGH] * 3) == ["int8", "fp16"] * 3


def test_interrupted_run_restarts_count():
    ctl = Controller(P(K_up=3, K_down=3))
    assert run(ctl, [LOW, LOW, HIGH, LOW, LOW, LOW]) == ["fp16"] * 5 + ["int8"]


def test_asymmetric_persistence():
    ctl = Controller(P(K_up=1, K_down=3))
    lv = run(ctl, [LOW] * 3 + [HIGH] + [LOW] * 3)
    assert lv == ["fp16", "fp16", "int8", "fp16", "fp16", "fp16", "int8"]  # up immediately, down after 3


def test_scene_change_override_fires_on_same_frame_and_holds_H():
    ctl = Controller(P(K_up=2, K_down=2, H=2))
    run(ctl, [LOW] * 4)                                 # settled on int8
    assert ctl.state["L"] == "int8"
    out = ctl.step(cues(0.0, d=0.9))                     # scene change, low complexity otherwise
    assert out["level"] == "fp16" and out["override"] == 1 and out["switched"] == 1
    # held at fp16 for H=2 frames even though proposals are int8, then K_down=2 more to go back down
    assert run(ctl, [LOW] * 5) == ["fp16", "fp16", "fp16", "int8", "int8"]


def test_override_disabled():
    ctl = Controller(P(K_up=2, K_down=2, scene_override=False, weights={"m": 0.25, "c": 0.25, "n": 0.25, "u": 0.25, "s": 0.0}))
    run(ctl, [LOW] * 4)
    assert ctl.step(cues(0.0, d=0.9))["level"] == "int8"


def test_ema_smooths_a_single_spike():
    ctl = Controller(P(alpha=0.3, K_up=1, K_down=1))
    lv = run(ctl, [LOW] * 5 + [HIGH] + [LOW] * 3)
    assert lv[5] == "int8"  # S_hat = 0.3*0.8 = 0.24 < theta: a one-frame spike does not upgrade
    ctl2 = Controller(P(alpha=1.0, K_up=1, K_down=1))
    assert run(ctl2, [LOW] * 5 + [HIGH])[5] == "fp16"


def test_reset_and_determinism():
    seq = [LOW, HIGH, LOW, LOW, cues(0.3, d=0.7), HIGH, LOW] * 3
    a = Controller(P())
    la = run(a, seq)
    a.reset()
    assert run(a, seq) == la == run(Controller(P()), seq)


@pytest.mark.parametrize("bad", [dict(weights={**W, "m": 0.3}), dict(alpha=0.0), dict(K_up=0),
                                 dict(norm={**NORM, "n": (1.0, 1.0)}), dict(levels=("int4", "int8", "fp16")),
                                 dict(start_level="fp32")])
def test_param_validation(bad):
    with pytest.raises(ValueError):
        P(**bad)


def test_params_from_config(default_cfg):
    p = params_from_config(default_cfg["controller"], theta=0.4, norm=NORM)
    assert p.K_up == p.K_down == default_cfg["controller"]["K"] and p.levels == ("int8", "fp16")
