import pytest

from adapres.data import subsets as S


def pool(n_day=30, n_night=15, n_dd=5):
    d = {f"d{i:03d}": "daytime" for i in range(n_day)}
    d.update({f"n{i:03d}": "night" for i in range(n_night)})
    d.update({f"x{i:03d}": "dawn_dusk" for i in range(n_dd)})
    return d


TARGETS = {"calib": {"daytime": 12, "night": 6, "dawn_dusk": 2},
           "test": {"daytime": 12, "night": 6, "dawn_dusk": 2}}


def test_deterministic_and_disjoint():
    a = S.select_disjoint_splits(pool(), TARGETS, seed=2027)
    b = S.select_disjoint_splits(pool(), TARGETS, seed=2027)
    assert a == b
    assert not set(a["splits"]["calib"]) & set(a["splits"]["test"])
    assert a["strata_counts"]["test"] == TARGETS["test"]
    assert a["strata_counts"]["calib"] == TARGETS["calib"]
    assert all(v == {} for v in a["shortfalls"].values())


def test_different_seed_differs():
    a = S.select_disjoint_splits(pool(), TARGETS, seed=1)
    b = S.select_disjoint_splits(pool(), TARGETS, seed=2)
    assert a["splits"] != b["splits"]


def test_input_order_does_not_matter():
    p = pool()
    rev = dict(reversed(list(p.items())))
    assert S.select_disjoint_splits(p, TARGETS, 7)["splits"] == S.select_disjoint_splits(rev, TARGETS, 7)["splits"]


def test_shortfall_filled_and_recorded():
    # only 3 dawn/dusk: test takes 2 first, calib gets 1 and is topped up from others
    out = S.select_disjoint_splits(pool(n_dd=3), TARGETS, seed=0)
    assert out["strata_counts"]["test"]["dawn_dusk"] == 2
    assert out["shortfalls"]["calib"] == {"dawn_dusk": 1}
    assert len(out["splits"]["calib"]) == 20 and len(out["splits"]["test"]) == 20
    assert not set(out["splits"]["calib"]) & set(out["splits"]["test"])


def test_not_enough_sequences():
    with pytest.raises(S.SubsetError, match="need 40"):
        S.select_disjoint_splits(pool(10, 5, 2), TARGETS, seed=0)


def test_unknown_stratum():
    with pytest.raises(S.SubsetError):
        S.select_disjoint_splits({"a": "evening"}, {"calib": {"daytime": 0}}, 0)


def test_sample_frames():
    keys = [(s, i) for s in ("a", "b") for i in range(100)]
    x = S.sample_frames(keys, 50, seed=3)
    assert x == S.sample_frames(list(reversed(keys)), 50, seed=3)
    assert len(set(x)) == 50 and set(x) <= set(keys)
    with pytest.raises(S.SubsetError):
        S.sample_frames(keys, 201, 0)
