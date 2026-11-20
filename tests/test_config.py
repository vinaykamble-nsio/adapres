import copy

import pytest
import yaml
from conftest import REPO

from adapres import config as C


def test_default_config_validates(default_cfg):
    assert default_cfg["seed"] == 2027
    assert default_cfg["bdd"]["subsets"]["calib"]["n_sequences"] == 20
    assert default_cfg["bdd"]["subsets"]["test"]["n_sequences"] == 40


def test_preregistration_parses():
    pre = yaml.safe_load((REPO / "configs" / "preregistration.yaml").read_text())
    assert pre["delta_map_points"] == 0.5
    assert set(pre["hypotheses"]) >= {"H1", "H2", "H3", "H4"}
    assert pre["seed"] == 2027


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c.pop("seed"), "missing config key: seed"),
    (lambda c: c.__setitem__("seed", "2027"), "seed: expected"),
    (lambda c: c.__setitem__("seed", True), "got bool"),
    (lambda c: c["bdd"]["subsets"]["test"]["strata_targets"].__setitem__("night", 99), "!= n_sequences"),
    (lambda c: c["bdd"]["subsets"]["calib"]["strata_targets"].__setitem__("dusk", 0), "unknown strata"),
    (lambda c: c["splice"].__setitem__("min_segments", 1), "min_segments must be >= 2"),
    (lambda c: c["classmap"].__setitem__("ignore_mode", "drop"), "ignore_mode"),
])
def test_invalid_configs_rejected(default_cfg, mutate, msg):
    cfg = copy.deepcopy(default_cfg)
    mutate(cfg)
    with pytest.raises(C.ConfigError, match=msg):
        C.validate(cfg)


def test_overrides_and_smoke_paths(tmp_path):
    cfg = C.load(REPO / "configs" / "default.yaml", {"paths": {"work_root": str(tmp_path)}})
    assert cfg["paths"]["input_root"] == "/kaggle/input"  # untouched sibling key
    assert C.resolve_out(cfg, "results_dir", smoke=False) == tmp_path / "results"
    assert C.resolve_out(cfg, "results_dir", smoke=True) == tmp_path / "smoke" / "results"


def test_config_hash_is_order_independent(default_cfg):
    a = C.config_sha256(default_cfg)
    b = C.config_sha256(dict(reversed(list(copy.deepcopy(default_cfg).items()))))
    assert a == b
