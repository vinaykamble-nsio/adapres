import csv
import json

import pytest

from adapres.results import ExperimentWriter, MetaError, check_meta


def meta(**kw):
    m = {"exp_id": "e1", "dataset": "ds", "split": "calib", "seed": 2027, "git_commit": "abc",
         "env": {"gpu": "fake"}, "config_sha256": "x", "timestamp_utc": "t"}
    m.update(kw)
    return m


@pytest.mark.parametrize("bad", [
    {k: v for k, v in meta().items() if k != "git_commit"},
    meta(git_commit=None), meta(git_commit=""), meta(env={}), meta(seed="2027"),
])
def test_refuses_incomplete_meta(bad, tmp_path):
    with pytest.raises(MetaError):
        check_meta(bad)
    with pytest.raises(MetaError):
        ExperimentWriter(tmp_path, bad)
    assert not any(tmp_path.iterdir())  # nothing written


def test_writes_meta_and_index(tmp_path):
    w = ExperimentWriter(tmp_path, meta())
    assert json.loads((w.dir / "meta.json").read_text())["git_commit"] == "abc"
    w.write_csv("x.csv", [{"a": 1}], ["a"])
    w.register(policy="p", mode="replay")
    with pytest.raises(MetaError, match="duplicate"):
        w.register(policy="p", mode="replay")
    w.register(policy="p2", mode="replay", replace=True)
    rows = list(csv.DictReader(open(tmp_path / "index.csv")))
    assert len(rows) == 1 and rows[0]["policy"] == "p2" and rows[0]["path"] == "ds/e1"
