"""Table builder reads results/ only and refuses smoke/duplicate/mixed-commit inputs."""

import csv
import json
import shutil
import sys

import pytest
from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
import make_step2_table as T  # noqa: E402

RES = REPO / "results"
pytestmark = pytest.mark.skipif(not (RES / "bdd100k_mot" / "calib_fixed_fp16").exists(),
                                reason="Step 2 results not downloaded yet")


def test_values_round_trip_from_metrics():
    rows, meta = T.build(RES)
    fp16 = next(r for r in rows if r["policy"] == "TensorRT FP16")
    m = json.loads((RES / "bdd100k_mot/calib_fixed_fp16/metrics.json").read_text())
    assert fp16["map50_95_all"] == pytest.approx(100 * m["all"]["map50_95"])
    assert fp16["map50_95_night"] == pytest.approx(100 * m["night"]["map50_95"])
    assert meta["n_frames"]["all"] == m["all"]["n_images"]


def _copy(tmp_path):
    dst = tmp_path / "results"
    shutil.copytree(RES, dst)
    return dst


def _edit_index(res, fn):
    rows = list(csv.DictReader(open(res / "index.csv")))
    rows = fn(rows)
    with open(res / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@pytest.mark.parametrize("mutate,msg", [
    (lambda rows: rows + [rows[-1]], "duplicate"),
    (lambda rows: [dict(r, smoke="True") if r["exp_id"] == "calib_fixed_int8" else r for r in rows], "SMOKE"),
    (lambda rows: [dict(r, git_commit="other") if r["exp_id"] == "calib_fixed_fp32" else r for r in rows], "commits"),
    (lambda rows: [r for r in rows if r["exp_id"] != "calib_fixed_fp16"], "missing"),
])
def test_refuses_bad_index(tmp_path, mutate, msg):
    res = _copy(tmp_path)
    _edit_index(res, mutate)
    with pytest.raises(T.TableError, match=msg):
        T.build(res)
