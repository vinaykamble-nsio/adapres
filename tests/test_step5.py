"""Step 5: hypotheses revision 2 logic, and tables/figures/numbers/recompute built from results/ only."""

import csv
import json
import shutil
import sys

import pytest
from conftest import REPO

from adapres import hypotheses as HYP

sys.path[:0] = [str(REPO / "paper"), str(REPO / "scripts")]
RES = REPO / "results"
HAVE = (RES / "bdd100k_mot" / "r04_test_adaptive_main").exists()


def _m(map_, frac, sw=0.0, n=100):
    sl = {"map50_95": map_, "n_images": n, "frac_level": {"int8": frac, "fp16": 1 - frac, "fp32": 0.0},
          "switch_rate_per_100": sw}
    return {"all": dict(sl), "easy": dict(sl), "crowded": {**sl, "frac_level": {"int8": 0.5, "fp16": 0.5, "fp32": 0}},
            "night": {**sl, "frac_level": {"int8": 0.6, "fp16": 0.4, "fp32": 0}}}


def _energy(ad_lat, ad_et, ad_ed):
    def d(x):
        return {"median": x}
    base = {"latency_total_ms_median": d(5.8), "E_total_mJ_per_frame": d(390), "E_dyn_mJ_per_frame": d(200)}
    ad = {"latency_total_ms_median": d(ad_lat), "E_total_mJ_per_frame": d(ad_et), "E_dyn_mJ_per_frame": d(ad_ed)}
    return {"fixed_fp16": base, "adaptive_sequential": ad, "adaptive_pipelined": ad}


def test_hypotheses_rev2_logic():
    m = {"adaptive_main": _m(0.258, 0.9, 1.5), "fixed_fp16": _m(0.267, 0.0)}
    agg = {"random_matched": _m(0.257, 0.9, 1.4)}
    boot = {"pairs": {"adaptive_main - random_matched": {"map50_95": {"mean": 0.001, "ci95": [-0.007, 0.003],
                                                                          "excludes_zero": False}}}}
    ks = {1: {"map50_95": 0.258, "switch_rate_per_100": 1.5}, 2: {"map50_95": 0.2581, "switch_rate_per_100": 1.3},
          5: {"map50_95": 0.2584, "switch_rate_per_100": 1.1}}
    eng = {"fp32": 7.8, "fp16": 2.7, "int8": 2.0}
    # E_dyn lower but E_total higher -> NOT lower energy under the primary metric (V1)
    h = HYP.compute(m, agg, boot, _energy(5.5, 400, 100), ks, eng, 0.5, locked_K=1)
    assert h["revision"] == 2
    assert h["H1"]["measured"]["pipelined"]["lower_energy_total"] is False
    assert h["H1"]["measured"]["pipelined"]["lower_energy_dyn_secondary"] is True
    assert h["H1"]["map_within_delta"] is False and h["H2"]["supported"] is False
    # V2: locked K = 1 no longer makes H3 degenerate; evaluated on the K-sweep
    assert h["H3"]["supported"] is True and set(h["H3"]["per_K"]) == {2, 5}
    # V3: engine-only saving share (positive savings; rev 1 would be negative with overhead)
    assert h["H4"]["engine_saving_ms_total"]["all"] > 0 and h["H4"]["saving_share_from_easy"] == pytest.approx(1.0)
    ks_bad = {**ks, 8: {"map50_95": 0.262, "switch_rate_per_100": 1.0}}       # +0.4 pts change -> fails
    assert HYP.compute(m, agg, boot, _energy(5.5, 380, 100), ks_bad, eng, 0.5, 1)["H3"]["supported"] is False


needs = pytest.mark.skipif(not HAVE, reason="notebook 04 results not present")


@needs
def test_tables_figs_numbers_from_results(tmp_path):
    import make_figs
    import make_numbers
    import make_tables

    made = make_tables.build(RES, tmp_path / "tables")
    assert made["test_main"] == 7 and made["hypotheses"] == 4
    for name in made:
        for ext in ("md", "tex", "csv"):
            assert (tmp_path / "tables" / f"{name}.{ext}").stat().st_size > 0
    rows = {r["policy"]: r for r in csv.DictReader(open(tmp_path / "tables" / "test_main.csv"))}
    m = json.loads((RES / "bdd100k_mot/r04_test_adaptive_main/metrics.json").read_text())
    assert float(rows["adaptive_main"]["map50_95"]) == pytest.approx(m["all"]["map50_95"])   # values round-trip
    figs = make_figs.build(RES, tmp_path / "figs")
    alt = json.loads((tmp_path / "figs" / "alt_text.json").read_text(encoding="utf-8"))
    assert set(figs) <= set(alt) and all(open(p, "rb").read(4) == b"%PDF" for p in figs.values())
    nums = make_numbers.build(RES, tmp_path / "numbers.tex")
    assert nums["MapAdaptive"] == f"{100 * m['all']['map50_95']:.2f}"
    assert all(k.isalpha() for k in nums)


@pytest.mark.skipif(not (RES / "bdd100k_mot" / "r05_test_v2_rho0.5").exists(), reason="notebook 05 results not present")
def test_v2_artifacts_from_results(tmp_path):
    import make_v2

    tabs, figs = tmp_path / "tables", tmp_path / "figs"
    made = make_v2.build(RES, tabs, figs, tmp_path / "numbers_v2.tex")
    assert made["v2_budgets"] == 18 and set(made["figures"]) == {"v2_budgets", "v2_energy"}
    m = json.loads((RES / "bdd100k_mot/r05_test_v2_rho0.5/metrics.json").read_text())
    nums = dict(line[len("\\newcommand{\\"):-1].split("}{") for line in
                (tmp_path / "numbers_v2.tex").read_text(encoding="utf-8").splitlines() if line.startswith("\\newcommand"))
    assert nums["MapVTwoFifty"] == f"{100 * m['all']['map50_95']:.2f}"
    h = json.loads((RES / "bdd100k_mot/r05_test_v2_hypotheses/hypotheses.json").read_text())
    assert nums["VPOneResult"] == ("supported" if h["P1_primary"]["supported"] else "not supported")
    assert all(k.isalpha() for k in nums)
    assert all(open(p, "rb").read(4) == b"%PDF" for p in figs.glob("v2_*.pdf"))


@needs
def test_generators_refuse_smoke_and_mixed_commits(tmp_path):
    import _results
    import make_tables

    res = tmp_path / "results"
    shutil.copytree(RES, res)
    rows = list(csv.DictReader(open(res / "index.csv")))
    for r in rows:
        if r["exp_id"] == "r04_test_fixed_int8":
            r["git_commit"] = "deadbeef"
    with open(res / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(_results.ResultsError, match="several commits"):
        make_tables.build(res, tmp_path / "t")
    for r in rows:
        if r["exp_id"] == "r04_test_fixed_int8":
            r["git_commit"], r["smoke"] = rows[-1]["git_commit"], "True"
    with open(res / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(_results.ResultsError):
        make_tables.build(res, tmp_path / "t")


@needs
def test_recompute_hypotheses_keeps_original(tmp_path):
    import recompute_hypotheses

    res = tmp_path / "results"
    shutil.copytree(RES, res)
    orig = (res / "bdd100k_mot/r04_test_hypotheses/hypotheses.json").read_bytes()
    h = recompute_hypotheses.main(res)
    assert (res / "bdd100k_mot/r04_test_hypotheses/hypotheses.json").read_bytes() == orig   # unchanged
    assert (res / "bdd100k_mot/r04_test_hypotheses/SUPERSEDED.txt").exists()
    meta = json.loads((res / "bdd100k_mot/r04_test_hypotheses_v2/meta.json").read_text())
    assert meta["supersedes"].startswith("r04_test_hypotheses") and len(meta["derived_from"]) >= 10
    assert h["revision"] == 2 and h["H3"]["analysis"].startswith("sensitivity")
