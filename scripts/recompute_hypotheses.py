"""Recompute H1-H4 (revision 2) from existing results/ files only (local-safe, no new data).

Writes results/bdd100k_mot/r04_test_hypotheses_v2/ (hypotheses.json + meta.json with the
SHA-256 of every input file) and a SUPERSEDED.txt note in the original r04_test_hypotheses/
(the original files are not modified). See DEVIATIONS.md V1-V3.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "paper")]

import yaml  # noqa: E402
from _results import RESULTS, index, load, r04_commit, run_dir, sha256  # noqa: E402

from adapres import config as C  # noqa: E402
from adapres import hypotheses as HYP  # noqa: E402
from adapres.repro import capture_env  # noqa: E402
from adapres.results import ExperimentWriter  # noqa: E402

POLICIES = ("adaptive_main", "fixed_fp16", "fixed_int8", "fixed_fp32")


def main(results: Path = RESULTS) -> dict:
    idx = index(results)
    commit = r04_commit(results, idx)
    m = {p: load(f"r04_test_{p}", "metrics.json", results, idx) for p in POLICIES}
    agg = {g: load(f"r04_test_{g}", "metrics.json", results, idx) for g in ("random_matched", "random_iid")}
    ks = {int(k[len("r04_test_adaptive_ksweep"):]): load(k, "metrics.json", results, idx)["all"]
          for k in idx if k.startswith("r04_test_adaptive_ksweep")}
    boot = load("r04_test_bootstrap", "bootstrap.json", results, idx)
    energy = load("r04_test_energy_latency", "energy.json", results, idx)
    mb = load("r04_microbench_yolo11s", "microbench.json", results, idx)
    cal = load("r04_calibration", "calibration.json", results, idx)
    engine_ms = {lv: mb["per_backend"][lv]["infer_ms"]["median"] for lv in ("fp32", "fp16", "int8")}
    delta = yaml.safe_load((REPO / "configs" / "preregistration.yaml").read_text())["delta_map_points"]
    hyp = HYP.compute(m, agg, boot, energy["summary"], ks, engine_ms, delta, int(cal["locks"]["main"]["params"]["K_down"]))
    inputs = [run_dir(f"r04_test_{p}", results, idx) / "metrics.json" for p in list(POLICIES) + list(agg)]
    inputs += [run_dir(k, results, idx) / "metrics.json" for k in idx if k.startswith("r04_test_adaptive_ksweep")]
    inputs += [run_dir("r04_test_bootstrap", results, idx) / "bootstrap.json",
               run_dir("r04_test_energy_latency", results, idx) / "energy.json",
               run_dir("r04_microbench_yolo11s", results, idx) / "microbench.json",
               run_dir("r04_calibration", results, idx) / "calibration.json"]
    cfg = C.load(REPO / "configs" / "default.yaml")
    env = capture_env(REPO, cfg["seed"])
    meta = {"exp_id": "r04_test_hypotheses_v2", "dataset": "bdd100k_mot", "split": "test", "seed": cfg["seed"],
            "git_commit": env["git"]["commit"], "git_dirty": env["git"]["dirty"], "env": env,
            "config_sha256": C.config_sha256(cfg), "timestamp_utc": env["timestamp_utc"], "smoke": False,
            "mode": "derived (local recompute from results/, no new data)", "results_commit": commit,
            "supersedes": "r04_test_hypotheses (revision 1)", "hypotheses_revision": HYP.REVISION,
            "derived_from": {str(p.relative_to(results.parent)).replace("\\", "/"): sha256(p) for p in inputs},
            "notes": "DEVIATIONS.md V1-V3: E_total primary; H3 on K-sweep (sensitivity); H4 engine-only saving"}
    w = ExperimentWriter(results, meta)
    w.write_json("hypotheses.json", hyp)
    w.register(policy="hypotheses_v2", mode="derived", replace=True)
    old = run_dir("r04_test_hypotheses", results, idx)
    (old / "SUPERSEDED.txt").write_text(
        "This hypotheses.json (revision 1, written by notebook 04) is SUPERSEDED by\n"
        "results/bdd100k_mot/r04_test_hypotheses_v2/hypotheses.json (revision 2).\n"
        "Reasons (DEVIATIONS.md V1-V3): H3 compared the locked K (=1) with K=1; H4's saving share\n"
        "included the constant controller overhead; energy primary metric changed to E_total.\n"
        "The original file is kept unchanged for the record.\n", encoding="utf-8")
    return hyp


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=1, default=str))
