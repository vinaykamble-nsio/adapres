"""Step 5: LaTeX number macros from results/ ONLY (local-safe) -> paper/numbers.tex.

The manuscript cites numbers only through these macros (PLAN §8), never typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _results import RESULTS, index, load, r04_commit  # noqa: E402

P = "r04_test_"


def build(results: Path = RESULTS, out: Path = HERE / "numbers.tex") -> dict:
    idx = index(results)
    commit = r04_commit(results, idx)
    pol = {"FixedFPsixteen": "fixed_fp16", "FixedINTeight": "fixed_int8", "FixedFPthirtytwo": "fixed_fp32",
           "Adaptive": "adaptive_main", "RandomMatched": "random_matched", "RandomIid": "random_iid",
           "Oracle": "oracle_cheap"}
    M = {k: load(P + v, "metrics.json", results, idx) for k, v in pol.items()}
    boot = load(P + "bootstrap", "bootstrap.json", results, idx)
    e = load(P + "energy_latency", "energy.json", results, idx)["summary"]
    live = load(P + "energy_latency", "live_validation.json", results, idx)
    cal = load("r04_calibration", "calibration.json", results, idx)
    hyp = load("r04_test_hypotheses_v2", "hypotheses.json", results, idx)
    mb = load("r04_microbench_yolo11s", "microbench.json", results, idx)["per_backend"]
    stages = json.loads((results / "r04_stages" / "stages.json").read_text())
    sel = load("bdd_subsets", "selection.json", results, idx)
    n: dict[str, str] = {}

    def p2(x):
        return f"{100 * x:.2f}"

    for k, m in M.items():
        n[f"Map{k}"] = p2(m["all"]["map50_95"])
        n[f"MapFifty{k}"] = p2(m["all"]["map50"])
        if m["all"].get("map50_95_ci"):
            lo, hi = m["all"]["map50_95_ci"]
            n[f"Map{k}CILo"], n[f"Map{k}CIHi"] = p2(lo), p2(hi)
        if "map50_95_sd" in m["all"]:
            n[f"Map{k}SD"] = p2(m["all"]["map50_95_sd"])
        for sl, tag in (("easy", "Easy"), ("crowded", "Crowded"), ("night", "Night"), ("scene_change", "SceneChange")):
            if m.get(sl, {}).get("map50_95") is not None:
                n[f"Map{k}{tag}"] = p2(m[sl]["map50_95"])
        n[f"FracINTeight{k}"] = f"{m['all']['frac_level']['int8']:.2f}"
        if m["all"].get("switch_rate_per_100") is not None:
            n[f"SwitchRate{k}"] = f"{m['all']['switch_rate_per_100']:.2f}"
    n["DropAdaptiveVsFPsixteen"] = f"{100 * (M['FixedFPsixteen']['all']['map50_95'] - M['Adaptive']['all']['map50_95']):.2f}"
    n["DropINTeightVsFPsixteen"] = f"{100 * (M['FixedFPsixteen']['all']['map50_95'] - M['FixedINTeight']['all']['map50_95']):.2f}"
    for key, tag in (("adaptive_main - fixed_fp16", "AdaptiveFPsixteen"), ("adaptive_main - fixed_int8", "AdaptiveINTeight"),
                     ("adaptive_main - random_matched", "AdaptiveRandomMatched"),
                     ("adaptive_main - random_iid", "AdaptiveRandomIid"), ("oracle_cheap - fixed_fp16", "OracleFPsixteen")):
        v = boot["pairs"][key]["map50_95"]
        n[f"Diff{tag}Mean"] = f"{100 * v['mean']:+.2f}"
        n[f"Diff{tag}CILo"], n[f"Diff{tag}CIHi"] = f"{100 * v['ci95'][0]:+.2f}", f"{100 * v['ci95'][1]:+.2f}"
    for k, tag in (("fixed_fp32", "FPthirtytwo"), ("fixed_fp16", "FPsixteen"), ("fixed_int8", "INTeight"),
                   ("adaptive_sequential", "AdaptSeq"), ("adaptive_pipelined", "AdaptPipe")):
        n[f"Lat{tag}"] = f"{e[k]['latency_total_ms_median']['median']:.2f}"
        n[f"LatPninetyfive{tag}"] = f"{e[k]['latency_total_ms_p95']['median']:.2f}"
        n[f"Fps{tag}"] = f"{e[k]['throughput_fps']['median']:.0f}"
        n[f"ETotal{tag}"] = f"{e[k]['E_total_mJ_per_frame']['median']:.0f}"
        n[f"EDyn{tag}"] = f"{e[k]['E_dyn_mJ_per_frame']['median']:.0f}"
    n["INTeightLatRedPct"] = f"{100 * (1 - e['fixed_int8']['latency_total_ms_median']['median'] / e['fixed_fp16']['latency_total_ms_median']['median']):.0f}"
    n["INTeightETotalRedPct"] = f"{100 * (1 - e['fixed_int8']['E_total_mJ_per_frame']['median'] / e['fixed_fp16']['E_total_mJ_per_frame']['median']):.0f}"
    n["EngineMsFPsixteen"] =f"{mb['fp16']['infer_ms']['median']:.2f}"
    n["EngineMsINTeight"] = f"{mb['int8']['infer_ms']['median']:.2f}"
    n["EngineMsFPthirtytwo"] = f"{mb['fp32']['infer_ms']['median']:.2f}"
    n["EngineSavingINTeightMs"] = f"{mb['fp16']['infer_ms']['median'] - mb['int8']['infer_ms']['median']:.2f}"
    n["EngineSpeedupINTeightPct"] = f"{100 * (1 - mb['int8']['infer_ms']['median'] / mb['fp16']['infer_ms']['median']):.0f}"
    n["CueCostMs"] = f"{cal['ctrl_overhead_ms_sequential']:.2f}"
    n["SwitchPenaltyMs"] = f"{cal['switch_penalty_ms']:.2f}"
    n["CalibConfigs"] = f"{cal['n_stage_a_configs']:,}".replace(",", "{,}")
    n["CalibDropMain"] = f"{100 * (cal['fp16_calib_map50_95'] - cal['locks']['main']['map50_95']):.2f}"
    n["DeltaPts"] = f"{hyp['H1']['delta_pts']}"
    drift = {c["check"].split()[1]: max(c["detail"].values()) for c in stages["1_engines_calib_cache"]["checks"]
             if c["check"].startswith("rebuilt")}
    n["DriftINTeight"], n["DriftFPsixteen"] = f"{drift['int8']:.2f}", f"{drift['fp16']:.2f}"
    n["TestFrames"] = f"{M['FixedFPsixteen']['all']['n_images']:,}".replace(",", "{,}")
    n["SceneChangeFrames"] = str(M["FixedFPsixteen"]["scene_change"]["n_images"])
    n["TestClips"], n["CalibClips"] = str(len(sel["splits"]["test"])), str(len(sel["splits"]["calib"]))
    n["BootResamples"] = f"{boot['n_resamples']:,}".replace(",", "{,}")
    n["LiveAgreementPct"] = f"{100 * live['level_agreement']:.0f}"
    n["LiveFrames"] = f"{live['n_frames']:,}".replace(",", "{,}")
    ks = hyp["H3"]["per_K"]
    n["SwitchRateKOne"] = f"{next(iter(ks.values()))['switch_rate_K1']:.2f}"
    for K, v in ks.items():
        word = {"2": "Two", "3": "Three", "5": "Five", "8": "Eight"}.get(str(K))
        if word:
            n[f"SwitchRateK{word}"] = f"{v['switch_rate']:.2f}"
            n[f"MapChangeK{word}"] = f"{v['map_change_pts_vs_K1']:+.2f}"
    n["HOneResult"] = "not supported" if not (hyp["H1"]["supported_pipelined"] or hyp["H1"]["supported_sequential"]) else "supported"
    n["HTwoResult"] = "supported" if hyp["H2"]["supported"] else "not supported"
    n["HThreeResult"] = "supported" if hyp["H3"]["supported"] else "not supported"
    n["HFourResult"] = "supported" if hyp["H4"]["supported"] else "not supported"
    n["HFourEasyShare"] = f"{hyp['H4']['saving_share_from_easy']:.3f}"
    n["CommitRun"] = commit[:8]
    bad = [k for k in n if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    lines = ["% generated by paper/make_numbers.py from results/ -- do not edit",
             f"% results commit {commit}"] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(n.items())]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


if __name__ == "__main__":
    nums = build()
    print(len(nums), "macros ->", HERE / "numbers.tex")
