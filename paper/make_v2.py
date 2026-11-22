"""Controller v2 (notebook 05, prereg addendum) tables, figures and number macros from results/ ONLY.

Writes paper/tables/v2_*.{md,tex,csv}, paper/figs/v2_*.pdf (+ alt text merged into alt_text.json)
and paper/numbers_v2.tex. Energy: E_total primary, E_dyn secondary (DEVIATIONS.md V1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figs as MF  # noqa: E402  (shared style + palette)
from _results import RESULTS, index, load, prefix_commit  # noqa: E402
from make_tables import ci, fmt, pts, write_table  # noqa: E402

P = "r05_test_"
RHOS = ("0.25", "0.5", "0.75")
LAB = {"v2": "v2 (learned risk)", "random": "Matched random", "v1": "v1 (r04 cues)", "lagged_oracle": "Lagged oracle (GT, t−1)",
       "oracle": "Oracle (GT, upper bound)"}
STY = {"v2": ("#1baf7a", "^", "-"), "random": ("#eda100", "D", "--"), "v1": ("#6250d6", "P", "-."),
       "lagged_oracle": ("#e87ba4", "v", ":"), "oracle": ("#008300", "*", "-")}


def _load_all(results, idx):
    M = {}
    for r in RHOS:
        for k in LAB:
            M[(k, r)] = load(f"{P}{k}_rho{r}", "metrics.json", results, idx)
    F = {lv: load(f"{P}fixed_{lv}", "metrics.json", results, idx) for lv in ("fp32", "fp16", "int8")}
    return M, F


def build(results: Path = RESULTS, out_tables: Path = HERE / "tables", out_figs: Path = HERE / "figs",
          out_numbers: Path = HERE / "numbers_v2.tex") -> dict:
    idx = index(results)
    commit = prefix_commit("r05_", results, idx)
    M, F = _load_all(results, idx)
    boot = load(P + "v2_bootstrap", "bootstrap.json", results, idx)
    hyp = load(P + "v2_hypotheses", "hypotheses.json", results, idx)
    head = load(P + "v2_hypotheses", "headroom.json", results, idx)
    en = load(P + "v2_energy_latency", "energy.json", results, idx)
    live = load(P + "v2_energy_latency", "live_validation.json", results, idx)
    cal = load("r05_v2_calibration", "v2_calibration.json", results, idx)
    stages = json.loads((results / "r05_stages" / "stages.json").read_text())
    made = {}

    # ---- T1 budget table
    rows, raw = [], []
    for lv, lab in (("fp16", "Fixed FP16"), ("int8", "Fixed INT8"), ("fp32", "Fixed FP32")):
        a = F[lv]["all"]
        rows.append({"b": "–", "p": lab, "map": fmt(pts(a["map50_95"])), "ci": ci(a.get("map50_95_ci")),
                     "int8": fmt(a["frac_level"]["int8"]), "sw": fmt(a.get("switch_rate_per_100", 0.0))})
        raw.append({"budget": None, "policy": f"fixed_{lv}", "map50_95": a["map50_95"], "frac_int8": a["frac_level"]["int8"]})
    for r in RHOS:
        for k, lab in LAB.items():
            a = M[(k, r)]["all"]
            sd = a.get("map50_95_sd")
            rows.append({"b": f"{float(r):.0%}", "p": lab, "map": fmt(pts(a["map50_95"])) + ("" if sd is None else f" ± {100 * sd:.2f}"),
                         "ci": ci(a.get("map50_95_ci")), "int8": fmt(a["frac_level"]["int8"]),
                         "sw": fmt(a.get("switch_rate_per_100"))})
            raw.append({"budget": float(r), "policy": k, "map50_95": a["map50_95"], "map50_95_sd": sd,
                        "ci_lo": (a.get("map50_95_ci") or [None, None])[0], "ci_hi": (a.get("map50_95_ci") or [None, None])[1],
                        "frac_int8": a["frac_level"]["int8"], "switch_rate_per_100": a.get("switch_rate_per_100")})
    cap = ("Controller v2 follow-up (pre-registered addendum), test split, 40 clips. Budget = target INT8 share set on "
           "calib; INT8 frac. = realised test share. Matched random has the realised share and switch rate of v2 at "
           "the same budget (mean ± SD over 10 seeds). Oracle and lagged oracle use ground truth (references, not "
           "deployable). 95% cluster-bootstrap CI over clips. Generated from results/ at commit " + commit[:8] + ".")
    write_table("v2_budgets", [("b", "Budget"), ("p", "Policy"), ("map", "mAP50-95"), ("ci", "95% CI"),
                               ("int8", "INT8 frac."), ("sw", "switches/100")], rows, cap, "tab:v2-budgets", raw, out_tables)
    made["v2_budgets"] = len(rows)

    # ---- T2 paired differences (P1-P3 + vs fixed)
    names = {"v2_rho0.5 - random_rho0.5": "P1 (primary): v2 − matched random, 50%",
             "v2_rho0.25 - random_rho0.25": "P2: v2 − matched random, 25%",
             "v2_rho0.75 - random_rho0.75": "P2: v2 − matched random, 75%",
             "v2_rho0.5 - v1_rho0.5": "P3: v2 − v1, 50%", "v2_rho0.5 - fixed_fp16": "v2 (50%) − fixed FP16",
             "v2_rho0.5 - fixed_int8": "v2 (50%) − fixed INT8"}
    rows, raw = [], []
    for k, lab in names.items():
        v = boot["pairs"][k]["map50_95"]
        rows.append({"p": lab, "m": fmt(pts(v["mean"]), 3), "ci": ci(v["ci95"], 3),
                     "r": "CI > 0" if v["ci95"][0] > 0 else ("CI < 0" if v["ci95"][1] < 0 else "includes 0")})
        raw.append({"pair": k, "mean": v["mean"], "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]})
    cap = (f"Paired cluster-bootstrap differences on the test split, mAP@0.5:0.95 (points), {boot['n_resamples']} "
           f"resamples over clips, seed {boot['seed']}. P1 is the pre-registered primary test of the addendum.")
    write_table("v2_pairs", [("p", "Comparison"), ("m", "Δ (boot. mean)"), ("ci", "95% CI"), ("r", "Result")], rows,
                cap, "tab:v2-pairs", raw, out_tables)
    made["v2_pairs"] = len(rows)

    # ---- T3 energy / latency under mixed switching
    s = en["summary"]
    order = ["fixed_fp16", "fixed_int8", "forced_alternation", "adaptive_v2_pipelined", "adaptive_v2_sequential"]
    lab = {"fixed_fp16": "Fixed FP16", "fixed_int8": "Fixed INT8", "forced_alternation": "Forced alternation (every frame)",
           "adaptive_v2_pipelined": "v2 50%, pipelined cues", "adaptive_v2_sequential": "v2 50%, sequential cues"}
    brk = {}
    for k in ("adaptive_v2_pipelined", "adaptive_v2_sequential"):
        rr = [r for r in en["runs"] if r["config"] == k]
        brk[k] = {"cue_wait": float(np.median([r["feat_wait_ms"]["median"] for r in rr])),
                  "controller": float(np.median([r["ctrl_ms"]["median"] for r in rr])),
                  "total": float(np.median([r["latency_total_ms"]["median"] for r in rr]))}

    def rng(d, nd=2):
        return "–" if not d else f"{d['median']:.{nd}f} [{d['min']:.{nd}f}–{d['max']:.{nd}f}]"

    rows, raw = [], []
    for k in order:
        v = s[k]
        rows.append({"c": lab[k], "lat": rng(v["latency_total_ms_median"]), "etot": rng(v["E_total_mJ_per_frame"], 1),
                     "edyn": rng(v["E_dyn_mJ_per_frame"], 1), "int8": fmt(v.get("frac_int8")),
                     "sw": fmt(v.get("switch_rate_per_100"), 1),
                     "ctl": fmt(brk[k]["controller"]) if k in brk else "–", "cue": fmt(brk[k]["cue_wait"]) if k in brk else "–"})
        raw.append({"config": k, **{f"{m}_median": (v[m] or {}).get("median") for m in
                    ("latency_total_ms_median", "E_total_mJ_per_frame", "E_dyn_mJ_per_frame", "throughput_fps")},
                    "frac_int8": v.get("frac_int8"), "switch_rate_per_100": v.get("switch_rate_per_100"),
                    **({f"{b}_ms": brk[k][b] for b in ("cue_wait", "controller")} if k in brk else {})})
    cap = (f"Live measurement under mixed switching on {len(live['clips'])} seeded test clips ({live['n_frames']} frames, "
           f"pre-decoded), one Tesla T4, {en['repeats']} interleaved repeats; median [min–max]. E_total/frame primary, "
           f"E_dyn/frame secondary (NVML board energy counter). Cue wait and controller step (feature statistics + "
           f"model prediction) are median ms per frame. Live decisions matched replay on "
           f"{100 * live['level_agreement_live_vs_replay']:.0f}% of frames.")
    write_table("v2_energy", [("c", "Configuration"), ("lat", "latency ms"), ("etot", "E_total mJ/frame"),
                              ("edyn", "E_dyn mJ/frame"), ("int8", "INT8 frac."), ("sw", "switches/100"),
                              ("cue", "cue wait ms"), ("ctl", "controller ms")], rows, cap, "tab:v2-energy", raw, out_tables)
    made["v2_energy"] = len(rows)

    # ---- T4 calibration of v2
    rows = [{"k": "risk model", "v": "gradient-boosted regression trees " +
             ", ".join(f"{k}={v}" for k, v in sorted(cal["params"].items()))},
            {"k": "training rows (calib frames × 2 histories)", "v": str(cal["n_train_rows"])},
            {"k": "held-out Spearman(pred., true gain), clip-grouped CV", "v": f"{cal['best_spearman']:.3f}"},
            {"k": "FP16 calib mAP50-95", "v": fmt(pts(cal["fp16_calib_map50_95"]))}]
    for r, v in cal["per_rho"].items():
        b = v["v2"]
        rows.append({"k": f"budget {float(r):.0%}: K / calib INT8 frac. / calib mAP50-95",
                     "v": f"{b['K']} / {b['calib_frac_int8']:.3f} / {100 * b['calib_map50_95']:.2f}"})
    drift = {c["check"].split()[1]: max(c["detail"].values()) for c in stages["1_engines_calib_cache"]["checks"]
             if c["check"].startswith("rebuilt")}
    rows.append({"k": "rebuilt-engine drift vs Step 2 (calib mAP50-95, pts)", "v": ", ".join(f"{k} {v:.2f}" for k, v in drift.items())})
    write_table("v2_calibration", [("k", "Item"), ("v", "Value")], rows,
                "Controller v2 fitted and locked on the calib split only (prereg addendum).", "tab:v2-calibration", None, out_tables)
    made["v2_calibration"] = len(rows)

    # ---- figures
    alt = {}
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for k, (col, mk, ls) in STY.items():
        xs = [M[(k, r)]["all"]["frac_level"]["int8"] for r in RHOS]
        ys = [100 * M[(k, r)]["all"]["map50_95"] for r in RHOS]
        if k == "random":
            sd = [100 * M[(k, r)]["all"]["map50_95_sd"] for r in RHOS]
            ax.errorbar(xs, ys, yerr=sd, color=col, marker=mk, ls=ls, lw=1.4, ms=5, capsize=2, label=LAB[k], zorder=3)
        else:
            ax.plot(xs, ys, color=col, marker=mk, ls=ls, lw=1.4, ms=6 if mk != "*" else 9, label=LAB[k], zorder=3 if k == "v2" else 2,
                    mfc="none" if k == "v2" else col, mew=1.5)
    for lv, x, (lb, col, mk, _h) in (("fp16", 0.0, MF.STYLE["fixed_fp16"]), ("int8", 1.0, MF.STYLE["fixed_int8"])):
        ax.scatter([x], [100 * F[lv]["all"]["map50_95"]], marker=mk, s=50, color=col, zorder=4)
        ax.annotate(lb, (x, 100 * F[lv]["all"]["map50_95"]), xytext=(x + (0.04 if x == 0 else -0.04), 100 * F[lv]["all"]["map50_95"]),
                    ha="left" if x == 0 else "right", va="center", fontsize=6.5)
    ax.set_xlabel("realised INT8 share on test")
    ax.set_ylabel("test mAP@0.5:0.95 (pts)")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=5.8, loc="lower left")
    MF._save(fig, "v2_budgets", out_figs, alt,
             "Line chart of test mAP@0.5:0.95 against the realised INT8 share for the three budgets. " + "; ".join(
                 f"{LAB[k]}: " + ", ".join(f"{M[(k, r)]['all']['frac_level']['int8']:.2f} -> {100 * M[(k, r)]['all']['map50_95']:.2f}"
                                           for r in RHOS) for k in LAB) +
             f". Fixed FP16 {100 * F['fp16']['all']['map50_95']:.2f} at share 0, fixed INT8 {100 * F['int8']['all']['map50_95']:.2f} at share 1. "
             "The v2 and matched-random lines overlap; the oracle lies above both.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    labs = ["FP16", "INT8", "Altern.", "v2\npipe.", "v2\nseq."]
    cols = [MF.STYLE["fixed_fp16"][1], MF.STYLE["fixed_int8"][1], "#6250d6", MF.STYLE["adaptive_main"][1], MF.STYLE["adaptive_main"][1]]
    hats = ["", "//", "--", "xx", "\\\\"]
    x = np.arange(len(order))
    eng = [s[k]["latency_total_ms_median"]["median"] for k in order]
    cue = [brk[k]["cue_wait"] if k in brk else 0.0 for k in order]
    ctl = [brk[k]["controller"] if k in brk else 0.0 for k in order]
    rest = [e - c - t for e, c, t in zip(eng, cue, ctl, strict=True)]
    a1.bar(x, rest, 0.62, color=cols, hatch=hats, edgecolor=MF.SURF, label="inference + pre/post")
    a1.bar(x, ctl, 0.62, bottom=rest, color=MF.INK2, edgecolor=MF.SURF, label="controller step")
    a1.bar(x, cue, 0.62, bottom=[r + c for r, c in zip(rest, ctl, strict=True)], color="#c3c2b7", edgecolor=MF.SURF, label="cue wait")
    for i, v in enumerate(eng):
        a1.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=6.5)
    a1.set_xticks(x, labs)
    a1.set_ylabel("latency per frame, median (ms)")
    a1.set_title("(a) latency and its parts")
    a1.legend(fontsize=6, loc="upper left")
    w = 0.38
    et = [s[k]["E_total_mJ_per_frame"]["median"] for k in order]
    ed = [s[k]["E_dyn_mJ_per_frame"]["median"] for k in order]
    a2.bar(x - w / 2, et, w, color=cols, edgecolor=MF.SURF, label="E_total (primary)")
    a2.bar(x + w / 2, ed, w, color=cols, alpha=0.45, hatch="////", edgecolor=MF.SURF, label="E_dyn (secondary)")
    a2.set_xticks(x, labs)
    a2.set_ylabel("GPU board energy per frame (mJ)")
    a2.set_title("(b) energy under mixed switching")
    a2.set_ylim(0, 1.3 * max(et))
    a2.legend(fontsize=6, loc="upper left", ncol=2)
    MF._save(fig, "v2_energy", out_figs, alt,
             "Two bar charts measured live under mixed switching. (a) median latency per frame split into inference, "
             "controller step and cue wait: " + ", ".join(f"{lb.replace(chr(10), ' ')} {e:.2f} ms" for lb, e in zip(labs, eng, strict=True)) +
             ". (b) energy per frame, E_total / E_dyn: " + ", ".join(f"{lb.replace(chr(10), ' ')} {a:.0f}/{b:.0f} mJ" for lb, a, b in zip(labs, et, ed, strict=True)) + ".")
    altp = out_figs / "alt_text.json"
    allalt = json.loads(altp.read_text(encoding="utf-8")) if altp.exists() else {}
    allalt.update({"v2_results_commit": commit, **alt})
    altp.write_text(json.dumps(allalt, indent=1, ensure_ascii=False), encoding="utf-8")
    made["figures"] = list(alt)

    # ---- numbers
    n = {}
    for r, word in zip(RHOS, ("TwentyFive", "Fifty", "SeventyFive"), strict=True):
        for k, tag in (("v2", "VTwo"), ("random", "Random"), ("v1", "VOne"), ("oracle", "Oracle"), ("lagged_oracle", "Lagged")):
            n[f"Map{tag}{word}"] = f"{100 * M[(k, r)]['all']['map50_95']:.2f}"
            n[f"Frac{tag}{word}"] = f"{M[(k, r)]['all']['frac_level']['int8']:.2f}"
        n[f"MapRandomSD{word}"] = f"{100 * M[('random', r)]['all']['map50_95_sd']:.2f}"
        h = head[str(float(r))] if str(float(r)) in head else head[r]
        n[f"Captured{word}Pct"] = "n/a" if h["captured"] is None else f"{100 * h['captured']:.0f}"
        n[f"SwitchVTwo{word}"] = f"{M[('v2', r)]['all']['switch_rate_per_100']:.1f}"
    for k, tag in (("v2_rho0.5 - random_rho0.5", "POne"), ("v2_rho0.25 - random_rho0.25", "PTwoLow"),
                   ("v2_rho0.75 - random_rho0.75", "PTwoHigh"), ("v2_rho0.5 - v1_rho0.5", "PThree"),
                   ("v2_rho0.5 - fixed_fp16", "VTwoFPsixteen"), ("v2_rho0.5 - fixed_int8", "VTwoINTeight")):
        v = boot["pairs"][k]["map50_95"]
        n[f"Diff{tag}Mean"] = f"{100 * v['mean']:+.2f}"
        n[f"Diff{tag}CILo"], n[f"Diff{tag}CIHi"] = f"{100 * v['ci95'][0]:+.2f}", f"{100 * v['ci95'][1]:+.2f}"
    for lv, tag in (("fp16", "FPsixteen"), ("int8", "INTeight"), ("fp32", "FPthirtytwo")):
        n[f"MapVFixed{tag}"] = f"{100 * F[lv]['all']['map50_95']:.2f}"
    for k, tag in (("fixed_fp16", "FPsixteen"), ("fixed_int8", "INTeight"), ("forced_alternation", "Alternation"),
                   ("adaptive_v2_pipelined", "VTwoPipe"), ("adaptive_v2_sequential", "VTwoSeq")):
        n[f"VLat{tag}"] = f"{s[k]['latency_total_ms_median']['median']:.2f}"
        n[f"VETotal{tag}"] = f"{s[k]['E_total_mJ_per_frame']['median']:.0f}"
        n[f"VEDyn{tag}"] = f"{s[k]['E_dyn_mJ_per_frame']['median']:.0f}"
    n["VMeasFracINTeight"] = f"{s['adaptive_v2_pipelined']['frac_int8']:.2f}"
    n["VMeasSwitchRate"] = f"{s['adaptive_v2_pipelined']['switch_rate_per_100']:.1f}"
    n["VControllerMsPipe"] = f"{brk['adaptive_v2_pipelined']['controller']:.2f}"
    n["VCueWaitMsSeq"] = f"{brk['adaptive_v2_sequential']['cue_wait']:.2f}"
    n["VCueWaitMsPipe"] = f"{brk['adaptive_v2_pipelined']['cue_wait']:.2f}"
    n["VSpearman"] = f"{cal['best_spearman']:.2f}"
    n["VHeadroomFiftyPts"] = f"{100 * (M[('oracle', '0.5')]['all']['map50_95'] - M[('random', '0.5')]['all']['map50_95']):.2f}"
    n["VDropINTeightPts"] = f"{100 * (F['fp16']['all']['map50_95'] - F['int8']['all']['map50_95']):.2f}"
    lat16, lat8 = s["fixed_fp16"]["latency_total_ms_median"]["median"], s["fixed_int8"]["latency_total_ms_median"]["median"]
    e16, e8 = s["fixed_fp16"]["E_total_mJ_per_frame"]["median"], s["fixed_int8"]["E_total_mJ_per_frame"]["median"]
    n["VINTeightLatRedPct"] = f"{100 * (1 - lat8 / lat16):.0f}"
    n["VINTeightETotalRedPct"] = f"{100 * (1 - e8 / e16):.0f}"
    n["VDriftINTeight"] = f"{drift['int8']:.2f}"
    n["VLiveAgreementPct"] = f"{100 * live['level_agreement_live_vs_replay']:.0f}"
    n["VEnergyClips"] = str(len(live["clips"]))
    n["VPOneResult"] = "supported" if hyp["P1_primary"]["supported"] else "not supported"
    n["VPFiveResult"] = "supported" if hyp["P5_energy"]["supported"] else "not supported"
    n["CommitVTwoRun"] = commit[:8]
    bad = [k for k in n if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    out_numbers.write_text("% generated by paper/make_v2.py from results/ -- do not edit\n% results commit " + commit + "\n" +
                           "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(n.items())) + "\n", encoding="utf-8")
    made["macros"] = len(n)
    return made


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        MF.PREVIEW = Path(sys.argv[2])
    print(build())
