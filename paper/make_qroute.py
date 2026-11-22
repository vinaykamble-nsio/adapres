"""Q-ROUTE (QROUTE_COMPLETE_EXPERIMENT.ipynb, prefix r06_) figures + tables from results/ ONLY (local-safe).

Run in the notebook (S17, into results/r06_figures so they ship in the zip) and locally after the zip is
unpacked (into paper/figs, paper/tables). Every value is read from r06_* records; nothing is typed in.
Style: the validated paper palette (make_figs.STYLE colours, fixed order) + marker/hatch secondary encoding;
one y-axis per panel; estimated (replay) and measured (live) values are labelled as such.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figs as MF  # noqa: E402  (rcParams, palette, _save)
from make_tables import fmt, write_table  # noqa: E402

P = "r06_"
# fixed config -> (colour, hatch); fixed order = action-space order (identity never colour-alone)
CFG_ORDER = ("fp16_640", "int8_640", "fp16_480", "int8_480", "fp16_320", "int8_320", "fp32_640")
PAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
HATCH = ("", "//", "\\\\", "xx", "..", "++", "--")
CFG_STY = {c: (PAL[i], HATCH[i]) for i, c in enumerate(CFG_ORDER)}
POL = {  # policy -> (label, colour, marker)
    "qroute_main_B_int8": ("Q-ROUTE @B_int8", "#1baf7a", "^"),
    "qroute_main_B_mid": ("Q-ROUTE @B_mid", "#1baf7a", "v"),
    "best_fixed_B_int8": ("Best fixed @B_int8", "#eb6834", "s"),
    "random_B_int8": ("Matched random @B_int8", "#eda100", "D"),
    "oracle_B_int8": ("Oracle @B_int8 (GT)", "#008300", "*"),
    "v1_r04": ("v1 (r04)", "#6250d6", "P"),
    "v2_r05": ("v2 (r05)", "#e87ba4", "X"),
}


class QResultsError(RuntimeError):
    pass


def _index(results: Path, allow_smoke: bool) -> dict:
    rows = [r for r in csv.DictReader(open(results / "index.csv", newline="", encoding="utf-8")) if r["exp_id"].startswith(P)]
    idx = {r["exp_id"]: r for r in rows}
    if not idx:
        raise QResultsError(f"no {P}* runs in {results / 'index.csv'}")
    if not allow_smoke and any(r["smoke"] == "True" for r in rows):
        raise QResultsError("r06 index contains SMOKE runs; pass allow_smoke=True for smoke previews only")
    commits = {r["git_commit"] for r in rows}
    if len(commits) != 1:
        raise QResultsError(f"r06 runs come from {len(commits)} commits: {commits}")
    return idx


def _load(idx, results, exp, name):
    if P + exp not in idx:
        raise QResultsError(f"missing run {P + exp}: the stage that writes it has not completed for this results dir. "
                            f"r06 runs present: {sorted(idx)}")
    p = results / idx[P + exp]["path"] / name
    if name.endswith(".csv"):
        return list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return json.loads(p.read_text(encoding="utf-8"))


def _has(idx, exp):
    return P + exp in idx


def build(results: Path, out_figs: Path = HERE / "figs", out_tables: Path = HERE / "tables", allow_smoke: bool = False) -> dict:
    results = Path(results)
    idx = _index(results, allow_smoke)
    commit = next(iter(idx.values()))["git_commit"]
    finals = {k[len(P + "final_"):]: k for k in idx if k.startswith(P + "final_") and not k.endswith(("_bootstrap", "_hypotheses"))}
    M = {n: _load(idx, results, k[len(P):], "metrics.json") for n, k in finals.items()}
    boot = _load(idx, results, "final_bootstrap", "bootstrap.json")
    hyp = _load(idx, results, "final_hypotheses", "hypotheses.json")
    hyp_live = _load(idx, results, "live_hypotheses", "hypotheses_live.json") if _has(idx, "live_hypotheses") else {}
    head = _load(idx, results, "dev_headroom", "headroom.json")
    pred = _load(idx, results, "dev_predictor", "predictor.json")
    oof = _load(idx, results, "dev_predictor", "oof_sample.csv")
    locks = _load(idx, results, "controller_locks", "locks.json")
    hw = _load(idx, results, "hardware_profile", "contention_profile.json")
    live = {n: _load(idx, results, "live", n) for n in ("energy.json", "contention_live.json", "live_validation.json", "memory.json")} \
        if _has(idx, "live") else None
    alt, made = {}, {}

    def m_all(n):
        a = M[n]["all"] if "all" in M[n] else M[n]
        return a["map50_95"], a["mean_cost_ms_est"]

    # ------------------------------------------------------------ tables
    rows, raw = [], []
    for n in sorted(M):
        a = M[n].get("all", M[n])
        alloc = a.get("allocation", {})
        top = ", ".join(f"{c} {100 * v:.0f}%" for c, v in sorted(alloc.items(), key=lambda kv: -kv[1]) if v >= 0.05)
        rows.append({"p": n, "map": fmt(100 * a["map50_95"]), "map50": fmt(100 * a["map50"]) if "map50" in a else "–",
                     "cost": fmt(a["mean_cost_ms_est"]), "sw": fmt(a.get("switch_rate_per_100")), "alloc": top or "–"})
        raw.append({"policy": n, "map50_95": a["map50_95"], "map50": a.get("map50"), "mean_cost_ms_est": a["mean_cost_ms_est"],
                    "switch_rate_per_100": a.get("switch_rate_per_100"), **{f"alloc_{c}": v for c, v in alloc.items()},
                    "info": json.dumps(M[n].get("info", {}), default=str), "git_commit": commit})
    write_table("qroute_final_policies", [("p", "Policy"), ("map", "mAP@.5:.95"), ("map50", "mAP@.5"),
                                          ("cost", "Est. cost (ms)"), ("sw", "Switch/100"), ("alloc", "Allocation (>=5%)")],
                rows, "Q-ROUTE final test (untouched clips): every policy. mAP exact on cached detections; cost = isolated "
                "profiled latency of the chosen engines (+ measured controller overhead for Q-ROUTE), i.e. estimated.",
                "tab:qroute_final", raw=raw, out=out_tables)
    hrows = []
    for k, v in {**hyp, **hyp_live}.items():
        hrows.append({"h": k, "s": "yes" if v.get("supported") else "no",
                      "d": json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "supported"},
                                      default=str)})
    write_table("qroute_hypotheses", [("h", "Hypothesis"), ("s", "Supported"), ("d", "Evidence")], hrows,
                "Q-ROUTE pre-registered hypotheses (configs/prereg_qroute.yaml).", "tab:qroute_hyp", out=out_tables)
    made["tables"] = ["qroute_final_policies", "qroute_hypotheses"]

    # ------------------------------------------------------------ F1 accuracy-latency (estimated) Pareto
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    fx = [(c, *m_all(f"fixed_{c}")) for c in CFG_ORDER if f"fixed_{c}" in M]
    pts_ = sorted([(cost, mp) for _, mp, cost in fx])
    front, best = [], -1
    for cost, mp in pts_:
        if mp > best:
            front.append((cost, mp))
            best = mp
    ax.plot([c for c, _ in front], [100 * m for _, m in front], color=MF.INK2, lw=1, ls="--", zorder=1, label="Fixed Pareto front")
    for k, (c, mp, cost) in enumerate(sorted(fx, key=lambda t: t[2])):
        col, _ = CFG_STY[c]
        ax.scatter(cost, 100 * mp, s=34, color=col, marker="o", edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(c, (cost, 100 * mp), xytext=(4, -9 if k % 2 else 4), textcoords="offset points", fontsize=6, color=MF.INK2)
    for n, (lab, col, mk) in POL.items():
        if n in M and not n.startswith("best_fixed"):
            mp, cost = m_all(n)
            ax.scatter(cost, 100 * mp, s=110 if mk == "*" else 50, color=col, marker=mk, edgecolor="white", linewidth=1.0,
                       zorder=4, label=lab)
    ax.set_xlabel("Mean per-frame cost (ms, estimated from isolated profiles)")
    ax.set_ylabel("Test mAP@0.5:0.95 (%)")
    ax.legend(loc="lower right", fontsize=6)
    MF._save(fig, "qroute_pareto_latency", out_figs, alt, "Accuracy vs estimated latency on the untouched test clips: fixed "
             "precision-resolution configs with their Pareto front, Q-ROUTE, matched random, v1/v2 and the GT oracle.")

    # ------------------------------------------------------------ F2 accuracy-energy (measured live)
    if live:
        s = live["energy.json"]["summary"]
        fig, ax = plt.subplots(figsize=(3.6, 2.7))
        for k, v in s.items():
            if not v.get("E_total_mJ_per_frame"):
                continue
            n = k if k in M else None
            if n is None:
                continue
            e = v["E_total_mJ_per_frame"]["median"]
            mp = m_all(n)[0]
            if k.startswith("fixed_"):
                col, _ = CFG_STY[k[6:]]
                mk, lab = "o", k[6:]
            else:
                lab, col, mk = POL.get(k, (k, MF.INK, "^"))
            ax.scatter(e, 100 * mp, s=40, color=col, marker=mk, edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(lab, (e, 100 * mp), xytext=(4, 3), textcoords="offset points", fontsize=6, color=MF.INK2)
        ax.set_xlabel("E_total per frame (mJ, measured live, NVML)")
        ax.set_ylabel("Test mAP@0.5:0.95 (%)")
        MF._save(fig, "qroute_pareto_energy", out_figs, alt, "Accuracy vs measured GPU-board energy per frame (live runs on 6 "
                 "test clips) for the fixed configurations and Q-ROUTE.")

    # ------------------------------------------------------------ F3 oracle headroom (dev, per action space)
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.3), sharey=True)
    for ax, b in zip(axs, ("B_int8", "B_mid"), strict=True):
        o = head["oracle"][b]
        names = ["best fixed", "precision only", "resolution only", "joint"]
        vals = [o["best_fixed_map"], o["precision_only"]["map"], o["resolution_only"]["map"], o["joint"]["map"]]
        cols = ["#eb6834", "#2a78d6", "#1baf7a", "#008300"]
        ax.bar(range(4), [100 * v for v in vals], color=cols, edgecolor="white", linewidth=2, hatch=None)
        for i, v in enumerate(vals):
            ax.annotate(f"{100 * v:.1f}", (i, 100 * v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)
        ax.set_xticks(range(4), names, rotation=25, ha="right")
        ax.set_title(f"Oracle at {b} (dev)")
        lo = min(vals)
        ax.set_ylim(100 * lo - 3, 100 * max(vals) + 2)
    axs[0].set_ylabel("Dev mAP@0.5:0.95 (%)")
    MF._save(fig, "qroute_oracle_headroom", out_figs, alt, "Budgeted oracle (uses ground truth) on development clips per "
             "action space vs the best fixed configuration at the same budget: the headroom any router could capture.")

    # ------------------------------------------------------------ F4 predicted vs observed regret (OOF, dev)
    cfgs = sorted({r["config"] for r in oof}, key=CFG_ORDER.index)
    fig, axs = plt.subplots(1, len(cfgs), figsize=(1.45 * len(cfgs) + 0.6, 1.9), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, cfgs, strict=True):
        x = np.array([float(r["pred"]) for r in oof if r["config"] == c])
        y = np.array([float(r["true"]) for r in oof if r["config"] == c])
        ax.scatter(x, y, s=3, color=CFG_STY[c][0], alpha=0.35, linewidths=0)
        ax.axhline(0, color=GRID_C, lw=0.6)
        rho = pred["per_config_oof_spearman"].get(c)
        ax.set_title(f"{c}\nρ={rho:.2f}" if rho is not None else c, fontsize=7)
        ax.set_xlabel("predicted", fontsize=6)
    axs[0].set_ylabel("observed regret (ΔF1)")
    MF._save(fig, "qroute_pred_vs_actual", out_figs, alt, "Out-of-fold predicted vs observed per-frame regret (F1 of FP16-640 "
             "minus F1 of the config) on development clips, with held-out Spearman per configuration.")

    # ------------------------------------------------------------ F5 per-scene (slices)
    sl = [k for k in ("all", "easy", "crowded", "night", "scene_change_natural") if k in M.get("qroute_main_B_int8", {})]
    pols = [n for n in ("best_fixed_B_int8", "qroute_main_B_int8", "oracle_B_int8", "fixed_fp16_640") if n in M]
    if sl and pols:
        fig, ax = plt.subplots(figsize=(4.8, 2.4))
        w = 0.8 / len(pols)
        for j, n in enumerate(pols):
            lab, col, _ = POL.get(n, ("Fixed FP16-640", "#2a78d6", "o"))
            vals = [100 * M[n][s_]["map50_95"] if M[n][s_]["n_images"] else np.nan for s_ in sl]
            ax.bar(np.arange(len(sl)) + (j - (len(pols) - 1) / 2) * w, vals, w, color=col, edgecolor="white", linewidth=1.5,
                   hatch=HATCH[j], label=lab)
        ax.set_xticks(range(len(sl)), [s_.replace("_", " ") for s_ in sl])
        ax.set_ylabel("Test mAP@0.5:0.95 (%)")
        ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
        MF._save(fig, "qroute_per_scene", out_figs, alt, "Per-scene (slice) test mAP of Q-ROUTE, best fixed config at the same "
                 "budget, the GT oracle and fixed FP16-640.")

    # ------------------------------------------------------------ F6 overhead breakdown
    fc = pred["feature_cost"]
    bars = [("light features", fc["light_features_ms"]), ("image cues", fc["image_cues_ms"]), ("predictor", fc["predictor_ms"])]
    live_bd = None
    if live:
        r = [x for x in live["energy.json"]["runs"] if x["config"] == "qroute_main_B_int8" and x.get("ctrl_ms")]
        if r:
            fw = float(np.median([x["feat_wait_ms"]["median"] for x in r]))
            ct = float(np.median([x["ctrl_ms"]["median"] for x in r]))
            tot = float(np.median([x["latency_total_ms"]["median"] for x in r]))
            live_bd = [("features (live)", fw), ("controller (live)", ct), ("inference (live)", tot - fw - ct)]
    fig, axs = plt.subplots(1, 2 if live_bd else 1, figsize=(5.2 if live_bd else 2.8, 2.2))
    axs = np.atleast_1d(axs)
    axs[0].barh([b[0] for b in bars], [b[1] for b in bars], color=["#2a78d6", "#eda100", "#1baf7a"], edgecolor="white")
    axs[0].set_xlabel("ms per frame (CPU, dev)")
    axs[0].set_title("Controller cost components" + (" (image cues gated in)" if pred["use_image_cues"] else " (image cues gated out)"),
                     fontsize=7)
    if live_bd:
        left = 0.0
        for (lab, v), col, h in zip(live_bd, ("#eda100", "#1baf7a", "#2a78d6"), ("", "//", ""), strict=True):
            axs[1].barh([0], [v], left=left, color=col, edgecolor="white", linewidth=2, hatch=h, label=f"{lab} {v:.2f} ms")
            left += v
        axs[1].set_yticks([])
        axs[1].set_xlabel("ms per frame (live, median)")
        axs[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    MF._save(fig, "qroute_overhead", out_figs, alt, "Q-ROUTE decision overhead: per-component CPU cost measured on development "
             "frames, and the live end-to-end breakdown (features, controller, inference) on the test clips.")

    # ------------------------------------------------------------ F7 allocation distribution
    pols = [n for n in ("qroute_main_B_int8", "qroute_main_B_mid", "qroute_precision_only", "qroute_resolution_only",
                        "qroute_hardware_only", "qroute_abl_no_temporal", "qroute_abl_no_persistence", "qroute_abl_energy_cost",
                        "qroute_abl_full_features", "oracle_B_int8", "contention_qroute_main_B_int8") if n in M]
    fig, ax = plt.subplots(figsize=(4.8, 0.32 * len(pols) + 0.9))
    for j, n in enumerate(pols):
        left = 0.0
        for c in CFG_ORDER:
            v = M[n]["all"]["allocation"].get(c, 0.0)
            if v > 0:
                col, h = CFG_STY[c]
                ax.barh(j, v, left=left, color=col, hatch=h, edgecolor="white", linewidth=1.5, label=c)
                left += v
    hs, ls = ax.get_legend_handles_labels()
    uniq = dict(zip(ls, hs, strict=True))
    ax.legend([uniq[c] for c in CFG_ORDER if c in uniq], [c for c in CFG_ORDER if c in uniq], fontsize=6, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, 1.0 + 1.2 / len(pols)))
    ax.set_yticks(range(len(pols)), [n.replace("qroute_", "") for n in pols], fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of test frames")
    ax.invert_yaxis()
    MF._save(fig, "qroute_allocation", out_figs, alt, "Share of test frames routed to each precision-resolution config by "
             "Q-ROUTE, its variants and ablations, and the GT oracle.")

    # ------------------------------------------------------------ F8 contention sensitivity (live + profile)
    lv_keys = list(hw["slowdown_ratios"])
    fig, axs = plt.subplots(1, 2 if live else 1, figsize=(5.8 if live else 2.8, 2.3))
    fig.subplots_adjust(wspace=0.45)
    axs = np.atleast_1d(axs)
    for c in [c for c in CFG_ORDER if c in hw["slowdown_ratios"][lv_keys[0]]]:
        col, _ = CFG_STY[c]
        axs[0].plot([float(k) for k in lv_keys], [hw["slowdown_ratios"][k][c] for k in lv_keys], marker="o", color=col, label=c, ms=4)
    axs[0].set_xlabel("Contention duty cycle")
    axs[0].set_ylabel("Latency / isolated")
    axs[0].legend(fontsize=5.5, ncol=2)
    axs[0].set_title("Measured slowdown per config", fontsize=7)
    if live:
        ct = live["contention_live.json"]
        names = list(ct)
        lvls = list(next(iter(ct.values()))["by_level"])
        w = 0.8 / len(names)
        cols = {"qroute_hw": "#1baf7a", "qroute_no_hw": "#6250d6", "hardware_only": "#eda100", "fixed_fp16_640": "#2a78d6",
                "fixed_int8_640": "#eb6834"}
        for j, n in enumerate(names):
            axs[1].bar(np.arange(len(lvls)) + (j - (len(names) - 1) / 2) * w, [100 * ct[n]["by_level"][x]["miss_rate"] for x in lvls], w,
                       color=cols.get(n, MF.INK2), hatch=HATCH[j], edgecolor="white", linewidth=1, label=n)
        axs[1].set_xticks(range(len(lvls)), lvls)
        axs[1].set_xlabel("Contention duty cycle (live)")
        axs[1].set_ylabel("Deadline misses (%)")
        axs[1].legend(fontsize=5.5)
    MF._save(fig, "qroute_contention", out_figs, alt, "Hardware contention: measured slowdown ratio per config, and live "
             "deadline-miss rate per contention level for Q-ROUTE with and without hardware adaptation and fixed configs.")

    # ------------------------------------------------------------ F9 ablations (mAP and cost vs main)
    abl = [n for n in ("qroute_hardware_only", "qroute_abl_no_temporal", "qroute_abl_no_persistence", "qroute_abl_full_features",
                       "qroute_abl_no_hw", "qroute_abl_energy_cost", "qroute_precision_only", "qroute_resolution_only") if n in M]
    if "qroute_main_B_int8" in M and abl:
        m0, c0 = m_all("qroute_main_B_int8")
        fig, axs = plt.subplots(1, 2, figsize=(5.4, 0.3 * len(abl) + 1.0), sharey=True)
        dm = [100 * (m_all(n)[0] - m0) for n in abl]
        dc = [m_all(n)[1] - c0 for n in abl]
        labs = [n.replace("qroute_", "").replace("hardware_only", "no regret prediction (hw only)") for n in abl]
        axs[0].barh(range(len(abl)), dm, color=["#1baf7a" if v >= 0 else "#eb6834" for v in dm], edgecolor="white")
        axs[0].axvline(0, color=MF.INK2, lw=0.8)
        axs[0].set_xlabel("Δ test mAP vs Q-ROUTE (pts)")
        axs[1].barh(range(len(abl)), dc, color=MF.INK2, edgecolor="white")
        axs[1].axvline(0, color=MF.INK2, lw=0.8)
        axs[1].set_xlabel("Δ est. cost vs Q-ROUTE (ms)")
        axs[0].set_yticks(range(len(abl)), labs, fontsize=6)
        axs[0].invert_yaxis()
        MF._save(fig, "qroute_ablations", out_figs, alt, "Ablations (each re-tuned on development): change in test mAP and "
                 "estimated cost relative to full Q-ROUTE at B_int8.")

    # ------------------------------------------------------------ F10 statistics (bootstrap CIs)
    pairs = list(boot["pairs"])
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(pairs) + 0.8))
    for j, k in enumerate(pairs):
        v = boot["pairs"][k]["map50_95"]
        lo, hi = 100 * v["ci95"][0], 100 * v["ci95"][1]
        col = "#1baf7a" if lo > 0 else ("#eb6834" if hi < 0 else MF.INK2)
        ax.plot([lo, hi], [j, j], color=col, lw=2)
        ax.scatter(100 * v["mean"], j, color=col, s=28, zorder=3, edgecolor="white", linewidth=1.5)
    ax.axvline(0, color=MF.INK2, lw=0.8, ls="--")
    ax.set_yticks(range(len(pairs)), [k.replace("qroute_", "") for k in pairs], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Δ test mAP@0.5:0.95 (pts), 95% cluster-bootstrap CI")
    MF._save(fig, "qroute_bootstrap", out_figs, alt, f"Paired cluster-bootstrap ({boot['n_resamples']} resamples over test clips) "
             "95% confidence intervals of mAP differences used by Q1-Q3.")

    # alt text and provenance
    out_figs.mkdir(parents=True, exist_ok=True)
    ap = out_figs / "alt_text.json"
    old = json.loads(ap.read_text(encoding="utf-8")) if ap.exists() else {}
    old.update(alt)
    ap.write_text(json.dumps(old, indent=2), encoding="utf-8")
    made["figures"] = sorted(alt)
    made["git_commit"] = commit
    made["locks"] = sorted(locks)
    if live:
        made["live_validation"] = live["live_validation.json"]
    return made


GRID_C = MF.GRID

if __name__ == "__main__":
    import argparse

    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--results", default=str(HERE.parent / "results"))
    ap_.add_argument("--allow-smoke", action="store_true")
    a = ap_.parse_args()
    print(json.dumps(build(Path(a.results), allow_smoke=a.allow_smoke), indent=1, default=str))
