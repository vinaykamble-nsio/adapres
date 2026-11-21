"""Step 5 figures from results/ ONLY (local-safe). Vector PDFs in paper/figs/ + alt_text.json.

Encoding (dataviz skill; palette validated with validate_palette.js, light mode): each policy keeps
ONE fixed colour and ONE marker/hatch in every figure, so identity is never colour-alone (Springer
requirement); key points are direct-labelled; values are also in paper/tables/. One y-axis per panel.
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
from _results import RESULTS, index, load, r04_commit, run_dir  # noqa: E402

OUT = HERE / "figs"
P = "r04_test_"
INK, INK2, GRID, SURF = "#1f1f1e", "#5c5c58", "#e6e5e0", "#ffffff"
# fixed entity -> (label, colour, marker, hatch); categorical slots in fixed order
STYLE = {
    "fixed_fp16": ("Fixed FP16", "#2a78d6", "o", ""),
    "fixed_int8": ("Fixed INT8", "#eb6834", "s", "//"),
    "adaptive_main": ("Adaptive (locked)", "#1baf7a", "^", "\\\\"),
    "random_matched": ("Matched random", "#eda100", "D", "xx"),
    "random_iid": ("i.i.d. random", "#e87ba4", "v", ".."),
    "oracle_cheap": ("Oracle (uses GT)", "#008300", "*", "++"),
    "fixed_fp32": ("Fixed FP32", "#6250d6", "P", "--"),
}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "lines.linewidth": 1.5, "lines.markersize": 6, "legend.frameon": False, "savefig.bbox": "tight",
})


PREVIEW: Path | None = None  # optional PNG previews (for visual checks; not part of the paper)


def _save(fig, name: str, out: Path, alt: dict, text: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    if PREVIEW is not None:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f"{name}.png", dpi=170)
    plt.close(fig)
    alt[name] = text


def build(results: Path = RESULTS, out: Path = OUT) -> dict:
    idx = index(results)
    commit = r04_commit(results, idx)
    M = {p: load(P + p, "metrics.json", results, idx) for p in STYLE}
    mb = load("r04_microbench_yolo11s", "microbench.json", results, idx)
    eng = {lv: mb["per_backend"][lv]["infer_ms"]["median"] for lv in ("fp32", "fp16", "int8")}
    energy = load(P + "energy_latency", "energy.json", results, idx)
    boot = load(P + "bootstrap", "bootstrap.json", results, idx)
    cal = load("r04_calibration", "calibration.json", results, idx)
    alt: dict[str, str] = {}

    def engine_ms(m):
        fr = m["all"]["frac_level"]
        return sum(fr.get(lv, 0.0) * eng[lv] for lv in eng)

    # ---- F1 accuracy vs estimated engine latency
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    abl = [k for k in idx if k.startswith(P + "adaptive_abl_") or k.startswith(P + "adaptive_delta")]
    xs = [engine_ms(load(k, "metrics.json", results, idx)) for k in abl]
    ys = [100 * load(k, "metrics.json", results, idx)["all"]["map50_95"] for k in abl]
    ax.scatter(xs, ys, s=14, marker="x", color=INK2, linewidths=0.8, label="ablations / δ locks", zorder=2)
    pts_txt = []
    hollow = {"adaptive_main", "random_matched"}  # drawn over fixed INT8; hollow so the overlap stays visible
    for p, (lab, col, mk, _) in STYLE.items():
        x, y = engine_ms(M[p]), 100 * M[p]["all"]["map50_95"]
        big = 90 if mk == "*" else (64 if p in hollow else 46)
        if p in hollow:
            ax.scatter([x], [y], s=big, marker=mk, facecolors="none", edgecolors=col, linewidths=1.6, label=lab, zorder=4)
        else:
            ax.scatter([x], [y], s=big, marker=mk, color=col, edgecolors=SURF, linewidths=0.8, label=lab, zorder=3)
        pts_txt.append(f"{lab} {y:.2f} pts at {x:.2f} ms")

    def lab_at(p, dx, dy, ha="left", text=None):
        x, y = engine_ms(M[p]), 100 * M[p]["all"]["map50_95"]
        ax.annotate(text or STYLE[p][0], (x, y), xytext=(x + dx, y + dy), fontsize=6.5, color=INK, ha=ha, va="center")

    lab_at("fixed_fp16", 0.18, 0.0)
    lab_at("oracle_cheap", -0.18, 0.0, ha="right")
    lab_at("fixed_fp32", -0.25, 0.0, ha="right")
    lab_at("fixed_int8", 0.25, -0.1, text="INT8 ≈ adaptive ≈ matched random")
    ax.set_xlabel("estimated engine latency per frame (ms, T4 microbench)")
    ax.set_ylabel("test mAP@0.5:0.95 (pts)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.36, 0.9), ncol=1, handletextpad=0.3, fontsize=6)
    ax.set_xlim(1.0, 8.6)
    _save(fig, "accuracy_cost", out, alt,
          "Scatter of test mAP@0.5:0.95 against estimated engine latency per frame. " + "; ".join(pts_txt) +
          f". Grey crosses are {len(abl)} ablation and delta-locked controllers, all clustered near fixed INT8. "
          "Engine-only latency; the adaptive controller's cue cost is excluded here and reported separately.")

    # ---- F2 measured latency and energy (two panels, one axis each)
    s = energy["summary"]
    order = ["fixed_fp32", "fixed_fp16", "fixed_int8", "adaptive_sequential", "adaptive_pipelined"]
    labs = ["FP32", "FP16", "INT8", "Adapt.\nseq.", "Adapt.\npipe."]
    cols = [STYLE["fixed_fp32"][1], STYLE["fixed_fp16"][1], STYLE["fixed_int8"][1], STYLE["adaptive_main"][1],
            STYLE["adaptive_main"][1]]
    hats = ["--", "", "//", "\\\\", "xx"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.4))
    x = np.arange(len(order))
    lat = [s[k]["latency_total_ms_median"] for k in order]
    a1.bar(x, [v["median"] for v in lat], 0.62, color=cols, hatch=hats, edgecolor=SURF, linewidth=1.0,
           yerr=[[v["median"] - v["min"] for v in lat], [v["max"] - v["median"] for v in lat]], ecolor=INK, capsize=2)
    for i, v in enumerate(lat):
        a1.text(i, v["median"] + 0.3, f"{v['median']:.2f}", ha="center", fontsize=6.5, color=INK)
    a1.set_xticks(x, labs)
    a1.set_ylabel("latency per frame, median (ms)")
    a1.set_title("(a) measured latency")
    w = 0.38
    et = [s[k]["E_total_mJ_per_frame"]["median"] for k in order]
    ed = [s[k]["E_dyn_mJ_per_frame"]["median"] for k in order]
    a2.bar(x - w / 2, et, w, color=cols, edgecolor=SURF, linewidth=1.0, label="E_total (primary)")
    a2.bar(x + w / 2, ed, w, color=cols, alpha=0.45, hatch="////", edgecolor=SURF, linewidth=1.0,
           label="E_dyn (secondary)")
    for i in range(len(order)):
        a2.text(i - w / 2, et[i] + 8, f"{et[i]:.0f}", ha="center", fontsize=6, color=INK)
    a2.set_xticks(x, labs)
    a2.set_ylabel("GPU board energy per frame (mJ)")
    a2.set_title("(b) measured energy")
    a2.legend(loc="upper right")
    _save(fig, "measured_latency_energy", out, alt,
          "Two bar charts measured live on one Tesla T4. (a) median latency per frame: " +
          ", ".join(f"{lb.replace(chr(10), ' ')} {v['median']:.2f} ms" for lb, v in zip(labs, lat, strict=True)) +
          ". (b) energy per frame, E_total (solid, primary) and E_dyn (hatched, secondary): " +
          ", ".join(f"{lb.replace(chr(10), ' ')} {a:.0f} / {b:.0f} mJ" for lb, a, b in zip(labs, et, ed, strict=True)) +
          ". Error bars span the three repeats.")

    # ---- F3 per-slice difference vs fixed FP16 (the claims are relative; bars from a zero baseline)
    sl = [("all", "all"), ("easy", "easy"), ("crowded", "crowded"), ("night", "night"), ("scene_change", "scene-chg.†")]
    pols = ["fixed_int8", "adaptive_main", "random_matched", "oracle_cheap"]
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    w = 0.19
    xx = np.arange(len(sl))
    txt = []
    for j, p in enumerate(pols):
        lab, col, _, hat = STYLE[p]
        vals = [100 * (M[p][k]["map50_95"] - M["fixed_fp16"][k]["map50_95"]) for k, _ in sl]
        ax.bar(xx + (j - 1.5) * w, vals, w * 0.9, color=col, hatch=hat, edgecolor=SURF, linewidth=1.0, label=lab)
        txt.append(f"{lab}: " + ", ".join(f"{n} {v:+.2f}" for (_, n), v in zip(sl, vals, strict=True)))
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xx, [n for _, n in sl])
    ax.set_ylabel("Δ mAP@0.5:0.95 vs fixed FP16 (pts)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    _save(fig, "slices", out, alt, "Grouped bars of the test mAP@0.5:0.95 difference to fixed FP16 per slice "
          "(FP16 itself is the zero line). " + ". ".join(txt) + ". †Scene changes are synthetic (spliced clips).")

    # ---- F4 K-sweep sensitivity (two panels)
    ks = sorted(int(k[len(P + "adaptive_ksweep"):]) for k in idx if k.startswith(P + "adaptive_ksweep"))
    kv = [load(f"{P}adaptive_ksweep{K}", "metrics.json", results, idx)["all"] for K in ks]
    col = STYLE["adaptive_main"][1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.0))
    fig.subplots_adjust(wspace=0.5)
    a1.plot(ks, [v["switch_rate_per_100"] for v in kv], marker="^", color=col)
    a1.set_xlabel("persistence K (frames)")
    a1.set_ylabel("switches per 100 frames")
    a1.set_title("(a) switching")
    a2.plot(ks, [100 * v["map50_95"] for v in kv], marker="^", color=col)
    a2.axhline(100 * M["fixed_fp16"]["all"]["map50_95"], color=STYLE["fixed_fp16"][1], lw=1, ls="--")
    a2.text(ks[-1], 100 * M["fixed_fp16"]["all"]["map50_95"] - 0.12, "Fixed FP16", ha="right", fontsize=6.5)
    a2.set_xlabel("persistence K (frames)")
    a2.set_ylabel("test mAP@0.5:0.95 (pts)")
    a2.set_title("(b) accuracy")
    for a in (a1, a2):
        a.set_xticks(ks)
    _save(fig, "ksweep", out, alt, "Sensitivity analysis: two line charts over persistence K with theta fixed at the "
          "locked value. " + "; ".join(f"K={K}: {v['switch_rate_per_100']:.2f} switches/100, {100 * v['map50_95']:.2f} pts"
                                       for K, v in zip(ks, kv, strict=True)) + ". Dashed line: fixed FP16.")

    # ---- F5 paired differences (forest plot)
    names = {"adaptive_main - fixed_fp16": "Adaptive − FP16", "adaptive_main - fixed_int8": "Adaptive − INT8",
             "adaptive_main - random_matched": "Adaptive − matched random",
             "adaptive_main - random_iid": "Adaptive − i.i.d. random", "oracle_cheap - fixed_fp16": "Oracle − FP16"}
    items = [(names[k], boot["pairs"][k]["map50_95"]) for k in names if k in boot["pairs"]]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for i, (_n, v) in enumerate(items):
        lo, hi = 100 * v["ci95"][0], 100 * v["ci95"][1]
        ax.plot([lo, hi], [i, i], color=INK, lw=1.5)
        ax.plot([100 * v["mean"]], [i], marker="o", color=INK, ms=5)
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(range(len(items)), [n for n, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("Δ test mAP@0.5:0.95 (pts), bootstrap mean and 95% CI")
    _save(fig, "ci_pairs", out, alt, f"Forest plot of paired differences, cluster bootstrap over 40 clips "
          f"({boot['n_resamples']} resamples): " + "; ".join(f"{n} {100 * v['mean']:+.2f} [{100 * v['ci95'][0]:+.2f}, "
                                                              f"{100 * v['ci95'][1]:+.2f}]" for n, v in items) + ".")

    # ---- F6 level-over-time trace for the test clip with the most switches
    rows = list(csv.DictReader(open(run_dir(P + "adaptive_main", results, idx) / "per_frame.csv", newline="")))
    by = {}
    for r in rows:
        by.setdefault(r["seq"], []).append(r)
    seq = max(sorted(by), key=lambda s: sum(int(r["switched"]) for r in by[s]))
    rs = sorted(by[seq], key=lambda r: int(r["frame_idx"]))
    t = [int(r["frame_idx"]) for r in rs]
    theta = cal["locks"]["main"]["params"]["theta"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 2.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(t, [float(r["score_hat"]) for r in rs], color=STYLE["adaptive_main"][1], lw=1.2)
    a1.axhline(theta, color=INK2, lw=0.9, ls="--")
    a1.text(t[-1], theta + 0.01, "θ (locked)", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ov = [int(r["frame_idx"]) for r in rs if r["override"] == "1"]
    for f in ov:
        a1.axvline(f, color=STYLE["fixed_fp16"][1], lw=0.6, alpha=0.6)
    if ov:
        a1.plot([], [], color=STYLE["fixed_fp16"][1], lw=0.8, label="scene-change override (→ FP16)")
        a1.legend(loc="lower right")
    a1.set_ylabel("score Ŝ")
    lv = [1 if r["level"] == "fp16" else 0 for r in rs]
    a2.step(t, lv, where="post", color=INK, lw=1.2)
    a2.set_yticks([0, 1], ["INT8", "FP16"])
    a2.set_xlabel("frame index (5 fps)")
    _save(fig, "trace", out, alt, f"Per-frame trace for test clip {seq} (the clip with the most switches): smoothed "
          f"score with the locked threshold {theta:.3f} (top) and the chosen precision (bottom). "
          f"{sum(lv)} of {len(lv)} frames ran at FP16; {len(ov)} scene-change overrides are marked.")

    (out / "alt_text.json").write_text(json.dumps({"results_commit": commit, **alt}, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    return {k: str(out / f"{k}.pdf") for k in alt}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        PREVIEW = Path(sys.argv[2])
    print(json.dumps(build(), indent=1))
