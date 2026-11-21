"""Mechanical evaluation of the pre-registered hypotheses H1-H4 (configs/preregistration.yaml)
from test-split result dicts. Pure: used by notebook 04 and by scripts/recompute_hypotheses.py.

Revision 2 (2026-11-21, decided AFTER seeing results; DEVIATIONS.md V1-V3):
  * energy: E_total/frame is primary, E_dyn/frame secondary (idle-subtraction artifact, V1);
  * H3 is evaluated on the K-sweep (theta fixed at the locked value) as a SENSITIVITY ANALYSIS,
    each K >= 2 against K = 1 (rev. 1 compared the locked K with K = 1, undefined when locked K = 1, V2);
  * H4 saving share uses engine latency only (rev. 1 included the constant controller overhead,
    which made every slice's 'saving' negative, V3).
"""

from __future__ import annotations

from typing import Mapping

REVISION = 2


def _med(d):
    return None if not d else d.get("median")


def compute(m: Mapping[str, dict], random_agg: Mapping[str, dict], boot: dict, energy_summary: Mapping[str, dict],
            ksweep: Mapping[int, dict], engine_ms: Mapping[str, float], delta_pts: float, locked_K: int) -> dict:
    """m: policy -> metrics.json dict (slices). random_agg: 'random_matched'/'random_iid' -> metrics dict.
    ksweep: K -> metrics 'all' dict. engine_ms: level -> microbench median engine latency."""
    a, f16 = m["adaptive_main"], m["fixed_fp16"]
    drop = 100 * (f16["all"]["map50_95"] - a["all"]["map50_95"])
    meas = {}
    for mode in ("sequential", "pipelined"):
        s = energy_summary.get(f"adaptive_{mode}", {})
        ref = energy_summary["fixed_fp16"]
        meas[mode] = {
            "latency_median_ms": _med(s.get("latency_total_ms_median")),
            "fp16_latency_median_ms": _med(ref["latency_total_ms_median"]),
            "E_total_mJ_per_frame": _med(s.get("E_total_mJ_per_frame")),
            "fp16_E_total_mJ_per_frame": _med(ref["E_total_mJ_per_frame"]),
            "E_dyn_mJ_per_frame_secondary": _med(s.get("E_dyn_mJ_per_frame")),
            "fp16_E_dyn_mJ_per_frame_secondary": _med(ref["E_dyn_mJ_per_frame"]),
        }
        x = meas[mode]
        x["lower_latency"] = bool(x["latency_median_ms"] is not None and x["latency_median_ms"] < x["fp16_latency_median_ms"])
        x["lower_energy_total"] = bool(x["E_total_mJ_per_frame"] is not None and
                                       x["E_total_mJ_per_frame"] < x["fp16_E_total_mJ_per_frame"])
        x["lower_energy_dyn_secondary"] = bool(x["E_dyn_mJ_per_frame_secondary"] is not None and
                                               x["E_dyn_mJ_per_frame_secondary"] < x["fp16_E_dyn_mJ_per_frame_secondary"])
        x["supported"] = bool(drop <= delta_pts and x["lower_latency"] and x["lower_energy_total"])
    H1 = {"map_drop_pts": drop, "delta_pts": delta_pts, "map_within_delta": drop <= delta_pts,
          "energy_metric": "E_total (primary); E_dyn secondary", "measured": meas,
          "supported_sequential": meas["sequential"]["supported"], "supported_pipelined": meas["pipelined"]["supported"]}

    pr = boot["pairs"].get("adaptive_main - random_matched", {}).get("map50_95")
    H2 = {"diff_mean_pts": 100 * pr["mean"] if pr else None,
          "diff_ci95_pts": [100 * x for x in pr["ci95"]] if pr else None,
          "supported": bool(pr and pr["excludes_zero"] and pr["ci95"][0] > 0),
          "matched_frac_int8": {"adaptive": a["all"]["frac_level"]["int8"],
                                "random_matched": random_agg["random_matched"]["all"]["frac_level"]["int8"]},
          "matched_switch_rate": {"adaptive": a["all"]["switch_rate_per_100"],
                                  "random_matched": random_agg["random_matched"]["all"].get("switch_rate_per_100")}}

    H3 = {"analysis": "sensitivity (K-sweep, theta fixed at the locked value; no recalibration)", "locked_K": locked_K,
          "per_K": {}, "supported": None}
    if 1 in ksweep:
        k1 = ksweep[1]
        for K in sorted(k for k in ksweep if k >= 2):
            kk = ksweep[K]
            dm = 100 * (kk["map50_95"] - k1["map50_95"])
            H3["per_K"][K] = {"switch_rate": kk["switch_rate_per_100"], "switch_rate_K1": k1["switch_rate_per_100"],
                              "map_change_pts_vs_K1": dm,
                              "fewer_switches": kk["switch_rate_per_100"] < k1["switch_rate_per_100"],
                              "map_change_below_0p3": abs(dm) < 0.3}
        if H3["per_K"]:
            H3["supported"] = all(v["fewer_switches"] and v["map_change_below_0p3"] for v in H3["per_K"].values())

    fp16_frac = {sl: a[sl]["frac_level"]["fp16"] for sl in ("easy", "crowded", "night") if a.get(sl, {}).get("n_images")}
    if a.get("scene_change", {}).get("frac_level"):
        fp16_frac["scene_change"] = a["scene_change"]["frac_level"]["fp16"]
    easy = fp16_frac.get("easy")

    def engine_saving(sl):
        fr = a[sl]["frac_level"]
        per_frame = engine_ms["fp16"] - sum(fr.get(lv, 0.0) * engine_ms[lv] for lv in engine_ms)
        return per_frame * a[sl]["n_images"]

    s_easy, s_all = engine_saving("easy"), engine_saving("all")
    H4 = {"frac_fp16_by_slice": fp16_frac,
          "fp16_more_often_than_easy": {sl: v > easy for sl, v in fp16_frac.items() if sl != "easy"} if easy is not None else {},
          "engine_saving_ms_total": {"easy": s_easy, "all": s_all},
          "saving_share_from_easy": (s_easy / s_all) if s_all else None,
          "saving_definition": "engine latency only (microbench medians), controller overhead excluded"}
    H4["supported"] = bool(H4["fp16_more_often_than_easy"] and all(H4["fp16_more_often_than_easy"].values())
                           and (H4["saving_share_from_easy"] or 0) > 0.5)
    return {"revision": REVISION, "H1": H1, "H2": H2, "H3": H3, "H4": H4,
            "note": "computed mechanically from test results; H5 (contention) not run (out of scope)"}
