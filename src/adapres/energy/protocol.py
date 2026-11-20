"""Latency / energy / memory protocol (PLAN §7.3-7.4). GPU board energy only.

Per repeat (3 by default, order of configs shuffled with a seed):
  idle window (engines resident, no work) -> P_idle
  for each config: warm-up (warmup_s, then up to settle_s more until temperature changes
  < 1 C over 20 s), then a measurement window of >= measure_s of continuous inference.
Per config and repeat: N frames, T, dE (counter), E_total/frame = dE/N,
E_dyn/frame = (dE - P_idle*T)/N, mean power, throughput, per-frame latency median/p95.
Deviation (README D8): idle is measured once per repeat (all engines stay resident for every
config, as in deployment) and the settle wait is capped at settle_s.
"""

from __future__ import annotations

import time
from typing import Callable, Mapping

import numpy as np


def _lat_stats(x) -> dict:
    a = np.asarray([v for v in x if v is not None], float)
    if not a.size:
        return {"n": 0}
    return {"median": float(np.median(a)), "mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
            "n": int(a.size)}


def run_for(stepper, seconds: float) -> list[dict]:
    rows = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        rows.append(stepper.step())
    return rows


def settle(stepper, meter, max_s: float, window_s: float = 20.0, delta_c: float = 1.0) -> dict:
    """Keep running until temperature changes < delta_c over window_s (max max_s)."""
    t0 = time.perf_counter()
    temps = [(t0, meter.temperature())]
    while time.perf_counter() - t0 < max_s:
        run_for(stepper, min(2.0, max_s))
        now = time.perf_counter()
        temps.append((now, meter.temperature()))
        old = [tc for (ts, tc) in temps if now - ts >= window_s]
        if old and abs(temps[-1][1] - old[-1]) < delta_c:
            return {"settled": True, "settle_s": now - t0, "temp_c": temps[-1][1]}
    return {"settled": False, "settle_s": time.perf_counter() - t0, "temp_c": temps[-1][1]}


def run_protocol(steppers: Mapping[str, object], meter, warmup_s: float, settle_s: float, idle_s: float,
                 measure_s: float, repeats: int, seed: int, log: Callable = print) -> dict:
    rng = np.random.default_rng(seed)
    names = sorted(steppers)
    runs = []
    for r in range(repeats):
        order = [names[i] for i in rng.permutation(len(names))]
        with meter.window() as idle:
            time.sleep(idle_s)
        p_idle_w = (idle["energy_mJ"] / 1000.0) / idle["T_s"] if idle["energy_mJ"] is not None else None
        log(f"repeat {r}: idle {p_idle_w if p_idle_w is None else round(p_idle_w, 2)} W; order {order}")
        for name in order:
            st = steppers[name]
            run_for(st, warmup_s)
            stl = settle(st, meter, settle_s) if settle_s > 0 else {"settled": None, "settle_s": 0.0,
                                                                      "temp_c": meter.temperature()}
            with meter.window() as w:
                rows = run_for(st, measure_s)
            N, T = len(rows), w["T_s"]
            E = w["energy_mJ"]
            rec = {"repeat": r, "config": name, "N": N, "T_s": T, "dE_mJ": E, "dE_sampled_mJ": w["energy_sampled_mJ"],
                   "method": w["method"], "P_idle_W": p_idle_w, "throughput_fps": N / T if T else None,
                   "E_total_mJ_per_frame": E / N if (E is not None and N) else None,
                   "E_dyn_mJ_per_frame": ((E - p_idle_w * 1000.0 * T) / N) if (E is not None and N and p_idle_w is not None) else None,
                   "mean_power_W": (E / 1000.0) / T if (E is not None and T) else None,
                   "latency_total_ms": _lat_stats([x["total_ms"] for x in rows]),
                   "latency_engine_ms": _lat_stats([x.get("engine_infer_ms") for x in rows]),
                   "temp_c_start": stl["temp_c"], "settle": stl,
                   "temp_c_end": w["samples"][-1]["temp_c"] if w["samples"] else None,
                   "sm_clock_mhz_median": float(np.median([s["sm_clock_mhz"] for s in w["samples"] if s["sm_clock_mhz"] is not None]))
                   if any(s["sm_clock_mhz"] is not None for s in w["samples"]) else None,
                   "util_gpu_median": float(np.median([s["util_gpu"] for s in w["samples"] if s["util_gpu"] is not None]))
                   if any(s["util_gpu"] is not None for s in w["samples"]) else None}
            if name.startswith("adaptive"):
                lv = [x["level"] for x in rows]
                rec["frac_int8"] = float(np.mean([v == "int8" for v in lv])) if lv else None
                rec["feat_wait_ms"] = _lat_stats([x["feat_wait_ms"] for x in rows])
                rec["ctrl_ms"] = _lat_stats([x["ctrl_ms"] for x in rows])
            runs.append(rec)
            log(f"  {name:22s} N={N:5d} {rec['throughput_fps'] or 0:7.1f} fps  E_dyn/frame="
                f"{rec['E_dyn_mJ_per_frame'] if rec['E_dyn_mJ_per_frame'] is None else round(rec['E_dyn_mJ_per_frame'], 2)} mJ  "
                f"lat med {rec['latency_total_ms'].get('median', float('nan')):.2f} ms")
    summary = {}
    for name in names:
        rs = [x for x in runs if x["config"] == name]

        def med_range(key, rs=rs):
            v = [x[key] for x in rs if x[key] is not None]
            return {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))} if v else None

        summary[name] = {k: med_range(k) for k in ("E_total_mJ_per_frame", "E_dyn_mJ_per_frame", "mean_power_W",
                                                     "throughput_fps", "P_idle_W")}
        med = [x["latency_total_ms"].get("median") for x in rs]
        summary[name]["latency_total_ms_median"] = {"median": float(np.median(med)), "min": float(min(med)),
                                                    "max": float(max(med))}
        p95 = [x["latency_total_ms"].get("p95") for x in rs]
        summary[name]["latency_total_ms_p95"] = {"median": float(np.median(p95)), "min": float(min(p95)), "max": float(max(p95))}
        if name.startswith("adaptive"):
            summary[name]["frac_int8"] = float(np.median([x["frac_int8"] for x in rs]))
    return {"runs": runs, "summary": summary, "method": meter.method, "repeats": repeats,
            "warmup_s": warmup_s, "settle_max_s": settle_s, "idle_s": idle_s, "measure_s": measure_s, "seed": seed}


def switch_penalty(backends: Mapping[str, object], frames, n_frames: int, repeats: int = 3) -> dict:
    """Mean extra latency per switch: alternate int8/fp16 every frame vs constant levels over
    the same frames. penalty = mean_alt - (mean_int8 + mean_fp16) / 2 (median over repeats)."""
    fr = [frames[i % len(frames)] for i in range(n_frames)]

    def timed(seq_levels):
        out = []
        for img, lv in zip(fr, seq_levels, strict=True):
            t0 = time.perf_counter()
            backends[lv].predict(img)
            out.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(out))

    reps = []
    for _ in range(repeats):
        m8 = timed(["int8"] * n_frames)
        m16 = timed(["fp16"] * n_frames)
        alt = timed(["int8" if i % 2 == 0 else "fp16" for i in range(n_frames)])
        reps.append({"mean_int8_ms": m8, "mean_fp16_ms": m16, "mean_alternating_ms": alt,
                     "penalty_ms": alt - (m8 + m16) / 2})
    return {"repeats": reps, "penalty_ms": float(np.median([r["penalty_ms"] for r in reps])), "n_frames": n_frames}


def memory_profile(meter, load_fn: Callable[[str], object], levels_alone=("fp32", "fp16", "int8"),
                   adaptive_set=("int8", "fp16"), release_fn: Callable[[], None] | None = None) -> dict:
    """GPU memory (NVML used, MiB) after context init, after each engine alone, and with the
    adaptive set resident (PLAN §7.4). load_fn(level) must return a backend that is fully
    loaded on the GPU (Ultralytics loads lazily, so load_fn runs one prediction)."""
    release_fn = release_fn or (lambda: None)
    base = meter.memory_used_mib()
    out = {"baseline_mib": base, "alone_delta_mib": {}}
    for lv in levels_alone:
        b = load_fn(lv)
        out["alone_delta_mib"][lv] = meter.memory_used_mib() - base
        del b
        release_fn()
    bs = [load_fn(lv) for lv in adaptive_set]
    out["adaptive_set"] = list(adaptive_set)
    out["adaptive_set_delta_mib"] = meter.memory_used_mib() - base
    del bs
    release_fn()
    return out
