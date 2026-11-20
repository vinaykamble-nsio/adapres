"""Notebook 02b: is the pipeline-vs-model.val() gap caused by the predictions or by the
metric implementation? Calib frames only (the same seeded 200-frame mini-set as
notebook 02's val check); the test split is never touched.

Three numbers on identical frames and identical GT (crowd/ignore boxes removed):
  A = our predictions  (YOLO.predict)       scored by our evaluator (pycocotools)
  B = their predictions (model.val jdict)   scored by our evaluator
  C = their predictions                     scored by Ultralytics' own metric (model.val)
  prediction component = B - A   (same metric, different predictions)
  metric component     = C - B   (same predictions, different metric)
Decision rule (fixed before the run): per metric, the cause is the component with the
larger |value|. Binding agreement check (user decision 2026-11-20):
  - default: end-to-end |C - A| <= tol on BOTH mAP50-95 and mAP50;
  - if every end-to-end failure is caused by the metric component, the tolerance is applied
    instead to the identical-predictions comparison |C - B| on BOTH metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..data.classmap import COCO_TO_EVAL
from ..eval.map_eval import CocoEvaluator, dets_to_coco_results
from ..eval.perframe import iou_matrix
from ..results import ExperimentWriter

METRICS = ("map50_95", "map50")


class DiagError(RuntimeError):
    pass


def parse_ultralytics_jdict(jdict: list[dict], stem_to_id: dict[str, int], category_offset: int = 1
                            ) -> dict[int, np.ndarray]:
    """Ultralytics DetectionValidator.jdict (non-COCO dataset) -> {image_id: (N,6) xyxy,score,coco80}.
    For non-COCO data Ultralytics writes category_id = class_index + 1 and image_id = file stem
    (or int(stem) if numeric). Boxes are xywh (top-left) in original-image pixels."""
    out: dict[int, list] = {i: [] for i in stem_to_id.values()}
    unknown = set()
    for p in jdict:
        key = Path(p.get("file_name", "")).stem or str(p["image_id"])
        if key not in stem_to_id:
            key = str(p["image_id"])
        if key not in stem_to_id:
            unknown.add(key)
            continue
        x, y, w, h = p["bbox"]
        out[stem_to_id[key]].append([x, y, x + w, y + h, p["score"], int(p["category_id"]) - category_offset])
    if unknown:
        raise DiagError(f"{len(unknown)} predicted image keys not in the mini-set, e.g. {sorted(unknown)[:3]}")
    return {k: np.asarray(v, float).reshape(-1, 6) for k, v in out.items()}


def compare_predictions(ours: dict[int, np.ndarray], theirs: dict[int, np.ndarray], tau: float = 0.25,
                        iou_thr: float = 0.9) -> dict:
    """How similar are the two prediction sets? Evaluated classes only, score >= tau.
    A box 'matches' if the other set has a same-class box with IoU >= iou_thr."""
    n_o = n_t = matched_o = matched_t = 0
    score_diffs, count_all_o, count_all_t = [], 0, 0
    for i in ours:
        o, t = ours[i], theirs.get(i, np.zeros((0, 6)))
        count_all_o += len(o)
        count_all_t += len(t)

        def keep(a):
            m = np.array([COCO_TO_EVAL.get(int(c)) is not None for c in a[:, 5]], bool) if len(a) else np.zeros(0, bool)
            return a[m & (a[:, 4] >= tau)] if len(a) else a

        o, t = keep(o), keep(t)
        n_o += len(o)
        n_t += len(t)
        if len(o) and len(t):
            iou = iou_matrix(o[:, :4], t[:, :4])
            same = o[:, 5][:, None] == t[:, 5][None, :]
            iou = np.where(same, iou, 0.0)
            best_t = iou.argmax(1)
            ok_o = iou.max(1) >= iou_thr
            matched_o += int(ok_o.sum())
            matched_t += int((iou.max(0) >= iou_thr).sum())
            score_diffs += list(np.abs(o[ok_o, 4] - t[best_t[ok_o], 4]))
    return {"tau": tau, "iou_thr": iou_thr, "n_ours": n_o, "n_theirs": n_t,
            "n_ours_all_conf": count_all_o, "n_theirs_all_conf": count_all_t,
            "frac_ours_matched": matched_o / n_o if n_o else None,
            "frac_theirs_matched": matched_t / n_t if n_t else None,
            "median_abs_score_diff": float(np.median(score_diffs)) if score_diffs else None}


def decompose(a: dict, b: dict, c: dict, tol_pts: float) -> dict:
    """a/b/c: {'map50_95': x, 'map50': y} in [0,1]. Returns components (pts), causes, checks, verdict."""
    comp = {}
    for k in METRICS:
        pred, metric, e2e = 100 * (b[k] - a[k]), 100 * (c[k] - b[k]), 100 * (c[k] - a[k])
        cause = "metric_implementation" if abs(metric) > abs(pred) else "predictions"
        comp[k] = {"A_ours_pred_ours_metric": 100 * a[k], "B_their_pred_ours_metric": 100 * b[k],
                   "C_their_pred_their_metric": 100 * c[k], "prediction_component_pts": pred,
                   "metric_component_pts": metric, "end_to_end_pts": e2e, "cause": cause}
    checks = {
        "end_to_end": {k: abs(comp[k]["end_to_end_pts"]) <= tol_pts for k in METRICS},
        "identical_predictions": {k: abs(comp[k]["metric_component_pts"]) <= tol_pts for k in METRICS},
        "prediction_agreement": {k: abs(comp[k]["prediction_component_pts"]) <= tol_pts for k in METRICS},
    }
    failing = [k for k in METRICS if not checks["end_to_end"][k]]
    if failing and all(comp[k]["cause"] == "metric_implementation" for k in failing):
        binding = "identical_predictions"
    else:
        binding = "end_to_end"
    verdict = all(checks[binding].values())
    return {"tolerance_pts": tol_pts, "components": comp, "checks": checks, "binding_check": binding,
            "binding_pass": verdict, "end_to_end_failing_metrics": failing}


def run(step, predict_fn: Callable, val_fn: Callable, tol_pts: float, committed_val_check: dict | None) -> dict:
    """step: a prepared eval_fixed.Step2. predict_fn(path) -> (N,6) dets (our YOLO.predict, pt FP32).
    val_fn(yaml) -> {'map50_95','map50','jdict','args'} from Ultralytics model.val(save_json=True)."""
    vs = step.val_check_set("val_mini_02b")
    ev = CocoEvaluator(vs["gt"], step.cfg["eval"]["max_dets"])
    ours = {i: np.asarray(predict_fn(str(vs["paths"][i])), float).reshape(-1, 6) for i in vs["ids"]}
    a = ev.evaluate(dets_to_coco_results(ours))
    v = val_fn(vs["yaml"])
    if not v.get("jdict"):
        raise DiagError("model.val() returned no predictions (jdict empty): save_json did not take effect")
    theirs = parse_ultralytics_jdict(v["jdict"], vs["stem_to_id"])
    b = ev.evaluate(dets_to_coco_results(theirs))
    c = {k: v[k] for k in METRICS}
    out = decompose({k: a[k] for k in METRICS}, {k: b[k] for k in METRICS}, c, tol_pts)
    out["prediction_similarity"] = compare_predictions(ours, theirs)
    out["n_frames"] = len(vs["ids"])
    out["image_ids"] = vs["ids"]
    out["ultralytics_val_args"] = v.get("args")
    out["per_class_ap50"] = {"A": a["per_class_ap50"], "B": b["per_class_ap50"]}
    out["per_class_ap50_95"] = {"A": a["per_class_ap50_95"], "B": b["per_class_ap50_95"]}
    if committed_val_check and not step.smoke:
        out["reproduces_notebook02"] = {
            k: {"notebook02_ours": committed_val_check["ours"][k], "now_A": a[k],
                "notebook02_ultralytics": committed_val_check["ultralytics_val"][k], "now_C": c[k]} for k in METRICS}
    w = ExperimentWriter(step.results_root, step._meta(f"val_diag_02b_{step.model_name}", "val_diagnostic",
                                                       dataset="sanity"))
    w.write_json("diag.json", out)
    w.register(policy="val_diagnostic", mode="check", replace=True)
    return out
