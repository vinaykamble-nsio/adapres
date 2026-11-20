"""Config loading and schema validation (no external schema library)."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

# key path -> expected type(s). Every key listed here is required.
_SCHEMA: dict[str, type | tuple[type, ...]] = {
    "seed": int,
    "paths.input_root": str,
    "paths.bdd_dataset_slug": str,
    "paths.work_root": str,
    "paths.results_dir": str,
    "paths.data_dir": str,
    "paths.logs_dir": str,
    "bdd.images_dir_name": str,
    "bdd.labels_dir_name": str,
    "bdd.label_split": str,
    "bdd.subsets.calib.n_sequences": int,
    "bdd.subsets.calib.strata_targets": dict,
    "bdd.subsets.test.n_sequences": int,
    "bdd.subsets.test.strata_targets": dict,
    "bdd.subsets.int8_calib_frames": int,
    "bdd.smoke.calib.n_sequences": int,
    "bdd.smoke.calib.strata_targets": dict,
    "bdd.smoke.test.n_sequences": int,
    "bdd.smoke.test.strata_targets": dict,
    "bdd.smoke.int8_calib_frames": int,
    "slices.crowded_percentile": (int, float),
    "slices.natural_scene_change.new_track_ratio": (int, float),
    "slices.natural_scene_change.min_tracks": int,
    "slices.easy_excludes_natural_scene_change": bool,
    "splice.n_streams": int,
    "splice.segment_len": int,
    "splice.min_segments": int,
    "splice.max_segments": int,
    "splice.window": int,
    "splice.smoke_n_streams": int,
    "classmap.ignore_mode": str,
    "model.name": str,
    "model.imgsz": int,
    "env.cuda_visible_devices": str,
    "env.require_ort_cuda": bool,
    "model.weights": str,
    "model.conf": (int, float),
    "model.iou": (int, float),
    "model.max_det": int,
    "model.conf_cue": (int, float),
    "model.rect": bool,
    "model.size_rule_min_int8_speedup": (int, float),
    "eval.max_dets": int,
    "eval.perframe_iou": (int, float),
    "eval.ignore_ioa": (int, float),
    "step2.split": str,
    "step2.levels": list,
    "step2.pt_reference": bool,
    "step2.n_sequences": int,
    "step2.smoke_n_sequences": int,
    "step2.warmup_frames": int,
    "step2.microbench_iters": int,
    "step2.microbench_warmup": int,
    "step2.val_check_frames": int,
    "step2.smoke_val_check_frames": int,
    "step2.int8_batch": int,
    "step2.sanity.fp16_vs_fp32_max_abs_pts": (int, float),
    "step2.sanity.trt_fp32_vs_pt_max_abs_pts": (int, float),
    "step2.sanity.int8_max_drop_pts": (int, float),
    "step2.sanity.int8_max_gain_pts": (int, float),
    "step2.sanity.val_check_max_abs_pts": (int, float),
    "step2.sanity.val_diag.ours_dup_other_class_max": (int, float),
    "step2.sanity.val_diag.theirs_dup_other_class_min": (int, float),
    "step2.sanity.val_diag.high_conf_match_min": (int, float),
    "features.width": int,
    "features.canny_low": int,
    "features.canny_high": int,
    "features.hist_bins": int,
    "controller.levels": list,
    "controller.start_level": str,
    "controller.weights": dict,
    "controller.alpha": (int, float),
    "controller.theta_quantile": (int, float),
    "controller.K": int,
    "controller.H": int,
    "controller.theta_sc": (int, float),
    "controller.u_empty": (int, float),
    "controller.scene_override": bool,
    "controller.w_h": (int, float),
    "controller.norm_percentiles": list,
    "step3.split": str,
    "step3.smoke_n_sequences": int,
    "step3.K_sweep": list,
    "step3.random_seeds": int,
    "step3.iid_p_int8": (int, float),
    "full.exp_prefix": str,
    "full.gate_calib_map_tol_pts": (int, float),
    "full.delta_pts": (int, float),
    "full.calib": dict,
    "full.smoke_calib": dict,
    "full.test": dict,
    "full.timing": dict,
    "full.smoke_timing": dict,
    "full.live_validation": dict,
}

VALID_LEVELS = ("fp32", "fp16", "int8")

VALID_STRATA = {"daytime", "night", "dawn_dusk"}


class ConfigError(ValueError):
    pass


def get(cfg: dict, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"missing config key: {dotted}")
        cur = cur[part]
    return cur


def validate(cfg: dict) -> dict:
    """Raise ConfigError listing every problem; return cfg unchanged if valid."""
    errors: list[str] = []
    for key, typ in _SCHEMA.items():
        try:
            val = get(cfg, key)
        except ConfigError as e:
            errors.append(str(e))
            continue
        # bool is a subclass of int; reject bools where ints are expected.
        if isinstance(val, bool) and typ is not bool and bool not in (typ if isinstance(typ, tuple) else (typ,)):
            errors.append(f"{key}: expected {typ}, got bool")
        elif not isinstance(val, typ):
            errors.append(f"{key}: expected {typ}, got {type(val).__name__}")
    for mode in ("subsets", "smoke"):
        for split in ("calib", "test"):
            try:
                n = get(cfg, f"bdd.{mode}.{split}.n_sequences")
                targets = get(cfg, f"bdd.{mode}.{split}.strata_targets")
            except ConfigError:
                continue
            if not isinstance(targets, dict):
                continue
            bad = set(targets) - VALID_STRATA
            if bad:
                errors.append(f"bdd.{mode}.{split}.strata_targets: unknown strata {sorted(bad)}")
            if isinstance(n, int) and sum(targets.values()) != n:
                errors.append(
                    f"bdd.{mode}.{split}: strata_targets sum {sum(targets.values())} != n_sequences {n}"
                )
    try:
        if get(cfg, "splice.min_segments") > get(cfg, "splice.max_segments"):
            errors.append("splice.min_segments > splice.max_segments")
        if get(cfg, "splice.min_segments") < 2:
            errors.append("splice.min_segments must be >= 2 (a splice needs two segments)")
    except ConfigError:
        pass
    try:
        lv = get(cfg, "step2.levels")
        if isinstance(lv, list) and (not lv or any(x not in VALID_LEVELS for x in lv)):
            errors.append(f"step2.levels must be a non-empty subset of {VALID_LEVELS}, got {lv}")
        if get(cfg, "step2.split") != "calib":
            errors.append("step2.split must be 'calib' (test stays untouched until configs are locked)")
    except ConfigError:
        pass
    try:
        if get(cfg, "step3.split") != "calib":
            errors.append("step3.split must be 'calib' (test stays untouched until configs are locked)")
        c = get(cfg, "controller")
        w = c["weights"]
        if set(w) != {"m", "c", "n", "u", "s"} or any(v < 0 for v in w.values()) or abs(sum(w.values()) - 1) > 1e-9:
            errors.append(f"controller.weights must be non-negative over m,c,n,u,s and sum to 1, got {w}")
        if not 0 < c["alpha"] <= 1:
            errors.append("controller.alpha must be in (0, 1]")
        if c["start_level"] not in c["levels"] or any(x not in VALID_LEVELS for x in c["levels"]):
            errors.append("controller.levels/start_level invalid")
        for k in ("K", "K_up", "K_down"):
            v = c.get(k)
            if v is not None and (not isinstance(v, int) or v < 1):
                errors.append(f"controller.{k} must be an int >= 1 or null")
        if not 0 <= c["theta_quantile"] <= 1:
            errors.append("controller.theta_quantile must be in [0, 1]")
    except (ConfigError, KeyError, TypeError):
        pass
    try:
        if get(cfg, "classmap.ignore_mode") not in ("all_eval_classes", "none"):
            errors.append("classmap.ignore_mode must be 'all_eval_classes' or 'none'")
    except ConfigError:
        pass
    if errors:
        raise ConfigError("invalid config:\n  " + "\n  ".join(errors))
    return cfg


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load(path: str | Path, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return validate(cfg)


def config_sha256(cfg: dict) -> str:
    blob = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def resolve_out(cfg: dict, key: str, smoke: bool) -> Path:
    """Output directory under work_root. Smoke runs go under work_root/smoke/ so
    they can never be mistaken for, or overwrite, full-run results."""
    root = Path(cfg["paths"]["work_root"])
    if smoke:
        root = root / "smoke"
    return root / cfg["paths"][key]
