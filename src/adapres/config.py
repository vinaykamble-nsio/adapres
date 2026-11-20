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
}

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
