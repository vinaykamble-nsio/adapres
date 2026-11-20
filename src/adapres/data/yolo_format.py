"""Write a YOLO-format mini dataset (symlinked images + label txt + data yaml) from
manifest rows. Used for (a) INT8 calibration images and (b) the model.val() pipeline
check. Ignore/crowd boxes cannot be expressed in YOLO labels and are omitted (the
model.val() comparison therefore uses a COCO GT without them as well)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .bdd_mot import Frame
from .classmap import EVAL_CATEGORIES


def yolo_label_lines(frame: Frame, width: int, height: int) -> list[str]:
    """Lines 'coco_idx cx cy w h' (normalised), eval non-crowd boxes only, clipped to the image."""
    lines = []
    for b in frame.boxes:
        if b.kind != "eval" or b.crowd:
            continue
        x1, y1 = max(0.0, b.x1), max(0.0, b.y1)
        x2, y2 = min(float(width), b.x2), min(float(height), b.y2)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        coco_idx = EVAL_CATEGORIES[b.eval_id][1]
        lines.append(f"{coco_idx} {(x1 + w / 2) / width:.6f} {(y1 + h / 2) / height:.6f} "
                     f"{w / width:.6f} {h / height:.6f}")
    return lines


def flat_name(rel_path: str) -> str:
    return rel_path.replace("/", "__").replace("\\", "__")


def write_yolo_dataset(out_dir: str | Path, items: list[tuple[str, Frame, int, int]], image_root: str | Path,
                       names: dict[int, str], copy: bool = False) -> Path:
    """items: (rel_path, frame, W, H). Images are symlinked (copied if copy=True) to
    out_dir/images/, labels written to out_dir/labels/. Returns the data yaml path."""
    out_dir = Path(out_dir)
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for rel, fr, w, h in items:
        src = Path(image_root) / rel
        if not src.exists():
            raise FileNotFoundError(f"image missing: {src}")
        name = flat_name(rel)
        dst = img_dir / name
        if not dst.exists():
            if copy:
                dst.write_bytes(src.read_bytes())
            else:
                try:
                    os.symlink(src, dst)
                except OSError:  # e.g. Windows without symlink privilege: copy instead
                    dst.write_bytes(src.read_bytes())
        (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(yolo_label_lines(fr, w, h)) + "\n")
    data = {"path": str(out_dir), "train": "images", "val": "images",
            "names": {int(k): str(v) for k, v in names.items()}}
    p = out_dir / "data.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p
