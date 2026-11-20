"""BDD100K MOT 2020 (box_track) parsing and dataset-layout discovery.

The Kaggle mirror's folder layout is not documented, so `discover_layout` finds the
label and image folders by name and fails loudly with the tree it saw if they are
missing. The label format assumed is Scalabel (per-video JSON list of frames with
`name`, `videoName`, `frameIndex`, `labels[{id, category, attributes, box2d}]`);
`parse_video_labels` raises on anything it does not understand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .classmap import bdd_category


class LayoutError(FileNotFoundError):
    pass


class LabelFormatError(ValueError):
    pass


@dataclass
class Box:
    track_id: str
    category: str
    kind: str              # "eval" | "ignore"
    eval_id: int | None
    x1: float
    y1: float
    x2: float
    y2: float
    crowd: bool = False


@dataclass
class Frame:
    seq: str
    frame_idx: int
    name: str
    boxes: list[Box] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- layout

def _find_dirs_named(root: Path, name: str, max_depth: int) -> list[Path]:
    hits: list[Path] = []
    root = Path(root)
    base_depth = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if len(p.parts) - base_depth >= max_depth:
            dirnames[:] = []
        for d in dirnames:
            if d == name:
                hits.append(p / d)
        # do not descend into frame folders: they hold thousands of jpgs
        dirnames[:] = [d for d in dirnames if d != name]
    return sorted(hits)


def tree_preview(root: Path, max_depth: int = 4, max_entries: int = 25) -> str:
    lines: list[str] = []
    root = Path(root)
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        ind = "  " * depth
        lines.append(f"{ind}{Path(dirpath).name}/  ({len(dirnames)} dirs, {len(filenames)} files)")
        if len(dirnames) > max_entries:
            dirnames[:] = dirnames[:3]
            lines.append(f"{ind}  ... (showing 3 of many subdirs)")
    return "\n".join(lines)


@dataclass
class Layout:
    label_dir: Path        # folder holding per-video *.json for the chosen split
    image_root: Path       # folder holding one subfolder per video
    n_label_files: int
    n_image_dirs: int


def discover_layout(search_root: str | Path, images_dir_name: str, labels_dir_name: str,
                    split: str = "val", max_depth: int = 6) -> Layout:
    search_root = Path(search_root)
    if not search_root.exists():
        raise LayoutError(f"search root does not exist: {search_root}")

    img_hits = _find_dirs_named(search_root, images_dir_name, max_depth)
    lbl_hits = _find_dirs_named(search_root, labels_dir_name, max_depth)
    problems = []
    if not img_hits:
        problems.append(f"no folder named {images_dir_name!r} under {search_root}")
    if not lbl_hits:
        problems.append(f"no folder named {labels_dir_name!r} under {search_root}")
    if problems:
        raise LayoutError("; ".join(problems) + "\nTree seen:\n" + tree_preview(search_root))

    # Label dir: the folder under labels root with the most *.json, whose path
    # contains a component equal to the split name.
    best: tuple[int, Path] | None = None
    for lroot in lbl_hits:
        for dirpath, _, files in os.walk(lroot):
            if split not in Path(dirpath).parts[len(lroot.parts):] and Path(dirpath).name != split:
                continue
            n = sum(1 for f in files if f.endswith(".json"))
            if n and (best is None or n > best[0]):
                best = (n, Path(dirpath))
    if best is None:
        raise LayoutError(
            f"no '{split}' folder with *.json found under {lbl_hits}\nTree seen:\n"
            + "\n".join(tree_preview(h) for h in lbl_hits)
        )

    # Image root: the folder under images root with the most subfolders that
    # contain .jpg files directly (one folder per video).
    best_img: tuple[int, Path] | None = None
    for iroot in img_hits:
        for dirpath, dirnames, _ in os.walk(iroot):
            n = 0
            for d in dirnames:
                try:
                    with os.scandir(Path(dirpath) / d) as it:
                        if any(e.name.lower().endswith(".jpg") for e in it):
                            n += 1
                except OSError:
                    pass
            if n and (best_img is None or n > best_img[0]):
                best_img = (n, Path(dirpath))
            if n:
                dirnames[:] = []  # its children are frame folders
    if best_img is None:
        raise LayoutError(
            f"no per-video jpg folders found under {img_hits}\nTree seen:\n"
            + "\n".join(tree_preview(h) for h in img_hits)
        )
    return Layout(label_dir=best[1], image_root=best_img[1], n_label_files=best[0], n_image_dirs=best_img[0])


# --------------------------------------------------------------------------- labels

def parse_video_labels(obj: list | dict, source: str = "<obj>") -> list[Frame]:
    """Parse one per-video Scalabel label file (already json-loaded)."""
    if isinstance(obj, dict) and "frames" in obj:
        obj = obj["frames"]
    if not isinstance(obj, list):
        raise LabelFormatError(f"{source}: expected a list of frames, got {type(obj).__name__}")
    frames: list[Frame] = []
    for i, fr in enumerate(obj):
        try:
            seq = fr["videoName"]
            fidx = int(fr["frameIndex"])
            name = fr["name"]
        except (KeyError, TypeError, ValueError) as e:
            raise LabelFormatError(f"{source}: frame {i} missing videoName/frameIndex/name ({e!r})") from e
        boxes: list[Box] = []
        for j, lab in enumerate(fr.get("labels") or []):
            if "box2d" not in lab or lab["box2d"] is None:
                continue  # non-box label types are not used
            b = lab["box2d"]
            try:
                x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError) as e:
                raise LabelFormatError(f"{source}: frame {i} label {j} bad box2d {b!r}") from e
            if x2 < x1 or y2 < y1:
                raise LabelFormatError(f"{source}: frame {i} label {j} inverted box {b!r}")
            kind, eid = bdd_category(lab["category"])
            attrs = lab.get("attributes") or {}
            boxes.append(Box(track_id=str(lab.get("id", f"{i}_{j}")), category=lab["category"],
                             kind=kind, eval_id=eid, x1=x1, y1=y1, x2=x2, y2=y2,
                             crowd=bool(attrs.get("crowd", False))))
        frames.append(Frame(seq=seq, frame_idx=fidx, name=name, boxes=boxes,
                            attributes=dict(fr.get("attributes") or {})))
    seqs = {f.seq for f in frames}
    if len(seqs) > 1:
        raise LabelFormatError(f"{source}: expected one video per file, found {sorted(seqs)[:5]}")
    frames.sort(key=lambda f: f.frame_idx)
    idxs = [f.frame_idx for f in frames]
    if len(set(idxs)) != len(idxs):
        raise LabelFormatError(f"{source}: duplicate frameIndex values")
    return frames


def load_video_labels(path: str | Path) -> list[Frame]:
    with open(path, encoding="utf-8") as f:
        return parse_video_labels(json.load(f), source=str(path))


def sequence_timeofday(frames: list[Frame]) -> str | None:
    """Majority `timeofday` attribute over frames, or None if absent everywhere.
    Returned raw (e.g. 'daytime', 'night', 'dawn/dusk', 'undefined')."""
    vals = [f.attributes.get("timeofday") for f in frames if f.attributes.get("timeofday")]
    if not vals:
        return None
    return max(sorted(set(vals)), key=vals.count)


def normalise_timeofday(raw: str | None) -> str | None:
    if raw is None:
        return None
    r = raw.strip().lower()
    if r == "daytime":
        return "daytime"
    if r == "night":
        return "night"
    if r in ("dawn/dusk", "dawn_dusk", "dawn", "dusk"):
        return "dawn_dusk"
    return None  # 'undefined' or unknown -> resolved by luma fallback
