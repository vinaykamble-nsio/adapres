"""VisDrone-VID annotations (optional second benchmark, PLAN §4.2 / §5.2).

Annotation lines: frame_index, target_id, x, y, w, h, score, category, truncation, occlusion.
category 0 = ignored region, 11 = others; score 0 marks a box to ignore (VisDrone toolkit convention,
flagged [VERIFY] in PLAN). Classes map to the COCO-derived evaluation set (PLAN §5.2); tricycle,
awning-tricycle, ignored regions and others become ignore boxes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .bdd_mot import Box, Frame, LabelFormatError

# VisDrone category -> eval id (1 person, 2 car, 3 truck, 4 bus, 6 motorcycle, 7 bicycle); None = ignore
VISDRONE_TO_EVAL = {1: 1, 2: 1, 3: 7, 4: 2, 5: 2, 6: 3, 7: None, 8: None, 9: 4, 10: 6, 0: None, 11: None}


def parse_sequence(path: str | Path, seq: str | None = None) -> list[Frame]:
    path = Path(path)
    seq = seq or path.stem
    by_frame: dict[int, list[Box]] = defaultdict(list)
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.replace(" ", "").split(",") if p != ""]
        if len(parts) < 8:
            raise LabelFormatError(f"{path}:{ln}: expected >= 8 comma-separated fields, got {len(parts)}")
        try:
            fidx, tid, x, y, w, h, score, cat = (int(float(v)) for v in parts[:8])
        except ValueError as e:
            raise LabelFormatError(f"{path}:{ln}: non-numeric field") from e
        if cat not in VISDRONE_TO_EVAL:
            raise LabelFormatError(f"{path}:{ln}: unknown VisDrone category {cat}")
        eid = VISDRONE_TO_EVAL[cat]
        kind = "eval" if (eid is not None and score != 0) else "ignore"
        by_frame[fidx].append(Box(track_id=f"{seq}_{tid}", category=f"visdrone_{cat}", kind=kind,
                                  eval_id=eid if kind == "eval" else None, x1=float(x), y1=float(y),
                                  x2=float(x + w), y2=float(y + h)))
    return [Frame(seq=seq, frame_idx=f, name=f"{f:07d}.jpg", boxes=by_frame[f]) for f in sorted(by_frame)]


def discover(root: str | Path) -> dict:
    """Find VisDrone-VID 'annotations' and 'sequences' folders; returns paths and counts (no reading)."""
    root = Path(root)
    ann = sorted(p for p in root.rglob("annotations") if p.is_dir() and any(p.glob("*.txt")))
    seqs = sorted(p for p in root.rglob("sequences") if p.is_dir())
    return {"annotation_dirs": [str(p) for p in ann], "sequence_dirs": [str(p) for p in seqs],
            "n_annotation_files": sum(len(list(p.glob("*.txt"))) for p in ann)}
