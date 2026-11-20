"""Synthetic BDD100K-MOT-like fixtures (a few KB, no real data)."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

IMG_W, IMG_H = 64, 36
NIGHT_LUMA, DAY_LUMA = 30, 150


def make_frame(seq: str, idx: int, boxes: list[dict], tod: str | None) -> dict:
    fr = {"name": f"{seq}-{idx + 1:07d}.jpg", "videoName": seq, "frameIndex": idx, "labels": boxes}
    if tod is not None:
        fr["attributes"] = {"timeofday": tod}
    return fr


def box(tid: str, cat: str, x1, y1, x2, y2, crowd=False) -> dict:
    return {"id": tid, "category": cat, "attributes": {"occluded": False, "truncated": False, "crowd": crowd},
            "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def make_fake_bdd(root: Path, strata: dict[str, str], n_frames: int = 45, with_tod: bool = True,
                  seed: int = 0) -> Path:
    """Create <root>/tcjpaperdatasets/{track_images_val/.../val/<seq>/*.jpg (empty),
    box_track_labels_trainval/.../val/<seq>.json}. strata: seq -> timeofday raw value."""
    rng = np.random.default_rng(seed)
    ds = root / "tcjpaperdatasets"
    lbl = ds / "box_track_labels_trainval" / "bdd100k" / "labels" / "box_track_20" / "val"
    img = ds / "track_images_val" / "bdd100k" / "images" / "track" / "val"
    lbl.mkdir(parents=True)
    img.mkdir(parents=True)
    cats = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "car", "car"]
    for seq, tod in strata.items():
        (img / seq).mkdir()
        frames = []
        n_obj = int(rng.integers(1, 6))
        for t in range(n_frames):
            # occasional burst of new tracks -> natural scene change
            k = n_obj + (6 if t in (20, 21) else 0)
            boxes = []
            for j in range(k):
                x = float(rng.uniform(0, 40))
                y = float(rng.uniform(0, 20))
                tid = f"{seq}_{j}" if t < 20 or j < n_obj else f"{seq}_new{t}_{j}"
                boxes.append(box(tid, cats[j % len(cats)], x, y, x + 10, y + 8))
            if t % 5 == 0:
                boxes.append(box(f"{seq}_ign", "other vehicle", 1, 1, 12, 10))
            frames.append(make_frame(seq, t, boxes, tod if with_tod else None))
            (img / seq / frames[-1]["name"]).write_bytes(b"")
        (lbl / f"{seq}.json").write_text(json.dumps(frames))
    return ds


def fake_reader(strata: dict[str, str]):
    """Mocked image reader: night sequences dark, others bright, tiny noise."""
    def read(path: Path) -> np.ndarray:
        seq = Path(path).parent.name
        base = NIGHT_LUMA if strata.get(seq) == "night" else DAY_LUMA
        val = base + (zlib.crc32(Path(path).name.encode()) % 7)  # deterministic across processes
        return np.full((IMG_H, IMG_W, 3), val, dtype=np.uint8)
    return read


@pytest.fixture
def default_cfg():
    from adapres import config as C

    return C.load(REPO / "configs" / "default.yaml")
