"""TensorRT engine export and inference via Ultralytics. KAGGLE-ONLY (GPU).

Backends share one interface so the pipeline can be dry-run locally with a fake:
    backend.predict(path_or_array) -> (dets (N,6) x1,y1,x2,y2,score,coco_cls, timing dict)
"""

from __future__ import annotations

import json
import shutil
import struct
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from ..results import sha256_file


class Backend(Protocol):
    name: str

    def predict(self, src) -> tuple[np.ndarray, dict]: ...


def read_engine_metadata(path: str | Path) -> dict:
    """Ultralytics prefixes .engine files with a 4-byte little-endian length + JSON
    metadata. Returns {} with an error note if the layout is different."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<i", f.read(4))[0]
            if not 0 < n < 1_000_000:
                return {"_error": f"implausible metadata length {n}"}
            return json.loads(f.read(n).decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - recorded
        return {"_error": repr(e)}


def export_engine(weights: str | Path, level: str, imgsz: int, out_dir: str | Path,
                  int8_data: str | Path | None = None, batch: int = 1, tag: str | None = None) -> dict:
    """Export one TensorRT engine. Ultralytics writes <stem>.engine next to the weights,
    so it is moved to out_dir/<stem>_<tag or level>.engine immediately (tag e.g. 'int8_480'
    keeps engines of different input sizes apart)."""
    from ultralytics import YOLO

    if level not in ("fp32", "fp16", "int8"):
        raise ValueError(level)
    if level == "int8" and not int8_data:
        raise ValueError("INT8 export needs int8_data (calibration yaml from the calib split)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(format="engine", imgsz=imgsz, batch=batch, dynamic=False, device=0, verbose=False)
    if level == "fp16":
        kw["half"] = True
    if level == "int8":
        kw.update(int8=True, data=str(int8_data), fraction=1.0)
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    produced = Path(model.export(**kw))
    build_s = time.perf_counter() - t0
    dest = out_dir / f"{Path(weights).stem}_{tag or level}.engine"
    shutil.move(str(produced), dest)
    import tensorrt as trt

    return {"level": level, "path": str(dest), "sha256": sha256_file(dest), "bytes": dest.stat().st_size,
            "build_s": round(build_s, 2), "export_kwargs": kw, "tensorrt": trt.__version__,
            "ultralytics_metadata": read_engine_metadata(dest)}


class UltralyticsBackend:
    """Wraps YOLO(<.pt|.engine>).predict with fixed eval settings."""

    def __init__(self, name: str, model_path: str | Path, imgsz: int, conf: float, iou: float, max_det: int,
                 rect: bool):
        import torch
        from ultralytics import YOLO

        self.name = name
        self.torch = torch
        self.model = YOLO(str(model_path), task="detect")
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, rect=rect, verbose=False, device=0)
        if str(model_path).endswith(".pt"):
            self.kw["half"] = False  # FP32 PyTorch reference

    def predict(self, src) -> tuple[np.ndarray, dict]:
        self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = self.model.predict(src, **self.kw)[0]
        self.torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000
        b = r.boxes
        dets = np.column_stack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()]) \
            if len(b) else np.zeros((0, 6))
        sp = r.speed
        return dets.astype(np.float32), {"pre_ms": sp.get("preprocess"), "infer_ms": sp.get("inference"),
                                         "post_ms": sp.get("postprocess"), "wall_ms": wall}


def microbench(backend: Backend, image: np.ndarray, iters: int, warmup: int) -> dict:
    """Repeated inference on one pre-decoded frame (decode excluded). Engine-level
    latency from Ultralytics speed['inference'] plus synchronised wall time."""
    for _ in range(warmup):
        backend.predict(image)
    inf, wall = [], []
    for _ in range(iters):
        _, t = backend.predict(image)
        inf.append(t["infer_ms"])
        wall.append(t["wall_ms"])

    def stats(x):
        a = np.asarray(x, float)
        return {"median": float(np.median(a)), "mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
                "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}

    return {"backend": backend.name, "infer_ms": stats(inf), "wall_ms": stats(wall), "iters": iters,
            "warmup": warmup}


def size_rule(fp16_median_ms: float, int8_median_ms: float, min_speedup: float) -> dict:
    """PLAN §1.2: triggered (also run yolo11m) if INT8 is < min_speedup faster than FP16."""
    speedup = 1.0 - int8_median_ms / fp16_median_ms
    return {"int8_speedup_vs_fp16": speedup, "threshold": min_speedup, "triggered": bool(speedup < min_speedup)}
