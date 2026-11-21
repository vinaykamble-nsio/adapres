"""Seeds, git hash and environment capture. GPU queries degrade to recorded errors,
never to invented values."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit(repo_dir: str | Path) -> dict:
    """Return {'commit', 'dirty', 'source'}. Falls back to a GIT_COMMIT file written
    by scripts/make_repo_zip.py when the repo arrived as a zip without .git."""
    repo_dir = Path(repo_dir)
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty, "source": "git"}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        marker = repo_dir / "GIT_COMMIT"
        if marker.exists():
            return {"commit": marker.read_text().strip(), "dirty": False, "source": "GIT_COMMIT file"}
        return {"commit": None, "dirty": None, "source": f"unavailable: {e!r}"}


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def package_versions() -> dict:
    names = [
        "torch", "torchvision", "ultralytics", "tensorrt", "tensorrt-cu12", "tensorrt-cu13", "onnx",
        "onnxruntime", "onnxruntime-gpu", "onnxslim", "pycocotools", "nvidia-ml-py",
        "numpy", "pandas", "pyarrow", "opencv-python", "opencv-python-headless", "opencv-contrib-python", "PyYAML",
    ]
    return {n: _pkg(n) for n in names}


def nvidia_smi_query() -> dict:
    fields = "name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,persistence_mode,pstate"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - recorded, not hidden
        return {"error": repr(e)}
    gpus = []
    for line in out.splitlines():
        vals = [v.strip() for v in line.split(",")]
        gpus.append(dict(zip(fields.split(","), vals, strict=False)))
    return {"gpus": gpus}


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def capture_env(repo_dir: str | Path, seed: int) -> dict:
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": detect_platform(),
        "python": sys.version,
        "os": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": seed,
        "git": git_commit(repo_dir),
        "packages": package_versions(),
        "nvidia_smi": nvidia_smi_query(),
    }
    try:
        import torch

        env["torch_cuda"] = {
            "available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError as e:
        env["torch_cuda"] = {"error": repr(e)}
    return env
