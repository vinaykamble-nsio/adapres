"""Results writer (PLAN §8). Refuses to write anything without complete metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

REQUIRED_META = ("exp_id", "dataset", "split", "seed", "git_commit", "env", "config_sha256", "timestamp_utc")
INDEX_FIELDS = ["exp_id", "dataset", "split", "policy", "mode", "path",
                "locked_config_sha256", "git_commit", "timestamp", "smoke"]


class MetaError(ValueError):
    pass


def check_meta(meta: dict) -> None:
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise MetaError(f"meta missing required keys: {missing}")
    if not isinstance(meta["env"], dict) or not meta["env"]:
        raise MetaError("meta['env'] must be a non-empty dict (hardware/software info)")
    if meta["git_commit"] in (None, ""):
        raise MetaError("meta['git_commit'] is empty; refusing to write unattributable results")
    if not isinstance(meta["seed"], int):
        raise MetaError("meta['seed'] must be an int")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    tmp.replace(path)  # atomic on POSIX; avoids half-written files on crash


class ExperimentWriter:
    """results/<dataset>/<exp_id>/ with meta.json written first."""

    def __init__(self, results_root: str | Path, meta: dict):
        check_meta(meta)
        self.root = Path(results_root)
        self.meta = dict(meta)
        self.dir = self.root / meta["dataset"] / meta["exp_id"]
        self.dir.mkdir(parents=True, exist_ok=True)
        _dump_json(self.meta, self.dir / "meta.json")

    def write_json(self, name: str, obj) -> Path:
        p = self.dir / name
        _dump_json(obj, p)
        return p

    def write_csv(self, name: str, rows: Iterable[dict], fieldnames: list[str]) -> Path:
        p = self.dir / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        tmp.replace(p)
        return p

    def register(self, policy: str = "", mode: str = "", locked_sha: str = "", replace: bool = False) -> None:
        """Append to results/index.csv. A duplicate exp_id raises unless replace=True
        (used only for idempotent data-prep re-runs), which drops the old row."""
        index = self.root / "index.csv"
        new = not index.exists()
        row = {
            "exp_id": self.meta["exp_id"], "dataset": self.meta["dataset"],
            "split": self.meta["split"], "policy": policy, "mode": mode,
            "path": str(self.dir.relative_to(self.root)).replace("\\", "/"),
            "locked_config_sha256": locked_sha, "git_commit": self.meta["git_commit"],
            "timestamp": self.meta["timestamp_utc"], "smoke": self.meta.get("smoke", False),
        }
        rows: list[dict] = []
        if not new:
            with open(index, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["exp_id"] == row["exp_id"] and r["dataset"] == row["dataset"]:
                        if not replace:
                            raise MetaError(f"duplicate index entry for {row['exp_id']}; delete the old run explicitly")
                        continue
                    rows.append(r)
        rows.append(row)
        tmp = index.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(index)
