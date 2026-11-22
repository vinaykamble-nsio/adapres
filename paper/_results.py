"""Shared, read-only access to results/ for the Step 5 scripts (local-safe).

Runs are selected ONLY through results/index.csv. Every selected run must be non-smoke and
all r04 runs must share one git commit; anything else raises.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


class ResultsError(RuntimeError):
    pass


def index(results: Path = RESULTS) -> dict[str, dict]:
    rows = list(csv.DictReader(open(results / "index.csv", newline="", encoding="utf-8")))
    out: dict[str, dict] = {}
    for r in rows:
        if r["exp_id"] in out:
            raise ResultsError(f"duplicate index entry {r['exp_id']}")
        out[r["exp_id"]] = r
    return out


def run_dir(exp_id: str, results: Path = RESULTS, idx: dict | None = None) -> Path:
    idx = idx or index(results)
    if exp_id not in idx:
        raise ResultsError(f"missing run in index.csv: {exp_id}")
    r = idx[exp_id]
    if r["smoke"] == "True":
        raise ResultsError(f"{exp_id} is a SMOKE run")
    d = results / r["path"]
    if not d.is_dir():
        raise ResultsError(f"{exp_id}: directory {d} missing")
    return d


def load(exp_id: str, name: str, results: Path = RESULTS, idx: dict | None = None):
    p = run_dir(exp_id, results, idx) / name
    return json.loads(p.read_text(encoding="utf-8"))


def r04_commit(results: Path = RESULTS, idx: dict | None = None) -> str:
    idx = idx or index(results)
    commits = {r["git_commit"] for k, r in idx.items() if k.startswith("r04_") and not k.endswith("_v2")}
    if len(commits) != 1:
        raise ResultsError(f"r04 runs come from several commits: {commits}")
    return commits.pop()


def prefix_commit(prefix: str, results: Path = RESULTS, idx: dict | None = None) -> str:
    """Single git commit shared by every run whose exp_id starts with `prefix` (raises otherwise)."""
    idx = idx or index(results)
    commits = {r["git_commit"] for k, r in idx.items() if k.startswith(prefix)}
    if len(commits) != 1:
        raise ResultsError(f"{prefix} runs come from {len(commits)} commits: {commits}")
    return commits.pop()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
