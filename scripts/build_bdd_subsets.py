"""CLI for the BDD subset build. KAGGLE-ONLY (reads dataset images)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from adapres import config as C  # noqa: E402
from adapres.pipeline import build_bdd_subsets  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--search-root", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = C.load(args.config)
    out = build_bdd_subsets.run(cfg, smoke=args.smoke, repo_dir=REPO, search_root=args.search_root)
    print(json.dumps(out["report"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
