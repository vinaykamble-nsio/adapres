"""Build adapres_repo.zip for the Kaggle zip-dataset fallback (local-safe).

Uses `git archive HEAD` (committed files only) and adds a GIT_COMMIT file so the
notebook can still record the commit hash. Refuses if tracked files are modified,
because the zip would not match the recorded hash.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dist" / "adapres_repo.zip"))
    args = ap.parse_args()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"tracked files modified; commit first:\n{dirty}")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(REPO), "archive", "--format=zip", "--prefix=adapres/",
                    "-o", str(out), "HEAD"], check=True)
    with zipfile.ZipFile(out, "a") as z:
        z.writestr("adapres/GIT_COMMIT", commit + "\n")
    print(f"wrote {out} at commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
