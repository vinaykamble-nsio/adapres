"""Regenerate every Step 5 artefact from results/ (local-safe):
hypotheses revision 2 -> tables -> figures -> numbers. Refuses to run with uncommitted
tracked changes (so the derived record carries a clean commit) unless --allow-dirty."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "scripts"), str(REPO / "src")]


def main() -> int:
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and "--allow-dirty" not in sys.argv:
        print("refusing: tracked files have uncommitted changes (commit first, or pass --allow-dirty):\n" + dirty)
        return 1
    import make_figs
    import make_numbers
    import make_tables
    import recompute_hypotheses

    h = recompute_hypotheses.main()
    print("hypotheses rev", h["revision"], {k: h[k].get("supported", h[k].get("supported_pipelined")) for k in ("H1", "H2", "H3", "H4")})
    print("tables:", make_tables.build())
    print("figures:", list(make_figs.build()))
    print("numbers:", len(make_numbers.build()), "macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
