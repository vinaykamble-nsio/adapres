"""Assemble springer-paper/ for Overleaf (local-safe): copy GENERATED artefacts (never hand-edit them
there) and the Springer Nature template class, then run the manuscript checker.

  paper/figs/*.pdf            -> springer-paper/figs/
  paper/tables/*.tex          -> springer-paper/tables/
  paper/numbers*.tex          -> springer-paper/
  springer-paper/sn-article-template/{sn-jnl.cls, bst/} -> springer-paper/
  springer-paper/research/refs.bib -> springer-paper/refs.bib (verified references)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SP = REPO / "springer-paper"


def main() -> int:
    (SP / "figs").mkdir(exist_ok=True)
    (SP / "tables").mkdir(exist_ok=True)
    for f in (REPO / "paper" / "figs").glob("*.pdf"):
        shutil.copy2(f, SP / "figs" / f.name)
    for f in (REPO / "paper" / "tables").glob("*.tex"):
        shutil.copy2(f, SP / "tables" / f.name)
    for f in (REPO / "paper").glob("numbers*.tex"):
        shutil.copy2(f, SP / f.name)
    tpl = SP / "sn-article-template"
    shutil.copy2(tpl / "sn-jnl.cls", SP / "sn-jnl.cls")
    if (tpl / "bst").is_dir():
        for f in (tpl / "bst").iterdir():
            shutil.copy2(f, SP / f.name)
    rb = SP / "research" / "refs.bib"
    if rb.exists():
        shutil.copy2(rb, SP / "refs.bib")
    print("synced generated artefacts into", SP)
    return subprocess.run([sys.executable, str(REPO / "scripts" / "check_manuscript.py")]).returncode


if __name__ == "__main__":
    sys.exit(main())
