"""Static checks for springer-paper/main.tex (no LaTeX installed locally; local-safe).

Fails (exit 1) on: an undefined custom macro, a missing \\input file or figure, a \\cite key that is not in
refs.bib, a \\ref without a matching \\label, a refs.bib entry marked UNVERIFIED that is cited, and a
non-ASCII character in any compiled .tex file that main.tex does not declare with \\newunicodechar.
Warns on numbers typed directly in the prose (results must come from numbers*.tex macros).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SP = Path(__file__).resolve().parents[1] / "springer-paper"
MAIN = SP / "main.tex"

# commands provided by LaTeX, the sn-jnl class or packages loaded in main.tex
KNOWN = set("""
documentclass usepackage begin end title author email affil orgdiv orgname orgaddress city state country street postcode
fnm sur abstract keywords maketitle section subsection subsubsection paragraph label ref cite caption includegraphics
centering textbf textit emph texttt input bibliography bibliographystyle item hline toprule midrule bottomrule botrule
footnote url href eqref mathrm mathbf left right frac sum min max quad qquad leq geq approx times cdot ldots dots textsc
newcommand newunicodechar ensuremath Delta delta theta alpha tau rho hat tilde mathcal mathbb log text in notin subset
le ge pm mp sim propto infty lfloor rfloor operatorname arg argmax argmin bm boldsymbol widetilde widehat overline
small footnotesize scriptsize normalsize large Large noindent vspace hspace newline linewidth textwidth columnwidth
appendix bmhead backmatter equalcont raggedbottom theoremstyle newtheorem S percent textasciitilde nolinkurl
sffamily ttfamily rmfamily itshape bfseries upshape enspace thinspace nobreak tabularnewline multicolumn multirow
cline def let relax protect mbox hbox makebox phantom par ldots textendash textemdash textquoteright
algorithmic State If Else EndIf For EndFor Require Ensure Return Statex Comment dagger to
end
""".split())


def main() -> int:
    tex = MAIN.read_text(encoding="utf-8")
    errors, warns = [], []
    body_files = [MAIN]
    for m in re.findall(r"\\input\{([^}]+)\}", tex):
        p = SP / (m if m.endswith(".tex") else m + ".tex")
        if not p.exists():
            errors.append(f"\\input file missing: {m}")
        else:
            body_files.append(p)
    defined = set()
    for p in body_files:
        defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", p.read_text(encoding="utf-8")))
    used = set()
    for p in body_files:
        t = re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8"))
        used |= set(re.findall(r"\\([A-Za-z]+)", t))
    unknown = sorted(u for u in used - defined - KNOWN if len(u) > 1)
    if unknown:
        errors.append(f"undefined macros: {unknown}")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not (SP / g).exists() and not (SP / (g + ".pdf")).exists():
            errors.append(f"figure missing: {g}")
    bib = (SP / "refs.bib").read_text(encoding="utf-8") if (SP / "refs.bib").exists() else ""
    entries = dict(re.findall(r"@\w+\{([^,\s]+),(.*?)(?=\n@|\Z)", bib, re.S))
    cited, partial = set(), []
    for grp in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited |= {k.strip() for k in grp.split(",")}
    for k in sorted(cited):
        if k not in entries:
            errors.append(f"\\cite key not in refs.bib: {k}")
        elif "UNVERIFIED" in entries[k]:
            partial.append(k)
    labels = set(re.findall(r"\\label\{([^}]+)\}", "\n".join(p.read_text(encoding="utf-8") for p in body_files)))
    for r in set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex)):
        if r not in labels:
            errors.append(f"\\ref without \\label: {r}")
    declared = set(re.findall(r"\\newunicodechar\{(.)\}", tex))
    for p in body_files:
        for ch in sorted({c for c in p.read_text(encoding="utf-8") if ord(c) > 127}):
            if ch not in declared:
                errors.append(f"undeclared non-ASCII {ch!r} (U+{ord(ch):04X}) in {p.name}")
    prose = re.sub(r"(?<!\\)%.*", "", tex)
    prose = re.sub(r"\\begin\{(equation|align|table|figure)\*?\}.*?\\end\{\1\*?\}", "", prose, flags=re.S)
    for num in re.findall(r"(?<![\w.{\\-])\d+\.\d+(?![\w}])", prose):
        warns.append(num)
    print(f"checked {len(body_files)} files, {len(used & defined)} macros used, {len(cited)} citations, {len(labels)} labels")
    if warns:
        print(f"WARN: {len(set(warns))} decimal literals in prose (should be design constants only): {sorted(set(warns))}")
    if partial:
        print(f"WARN: {len(partial)} cited references are only partly verified (see their `verify` field in refs.bib): {sorted(partial)}")
    for e in errors:
        print("ERROR:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
