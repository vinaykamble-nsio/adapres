"""Every source/config/notebook file must be committable (not git-ignored).
Kaggle clones the repo, so an ignored module works locally but is missing there."""

import shutil
import subprocess

import pytest
from conftest import REPO

PATTERNS = ("src/**/*.py", "scripts/*.py", "configs/*.yaml", "notebooks/*.ipynb", "tests/*.py",
            "requirements-*.txt", "pyproject.toml")


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs git checkout")
def test_no_source_file_is_gitignored():
    files = sorted({str(p.relative_to(REPO)).replace("\\", "/") for pat in PATTERNS for p in REPO.glob(pat)
                    if "__pycache__" not in p.parts})
    assert files
    r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--no-index", "-v", *files],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "", f"git-ignored source files:\n{r.stdout}"
