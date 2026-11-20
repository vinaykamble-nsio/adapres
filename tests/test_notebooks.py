"""Notebook-structure checks (local-safe: parses JSON, compiles cells, runs nothing)."""

import ast
import re
import sys

import nbformat
import pytest
from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
import make_notebooks  # noqa: E402

NB_FILES = sorted((REPO / "notebooks").glob("*.ipynb"))


def test_notebooks_exist():
    assert [p.name for p in NB_FILES] == sorted(make_notebooks.NOTEBOOKS)


@pytest.mark.parametrize("path", NB_FILES, ids=lambda p: p.name)
def test_notebook_up_to_date(path):
    """Committed .ipynb must equal the generator output (no hand edits)."""
    fresh = make_notebooks.NOTEBOOKS[path.name]()
    on_disk = nbformat.read(path, as_version=4)
    assert [c.source for c in on_disk.cells] == [c.source for c in fresh.cells], \
        "stale notebook: run python scripts/make_notebooks.py"


@pytest.mark.parametrize("path", NB_FILES, ids=lambda p: p.name)
def test_notebook_structure(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    cells = nb.cells
    assert cells[0].cell_type == "markdown"
    readme = cells[0].source
    for must in ("Purpose", "Add Data", "Accelerator", "GPU T4", "Download afterwards", "SMOKE_TEST", "min"):
        assert must in readme, f"README cell missing {must!r}"
    c1 = cells[1].source
    for must in ("SMOKE_TEST = True", "CONFIG = {", "nvidia-smi", "CUDA_VISIBLE_DEVICES", "git commit",
                 "seed", "torch", "onnxruntime", "tensorrt", "GH_TOKEN", "zip"):
        assert must in c1, f"cell 1 missing {must!r}"
    assert c1.index("CUDA_VISIBLE_DEVICES") < c1.index("import torch")
    assert "pip" in cells[2].source and "requirements-kaggle.txt" in cells[2].source
    assert "/kaggle/working/results_" in cells[-1].source and ".zip" in cells[-1].source
    for c in cells:
        if c.cell_type != "code":
            continue
        ast.parse(c.source)  # syntax check, nothing executed
        assert not re.search(r"\b[A-Za-z]:\\|\b[A-Za-z]:/|/Users/|/home/", c.source), "local path in notebook"
        for p in re.findall(r"[\"'](/[A-Za-z_][^\"']*)[\"']", c.source):
            assert p.startswith(("/kaggle/", "/tmp/")), f"non-Kaggle absolute path {p!r}"
        assert c.outputs == [] if hasattr(c, "outputs") else True


def test_token_never_printed():
    src = make_notebooks.config_and_env_cell("x")
    # the token is only used inside a one-off git -c header and redacted in errors
    assert "print(token" not in src and "print(hdr" not in src
    assert 'replace(token, "***")' in src
