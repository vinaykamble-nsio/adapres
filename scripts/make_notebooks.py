"""Generate Kaggle notebooks from source strings (local-safe; writes .ipynb only).

Run: python scripts/make_notebooks.py
Notebooks are generated, not hand-edited, so their structure is reviewable in git
and checked by tests/test_notebooks.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
import yaml

REPO = Path(__file__).resolve().parents[1]
KAGGLE = yaml.safe_load((REPO / "configs" / "kaggle.yaml").read_text())

# --------------------------------------------------------------------------- shared cells

REV_PLACEHOLDER = "__NOTEBOOK_REV__"


def config_and_env_cell(nb_name: str, extra_config: str = "", rev: str = REV_PLACEHOLDER) -> str:
    return f'''# ===================== CELL 1: CONFIG + ENVIRONMENT CHECK =====================
# Edit nothing below for a normal run except SMOKE_TEST.
SMOKE_TEST = True   # True: tiny subset (run this first). False: full run.
NOTEBOOK_REV = "{rev}"   # hash of this notebook's generated source; checked against the cloned repo

CONFIG = {{
    "notebook": "{nb_name}",
    "repo_https": "{KAGGLE['repo_https']}",       # private repo, cloned with Secret GH_TOKEN
    "repo_ref": "{KAGGLE['repo_ref']}",
    "repo_dir": "/tmp/adapres",                     # /tmp: not saved into notebook output
    "repo_zip_marker": "adapres/GIT_COMMIT",        # fallback: repo zip attached as a dataset
    "config_path": "configs/default.yaml",          # relative to repo_dir
    "overrides": {{                                  # applied on top of default.yaml
        "paths": {{"input_root": "/kaggle/input", "work_root": "/kaggle/working"}},
    }},
    "cuda_visible_devices": "0",                    # T4 x2 -> pin to one GPU
    "required_gpu_substring": "T4",
{extra_config}}}

import os, sys, subprocess, json, platform, base64, glob, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]  # before any CUDA import
T0 = time.time()

def sh(cmd, check=True):
    """Run a shell command, print its output, raise with the real error on failure."""
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
    print(r.stdout[-6000:])
    if r.stderr.strip():
        print(r.stderr[-6000:], file=sys.stderr)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed with exit code {{r.returncode}}: {{cmd}}")
    return r

sh("nvidia-smi")
print("python", sys.version)
for mod in ("torch", "onnxruntime", "tensorrt", "ultralytics"):
    try:
        m = __import__(mod)
        print(f"{{mod}} {{getattr(m, '__version__', '?')}}")
    except Exception as e:
        print(f"{{mod}} NOT AVAILABLE YET ({{e!r}}); installed in cell 2 if pinned")
import torch
print("torch.cuda.is_available:", torch.cuda.is_available(), "| visible devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Settings -> Accelerator -> GPU T4 x2.")
gpu = torch.cuda.get_device_name(0)
print("GPU 0:", gpu)
if CONFIG["required_gpu_substring"] not in gpu:
    raise RuntimeError(f"GPU is {{gpu!r}}; this project requires a T4 (P100 has no fast INT8). "
                       "Settings -> Accelerator -> GPU T4 x2, then restart.")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected exactly 1 visible GPU after pinning, got {{torch.cuda.device_count()}}")

# ---- get the repo: private GitHub clone (Secret GH_TOKEN), else attached zip dataset
REPO = CONFIG["repo_dir"]
repo_source = None
if os.path.isdir(REPO):
    shutil.rmtree(REPO)
try:
    if "github.com/OWNER/" in CONFIG["repo_https"]:
        raise RuntimeError("repo_https is still the placeholder: set it in configs/kaggle.yaml and regenerate")
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GH_TOKEN")
    if not token:
        raise RuntimeError("Secret GH_TOKEN is empty")
    hdr = base64.b64encode(f"x-access-token:{{token}}".encode()).decode()
    # token passed as a one-off header: never written to .git/config or printed
    r = subprocess.run(["git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {{hdr}}",
                        "clone", "--quiet", CONFIG["repo_https"], REPO], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git clone failed: " + r.stderr.replace(token, "***").replace(hdr, "***"))
    sh(["git", "-C", REPO, "checkout", "--quiet", CONFIG["repo_ref"]])
    repo_source = "github"
    del token, hdr
except Exception as e:
    print(f"GitHub clone unavailable: {{e!r}}\\nTrying zip-dataset fallback ...")
    hits = sorted(glob.glob("/kaggle/input/**/" + CONFIG["repo_zip_marker"], recursive=True))
    if not hits:
        raise RuntimeError("Neither GitHub clone (Secret GH_TOKEN) nor an attached repo zip "
                           "(dataset containing adapres/GIT_COMMIT) is available. See README cell.") from e
    shutil.copytree(os.path.dirname(hits[0]), REPO)
    repo_source = "zip_dataset:" + os.path.dirname(hits[0])

sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import make_notebooks
expected_rev = make_notebooks.notebook_rev(CONFIG["notebook"] + ".ipynb")
if expected_rev != NOTEBOOK_REV:
    raise RuntimeError(f"STALE NOTEBOOK: this notebook is rev {{NOTEBOOK_REV}} but repo {{CONFIG['repo_ref']}} "
                       f"generates rev {{expected_rev}}. Re-import notebooks/{{CONFIG['notebook']}}.ipynb "
                       "from the repo (File -> Import Notebook), or push your local commits.")
print("notebook rev", NOTEBOOK_REV, "matches repo")
from adapres import config as C
from adapres.repro import git_commit, set_seed
cfg = C.load(os.path.join(REPO, CONFIG["config_path"]), CONFIG["overrides"])
SEED = cfg["seed"]
set_seed(SEED)
GIT = git_commit(REPO)
print("repo source:", repo_source)
print("git commit:", GIT)
print("seed:", SEED, "| SMOKE_TEST:", SMOKE_TEST)
if GIT["commit"] is None:
    raise RuntimeError("could not determine the git commit; refusing to produce unattributable results")
if GIT["dirty"]:
    raise RuntimeError("repo checkout is dirty; results would not match the recorded commit")
RESULTS = C.resolve_out(cfg, "results_dir", SMOKE_TEST)
LOGS = C.resolve_out(cfg, "logs_dir", SMOKE_TEST)
for d in (RESULTS, LOGS):
    os.makedirs(d, exist_ok=True)
print("results ->", RESULTS, "| logs ->", LOGS)
'''


PIP_CELL = r'''# ===================== CELL 2: INSTALLS (wheels only) + VERSION LOCK =====================
# Internet must be ON. Installs only what the Kaggle image lacks; torch is not touched.
# --only-binary=:all: : never compile from source (that is what failed on onnx/tensorrt).
import re, importlib.metadata as md
print("python", sys.version, "| torch", torch.__version__, "| torch CUDA", torch.version.cuda)
REQ = os.path.join(REPO, "requirements-kaggle.txt")
lines = [ln.split("#")[0].strip() for ln in open(REQ)]
opts = [ln for ln in lines if ln.startswith("-")]
reqs = [ln for ln in lines if ln and not ln.startswith("-")]
PIP = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
# Preflight: resolve each requirement on its own (no install) so ALL failures show in one run.
bad = {}
for req in reqs:
    p = subprocess.run(PIP + ["--dry-run", "--no-deps", "--quiet", *opts, req], capture_output=True, text=True)
    if p.returncode != 0:
        bad[req] = "\n".join(l for l in p.stderr.splitlines() if "ERROR" in l) or p.stderr[-2000:]
    print(f"preflight {req:45s} {'OK' if p.returncode == 0 else 'FAILED'}")
if bad:
    for req, err in bad.items():
        print(f"\n--- {req} ---\n{err}")
    raise RuntimeError(f"{len(bad)} requirement(s) have no wheel for this Python/CUDA: {list(bad)}. "
                       "Paste this whole cell output; the available versions are listed above.")
r = sh(PIP + ["--quiet", "-r", REQ], check=False)
with open(os.path.join(LOGS, f"{CONFIG['notebook']}_pip.log"), "w") as f:
    f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
if r.returncode != 0:
    raise RuntimeError("pip install failed (real error above).")
importlib.invalidate_caches()
# Version lock: record exactly what got installed so requirements-kaggle.txt can be pinned to it.
names = [re.split(r"[<>=!~\[ ]", q)[0] for q in reqs]
resolved = {n: md.version(n) for n in names}
os.makedirs(os.path.join(RESULTS, "env"), exist_ok=True)
lock = "\n".join(f"{n}=={v}" for n, v in resolved.items())
with open(os.path.join(RESULTS, "env", "pip_resolved.txt"), "w") as f:
    f.write(f"# python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda}\n{lock}\n")
print("RESOLVED VERSIONS (copy into requirements-kaggle.txt as exact pins):\n" + lock)
if resolved.get("tensorrt-cu12-bindings") != resolved.get("tensorrt-cu12-libs"):
    raise RuntimeError(f"TensorRT bindings/libs version mismatch: {resolved}")
import tensorrt, onnx, onnxruntime
print("import OK: tensorrt", tensorrt.__version__, "| onnx", onnx.__version__, "| onnxruntime", onnxruntime.__version__)
print("torch still", torch.__version__, "cuda", torch.version.cuda)
'''


def zip_cell(nb_name: str) -> str:
    return f'''# ===================== FINAL CELL: ZIP RESULTS =====================
import zipfile
suffix = "_smoke" if SMOKE_TEST else ""
zpath = f"/kaggle/working/results_{nb_name}{{suffix}}.zip"
base = os.path.dirname(RESULTS)  # contains results/ and logs/
n = 0
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for top in (RESULTS, LOGS):
        for root, _, files in os.walk(top):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, base))
                n += 1
print(f"wrote {{zpath}} ({{n}} files, {{os.path.getsize(zpath)/1e6:.2f}} MB)")
print(f"total notebook time: {{(time.time()-T0)/60:.1f}} min")
'''


# --------------------------------------------------------------------------- notebook 01

NB01_README = """# 01 — Environment check and BDD100K MOT subsets

**Purpose.** Verify the Kaggle GPU stack (T4, TensorRT, ONNX Runtime GPU, NVML energy counter) and
build the seeded, disjoint calib/test clip sets from BDD100K MOT 2020 **val**, with slice labels,
spliced scene-change streams, INT8-calibration frame list and validated COCO-format GT.

**Before running**
1. *Add Data* → search **`tcjpaperdatasets`** (owner `tcjpaperdatasets`) → Add. It must contain
   `track_images_val` and `box_track_labels_trainval`; cell 4 fails loudly if not.
2. *Settings* → **Accelerator: GPU T4 x2** (the notebook pins one GPU; P100 is rejected).
   **Internet: On**.
3. *Add-ons → Secrets* → add **`GH_TOKEN`** (a GitHub fine-grained token with read access to the
   private repo) and tick it for this notebook. Fallback: attach the repo zip dataset
   (see repo README, "Zip fallback").
4. First run with `SMOKE_TEST = True` (≈3–5 min). If every cell passes, set `SMOKE_TEST = False`
   and *Save Version → Save & Run All (Commit)* (≈10–20 min).

**Download afterwards** (Output tab): `results_01_env_and_data.zip` (or `..._smoke.zip`).
Unzip into the repo root so you get `results/env/` and `results/bdd100k_mot/bdd_subsets/` and
`logs/`. Commit `results/` (small CSV/JSON; no images, no boxes). The COCO GT JSONs stay in this
notebook's output under `data/bdd/` (not committed: they contain BDD labels); later notebooks
attach this notebook's output via *Add Data → Notebook Output*.

**Deviation D1 (PLAN §3.5):** calib and test are both drawn from MOT val as disjoint seeded sets,
because the mirror has no MOT train images.
"""

NB01_ENV_PROBES = '''# ===================== CELL 3: GPU STACK PROBES =====================
# TensorRT FP32/FP16 build of a tiny network, ONNX Runtime CUDA EP, NVML energy counter.
from adapres.repro import capture_env
from adapres.envcheck import run_all
env = capture_env(REPO, SEED)
env["repo_source"] = repo_source
env["probes"] = run_all(cfg["env"]["energy_probe_seconds"])
env_dir = os.path.join(RESULTS, "env"); os.makedirs(env_dir, exist_ok=True)
with open(os.path.join(env_dir, "env_kaggle.json"), "w") as f:
    json.dump(env, f, indent=2, default=str)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "traceback"} for k, v in env["probes"].items()},
                 indent=2, default=str))
print("packages:", json.dumps(env["packages"], indent=2))
failed = {k: v.get("error") for k, v in env["probes"].items() if not v.get("ok")}
for k, v in env["probes"].items():
    if not v.get("ok") and "traceback" in v:
        print(f"--- {k} traceback ---\\n{v['traceback']}")
if failed:
    raise RuntimeError(f"GPU stack probes failed (env_kaggle.json saved): {failed}")
nv = env["probes"]["nvml"]
print("NVML energy counter supported:", nv.get("energy_counter_supported"),
      "| counter mean W:", nv.get("energy_counter_mean_W"), "| sampled mean W:", nv.get("power_sample_mean_W"))
'''

NB01_TREE = '''# ===================== CELL 4: DATASET FILE-TREE CHECK =====================
from pathlib import Path
from adapres.data import bdd_mot
inp = Path(cfg["paths"]["input_root"])
print("attached inputs:", sorted(p.name for p in inp.iterdir()))
pref = inp / cfg["paths"]["bdd_dataset_slug"]
search_root = pref if pref.exists() else inp
print("searching under", search_root)
try:
    layout = bdd_mot.discover_layout(search_root, cfg["bdd"]["images_dir_name"],
                                     cfg["bdd"]["labels_dir_name"], split=cfg["bdd"]["label_split"])
except bdd_mot.LayoutError as e:
    print(str(e))
    raise RuntimeError("BDD100K MOT folders not found: attach the 'tcjpaperdatasets' dataset "
                       "(needs track_images_val and box_track_labels_trainval).") from e
print("label dir :", layout.label_dir, f"({layout.n_label_files} json)")
print("image root:", layout.image_root, f"({layout.n_image_dirs} video folders)")
print(bdd_mot.tree_preview(layout.label_dir.parent, max_depth=1))
sample = sorted(layout.label_dir.glob("*.json"))[0]
frames = bdd_mot.load_video_labels(sample)
print("sample", sample.name, "frames:", len(frames), "| first frame attrs:", frames[0].attributes,
      "| boxes in frame 0:", len(frames[0].boxes))
print("label categories in sample:", sorted({b.category for f in frames for b in f.boxes}))
if layout.n_label_files < 150:
    print(f"WARNING: only {layout.n_label_files} val label files (expected ~200 per PLAN §4.1 [VERIFY])")
'''

NB01_BUILD = '''# ===================== CELL 5: BUILD SUBSETS (checkpointed per sequence) =====================
import logging
from adapres.pipeline import build_bdd_subsets
logging.basicConfig(level=logging.INFO, force=True,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(os.path.join(LOGS, "01_build_subsets.log"))])
out = build_bdd_subsets.run(cfg, smoke=SMOKE_TEST, repo_dir=REPO, search_root=search_root, workers=8)
SUBSET_DIR = out["results_dir"]
print("results in", SUBSET_DIR)
'''

NB01_REPORT = '''# ===================== CELL 6: REPORT + SANITY CHECKS =====================
import pandas as pd
rep = out["report"]
print(json.dumps({k: rep[k] for k in ("n_videos_parsed", "n_eligible", "timeofday_source",
      "timeofday_counts_all_eligible", "selection_strata_counts", "shortfalls", "n_streams",
      "int8_calib_frames")}, indent=2))
print(pd.DataFrame(rep["frames"]).T)
print(pd.DataFrame(rep["coco_gt"]).T[["n_images", "n_annotations", "n_crowd_or_ignore", "gt_as_det_map50_95"]])
sel = json.load(open(os.path.join(SUBSET_DIR, "selection.json")))
cal, tst = set(sel["splits"]["calib"]), set(sel["splits"]["test"])
problems = []
if cal & tst: problems.append(f"calib/test overlap: {sorted(cal & tst)}")
want = cfg["bdd"]["smoke" if SMOKE_TEST else "subsets"]
for sp in ("calib", "test"):
    if len(sel["splits"][sp]) != want[sp]["n_sequences"]:
        problems.append(f"{sp}: {len(sel['splits'][sp])} sequences, wanted {want[sp]['n_sequences']}")
if any(rep["shortfalls"][sp] for sp in rep["shortfalls"]):
    print("NOTE: stratum shortfalls were filled from other strata:", rep["shortfalls"])
thr = json.load(open(os.path.join(SUBSET_DIR, "slice_thresholds.json")))
print("slice thresholds (fit on calib):", thr)
if problems:
    raise RuntimeError("sanity checks failed: " + "; ".join(problems))
print("selection sha256:", sel["selection_sha256"])
print("ALL CHECKS PASSED")
'''


def build_nb01(rev: str = REV_PLACEHOLDER) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                              "isGpuEnabled": True}}
    nb.cells = [
        nbf.v4.new_markdown_cell(NB01_README),
        nbf.v4.new_code_cell(config_and_env_cell("01_env_and_data", rev=rev)),
        nbf.v4.new_code_cell(PIP_CELL),
        nbf.v4.new_code_cell(NB01_ENV_PROBES),
        nbf.v4.new_code_cell(NB01_TREE),
        nbf.v4.new_code_cell(NB01_BUILD),
        nbf.v4.new_code_cell(NB01_REPORT),
        nbf.v4.new_code_cell(zip_cell("01_env_and_data")),
    ]
    return nb


BUILDERS = {"01_env_and_data.ipynb": build_nb01}


def notebook_rev(name: str) -> str:
    """Hash of the notebook's generated cell sources (with the rev placeholder), so a
    Kaggle copy can detect that it is older than the repo it clones."""
    import hashlib

    nb = BUILDERS[name](REV_PLACEHOLDER)
    blob = "\n---cell---\n".join(c.source for c in nb.cells)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def build(name: str) -> nbf.NotebookNode:
    return BUILDERS[name](notebook_rev(name))


NOTEBOOKS = {name: (lambda n=name: build(n)) for name in BUILDERS}


def main() -> int:
    out_dir = REPO / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name, fn in NOTEBOOKS.items():
        nb = fn()
        nbf.validate(nb)
        nbf.write(nb, out_dir / name)
        print("wrote", out_dir / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
