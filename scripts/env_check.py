"""Environment check -> results/env/env_<platform>.json. KAGGLE-ONLY for GPU probes;
`--no-gpu` runs only the software/git part (local-safe)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from adapres import config as C  # noqa: E402
from adapres.repro import capture_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    ap.add_argument("--out", required=True, help="output directory for env json")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()
    cfg = C.load(args.config)
    env = capture_env(REPO, cfg["seed"])
    if not args.no_gpu:
        from adapres.envcheck import run_all

        env["probes"] = run_all(cfg["env"]["energy_probe_seconds"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"env_{env['platform']}.json"
    p.write_text(json.dumps(env, indent=2, default=str))
    print(json.dumps(env, indent=2, default=str))
    print(f"wrote {p}")
    failed = [k for k, v in env.get("probes", {}).items() if not v.get("ok")]
    if failed:
        print(f"PROBES FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
