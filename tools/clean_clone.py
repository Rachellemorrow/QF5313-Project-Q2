"""Create an in-repository clean clone and execute full frozen offline reproduction."""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    clone = ROOT/".cache"/f"reproduction-{revision[:8]}-{stamp}"
    subprocess.run(["git", "clone", "--no-hardlinks", str(ROOT), str(clone)], check=True, cwd=ROOT)
    env = dict(os.environ, UV_CACHE_DIR=str(ROOT/".cache/uv"),
               TMPDIR=str(clone/".cache/tmp"), TORCH_HOME=str(clone/".cache/torch"),
               XDG_CACHE_HOME=str(clone/".cache"), UV_PYTHON_INSTALL_DIR=str(ROOT/".cache/python"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    # uv network access is disabled; cached wheels are allowed, clone venv is fresh.
    subprocess.run(["uv", "sync", "--frozen", "--offline"], cwd=clone, env=env, check=True)
    run = subprocess.run(["uv", "run", "--offline", "python", "tools/reproduce.py", "--seed", str(args.seed)], cwd=clone, env=env)
    output = ROOT/"results/clean_clone.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"revision": revision, "clone": str(clone.relative_to(ROOT)),
        "fresh_venv": True, "uv_offline": True, "returncode": run.returncode}, indent=2)+"\n")
    raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
