"""Create an in-repository clean clone and execute full frozen offline reproduction."""
import argparse
import hashlib
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    clone = ROOT/".cache"/f"reproduction-{revision[:8]}-{stamp}"
    evidence = ROOT/"results/clean_clone"/stamp
    evidence.mkdir(parents=True)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    record = {"revision": revision, "source_status": status, "clone": str(clone.relative_to(ROOT)),
              "fresh_venv": True, "dependency_access": "offline with shared pre-populated uv/Python caches; not a cold-machine offline installation",
              "lockfile_sha256": hashlib.sha256((ROOT/"uv.lock").read_bytes()).hexdigest(), "stages": []}
    def stage(cmd, cwd, env=None):
        with (evidence/f"stage{len(record['stages'])}.log").open("w") as handle:
            run = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        record["stages"].append({"command": cmd, "returncode": run.returncode})
        (evidence/"evidence.json").write_text(json.dumps(record, indent=2)+"\n")
        if run.returncode:
            raise SystemExit(run.returncode)
    if status:
        record["failure"] = "Source working tree is not clean; commit intended deliverables before final reproduction"
        (evidence/"evidence.json").write_text(json.dumps(record, indent=2)+"\n")
        raise SystemExit(record["failure"])
    stage(["git", "clone", "--no-hardlinks", str(ROOT), str(clone)], ROOT)
    env = dict(os.environ, UV_CACHE_DIR=str(ROOT/".cache/uv"),
               TMPDIR=str(clone/".cache/tmp"), TORCH_HOME=str(clone/".cache/torch"),
               XDG_CACHE_HOME=str(clone/".cache"), UV_PYTHON_INSTALL_DIR=str(ROOT/".cache/python"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    # uv network access is disabled; cached wheels are allowed, clone venv is fresh.
    stage(["uv", "sync", "--frozen", "--offline"], clone, env)
    stage(["uv", "run", "--offline", "python", "tools/reproduce.py", "--seed", str(args.seed)], clone, env)
    print(f"PASS full clean-clone reproduction: {evidence.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
