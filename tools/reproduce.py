"""Execute the entire documented offline workflow, including full neural training."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if sys.version_info[:2] != (3,13):
        raise RuntimeError("Python 3.13 required")
    env = dict(os.environ, UV_CACHE_DIR=str(ROOT/".cache/uv"),
               TMPDIR=str(ROOT/".cache/tmp"), TORCH_HOME=str(ROOT/".cache/torch"),
               XDG_CACHE_HOME=str(ROOT/".cache"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "data/fetch.py", "--offline"],
        [sys.executable, "-m", "pytest", "tests", "tests/integration_data.py", "-q"],
        [sys.executable, "tools/independent_reference.py"],
        [sys.executable, "tools/data_diagnostics.py"],
        [sys.executable, "code/approach1_pricing.py", "--seed", str(args.seed)],
        [sys.executable, "code/approach2_hedging.py", "--seed", str(args.seed)],
        [sys.executable, "code/compare.py"],
    ]
    out = ROOT/"results/reproduction"
    out.mkdir(parents=True, exist_ok=True)
    record = []
    for i, cmd in enumerate(commands):
        started = time.perf_counter()
        with (out/f"step{i}.log").open("w") as handle:
            run = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        record.append({"command": cmd, "returncode": run.returncode, "seconds": time.perf_counter()-started})
        (out/"execution.json").write_text(json.dumps(record, indent=2)+"\n")
        print(record[-1], flush=True)
        if run.returncode:
            raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
