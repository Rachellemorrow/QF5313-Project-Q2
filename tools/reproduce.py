"""Execute the entire documented offline workflow, including full neural training."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
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
        [sys.executable, "-m", "pytest", "tests", "-q"],
        [sys.executable, "tools/independent_reference.py"],
        [sys.executable, "tools/data_diagnostics.py"],
        [sys.executable, "code/approach1_pricing.py", "--seed", str(args.seed)],
        [sys.executable, "code/approach2_hedging.py", "--seed", str(args.seed)],
        [sys.executable, "tools/verify_results.py"],
        [sys.executable, "code/compare.py"],
    ]
    started_at = time.time()
    out = ROOT/"results/reproduction"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    out.mkdir(parents=True, exist_ok=True)
    record = {"seed": args.seed, "python": sys.version, "started_at": started_at,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "lockfile_sha256": hashlib.sha256((ROOT/"uv.lock").read_bytes()).hexdigest(), "stages": []}
    for i, cmd in enumerate(commands):
        started = time.perf_counter()
        with (out/f"step{i}.log").open("w") as handle:
            run = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        record["stages"].append({"command": cmd, "returncode": run.returncode, "seconds": time.perf_counter()-started})
        (out/"execution.json").write_text(json.dumps(record, indent=2)+"\n")
        print(record["stages"][-1], flush=True)
        if run.returncode:
            raise SystemExit(run.returncode)
    expected = [ROOT/"results/approach1/if_main/metrics.json", ROOT/"results/approach2/if_main/metrics.json",
                ROOT/"results/comparison.html", ROOT/"results/data/calibration.html", ROOT/"results/data/curve.html"]
    hashes = {}
    for path in expected:
        if not path.is_file() or path.stat().st_mtime < started_at:
            raise RuntimeError(f"Missing or stale output: {path}")
        if path.suffix == ".json":
            obj = json.loads(path.read_text())
            if not {"params", "metrics", "fit_seconds", "history"} <= obj.keys():
                raise RuntimeError(f"Missing metrics schema: {path}")
        if path.suffix == ".html" and "plotly.js" not in path.read_text():
            raise RuntimeError(f"Missing inline Plotly library: {path}")
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["output_hashes"] = hashes
    (out/"execution.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
