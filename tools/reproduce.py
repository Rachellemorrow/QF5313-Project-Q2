"""Run the complete frozen, offline Q2 pipeline and preserve command evidence.

Run from repository root: python tools/reproduce.py --output results/rebuild
No smoke flag is used: both default complete experiments are mandatory.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/rebuild"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (root / args.output).resolve()
    if not out.is_relative_to(root):
        raise ValueError("Outputs must stay inside repository")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UV_CACHE_DIR=str(root / ".cache/uv"),
               XDG_CACHE_HOME=str(root / ".cache"), TMPDIR=str(root / ".cache/tmp"),
               PYTHONPYCACHEPREFIX=str(root / ".cache/pycache"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    prefix = ["uv", "run", "--frozen", "--offline"]
    commands = [
        ["uv", "sync", "--frozen", "--offline"],
        prefix + ["python", "data/fetch.py", "--offline"],
        prefix + ["pytest", "-q"],
        prefix + ["python", "code/approach1_pricing.py", "--seed", "42", "--output", str(out / "approach1/IF")],
        prefix + ["python", "code/approach2_hedging.py", "--seed", "42", "--output", str(out / "approach2/IF")],
        prefix + ["python", "code/compare.py", "--root", str(out)],
    ]
    records = []
    for i, command in enumerate(commands):
        print("RUN", " ".join(command), flush=True)
        start = time.perf_counter()
        log = out / f"command_{i+1}.log"
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
        records.append({"command": command, "returncode": result.returncode,
                        "seconds": time.perf_counter()-start, "log": str(log.relative_to(root)),
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
        (out / "execution.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            print(log.read_text(), file=sys.stderr)
            raise SystemExit(result.returncode)
    print("Complete pipeline passed; see", out / "execution.json")


if __name__ == "__main__":
    main()
