"""Aggregate actual approach metrics; rank only comparable hedging outcomes."""
import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{i}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    files = sorted(args.results.glob("approach*/*/metrics.json"))
    if len(files) < 2:
        raise RuntimeError("Both executed approaches are required")
    rows = []
    for file in files:
        obj = json.loads(file.read_text())
        if not {"params", "metrics", "fit_seconds", "history"} <= obj.keys():
            raise ValueError(f"Invalid metrics schema: {file}")
        label = str(file.parent.relative_to(args.results))
        rows.append((label, "fit_seconds", obj["fit_seconds"]))
        rows.extend((label, key, val) for key, val in flatten(obj["metrics"]))
    table = Table(title="Executed approach comparison (different estimands are not ranked together)")
    for title in ("Approach/dataset", "Metric", "Value"):
        table.add_column(title)
    for label, key, value in rows:
        if any(token in key.lower() for token in ("price", "cvar", "z_score", "speedup", "slope")) or key == "fit_seconds":
            table.add_row(label, key, f"{value:.8g}")
    Console().print(table)
    fig = go.Figure(go.Table(header={"values": ["Approach/dataset", "Metric", "Value"]},
                            cells={"values": list(zip(*rows))}))
    fig.update_layout(title="Q2: complete numeric metrics from newly executed experiments", height=1000)
    fig.write_html(args.results/"comparison.html", include_plotlyjs=True)
    (args.results/"comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
