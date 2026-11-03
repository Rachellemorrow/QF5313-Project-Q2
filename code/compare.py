"""Aggregate newly executed Q2 results into standalone comparison tables."""
import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "comparison.html"
    files = sorted(args.root.glob("approach*/*/metrics.json"))
    if len(files) < 2:
        raise ValueError("Both approaches must have metrics before comparison")
    rows = []
    for path in files:
        payload = json.loads(path.read_text())
        for required in ("params", "metrics", "fit_seconds", "history"):
            if required not in payload:
                raise ValueError(f"{path}: missing {required}")
        rows.extend((str(path.relative_to(args.root)), key, val)
                    for key, val in flatten(payload["metrics"]))
    table = Table(title="Q2 executed numerical comparisons")
    for column in ("Artifact", "Metric", "Value"):
        table.add_column(column)
    for source, metric, val in rows:
        table.add_row(source, metric, f"{val:.8g}")
    Console().print(table)
    figure = go.Figure(go.Table(header=dict(values=["Artifact", "Metric", "Value"]),
        cells=dict(values=list(zip(*rows)))))
    figure.update_layout(title="Q2 numerical comparison (different units are not ranked together)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    # Rank like-for-like CVaR only; metric keys retain scenario/policy labels.
    cvar = sorted((r for r in rows if "cvar" in r[1].lower()), key=lambda r: r[2])
    if cvar:
        Console().print("CVaR entries (ascending; compare only matched cost/test scenarios)", cvar)
    (output.parent / "comparison.json").write_text(json.dumps({"source_files": [str(p) for p in files],
        "rows": rows, "cvar_entries_ascending": cvar}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
