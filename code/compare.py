#!/usr/bin/env python
"""Aggregate results/approach*/**/metrics.json into comparison tables + results/comparison.html.

Usage: uv run python code/compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_HTML = RESULTS / "comparison.html"

console = Console()


def find_metrics() -> list[tuple[str, dict]]:
    found = []
    for path in sorted(RESULTS.glob("approach*/**/metrics.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            console.print(f"[red]Skipping unreadable {path}: {exc}[/red]")
            continue
        label = str(path.parent.relative_to(RESULTS))
        found.append((label, data))
    return found


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def print_metrics_table(title: str, label: str, flat: dict) -> None:
    table = Table(title=f"{title} — {label}", show_lines=False)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")
    for k, v in sorted(flat.items()):
        if isinstance(v, float):
            table.add_row(k, f"{v:.6g}")
        else:
            table.add_row(k, str(v)[:80])
    console.print(table)


def cvar_rows(flat: dict) -> list[tuple[str, float, float, float]]:
    """Extract (method, mean_pnl, var_pnl, cvar95) rows if present (approach2 schema)."""
    rows = []
    # expect keys like metrics.neural_k0000.cvar95 or nested dicts already flattened
    prefixes = sorted({k.rsplit(".", 1)[0] for k in flat if k.endswith(".cvar95")})
    for p in prefixes:
        cvar = flat.get(f"{p}.cvar95")
        mean = flat.get(f"{p}.mean_pnl", flat.get(f"{p}.expected_pnl"))
        var = flat.get(f"{p}.var_pnl", flat.get(f"{p}.pnl_variance"))
        if cvar is not None:
            rows.append((p, float(mean) if mean is not None else float("nan"),
                         float(var) if var is not None else float("nan"), float(cvar)))
    return rows


def main() -> None:
    found = find_metrics()
    if not found:
        console.print("[yellow]No results/approach*/**/metrics.json found. Run the approach scripts first.[/yellow]")
        return

    html_sections = []
    for label, data in found:
        params = flatten(data.get("params", {}))
        metrics = flatten(data.get("metrics", {}))
        print_metrics_table("Params", label, params)
        print_metrics_table("Metrics", label, metrics)
        console.print(f"fit_seconds = {data.get('fit_seconds')}\n")
        rows = cvar_rows(metrics)
        if rows:
            table = Table(title=f"Ranked by CVaR(95%) ↑ — {label}")
            table.add_column("method")
            table.add_column("mean P&L", justify="right")
            table.add_column("P&L variance", justify="right")
            table.add_column("CVaR95", justify="right")
            for name, mean, var, cvar in sorted(rows, key=lambda r: r[3]):
                table.add_row(name, f"{mean:.4f}", f"{var:.2f}", f"{cvar:.4f}")
            console.print(table)
            fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[3] for r in rows],
                                   name="CVaR95"))
            fig.update_layout(title=f"CVaR(95%) by hedging method — {label}",
                              xaxis_title="method", yaxis_title="CVaR of -V_T")
            html_sections.append(fig.to_html(full_html=False, include_plotlyjs=True))

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    if html_sections:
        body = "<h1>Question 2 — Comparison of approach results</h1>\n" + "\n".join(html_sections)
    else:
        body = "<h1>Question 2 — Comparison of approach results</h1><p>No CVaR tables found; see approach metrics.json outputs.</p>"
    OUT_HTML.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Q2 comparison</title></head>"
        f"<body>{body}</body></html>"
    )
    console.print(f"[green]Wrote {OUT_HTML}[/green]")


if __name__ == "__main__":
    main()
