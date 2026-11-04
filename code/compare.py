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
    parser.add_argument("--pricing-dataset", default="if_main")
    parser.add_argument("--hedging-dataset", default="if_main")
    args = parser.parse_args()
    files = [args.results/"approach1"/args.pricing_dataset/"metrics.json",
             args.results/"approach2"/args.hedging_dataset/"metrics.json"]
    if not all(p.is_file() for p in files):
        raise RuntimeError("Both executed approaches are required")
    hedge = json.loads(files[1].read_text())
    if hedge["params"]["n_train"] < 2**17:
        raise ValueError("Refusing smoke-test hedging output as final comparison")
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
    rankings = []
    ranked = Table(title="Held-out CVaR ranking within matched frequency and cost (lower is better)")
    headings = ["Days", "Cost", "Rank", "Policy", "CVaR95", "E[P&L]", "Var[P&L]", "Mean cost", "Turnover"]
    for title in headings:
        ranked.add_column(title)
    for freq in (1,5,21):
        for cost in (0.,.001,.005):
            m = hedge["metrics"]
            candidates = [("BS",m["bs"][str(freq)][str(cost)]),
                          ("Neural",m["neural"][str(cost)][str(freq)][str(cost)]),
                          ("Zero",m["zero_hedge"])]
            for rank,(policy,values) in enumerate(sorted(candidates,key=lambda x:x[1]["cvar95"]),1):
                row = [freq,cost,rank,policy,*[values[k] for k in ("cvar95","expected_pnl","pnl_variance","avg_cost","turnover")]]
                rankings.append(row)
                ranked.add_row(*[f"{x:.7g}" if isinstance(x,float) else str(x) for x in row])
    Console().print(ranked)
    fig = go.Figure(go.Table(header={"values": headings}, cells={"values": list(zip(*rankings))}))
    fig.update_layout(title="Q2: matched held-out hedging comparison; rankings are not significance claims", height=1100)
    fig.write_html(args.results/"comparison.html", include_plotlyjs=True)
    (args.results/"comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    (args.results/"hedging_rankings.json").write_text(json.dumps(rankings,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
