"""Generate fresh, offline calibration figures from verified committed snapshots."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]


def main():
    f = pd.read_csv(ROOT/"data/clean/if_main.csv", parse_dates=["date"])
    i = pd.read_csv(ROOT/"data/clean/csi300_index.csv", parse_dates=["date"])
    merged = f.merge(i, on="date", suffixes=("_futures", "_index"))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Newly retrieved price history", "IF0 minus CSI300 basis", "Daily IF0 log return"))
    for col in ("close_futures", "close_index"):
        fig.add_trace(go.Scatter(x=merged.date, y=merged[col], name=col), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged.date, y=merged.close_futures-merged.close_index, name="basis (not roll identification)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=f.date, y=np.log(f.close).diff(), name="daily log return"), row=3, col=1)
    fig.update_layout(height=950, title="Historical calibration diagnostics")
    out = ROOT/"results/data"
    out.mkdir(parents=True, exist_ok=True)
    fig.write_html(out/"calibration.html", include_plotlyjs=True)
    metadata = json.loads((ROOT/"data/metadata.json").read_text())
    curve = metadata["calibration"]["zero_curve"]
    fig = go.Figure(go.Scatter(x=list(curve), y=list(curve.values()), mode="lines+markers"))
    fig.update_layout(title="SHIBOR deposit-implied continuous zero yields", yaxis_title="Annual decimal rate")
    fig.write_html(out/"curve.html", include_plotlyjs=True)
    print("Wrote results/data/calibration.html and curve.html")


if __name__ == "__main__":
    main()
