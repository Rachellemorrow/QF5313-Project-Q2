"""Plot the freshly acquired calibration snapshots without changing them."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/verification/data_diagnostics.html"))
    args = parser.parse_args()
    futures = pd.read_csv("data/clean/if_main.csv")
    index = pd.read_csv("data/clean/csi300_index.csv")
    rates = pd.read_csv("data/clean/shibor.csv")
    c = json.loads(Path("data/clean/calibration.json").read_text())
    fig = make_subplots(rows=2, cols=2, subplot_titles=("IF0 and index (raw levels)",
        "Successive-observation IF0 log returns", "Log-return distribution", "Latest SHIBOR quotes (%)"))
    for frame, name in ((futures, "IF0 continuous"), (index, "CSI300 index")):
        fig.add_trace(go.Scatter(x=frame.date, y=frame.close, name=name), row=1, col=1)
    returns = np.log(futures.close).diff()
    fig.add_trace(go.Scatter(x=futures.date, y=returns, name="Log return"), row=1, col=2)
    fig.add_trace(go.Histogram(x=returns.dropna(), nbinsx=100, name="Log-return histogram"), row=2, col=1)
    last = rates.loc[rates.date == c["rate_date"]].iloc[0]
    tenors = ["O/N","1W","2W","1M","3M","6M","9M","1Y"]
    fig.add_trace(go.Scatter(x=tenors, y=[last[t] for t in tenors], name=c["rate_date"], mode="lines+markers"), row=2, col=2)
    fig.update_layout(title="New snapshot diagnostics: unadjusted continuous-series roll risk remains", height=800)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True)
    print(args.output)


if __name__ == "__main__":
    main()
