"""Lead's independent snapshot and calibration checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_snapshots_and_calibration():
    metadata = json.loads((ROOT / "data/metadata.json").read_text())
    for name, spec in metadata["artifacts"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == spec["sha256"]
    f = pd.read_csv(ROOT/"data/clean/if_main.csv", parse_dates=["date"])
    assert f.date.is_monotonic_increasing and not f.date.duplicated().any()
    assert (f.date.iloc[-1]-f.date.iloc[0]).days > 5*365
    assert len(f) > 1250 and f.close.gt(0).all()
    p = json.loads((ROOT/"data/clean/calibration.json").read_text())
    window = metadata["calibration"]
    f = f[f.date.between(window["window_start"], window["window_end"])]
    assert (f.date.max()-f.date.min()).days >= 5*365 and len(f) >= 1250
    assert len(f) == window["observations"]
    assert p["S0"] == f.close.iloc[-1] and p["K"] == p["S0"]
    assert np.isclose(p["B"], 1.2*p["S0"])
    returns = np.log(f.close).diff().dropna()
    assert np.isclose(p["sigma"]**2, returns.var(ddof=1)*252, rtol=1e-12)
    assert np.isclose(p["mu"], returns.mean()*252+.5*p["sigma"]**2, rtol=1e-12)
    rates = pd.read_csv(ROOT/"data/clean/shibor.csv")
    quote = rates.loc[rates.date == window["rate_date"], "1Y"].iloc[0]
    assert np.isclose(p["r"], np.log1p(quote/100), rtol=1e-12)
    tenors = {"O/N": 1/365, "1W": 7/365, "2W": 14/365, "1M": 1/12, "3M": .25, "6M": .5, "9M": .75, "1Y": 1.}
    row = rates.loc[rates.date <= window["window_end"]].iloc[-1]
    assert row.date == window["rate_date"]
    for tenor, fraction in tenors.items():
        assert np.isclose(window["zero_curve"][tenor], np.log1p(row[tenor]/100*fraction)/fraction)
    assert p["steps"] >= 252*p["T"]
