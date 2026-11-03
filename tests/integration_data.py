"""Independent snapshot and calibration consistency checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_checksums_and_coverage():
    metadata = json.loads((ROOT / "data/metadata.json").read_text())
    for name, item in metadata["artifacts"].items():
        path = ROOT / "data/clean" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame.date)
        assert dates.is_unique and dates.is_monotonic_increasing
        assert len(frame) == item["rows"]
        assert (dates.iloc[-1] - dates.iloc[0]).days >= 365.25 * 5


def test_calibration_independent_recalculation():
    c = json.loads((ROOT / "data/clean/calibration.json").read_text())
    f = pd.read_csv(ROOT / "data/clean/if_main.csv")
    returns = np.log(f.close / f.close.shift()).dropna()
    assert np.isclose(c["sigma"], returns.std(ddof=1) * np.sqrt(252), rtol=1e-12)
    assert np.isclose(c["mu"], 252 * (returns.mean() + returns.var(ddof=1)/2), rtol=1e-12)
    assert c["S0"] == f.close.iloc[-1]
    assert np.isclose(c["r"], np.log1p(c["rate_quote_1Y_percent"] / 100))
