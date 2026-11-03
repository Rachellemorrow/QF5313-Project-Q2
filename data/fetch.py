"""Fresh AKShare retrieval, checksum-verified offline replay and calibration.

Remote: uv run python data/fetch.py --fetch
Offline: uv run python data/fetch.py --offline
SHIBOR deposit quotes converted to continuous zero rates using ACT/360.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data/clean"
META = ROOT / "data/metadata.json"
TENORS = {"O/N": 1, "1W": 7, "2W": 14, "1M": 30, "3M": 90,
          "6M": 180, "9M": 270, "1Y": 360}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(frame, names):
    frame = frame.rename(columns=names)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame.sort_values("date").reset_index(drop=True)


def inspect(frame, value):
    dates = pd.to_datetime(frame.date)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted dates")
    values = pd.to_numeric(frame[value], errors="raise")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Invalid {value}")
    return {"rows": len(frame), "start": frame.date.iloc[0], "end": frame.date.iloc[-1],
            "missing_cells": int(frame.isna().sum().sum()),
            "weekday_dates_without_observation": int(len(pd.bdate_range(dates.min(), dates.max()).difference(dates))),
            "missing_date_note": "Weekday gaps include exchange holidays; no forward fill; not a trading-calendar audit."}


def acquire():
    import akshare as ak
    CLEAN.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data/raw"
    raw.mkdir(exist_ok=True)
    # Fixed historical cutoff for a reproducible estimator, freshly retrieved.
    end = "2026-11-03"
    futures = ak.futures_main_sina(symbol="IF0", start_date="20170101", end_date="20261103")
    futures.to_csv(raw / "if_fresh.csv", index=False)
    futures = normalize(futures, {"日期": "date", "收盘价": "close"})
    index = ak.stock_zh_index_daily(symbol="sh000300")
    index.to_csv(raw / "index_fresh.csv", index=False)
    index = normalize(index, {})
    shibor = ak.macro_china_shibor_all()
    shibor.to_csv(raw / "shibor_fresh.csv", index=False)
    shibor = normalize(shibor, {"日期": "date", **{f"{t}-定价": t for t in TENORS}})
    datasets = {
        "if_main.csv": (futures, ["date", "close"], "ak.futures_main_sina(IF0)", "https://finance.sina.com.cn/futures/quotes/IF0.shtml"),
        "csi300_index.csv": (index, ["date", "close"], "ak.stock_zh_index_daily(sh000300)", "https://finance.sina.com.cn/realstock/company/sh000300/nc.shtml"),
        "shibor.csv": (shibor, ["date", *TENORS], "ak.macro_china_shibor_all()", "https://cdn.jin10.com/data_center/reports/il_1.json"),
    }
    artifacts = {}
    for name, (frame, columns, source, url) in datasets.items():
        frame = frame.loc[(frame.date >= "2017-01-01") & (frame.date <= end), columns]
        stats = inspect(frame, "1Y" if name == "shibor.csv" else "close")
        frame.to_csv(CLEAN / name, index=False, float_format="%.12g")
        artifacts[name] = {"sha256": digest(CLEAN / name), "source": source, "url": url,
                           "schema": {c: str(frame[c].dtype) for c in columns}, **stats}
    META.write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "akshare_version": ak.__version__, "cutoff": end, "artifacts": artifacts,
        "futures_series": "Sina IF0 vendor main continuous series; not individual deliverable contracts. No roll adjustment applied. Vendor snapshot has no constituent contract identifier, so precise roll dates and roll jumps cannot be independently identified. Large changes are flagged, never silently removed.",
        "index_source_note": "Sina AKShare index endpoint used instead of Eastmoney index_zh_a_hist; same CSI300 underlying.",
        "rate_convention": "SHIBOR percentage deposit rates, ACT/360 simple quotes; zero=log1p(quote*tenor)/tenor; linear interpolation in continuous zero rates. 1Y=360/360; no extrapolation."}, indent=2) + "\n")


def verify_calibrate():
    meta = json.loads(META.read_text())
    frames = {}
    for name, info in meta["artifacts"].items():
        if digest(CLEAN / name) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
        frames[name] = pd.read_csv(CLEAN / name)
        inspect(frames[name], "1Y" if name == "shibor.csv" else "close")
    f = frames["if_main.csv"]
    if (pd.Timestamp(f.date.iloc[-1]) - pd.Timestamp(f.date.iloc[0])).days < 365.25 * 5:
        raise ValueError("Less than five years of futures data")
    ret = np.diff(np.log(f.close.to_numpy()))
    index = frames["csi300_index.csv"]
    index_dates = set(index.loc[(index.date >= f.date.iloc[0]) & (index.date <= f.date.iloc[-1]), "date"])
    missing_against_index = sorted(index_dates - set(f.date))
    sigma = float(np.std(ret, ddof=1) * np.sqrt(252))
    log_drift = float(np.mean(ret) * 252)
    rates = frames["shibor.csv"]
    rates = rates.loc[rates.date <= f.date.iloc[-1]].dropna(subset=list(TENORS))
    last = rates.iloc[-1]
    tenors = np.array(list(TENORS.values())) / 360
    zeros = np.log1p(last[list(TENORS)].to_numpy(dtype=float) / 100 * tenors) / tenors
    r = float(np.interp(1., tenors, zeros))
    flags = np.flatnonzero(np.abs(ret) > .05)
    calibration = {"S0": float(f.close.iloc[-1]), "K": float(f.close.iloc[-1]),
        "B": float(f.close.iloc[-1]) * 1.2, "r": r, "sigma": sigma,
        "mu": log_drift + sigma**2 / 2, "annual_mean_log_return": log_drift,
        "T": 1., "steps": 252, "observations": len(f), "returns": len(ret),
        "start": f.date.iloc[0], "end": f.date.iloc[-1], "rate_date": last.date,
        "rate_quote_1Y_percent": float(last["1Y"]),
        "index_trading_dates_without_futures": missing_against_index,
        "gap_treatment": "No fill. Successive-observation returns include gaps; 252 annualization is an approximation. Index calendar is a cross-check, not an official CFFEX calendar.",
        "abs_log_return_above_5pct": [{"date": f.date.iloc[i+1], "log_return": float(ret[i])} for i in flags],
        "model": "assignment drift-r GBM, not zero-drift Black-76 futures",
        "return_convention": "Successive observed daily log close ratios; annualization 252; sample variance ddof=1; no holiday imputation or roll correction."}
    (CLEAN / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    return calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--fetch", action="store_true")
    choice.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        acquire()
    Console().print(verify_calibrate())


if __name__ == "__main__":
    main()
