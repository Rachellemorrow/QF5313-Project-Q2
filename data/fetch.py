"""Retrieve public observations or verify the committed offline snapshot.

Live: UV_CACHE_DIR=.cache/uv uv run python data/fetch.py --live
Offline: UV_CACHE_DIR=.cache/uv uv run python data/fetch.py --offline
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data/clean"
META = ROOT / "data/metadata.json"
TENORS = {"O/N": 1/365, "1W": 7/365, "2W": 14/365, "1M": 1/12,
          "3M": .25, "6M": .5, "9M": .75, "1Y": 1.0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(frame, name):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date)
    if frame.date.duplicated().any() or frame.date.isna().any():
        raise ValueError(f"{name}: duplicate or invalid date")
    frame = frame.sort_values("date").reset_index(drop=True)
    values = frame.drop(columns="date").apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name}: missing/nonfinite observations")
    if "close" in values and (values.close <= 0).any():
        raise ValueError("Nonpositive price")
    frame[values.columns] = values
    return frame


def calibrate(futures, index, rates):
    cutoff = min(futures.date.max(), index.date.max(), rates.date.max())
    f = futures[futures.date <= cutoff].copy()
    start = cutoff - pd.DateOffset(years=6)
    f = f[f.date >= start]
    if (f.date.max() - f.date.min()).days < 5*365 or len(f) < 1000:
        raise ValueError("Insufficient five-year futures history")
    ret = np.diff(np.log(f.close.to_numpy()))
    sigma = float(np.std(ret, ddof=1)*np.sqrt(252))
    mu = float(np.mean(ret)*252 + .5*sigma*sigma)
    curve = rates[rates.date <= cutoff].iloc[-1]
    times = np.array(list(TENORS.values()))
    quotes = np.array([curve[t] for t in TENORS])/100
    # Deposit-rate approximation: D(t)=1/(1+q(t)*t), interpolate zero yields.
    zeros = np.log1p(quotes*times)/times
    r = float(np.interp(1.0, times, zeros))
    s0 = float(f.close.iloc[-1])
    aligned = f[["date", "close"]].merge(index[["date", "close"]], on="date", suffixes=("_f", "_i"))
    aligned["basis"] = aligned.close_f-aligned.close_i
    aligned["basis_change"] = aligned.basis.diff()
    jumps = aligned.loc[aligned.basis_change.abs().nlargest(10).index]
    centered = ret-ret.mean()
    bandwidth = int(np.floor(4*(len(ret)/100)**(2/9)))
    long_run = np.dot(centered, centered)/len(ret)
    for lag in range(1, bandwidth+1):
        long_run += 2*(1-lag/(bandwidth+1))*np.dot(centered[lag:], centered[:-lag])/len(ret)
    largest = np.argsort(abs(ret))[-10:][::-1]
    quality = {
        "window_start": str(f.date.min().date()), "window_end": str(f.date.max().date()),
        "observations": len(f), "log_return_count": len(ret),
        "annual_log_drift": float(ret.mean()*252), "annual_log_drift_iid_se": float(sigma*np.sqrt(252/len(ret))),
        "annual_log_drift_hac_se": float(252*np.sqrt(long_run/len(ret))), "hac_bartlett_lags": bandwidth,
        "mu": mu, "sigma": sigma, "rate_date": str(curve.date.date()),
        "zero_curve": dict(zip(TENORS, zeros.tolist())),
        "missing_weekdays_including_exchange_holidays": len(pd.bdate_range(f.date.min(), f.date.max()).difference(f.date)),
        "missing_index_trading_dates": [str(x.date()) for x in index[(index.date >= f.date.min()) & (index.date <= f.date.max())].date if x not in set(f.date)],
        "largest_absolute_daily_log_return": float(abs(ret).max()),
        "largest_daily_log_returns": [{"date": str(f.date.iloc[int(j)+1].date()), "log_return": float(ret[j]), "roll_classification": "unavailable"} for j in largest],
        "largest_basis_changes": json.loads(jumps.to_json(orient="records", date_format="iso")),
        "roll_policy": "Sina IF0 main continuous as published, no back adjustment applied. Constituent/roll flags unavailable; basis-change diagnostics cannot identify every roll or separate genuine moves. Volatility includes possible roll jumps.",
        "rate_convention": "SHIBOR percent deposit quotes with fixed tenor fractions: O/N=1/365, weeks=days/365, months=months/12, 1Y=1. Continuously compounded zero yields log(1+q*t)/t linearly interpolated. Not literal calendar ACT/365 for months or a bootstrapped tradable zero curve.",
    }
    return dict(S0=s0, K=s0, B=1.2*s0, r=r, sigma=sigma, mu=mu, T=1., steps=252), quality


def live():
    CLEAN.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    url = "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/InnerFuturesNewService.getDailyKLine"
    f = ak.futures_main_sina(symbol="IF0", start_date="20190101", end_date="20261104")
    f.to_csv(raw / "if_main_akshare.csv", index=False)
    futures = validate(f.rename(columns={"日期": "date", "收盘价": "close"})[["date", "close"]], "IF0")
    errors = ["Initial CffexFuturesService.getCffexFuturesDailyKLine IF0 trial returned only 588 records for 2019-2024 and failed coverage validation; replaced with assignment-prescribed futures_main_sina, newly retrieved."]
    try:
        i = ak.index_zh_a_hist(symbol="000300", start_date="20190101", end_date="20261104")
        index = validate(i.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]], "CSI300")
        index_source = "ak.index_zh_a_hist(000300): Eastmoney"
    except Exception as exc:
        errors.append(f"index_zh_a_hist: {type(exc).__name__}: {exc}")
        i = ak.stock_zh_index_daily(symbol="sh000300")
        index = validate(i[["date", "close"]], "CSI300")
        index_source = "ak.stock_zh_index_daily(sh000300): Sina"
    s = ak.macro_china_shibor_all()
    rates = validate(s.rename(columns={"日期": "date", **{t+"-定价": t for t in TENORS}})[["date", *TENORS]], "SHIBOR")
    frames = {"if_main": futures, "csi300_index": index, "shibor": rates}
    sources = {"if_main": url+"?symbol=IF0", "csi300_index": index_source,
               "shibor": "ak.macro_china_shibor_all: https://cdn.jin10.com/data_center/reports/il_1.json"}
    artifacts = {}
    for name, frame in frames.items():
        frame = frame[(frame.date >= "2019-01-01") & (frame.date <= "2026-11-04")]
        frames[name] = frame
        path = CLEAN / f"{name}.csv"
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        artifacts[str(path.relative_to(ROOT))] = {"sha256": digest(path), "source": sources[name],
            "rows": len(frame), "start": str(frame.date.min().date()), "end": str(frame.date.max().date()),
            "schema": {c: ("ISO date" if c=="date" else "index points" if c=="close" else "percent per annum") for c in frame},
            "duplicates": 0, "missing_values": int(frame.isna().sum().sum())}
    params, quality = calibrate(frames["if_main"], frames["csi300_index"], frames["shibor"])
    path = CLEAN / "calibration.json"
    path.write_text(json.dumps(params, indent=2, allow_nan=False)+"\n")
    artifacts[str(path.relative_to(ROOT))] = {"sha256": digest(path), "source": "derived by data/fetch.py from verified snapshots", "schema": "spot/strike/barrier in index points; r,sigma,mu annual decimal; T years; steps integer"}
    META.write_text(json.dumps({"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts,
        "calibration": quality, "retrieval_failures": errors}, indent=2, allow_nan=False)+"\n")


def equivalent(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a, (int, float)):
        return isinstance(b, (int, float)) and bool(np.isclose(a,b,rtol=1e-10,atol=1e-10))
    return a == b


def offline(refresh_diagnostics=False):
    meta = json.loads(META.read_text())
    for file, spec in meta["artifacts"].items():
        if digest(ROOT/file) != spec["sha256"]:
            raise ValueError(f"Checksum mismatch: {file}")
    frames = [validate(pd.read_csv(CLEAN/f"{n}.csv"), n) for n in ("if_main", "csi300_index", "shibor")]
    params, quality = calibrate(*frames)
    if refresh_diagnostics:
        meta["calibration"] = quality
        META.write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
    if not equivalent(quality, meta["calibration"]):
        raise ValueError("Stale calibration quality metadata")
    for name, frame in zip(("if_main", "csi300_index", "shibor"), frames):
        spec = meta["artifacts"][f"data/clean/{name}.csv"]
        assert spec["rows"] == len(frame)
        assert spec["start"] == str(frame.date.min().date()) and spec["end"] == str(frame.date.max().date())
        assert spec["duplicates"] == 0 and spec["missing_values"] == 0
    saved = json.loads((CLEAN/"calibration.json").read_text())
    for key in params:
        if not np.isclose(params[key], saved[key], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Calibration mismatch: {key}")
    Console().print("[green]PASS[/green] offline SHA-256, metadata, data quality and deterministic recalibration", params)
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="Retrieve fresh public observations")
    group.add_argument("--offline", action="store_true", help="Verify existing snapshot (default)")
    parser.add_argument("--refresh-diagnostics", action="store_true", help="Recompute diagnostic metadata from hash-verified snapshots after methodology changes")
    args = parser.parse_args()
    if args.live:
        live()
    offline(args.refresh_diagnostics)
