#!/usr/bin/env python
"""Reproducible data retrieval + offline snapshot for Question 2 (Monte Carlo pricing & hedging on Chinese futures).

Sources (all public, no credentials, via AKShare):
  * CSI 300 stock-index futures main continuous contract (IF0, Sina finance via ``ak.futures_main_sina``)
  * CSI 300 index daily history (000300, ``ak.index_zh_a_hist``)
  * SHIBOR fixing history (``ak.macro_china_shibor_all``)

Usage
-----
    uv run python data/fetch.py --help
    uv run python data/fetch.py                 # verify committed snapshot (offline-safe)
    uv run python data/fetch.py --verify-only   # only check SHA-256 of data/clean artifacts
    uv run python data/fetch.py --refresh       # re-download from AKShare and rewrite snapshot+metadata

The committed snapshot in ``data/clean/`` is the authoritative offline fallback; a fresh clone
can run every downstream script without network access. ``--refresh`` re-fetches and rewrites
both ``data/clean/*.csv`` and ``data/metadata.json`` (source, retrieval date, schema, SHA-256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
META = ROOT / "data" / "metadata.json"

RETRIEVAL_DATE = date.today().isoformat()
START_DATE = "20170101"  # futures main-contract history begins 2017-01-17
END_DATE = date.today().strftime("%Y%m%d")

# ---------------------------------------------------------------------------
# Fetch + normalize
# ---------------------------------------------------------------------------

def fetch_if_main() -> pd.DataFrame:
    df = ak.futures_main_sina(symbol="IF0")
    df = df.rename(columns={
        "日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low",
        "收盘价": "close", "成交量": "volume", "持仓量": "open_interest",
    })
    df = df[["date", "open", "high", "low", "close", "volume", "open_interest"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    for c in ["open", "high", "low", "close", "volume", "open_interest"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_csi300_index() -> pd.DataFrame:
    df = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=START_DATE, end_date=END_DATE)
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change",
    })
    df = df[["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    for c in ["open", "high", "low", "close", "volume", "amount", "pct_change"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_shibor() -> pd.DataFrame:
    df = ak.macro_china_shibor_all()
    keep = {
        "日期": "date",
        "O/N-定价": "on", "1W-定价": "w1", "2W-定价": "w2", "1M-定价": "m1",
        "3M-定价": "m3", "6M-定价": "m6", "9M-定价": "m9", "1Y-定价": "y1",
    }
    df = df.rename(columns=keep)[list(keep.values())].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df[df["date"] >= "2017-01-01"]
    for c in keep.values():
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")  # percent, e.g. 1.44 == 1.44%
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Snapshot handling
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(frames: dict[str, pd.DataFrame]) -> dict:
    CLEAN.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, df in frames.items():
        out = CLEAN / name
        df.to_csv(out, index=False)
        artifacts[name] = {
            "rows": int(len(df)),
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
            "columns": list(df.columns),
            "sha256": sha256_of(out),
        }
    return artifacts


def load_metadata() -> dict:
    if META.exists():
        return json.loads(META.read_text())
    return {}


def verify_snapshot() -> bool:
    ok = True
    meta = load_metadata()
    artifacts = meta.get("artifacts", {})
    if not artifacts:
        print("FAIL: data/metadata.json missing or has no artifacts; run with --refresh")
        return False
    for name, info in artifacts.items():
        path = CLEAN / name
        if not path.exists():
            print(f"FAIL: missing {path}")
            ok = False
            continue
        digest = sha256_of(path)
        status = "OK" if digest == info.get("sha256") else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{status}  {name:24s} rows={info.get('rows')}  [{info.get('date_min')} .. {info.get('date_max')}]  sha256={digest[:12]}…")
    return ok


def build_metadata(artifacts: dict) -> dict:
    return {
        "project": "Advanced Programming 2026 — Question 2: Monte Carlo pricing and hedging on Chinese futures",
        "retrieval_date": RETRIEVAL_DATE,
        "seed_note": "Data retrieval is deterministic given the AKShare endpoints; no randomness involved.",
        "sources": {
            "if_main.csv": {
                "provider": "AKShare (ak.futures_main_sina) → Sina Finance",
                "series": "CSI 300 stock-index futures, IF main continuous contract (IF0)",
                "roll_convention": "AKShare 'main contract' continuous series; rolls on dominant open-interest/volume contract; artificial roll gaps possible in levels, mitigated by modelling log returns and disclosing the convention.",
                "units": "index points (CNY per point × 300 per contract)",
            },
            "csi300_index.csv": {
                "provider": "AKShare (ak.index_zh_a_hist) → Eastmoney",
                "series": "CSI 300 index (000300) daily OHLCV",
                "units": "index points",
            },
            "shibor.csv": {
                "provider": "AKShare (ak.macro_china_shibor_all) → China Foreign Exchange Trade System / Eastmoney mirror",
                "series": "Shanghai Interbank Offered Rate fixings, all tenors, daily",
                "units": "percent per annum (e.g. 1.44 == 1.44%); converted to decimals in code",
            },
        },
        "estimation_window": {
            "full_history": "2017-01-17 .. present (as retrieved)",
            "calibration_note": "Downstream scripts estimate μ, σ on trailing 5 calendar years of the retrieved history unless overridden; the window is stated in results/*/metrics.json.",
        },
        "missing_data_policy": "Rows with non-numeric or missing close are dropped; SHIBOR fixings are forward/back-filled only for curve interpolation, never for return estimation. Gaps (e.g. 2015-05 tabulated row) are reported by fetch.py --refresh.",
        "transformations": [
            "date normalized to ISO YYYY-MM-DD",
            "prices cast to float64; volumes to float64",
            "SHIBOR percent → decimal fraction only inside pricing code (r = y1 / 100)",
        ],
        "artifacts": artifacts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download from AKShare and rewrite snapshot + metadata.json")
    ap.add_argument("--verify-only", action="store_true", help="only verify SHA-256 of committed snapshot (offline-safe)")
    args = ap.parse_args()

    if args.verify_only:
        return 0 if verify_snapshot() else 1

    if args.refresh:
        print("Downloading IF0 main continuous, CSI 300 index, SHIBOR …")
        frames = {
            "if_main.csv": fetch_if_main(),
            "csi300_index.csv": fetch_csi300_index(),
            "shibor.csv": fetch_shibor(),
        }
        for name, df in frames.items():
            n_na = int(df.isna().sum().sum())
            print(f"  {name:20s} rows={len(df)}  range=[{df['date'].min()} .. {df['date'].max()}]  NaN cells={n_na}")
        artifacts = write_snapshot(frames)
        META.write_text(json.dumps(build_metadata(artifacts), indent=2, ensure_ascii=False))
        print(f"Wrote {META}")
        return 0

    # default: verify snapshot, then print coverage summary if possible
    ok = verify_snapshot()
    if ok:
        try:
            if_df = pd.read_csv(CLEAN / "if_main.csv")
            print(f"\nCoverage OK: {len(if_df)} trading days, {if_df['date'].min()} .. {if_df['date'].max()} (>= 5 years required).")
        except Exception as exc:  # pragma: no cover - informational only
            print(f"(coverage summary unavailable: {exc})")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
