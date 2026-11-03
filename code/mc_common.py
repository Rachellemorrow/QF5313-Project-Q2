#!/usr/bin/env python
"""Shared mathematical specification for Question 2 (Monte Carlo pricing & hedging on Chinese futures).

Owned by the lead agent. Subagents must not silently change the conventions defined here
without lead approval; report inconsistencies instead.

Conventions (authoritative)
---------------------------
* Underlying: CSI 300 index futures, main continuous contract (IF0), last retrieved close
  S0 = 3974.8 (2026-11-02). Log-return simulation under the risk-neutral measure:

      S_{t_{i+1}} = S_{t_i} * exp((r - 0.5*sigma^2) * dt + sigma * sqrt(dt) * Z_i),
      Z_i iid ~ N(0, 1),   dt = T / m,   m = 252 monitors per year,   T = 1.0 year.

  IMPORTANT (model distinction): the assignment explicitly requests this GBM with drift r.
  A futures *price* under standard risk-neutral assumptions is driftless (Black-76). We
  implement the assignment-prescribed GBM for pricing/validation and NEVER silently conflate
  it with a futures-price model; a Black-76 cross-check, if produced, is labelled separately.

* Historical (physical) estimates, from IF0 close log returns, trailing 5 calendar years
  (2021-10-28 .. 2026-11-02, n=1214): mu_hat = mean*252, sigma_hat = std(ddof=1)*sqrt(252).
  sigma_hat is used for BOTH physical-drift disclosure and risk-neutral simulation volatility.
  mu_hat is reported for completeness and is NOT used in risk-neutral pricing.

* Discounting: SHIBOR 1Y fixing 2026-11-03 = 1.4390% p.a. -> r = 0.01439 (flat for T = 1Y;
  no curve interpolation needed at this tenor; multi-tenor interpolation would use linear-in-r
  between fixings). No dividends/convenience yield modelled; the futures basis is not modelled
  (assignment fixes the GBM on the futures price level directly).

* Payoffs (all monitored on the SAME grid t_i = i*T/m, i = 1..m; the initial point S_0 is NOT
  monitored for the barrier or the average; stated explicitly wherever used):
    1. Arithmetic Asian call:  C_A = max(mean(S_{t_1..m}) - K, 0)
    2. Up-and-out call, discrete monitoring: C_B = max(S_T - K, 0) * 1{max(S_{t_1..m}) < B}
    3. Digital call:           C_D = 1{S_T > K}
    4. European vanilla call (validation benchmark): C_E = max(S_T - K, 0)
  Strike K = S0 (at-the-money spot). Barrier B = 1.20 * S0.

* Pricing: MC price = exp(-r*T) * mean(payoff); standard error = exp(-r*T)*std(ddof=1)/sqrt(N).
  Validation: European MC vs Black-Scholes closed form; report z = (MC - BS)/SE; |z| <= 3 PASS.

* Geometric Asian control variate: under Q, for the SAME discrete monitoring grid,
  G = (prod_i S_{t_i})^(1/m) is lognormal with
      mu_G   = ln(S0) + (r - sigma^2/2) * T * (m + 1) / (2 m)
      var_G  = sigma^2 * T * (m + 1) * (2 m + 1) / (6 * m^2)
  and E[G] = exp(mu_G + var_G / 2). The control-variate estimator for the arithmetic Asian
  call uses the closed-form geometric-Asian CALL price (not E[G]) as control:
      price_geo = exp(-rT) * (exp(mu_G + var_G/2) * N(d1) - K * N(d2)),
      d1 = (mu_G - ln K + var_G) / sqrt(var_G),  d2 = d1 - sqrt(var_G).
  In practice we use the per-path geometric payoff G_path = (prod S_i)^(1/m) as the control
  with the EXACT same expectation above, i.e. beta = Cov(A, G_path)/Var(G_path) and the
  closed-form E[geometric call payoff] replaces E[G_path] via its own closed form. Both forms
  are implemented; the call-on-geometric control is the primary estimator.

* Greeks: delta and vega by (i) pathwise derivatives and (ii) central finite differences with
  common random numbers, h in {0.1, 0.01, 0.001} * S0 for the spot bump; vega uses an absolute
  volatility bump eps in {0.1, 0.01, 0.001} (i.e. eps = h/S0 * 1.0, documented in outputs).
  Pathwise does NOT apply to digital/barrier payoffs (discontinuous); only FD is reported there.

* Seeds: CLI default 42; training/validation/test use independent seeds (42 / 123 / 2026).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"

T = 1.0
M = 252
B_MULT = 1.20
R_FIXING = 0.01439  # SHIBOR 1Y, 2026-11-03, as decimal
WINDOW_YEARS = 5


@dataclass
class MarketParams:
    s0: float
    k: float
    barrier: float
    r: float
    sigma: float
    mu_phys: float
    t: float
    m: int
    dt: float
    estimation_window: str
    n_returns: int
    source_last_date: str

    def to_dict(self):
        return asdict(self)


def load_if_main() -> pd.DataFrame:
    df = pd.read_csv(CLEAN / "if_main.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def estimate_params(window_years: int = WINDOW_YEARS) -> MarketParams:
    df = load_if_main()
    last_date = df["date"].max()
    start = last_date - pd.Timedelta(days=int(window_years * 365.25) + 5)
    sub = df[df["date"] >= start].copy()
    rets = np.log(sub["close"] / sub["close"].shift(1)).dropna()
    sigma = float(rets.std(ddof=1) * math.sqrt(252))
    mu_phys = float(rets.mean() * 252)
    s0 = float(df["close"].iloc[-1])
    return MarketParams(
        s0=s0,
        k=s0,
        barrier=B_MULT * s0,
        r=R_FIXING,
        sigma=sigma,
        mu_phys=mu_phys,
        t=T,
        m=M,
        dt=T / M,
        estimation_window=f"{sub['date'].min().date()} .. {last_date.date()} (trailing {window_years} calendar years, IF0 close log returns)",
        n_returns=int(len(rets)),
        source_last_date=str(last_date.date()),
    )


# ---------------------------------------------------------------------------
# GBM simulation (NumPy)
# ---------------------------------------------------------------------------

def gbm_increments(n_paths: int, m: int, seed: int, antithetic: bool = False) -> np.ndarray:
    """Return standard normal Z of shape (n_paths, m). If antithetic, n_paths must be even;
    the second half is the negation of the first half (paired rows)."""
    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("antithetic requires an even number of paths")
        half = rng.standard_normal((n_paths // 2, m))
        return np.concatenate([half, -half], axis=0)
    return rng.standard_normal((n_paths, m))


def gbm_paths(s0: float, r: float, sigma: float, t: float, m: int, z: np.ndarray) -> np.ndarray:
    """Simulate paths from increments Z. Returns array of shape (n_paths, m + 1), paths[:, 0] == s0."""
    dt = t / m
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt) * z
    log_increments = drift + diffusion
    log_paths = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return s0 * np.exp(log_paths)


def simulate_gbm(s0: float, r: float, sigma: float, t: float, m: int,
                 n_paths: int, seed: int, antithetic: bool = False) -> np.ndarray:
    z = gbm_increments(n_paths, m, seed, antithetic=antithetic)
    return gbm_paths(s0, r, sigma, t, m, z)


# ---------------------------------------------------------------------------
# Payoffs (monitored on t_1..t_m; S_0 NOT monitored)
# ---------------------------------------------------------------------------

def payoff_vanilla(paths: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(paths[:, -1] - k, 0.0)


def payoff_asian(paths: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(paths[:, 1:].mean(axis=1) - k, 0.0)


def payoff_barrier(paths: np.ndarray, k: float, b: float) -> np.ndarray:
    monitored = paths[:, 1:]
    alive = monitored.max(axis=1) < b
    return np.maximum(paths[:, -1] - k, 0.0) * alive


def payoff_digital(paths: np.ndarray, k: float) -> np.ndarray:
    return (paths[:, -1] > k).astype(np.float64)


def geometric_average(paths: np.ndarray) -> np.ndarray:
    return np.exp(np.log(paths[:, 1:]).mean(axis=1))


def payoff_geometric_asian_call(paths: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(geometric_average(paths) - k, 0.0)


# ---------------------------------------------------------------------------
# Black-Scholes closed form + discrete-grid geometric Asian closed form
# ---------------------------------------------------------------------------

def black_scholes_call(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s0 * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def geometric_asian_moments(s0: float, r: float, sigma: float, t: float, m: int):
    """Exact moments of the discrete geometric average over t_i = i*T/m, i=1..m, under Q."""
    mu_g = math.log(s0) + (r - 0.5 * sigma * sigma) * t * (m + 1) / (2 * m)
    var_g = sigma * sigma * t * (m + 1) * (2 * m + 1) / (6 * m * m)
    return mu_g, var_g


def geometric_asian_call_price(s0: float, k: float, r: float, sigma: float, t: float, m: int) -> float:
    mu_g, var_g = geometric_asian_moments(s0, r, sigma, t, m)
    sd = math.sqrt(var_g)
    d1 = (mu_g - math.log(k) + var_g) / sd
    d2 = d1 - sd
    eg = math.exp(mu_g + var_g / 2.0)
    return math.exp(-r * t) * (eg * norm.cdf(d1) - k * norm.cdf(d2))


def mc_price(payoff: np.ndarray, r: float, t: float):
    """Return (discounted mean price, standard error of that price)."""
    disc = math.exp(-r * t)
    n = payoff.shape[0]
    mean = float(payoff.mean())
    se = float(payoff.std(ddof=1) / math.sqrt(n)) * disc
    return disc * mean, se


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
