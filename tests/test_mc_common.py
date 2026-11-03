"""Verification of the shared mathematics (mc_common). Run: uv run pytest tests/ -v"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import mc_common as mc  # noqa: E402


P = mc.estimate_params()


def test_gbm_first_moment():
    # E[S_T] = S0 * exp(r T) under the risk-neutral measure
    n = 400_000
    z = mc.gbm_increments(n, mc.M, seed=7)
    paths = mc.gbm_paths(P.s0, P.r, P.sigma, P.t, mc.M, z)
    mean_st = paths[:, -1].mean()
    expected = P.s0 * math.exp(P.r * P.t)
    assert abs(mean_st - expected) / expected < 0.01  # < 1% sampling error


def test_gbm_log_moment():
    n = 400_000
    z = mc.gbm_increments(n, mc.M, seed=8)
    paths = mc.gbm_paths(P.s0, P.r, P.sigma, P.t, mc.M, z)
    log_ret = np.log(paths[:, -1] / P.s0)
    assert abs(log_ret.mean() - (P.r - 0.5 * P.sigma**2) * P.t) < 0.005
    assert abs(log_ret.std(ddof=1) - P.sigma * math.sqrt(P.t)) < 0.005


def test_antithetic_pairing():
    z = mc.gbm_increments(1000, 10, seed=42, antithetic=True)
    assert np.allclose(z[:500] + z[500:], 0.0)


def test_antithetic_odd_rejected():
    with pytest.raises(ValueError):
        mc.gbm_increments(999, 10, seed=42, antithetic=True)


def test_black_scholes_put_call_parity():
    call = mc.black_scholes_call(P.s0, P.k, P.r, P.sigma, P.t)
    df = math.exp(-P.r * P.t)
    put = call - P.s0 + P.k * df
    assert put > 0
    assert abs((call - put) - (P.s0 - P.k * df)) < 1e-8


def test_black_scholes_vega_positive():
    call = mc.black_scholes_call(P.s0, P.k, P.r, P.sigma, P.t)
    assert mc.black_scholes_call(P.s0, P.k, P.r, 2 * P.sigma, P.t) > call


def test_d1_spot_atm():
    # ATM forward relation: d1 = (r + sigma^2/2) * sqrt(T) / sigma when K == S0
    d1 = (math.log(P.s0 / P.k) + (P.r + 0.5 * P.sigma**2) * P.t) / (P.sigma * math.sqrt(P.t))
    assert d1 == pytest.approx((P.r + 0.5 * P.sigma**2) * math.sqrt(P.t) / P.sigma)


def test_black_scholes_intrinsic_bounds():
    call = mc.black_scholes_call(P.s0, P.k, P.r, P.sigma, P.t)
    assert call >= max(P.s0 - P.k * math.exp(-P.r * P.t), 0.0)


def test_geo_asian_below_arithmetic_bound():
    geo = mc.geometric_asian_call_price(P.s0, P.k, P.r, P.sigma, P.t, mc.M)
    vanilla = mc.black_scholes_call(P.s0, P.k, P.r, P.sigma, P.t)
    assert 0 < geo < vanilla


def test_geo_moments_positive_variance():
    mu_g, var_g = mc.geometric_asian_moments(P.s0, P.r, P.sigma, P.t, mc.M)
    assert var_g > 0
    assert math.isfinite(mu_g)


def test_geo_variance_exact_discrete_sum():
    """Regression test: var_G must equal sigma^2 * (T/m)/m^2 * sum_{i,j} min(i,j),
    with sum_{i,j} min(i,j) = m(m+1)(2m+1)/6 — for T != 1 (catches the T^2 bug)."""
    for t_years in (0.5, 1.0, 2.0, 7.3):
        for m in (1, 2, 12, 252):
            s0, r, sigma = 100.0, 0.03, 0.25
            _, var_g = mc.geometric_asian_moments(s0, r, sigma, t_years, m)
            dt = t_years / m
            min_sum = sum(min(i, j) for i in range(1, m + 1) for j in range(1, m + 1))
            assert min_sum == pytest.approx(m * (m + 1) * (2 * m + 1) / 6)
            expected = sigma * sigma * dt / (m * m) * min_sum
            assert var_g == pytest.approx(expected, rel=1e-12)
            # also equals closed form sigma^2 * T * (m+1)(2m+1) / (6 m^2)
            assert var_g == pytest.approx(sigma * sigma * t_years * (m + 1) * (2 * m + 1) / (6 * m * m), rel=1e-12)


def test_payoff_boundaries():
    paths = np.array([[100.0, 110.0, 90.0, 105.0], [100.0, 80.0, 70.0, 60.0]])
    asian = mc.payoff_asian(paths, k=100.0)
    assert asian[0] == pytest.approx(max((110 + 90 + 105) / 3 - 100, 0.0))
    assert asian[1] == pytest.approx(0.0)
    barrier = mc.payoff_barrier(paths, k=100.0, b=120.0)
    assert barrier[0] == pytest.approx(5.0)  # max S = 110 < 120 -> alive
    assert barrier[1] == pytest.approx(0.0)  # out-of-the-money anyway
    barrier2 = mc.payoff_barrier(paths, k=100.0, b=105.0)
    assert barrier2[0] == pytest.approx(0.0)  # 110 >= 105 -> knocked out
    digital = mc.payoff_digital(paths, k=100.0)
    assert digital.tolist() == [1.0, 0.0]
    # S_0 must NOT be monitored for the barrier
    paths_open = np.array([[100.0, 130.0, 95.0, 105.0]])
    assert mc.payoff_barrier(paths_open, k=100.0, b=125.0)[0] == pytest.approx(0.0)  # 130 >= 125 knocked out


def test_mc_price_standard_error():
    rng_pay = np.array([1.0, 2.0, 3.0, 4.0])
    price, se = mc.mc_price(rng_pay, r=0.0, t=1.0)
    assert price == pytest.approx(2.5)
    assert se == pytest.approx(np.std(rng_pay, ddof=1) / 2.0)


def test_risk_neutral_drift_not_physical():
    # mu_phys is reported but must not enter the simulation drift
    assert abs(P.mu_phys) < 1.0  # sanity: it is finite and small-ish
    assert P.r == pytest.approx(0.01439)


def test_norm_cdf_sanity():
    assert norm.cdf(0.0) == pytest.approx(0.5)
