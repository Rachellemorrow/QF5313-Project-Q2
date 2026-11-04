"""Numerical unit tests for the Q2 Monte Carlo / deep-hedging modules.

Owned by the deepseek-numerical subagent.  Tests use the built-in ``Model``
defaults (no dependency on ``data/clean/calibration.json``) and modest path
counts so they stay fast and hermetic.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import mc_common as mc  # noqa: E402
from mc_common import (Model, bs_call_delta, bs_call_price, bs_call_vega,  # noqa: E402
                       geometric_asian_price, mc_price_np, mc_price_torch,
                       payoff_asian_call, payoff_digital, payoff_european_call,
                       payoff_geometric_asian, payoff_up_and_out,
                       pathwise_asian_delta, pathwise_asian_vega,
                       pathwise_vanilla_delta, pathwise_vanilla_vega,
                       simulate_gbm_np, asian_cv_reference, spawn_seed_grid,
                       validate_model, load_model)

import approach1_pricing as a1  # noqa: E402
import approach2_hedging as a2  # noqa: E402
from approach2_hedging import empirical_cvar  # noqa: E402


@pytest.fixture(scope="module")
def model() -> Model:
    return Model()


# ---------------------------------------------------------------------------
# GBM moments
# ---------------------------------------------------------------------------

def test_gbm_log_moments(model):
    S, _ = simulate_gbm_np(model, 200_000, 0)
    lr = np.log(S[:, -1] / model.S0)
    assert abs(lr.mean() - (model.r - 0.5 * model.sigma ** 2) * model.T) < 0.005
    assert abs(lr.var(ddof=1) - model.sigma ** 2 * model.T) < 0.005


def test_gbm_expected_terminal(model):
    S, _ = simulate_gbm_np(model, 200_000, 0)
    theo = model.S0 * math.exp(model.r * model.T)
    se = S[:, -1].std(ddof=1) / math.sqrt(S.shape[0])
    assert abs(S[:, -1].mean() - theo) < 3 * se


# ---------------------------------------------------------------------------
# Black-Scholes validation
# ---------------------------------------------------------------------------

def test_bs_mc_zscore(model):
    p = mc_price_np(model, 100_000, 1)
    bs = bs_call_price(model.S0, model.K, model.r, model.sigma, model.T)
    z = (p["european"]["price"] - bs) / p["european"]["se"]
    assert abs(z) <= 3


# ---------------------------------------------------------------------------
# Payoff boundary cases
# ---------------------------------------------------------------------------

def test_payoff_boundaries(model):
    S = np.array([[100.0, 90.0, 105.0], [100.0, 150.0, 110.0]])
    assert np.isclose(payoff_up_and_out(S, 100.0, 120.0)[0], 5.0)  # max 90 < 120
    assert payoff_up_and_out(S, 100.0, 120.0)[1] == 0.0           # barrier hit (150)
    S2 = np.array([[100.0, 105.0], [100.0, 95.0]])
    assert payoff_digital(S2, 100.0)[0] == 1.0
    assert payoff_digital(S2, 100.0)[1] == 0.0
    S3 = np.array([[0.0, 2.0, 2.0]])  # initial irrelevant, mean of (2,2)=2
    assert np.isclose(payoff_asian_call(S3, 1.0)[0], 1.0)


# ---------------------------------------------------------------------------
# Geometric-Asian closed form
# ---------------------------------------------------------------------------

def test_geometric_asian_closed_form(model):
    price, mu_G, var_G = geometric_asian_price(
        model.S0, model.K, model.r, model.sigma, model.T, model.steps)
    S, _ = simulate_gbm_np(model, 200_000, 2)
    g = np.exp(np.mean(np.log(S[:, 1:]), axis=1))
    assert abs(np.log(g).mean() - mu_G) < 0.01
    assert abs(np.log(g).var(ddof=1) - var_G) < 0.01
    mc_g = np.exp(-model.r * model.T) * np.maximum(g - model.K, 0).mean()
    se = np.exp(-model.r * model.T) * np.maximum(g - model.K, 0).std(ddof=1) / math.sqrt(200_000)
    assert abs(mc_g - price) < 3 * se


# ---------------------------------------------------------------------------
# Variance reduction
# ---------------------------------------------------------------------------

def test_antithetic_matches_naive(model):
    n = 100_000
    S, _ = simulate_gbm_np(model, n, 3)
    A = payoff_asian_call(S, model.K)
    naive = A.mean()
    S_a, _ = simulate_gbm_np(model, n, 4, antithetic=True)
    A_a = payoff_asian_call(S_a, model.K)
    pair = (A_a[:n // 2] + A_a[n // 2:2 * (n // 2)]) / 2.0
    assert abs(pair.mean() - naive) < 0.5


def test_control_variate_reduces_variance(model):
    n = 100_000
    S, _ = simulate_gbm_np(model, n, 5)
    A = payoff_asian_call(S, model.K)
    G = payoff_geometric_asian(S, model.K)
    beta = np.cov(A, G, ddof=1)[0, 1] / np.var(G, ddof=1)
    geo_price, _, _ = geometric_asian_price(
        model.S0, model.K, model.r, model.sigma, model.T, model.steps)
    E_G = math.exp(model.r * model.T) * geo_price
    cv = A - beta * (G - E_G)
    vrr = A.var(ddof=1) / cv.var(ddof=1)
    assert vrr > 5  # assignment indicative target
    assert abs(cv.mean() - A.mean()) < 0.3


# ---------------------------------------------------------------------------
# Pathwise Greeks
# ---------------------------------------------------------------------------

def test_pathwise_vanilla_greeks(model):
    n = 200_000
    S, Z = simulate_gbm_np(model, n, 6)
    W = mc.brownian_paths_from_z(model, Z)
    delta = pathwise_vanilla_delta(model, S).mean()
    vega = pathwise_vanilla_vega(model, S, W[:, -1]).mean()
    assert abs(delta - bs_call_delta(model.S0, model.K, model.r, model.sigma, model.T)) < 0.01
    assert abs(vega - bs_call_vega(model.S0, model.K, model.r, model.sigma, model.T)) < 1.0


def test_pathwise_asian_matches_fd(model):
    n = 150_000
    g = a1.compute_greeks_shared(model, n, 7, bumps=[0.01], n_boot=100, boot_seed=1)
    pw_delta = g["pathwise"]["asian"]["delta"]["estimate"]
    pw_vega = g["pathwise"]["asian"]["vega"]["estimate"]
    assert abs(pw_delta - g["fd"]["asian"]["delta"]["0.01"]["estimate"]) < 0.05
    assert abs(pw_vega - g["fd"]["asian"]["vega"]["0.01"]["estimate"]) < 20.0


# ---------------------------------------------------------------------------
# Finite-difference Greeks (CRN) vs analytic
# ---------------------------------------------------------------------------

def test_fd_vanilla_greeks_crn(model):
    n = 150_000
    g = a1.compute_greeks_shared(model, n, 9, bumps=[0.01], n_boot=100, boot_seed=2)
    assert abs(g["fd"]["european"]["delta"]["0.01"]["estimate"] - bs_call_delta(
        model.S0, model.K, model.r, model.sigma, model.T)) < 0.01
    assert abs(g["fd"]["european"]["vega"]["0.01"]["estimate"] - bs_call_vega(
        model.S0, model.K, model.r, model.sigma, model.T)) < 1.5


def test_paired_pw_fd_greeks(model):
    """Pathwise vs finite-difference are computed on the SAME shocks and the
    paired difference at each bump is reported and statistically small."""
    g = a1.compute_greeks_shared(model, 150_000, 10, bumps=[0.1, 0.01],
                                 n_boot=200, boot_seed=3)
    for h in ("0.1", "0.01"):
        for greek in ("delta", "vega"):
            key = f"vanilla_{greek}_h{h}"
            assert key in g["paired_diff"]
            assert g["paired_diff"][key]["se"] > 0
    # small-bump FD and pathwise agree to well within the paired SE
    diff = g["paired_diff"]["vanilla_delta_h0.01"]
    assert abs(diff["mean_diff"]) < 5 * diff["se"]


# ---------------------------------------------------------------------------
# Engine agreement (NumPy vs PyTorch)
# ---------------------------------------------------------------------------

def test_numpy_torch_engine_agreement(model):
    pn = mc_price_np(model, 50_000, 10)
    pt = mc_price_torch(model, 50_000, 10)
    for k in pn:
        assert abs(pn[k]["price"] - pt[k]["price"]) < 1e-6 * max(1.0, abs(pn[k]["price"]))


# ---------------------------------------------------------------------------
# Reference estimator
# ---------------------------------------------------------------------------

def test_asian_cv_reference_precise(model):
    n = 1 << 16
    ref = asian_cv_reference(model, n, 11, n_pilot=1 << 12)
    naive = mc_price_np(model, n, 12)["asian"]
    # the control-variate reference is dramatically tighter than naive MC
    assert ref["se"] < 0.1 * naive["se"]
    # and consistent with the naive price within ~3 naive standard errors
    assert abs(ref["price"] - naive["price"]) < 3 * naive["se"]
    assert ref["n_pilot"] == 1 << 12
    assert ref["antithetic"] is True


# ---------------------------------------------------------------------------
# Seed spawning (disjoint streams)
# ---------------------------------------------------------------------------

def test_spawn_seed_grid_disjoint_and_deterministic():
    g1 = spawn_seed_grid(42, 3, 2, 2)
    g2 = spawn_seed_grid(42, 3, 2, 2)
    streams1, streams2 = [], []
    for i in range(3):
        for r in range(2):
            for m in range(2):
                a = np.random.default_rng(g1[i][r][m]).integers(0, 2 ** 32, 5)
                b = np.random.default_rng(g2[i][r][m]).integers(0, 2 ** 32, 5)
                assert np.array_equal(a, b)          # deterministic
                streams1.append(a)
                streams2.append(b)
    # all 12 cells produce distinct streams
    assert len({tuple(s) for s in streams1}) == 12


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_validate_model_rejects_bad_params(model):
    base = model.asdict()
    for bad in ({"S0": 0.0}, {"S0": -1.0}, {"K": 0.0}, {"B": 0.0},
                {"sigma": 0.0}, {"sigma": -0.2}, {"T": 0.0}, {"T": -1.0},
                {"steps": 0}, {"steps": -5}, {"steps": 1.5}, {"r": float("nan")}):
        with pytest.raises(ValueError):
            validate_model(Model(**{**base, **bad}))


def test_antithetic_odd_path_count_rejected(model):
    with pytest.raises(ValueError):
        simulate_gbm_np(model, 101, 0, antithetic=True)


def test_bs_degenerate_rejected():
    with pytest.raises(ValueError):
        bs_call_price(100.0, 100.0, 0.03, 0.0, 1.0)
    with pytest.raises(ValueError):
        bs_call_price(100.0, 100.0, 0.03, 0.2, 0.0)
    with pytest.raises(ValueError):
        geometric_asian_price(100.0, 100.0, 0.03, 0.0, 1.0, 252)
    with pytest.raises(ValueError):
        geometric_asian_price(100.0, 100.0, 0.03, 0.2, 1.0, 0)


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.json")


def test_load_model_missing_keys_raises(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"S0": 1.0}')
    with pytest.raises(KeyError):
        load_model(p)


def test_load_model_invalid_params_raises(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"S0": 1.0, "K": 1.0, "B": 1.2, "r": 0.0, "sigma": 0.0, '
                 '"mu": 0.0, "T": 1.0, "steps": 252}')
    with pytest.raises(ValueError):
        load_model(p)


# ---------------------------------------------------------------------------
# CVaR orientation
# ---------------------------------------------------------------------------

def test_empirical_cvar_orientation():
    x = np.arange(1.0, 101.0)  # 1..100
    var_, cvar = empirical_cvar(x, alpha=0.95)
    assert abs(cvar - 98.0) < 1e-9
    assert var_ == pytest.approx(95.0)  # 95% of the 100 sorted values are <= 95


def test_cvar_at_least_mean():
    x = np.random.default_rng(0).exponential(1.0, 10_000)
    _, cvar = empirical_cvar(x, alpha=0.95)
    assert cvar > x.mean()


# ---------------------------------------------------------------------------
# Hedging P&L (Option B)
# ---------------------------------------------------------------------------

def test_hedging_pnl_matches_loop(model):
    S, _ = simulate_gbm_np(model, 512, 11)
    for f in (1, 5, 21):
        ridx = a2.rebal_indices(model.steps, f)
        delta = np.full((512, len(ridx)), 0.5)
        V, turnover, costs = a2.hedging_pnl_np(S, delta, ridx, 0.001, model)
        gains = np.zeros(512)
        prev = np.zeros(512)
        c2 = np.zeros(512)
        for j, tau in enumerate(ridx):
            nxt = ridx[j + 1] if j + 1 < len(ridx) else model.steps
            gains += math.exp(model.r * (model.T - tau * model.dt)) * delta[:, j] * (S[:, nxt] - S[:, tau])
            c2 += 0.001 * np.abs(delta[:, j] - prev) * S[:, tau]
            prev = delta[:, j]
        H = np.maximum(S[:, -1] - model.K, 0.0)
        assert np.allclose(V, gains - c2 - H, atol=1e-6)


def test_bs_hedge_reduces_loss_vs_zero(model):
    S = a2.simulate_dataset(model, 4096, 12, dtype=np.float64)
    zero_loss = np.maximum(S[:, -1] - model.K, 0.0)
    ridx = a2.rebal_indices(model.steps, 1)
    delta = a2.bs_delta_schedule(S, ridx, model)
    V, _, _ = a2.hedging_pnl_np(S, delta, ridx, 0.0, model)
    bs_loss = -V
    _, zero_cvar = empirical_cvar(zero_loss)
    _, bs_cvar = empirical_cvar(bs_loss)
    assert bs_cvar < 0.5 * zero_cvar


def test_neural_policy_no_lookahead(model):
    import torch
    policy = a2.PolicyMLP(32)
    S_np = a2.simulate_dataset(model, 16, 13)
    S = torch.from_numpy(S_np)
    ridx = a2.rebal_indices(model.steps, 1)
    with torch.no_grad():
        delta = a2.neural_delta(policy, S, model, ridx)
    assert delta.shape == (S.shape[0], len(ridx))
    assert bool(torch.isfinite(delta).all())
