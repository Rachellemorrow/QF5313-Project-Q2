#!/usr/bin/env python
"""Option A — Monte Carlo pricing engine, variance reduction, Greeks, convergence, performance.

Self-contained, linear pipeline (load -> verify -> simulate/fit -> evaluate -> write outputs).

Mathematical specification (authoritative conventions live in ``mc_common``; see its docstring):
  * GBM under Q, assignment-prescribed drift ``r`` (NOT the driftless Black-76 futures model):
        S_{t_{i+1}} = S_{t_i} exp((r - 0.5*sigma^2) dt + sigma*sqrt(dt) Z_i),  Z_i ~ N(0,1) iid
    dt = T/m, T = 1.0, m = 252 monitors/year.  This is a spot-like GBM, used only for the
    assignment's Black-Scholes validation; it is never conflated with a futures-price model.
  * Historical mu_hat / sigma_hat from IF0 close log returns (trailing 5y).  sigma_hat feeds the
    risk-neutral volatility; mu_hat is disclosed but NOT used in pricing.

Seeds (all stochastic components seeded):
  * main pricing / FD / pathwise / CV:        seed 42          (--seed)
  * torch engine (independent paths):         seed 4242        (--torch-seed)
  * variance-reduction R replications:        seeds 1000..1000+R-1
  * convergence R replications:               seeds 2000..2000+R-1
  * high-precision Asian reference:           seed 42 (+ per-chunk offset)

Outputs (results/approach1/if/): metrics.json, convergence.html, var_reduction.html,
greeks.html, line_profile.txt.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import timeit
import tracemalloc
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.stats import linregress, norm

# Ensure code/ is on sys.path so ``import mc_common`` works regardless of how the script is run.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_common as mc  # noqa: E402

console = Console()

N_MAIN = 2 ** 17  # 131072
NS_CONVERGENCE = [2 ** k for k in range(8, 18)]  # 2^8 .. 2^17
PAYOFFS = ["vanilla", "asian", "barrier", "digital"]
DELTA_BUMPS = [0.1, 0.01, 0.001]  # as fraction of S0 -> h = bump * S0
VEGA_BUMPS = [0.1, 0.01, 0.001]   # absolute volatility bump (d sigma)
CVAR_NOTE = "vega bumps are ABSOLUTE volatility bumps (d sigma); delta bumps are h = fraction * S0"


# ---------------------------------------------------------------------------
# NumPy engine (thin wrappers over mc_common; kept local for clarity)
# ---------------------------------------------------------------------------

def price_numpy(payoff: np.ndarray, r: float, t: float):
    return mc.mc_price(payoff, r, t)


def compute_payoff(name: str, paths: np.ndarray, k: float, b: float) -> np.ndarray:
    if name == "vanilla":
        return mc.payoff_vanilla(paths, k)
    if name == "asian":
        return mc.payoff_asian(paths, k)
    if name == "barrier":
        return mc.payoff_barrier(paths, k, b)
    if name == "digital":
        return mc.payoff_digital(paths, k)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# PyTorch engine (independent RNG, device='cpu', float64)
# ---------------------------------------------------------------------------

def torch_simulate(s0: float, r: float, sigma: float, t: float, m: int,
                   n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn((n, m), generator=gen, dtype=torch.float64)
    dt = t / m
    drift = (r - 0.5 * sigma * sigma) * dt
    log_inc = drift + sigma * math.sqrt(dt) * z
    log_paths = torch.cat([torch.zeros(n, 1, dtype=torch.float64),
                           torch.cumsum(log_inc, dim=1)], dim=1)
    return s0 * torch.exp(log_paths)


def torch_payoff(name: str, paths: torch.Tensor, k: float, b: float) -> torch.Tensor:
    if name == "vanilla":
        return torch.clamp(paths[:, -1] - k, min=0.0)
    if name == "asian":
        return torch.clamp(paths[:, 1:].mean(dim=1) - k, min=0.0)
    if name == "barrier":
        alive = (paths[:, 1:].max(dim=1).values < b).to(torch.float64)
        return torch.clamp(paths[:, -1] - k, min=0.0) * alive
    if name == "digital":
        return (paths[:, -1] > k).to(torch.float64)
    raise ValueError(name)


def torch_price(payoff: torch.Tensor, r: float, t: float):
    d = math.exp(-r * t)
    n = payoff.numel()
    mean = float(payoff.mean().item())
    se = float(payoff.std(correction=1).item() / math.sqrt(n)) * d
    return d * mean, se


# ---------------------------------------------------------------------------
# Variance reduction (Asian payoff)
# ---------------------------------------------------------------------------

def asian_control_variate(paths: np.ndarray, s0: float, k: float, r: float,
                          sigma: float, t: float, m: int):
    """Geometric-Asian-call control variate (primary estimator, per mc_common).

    Control variate G_i = max(G_path_i - K, 0) with G_path_i the per-path geometric average.
    Its discounted expectation is ``mc.geometric_asian_call_price`` (discrete schedule).
    beta = Cov(A, G)/Var(G) estimated on the same run.
    Returns (beta, price, se, var_cv, var_naive, vr).
    """
    d = math.exp(-r * t)
    A = mc.payoff_asian(paths, k)                            # undiscounted arithmetic call payoff
    G = mc.payoff_geometric_asian_call(paths, k)             # undiscounted geometric call payoff
    mu_g_undisc = mc.geometric_asian_call_price(s0, k, r, sigma, t, m) / d
    n = A.shape[0]
    cov = np.cov(A, G, ddof=1)[0, 1]
    var_g = np.var(G, ddof=1)
    beta = float(cov / var_g)
    theta_cv = A.mean() - beta * (G.mean() - mu_g_undisc)
    price = float(d * theta_cv)
    var_a = float(np.var(A, ddof=1))
    var_cv_undisc = (var_a - beta * beta * var_g) / n
    var_cv = d * d * var_cv_undisc
    var_naive = d * d * var_a / n
    vr = var_a / max(var_a - beta * beta * var_g, 1e-300)
    return beta, price, math.sqrt(var_cv), var_cv, var_naive, float(vr)


def antithetic_price(paths: np.ndarray, k: float, r: float, t: float):
    """Antithetic estimate via paired (Z, -Z) rows; SE from pair-averaged payoffs.

    ``mc.gbm_increments(..., antithetic=True)`` returns ``[Z; -Z]``, i.e. the second half is
    the negation of the first half, so the paired rows are ``(j, j + N/2)``.
    """
    d = math.exp(-r * t)
    payoff = mc.payoff_asian(paths, k)  # undiscounted
    n = payoff.shape[0]
    half = n // 2
    pair_mean = (payoff[:half] + payoff[half:]) / 2.0
    est = float(d * pair_mean.mean())
    var_est = float(d * d * np.var(pair_mean, ddof=1) / half)
    return est, math.sqrt(var_est), var_est


# ---------------------------------------------------------------------------
# Pathwise Greeks
# ---------------------------------------------------------------------------

def brownian_increments(z: np.ndarray, t: float, m: int):
    """Return (W, times): W[i] = Brownian value at t_i, times = (t_1..t_m)."""
    dt = t / m
    W = math.sqrt(dt) * np.cumsum(z, axis=1)
    times = dt * np.arange(1, m + 1)
    return W, times


def pathwise_greeks(paths: np.ndarray, z: np.ndarray, p, name: str):
    """Pathwise delta & vega. Valid only for a.s. differentiable payoffs (vanilla, Asian)."""
    d = math.exp(-p.r * p.t)
    W, times = brownian_increments(z, p.t, p.m)
    n = paths.shape[0]
    if name == "vanilla":
        s_t = paths[:, -1]
        ind = (s_t > p.k).astype(np.float64)
        delta_terms = d * ind * s_t / p.s0
        vega_terms = d * ind * s_t * (W[:, -1] - p.sigma * p.t)
    elif name == "asian":
        s_mon = paths[:, 1:]                       # (n, m)
        avg = s_mon.mean(axis=1)
        ind = (avg > p.k).astype(np.float64)
        delta_deriv = s_mon.mean(axis=1) / p.s0
        vega_deriv = (s_mon * (W - p.sigma * times[None, :])).mean(axis=1)
        delta_terms = d * ind * delta_deriv
        vega_terms = d * ind * vega_deriv
    else:
        raise ValueError(f"pathwise not valid for {name}")
    delta = float(delta_terms.mean())
    vega = float(vega_terms.mean())
    delta_se = float(delta_terms.std(ddof=1) / math.sqrt(n))
    vega_se = float(vega_terms.std(ddof=1) / math.sqrt(n))
    return delta, delta_se, vega, vega_se


def fd_greek(name: str, z: np.ndarray, p, bump: float, which: str):
    """Central finite difference with common random numbers (same z for +/- bump).
    Returns (estimate, se) using the per-path difference so the SE reflects the CRN pairing."""
    d = math.exp(-p.r * p.t)
    if which == "delta":
        paths_up = mc.gbm_paths(p.s0 + bump, p.r, p.sigma, p.t, p.m, z)
        paths_dn = mc.gbm_paths(p.s0 - bump, p.r, p.sigma, p.t, p.m, z)
    else:  # vega
        paths_up = mc.gbm_paths(p.s0, p.r, p.sigma + bump, p.t, p.m, z)
        paths_dn = mc.gbm_paths(p.s0, p.r, p.sigma - bump, p.t, p.m, z)
    pu = compute_payoff(name, paths_up, p.k, p.barrier)
    pd = compute_payoff(name, paths_dn, p.k, p.barrier)
    g_i = d * (pu - pd) / (2.0 * bump)
    n = g_i.shape[0]
    est = float(g_i.mean())
    se = float(g_i.std(ddof=1) / math.sqrt(n))
    return est, se


def bs_greeks(p):
    d1 = (math.log(p.s0 / p.k) + (p.r + 0.5 * p.sigma ** 2) * p.t) / (p.sigma * math.sqrt(p.t))
    delta = norm.cdf(d1)
    vega = p.s0 * math.sqrt(p.t) * norm.pdf(d1)  # per unit sigma
    return float(delta), float(vega)


# ---------------------------------------------------------------------------
# High-precision Asian reference (chunked control variate)
# ---------------------------------------------------------------------------

def asian_reference(p, n_ref: int = 2 ** 21, chunk: int = 2 ** 18, seed: int = 42):
    d = math.exp(-p.r * p.t)
    mu_g = mc.geometric_asian_call_price(p.s0, p.k, p.r, p.sigma, p.t, p.m) / d
    # pilot beta from a single chunk
    z0 = mc.gbm_increments(chunk, p.m, seed)
    p0 = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, z0)
    A0 = mc.payoff_asian(p0, p.k)
    G0 = mc.payoff_geometric_asian_call(p0, p.k)
    beta = float(np.cov(A0, G0, ddof=1)[0, 1] / np.var(G0, ddof=1))
    tot = 0.0
    ssq = 0.0
    n_done = 0
    for c in range(n_ref // chunk):
        z = mc.gbm_increments(chunk, p.m, seed + 10000 + c)
        paths = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, z)
        A = mc.payoff_asian(paths, p.k)
        G = mc.payoff_geometric_asian_call(paths, p.k)
        cv_i = A - beta * (G - mu_g)
        tot += float(cv_i.sum())
        ssq += float((cv_i ** 2).sum())
        n_done += chunk
    est = tot / n_done
    var_cv = (ssq - tot * tot / n_done) / (n_done - 1)
    se = d * math.sqrt(var_cv / n_done)
    return float(d * est), float(se)


# ---------------------------------------------------------------------------
# Convergence experiment
# ---------------------------------------------------------------------------

def convergence_experiment(p, ref_asian: float, rng_seed_base: int = 2000, reps: int = 20):
    d = math.exp(-p.r * p.t)
    ref_vanilla = mc.black_scholes_call(p.s0, p.k, p.r, p.sigma, p.t)
    out = {
        "N": [], "asian_naive": [], "asian_antithetic": [], "asian_cv": [], "vanilla_naive": [],
        "_raw": {str(N): {"asian_naive": [], "asian_antithetic": [], "asian_cv": [], "vanilla_naive": []}
                 for N in NS_CONVERGENCE},
    }
    for N in NS_CONVERGENCE:
        a_naive, a_anti, a_cv, v_naive = [], [], [], []
        for r in range(reps):
            seed = rng_seed_base + r
            # naive + control variate share one set of paths
            z = mc.gbm_increments(N, p.m, seed)
            paths = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, z)
            A = mc.payoff_asian(paths, p.k)
            G = mc.payoff_geometric_asian_call(paths, p.k)
            price_naive = d * A.mean()
            mu_g = mc.geometric_asian_call_price(p.s0, p.k, p.r, p.sigma, p.t, p.m) / d
            beta = np.cov(A, G, ddof=1)[0, 1] / np.var(G, ddof=1)
            price_cv = d * (A.mean() - beta * (G.mean() - mu_g))
            # vanilla from the same paths
            vpay = mc.payoff_vanilla(paths, p.k)
            price_vanilla = d * vpay.mean()
            # antithetic (paired)
            za = mc.gbm_increments(N, p.m, seed, antithetic=True)
            pathsa = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, za)
            price_anti, _, _ = antithetic_price(pathsa, p.k, p.r, p.t)

            a_naive.append(float(price_naive)); a_cv.append(float(price_cv))
            v_naive.append(float(price_vanilla)); a_anti.append(float(price_anti))

        out["N"].append(N)
        out["asian_naive"].append(float(np.sqrt(np.mean((np.array(a_naive) - ref_asian) ** 2))))
        out["asian_cv"].append(float(np.sqrt(np.mean((np.array(a_cv) - ref_asian) ** 2))))
        out["asian_antithetic"].append(float(np.sqrt(np.mean((np.array(a_anti) - ref_asian) ** 2))))
        out["vanilla_naive"].append(float(np.sqrt(np.mean((np.array(v_naive) - ref_vanilla) ** 2))))
        out["_raw"][str(N)] = {"asian_naive": a_naive, "asian_cv": a_cv,
                               "asian_antithetic": a_anti, "vanilla_naive": v_naive}
    return out


# ---------------------------------------------------------------------------
# Performance: naive Python loop vs vectorized NumPy (vectorized fn is the hot loop)
# ---------------------------------------------------------------------------

def asian_vectorized(z, s0, r, sigma, t, m, k):
    """Vectorized Asian pricer (hot loop for line_profiler). Equivalent to mc.gbm_paths+payoff_asian."""
    dt = t / m
    drift = (r - 0.5 * sigma * sigma) * dt
    log_inc = drift + sigma * math.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(log_inc, axis=1)], axis=1)
    paths = s0 * np.exp(log_paths)
    avg = paths[:, 1:].mean(axis=1)
    payoff = np.maximum(avg - k, 0.0)
    return payoff.mean()


def asian_naive_loop(z, s0, r, sigma, t, m, k):
    """Pure-Python-loop Asian pricer (numpy used only for the pre-drawn RNG array z)."""
    dt = t / m
    drift = (r - 0.5 * sigma * sigma) * dt
    diff = sigma * math.sqrt(dt)
    n = z.shape[0]
    total = 0.0
    for i in range(n):
        s = s0
        ssum = 0.0
        zi = z[i]
        for j in range(m):
            s = s * math.exp(drift + diff * zi[j])
            ssum += s
        avg = ssum / m
        if avg > k:
            total += avg - k
    return total / n


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        return float(x)
    if isinstance(x, int):
        return int(x)
    return x


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Option A: Monte Carlo pricing, variance reduction, Greeks, convergence, performance.")
    ap.add_argument("--seed", type=int, default=42, help="primary numpy seed")
    ap.add_argument("--torch-seed", type=int, default=4242, help="independent torch engine seed")
    ap.add_argument("--n-paths", type=int, default=N_MAIN, help="main path count (2^17 default)")
    ap.add_argument("--reps", type=int, default=20, help="replication count for VR + convergence")
    ap.add_argument("--outdir", type=str, default="results/approach1/if", help="output directory")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # ---- 0. load params -----------------------------------------------------
    p = mc.estimate_params()
    console.print(Panel.fit(
        f"[bold cyan]Market parameters[/bold cyan]\n"
        f"S0 = {p.s0:,.2f}\nK = {p.k:,.2f}  (at-the-money spot)\nB = {p.barrier:,.2f}  (1.20 x S0)\n"
        f"r = {p.r:.5f}  (SHIBOR 1Y)\nsigma = {p.sigma:.6f}  (5y realized, annualized)\n"
        f"mu_phys = {p.mu_phys:.6f}  (disclosed only, NOT used in Q pricing)\n"
        f"T = {p.t}, m = {p.m} steps/yr, dt = {p.dt:.6f}\n"
        f"window = {p.estimation_window}\n"
        f"source_last_date = {p.source_last_date}   n_returns = {p.n_returns}"
    ))
    console.print("[yellow]Model note:[/yellow] assignment-prescribed GBM with drift r (spot-like); NOT the "
                  "driftless Black-76 futures model. Never conflated; a Black-76 check would be labelled separately.\n")

    d = math.exp(-p.r * p.t)
    n_paths = args.n_paths

    # ---- 1. validation (European vanilla, NumPy vs torch vs BS) --------------
    z_val = mc.gbm_increments(n_paths, p.m, args.seed)
    paths_val = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, z_val)
    vpay = mc.payoff_vanilla(paths_val, p.k)
    mc_np, se_np = mc.mc_price(vpay, p.r, p.t)

    tpaths = torch_simulate(p.s0, p.r, p.sigma, p.t, p.m, n_paths, args.torch_seed)
    tpay = torch_payoff("vanilla", tpaths, p.k, p.barrier)
    mc_torch, se_torch = torch_price(tpay, p.r, p.t)

    bs = mc.black_scholes_call(p.s0, p.k, p.r, p.sigma, p.t)
    z_score = (mc_np - bs) / se_np
    diff_np_torch = abs(mc_np - mc_torch)
    cross_ok = diff_np_torch <= 3.0 * math.sqrt(se_np ** 2 + se_torch ** 2)

    console.print(Panel(
        f"[bold]Validation — European vanilla (N={n_paths})[/bold]\n"
        f"BS closed form        = {bs:.6f}\n"
        f"NumPy MC              = {mc_np:.6f}  (SE {se_np:.4f})\n"
        f"PyTorch MC (seed {args.torch_seed}) = {mc_torch:.6f}  (SE {se_torch:.4f})\n"
        f"z = (MC_np - BS)/SE   = {z_score:+.3f}   [PASS |z|<=3: {abs(z_score) <= 3}]\n"
        f"|np - torch|          = {diff_np_torch:.4f}   [PASS <=3*sqrt(se^2+se^2): {cross_ok}]"
    ))

    # ---- 2. all four payoffs, both engines -----------------------------------
    table = Table(title=f"All four payoffs — NumPy vs PyTorch (N={n_paths})", header_style="bold")
    table.add_column("Payoff", justify="left")
    table.add_column("NumPy price", justify="right")
    table.add_column("NumPy SE", justify="right")
    table.add_column("PyTorch price", justify="right")
    table.add_column("PyTorch SE", justify="right")
    payoff_results = {}
    for name in PAYOFFS:
        pnp = compute_payoff(name, paths_val, p.k, p.barrier)
        pr_np, se_np2 = mc.mc_price(pnp, p.r, p.t)
        tpay = torch_payoff(name, tpaths, p.k, p.barrier)
        pr_t, se_t2 = torch_price(tpay, p.r, p.t)
        table.add_row(name, f"{pr_np:.6f}", f"{se_np2:.4f}", f"{pr_t:.6f}", f"{se_t2:.4f}")
        payoff_results[name] = {"numpy": pr_np, "numpy_se": se_np2, "torch": pr_t, "torch_se": se_t2}
    console.print(table)

    # ---- 3. Asian variance reduction -----------------------------------------
    beta, cv_price, cv_se, cv_var, naive_var, vr_cv = asian_control_variate(
        paths_val, p.s0, p.k, p.r, p.sigma, p.t, p.m)
    naive_price, naive_se = mc.mc_price(mc.payoff_asian(paths_val, p.k), p.r, p.t)
    anti_price, anti_se, anti_var = antithetic_price(
        mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m,
                     mc.gbm_increments(n_paths, p.m, args.seed, antithetic=True)),
        p.k, p.r, p.t)
    vr_anti = naive_var / anti_var

    # R replications
    rep_seeds = [1000 + i for i in range(args.reps)]
    rep = {"naive": [], "anti": [], "cv": [], "cv_beta": [],
           "naive_var": [], "anti_var": [], "cv_var": [], "vr_anti": [], "vr_cv": []}
    for s in rep_seeds:
        z = mc.gbm_increments(n_paths, p.m, s)
        paths = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, z)
        pr_n, _ = mc.mc_price(mc.payoff_asian(paths, p.k), p.r, p.t)
        b, pr_c, _, var_c, var_n, vrc = asian_control_variate(paths, p.s0, p.k, p.r, p.sigma, p.t, p.m)
        za = mc.gbm_increments(n_paths, p.m, s, antithetic=True)
        pathsa = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, za)
        pr_a, _, var_a = antithetic_price(pathsa, p.k, p.r, p.t)
        rep["naive"].append(pr_n); rep["naive_var"].append(var_n)
        rep["anti"].append(pr_a); rep["anti_var"].append(var_a)
        rep["cv"].append(pr_c); rep["cv_beta"].append(b); rep["cv_var"].append(var_c)
        rep["vr_anti"].append(var_n / var_a); rep["vr_cv"].append(vrc)

    def mstd(x):
        return float(np.mean(x)), float(np.std(x, ddof=1))

    vr_table = Table(title="Asian variance reduction", header_style="bold")
    vr_table.add_column("Estimator"); vr_table.add_column("price (single)"); vr_table.add_column("SE (single)")
    vr_table.add_column("price (R=20 mean±std)"); vr_table.add_column("var (R=20 mean±std)")
    vr_table.add_column("VR vs naive (mean±std)")
    vr_table.add_row("naive", f"{naive_price:.4f}", f"{naive_se:.4f}",
                     f"{mstd(rep['naive'])[0]:.4f}±{mstd(rep['naive'])[1]:.4f}",
                     f"{mstd(rep['naive_var'])[0]:.5f}±{mstd(rep['naive_var'])[1]:.5f}", "1.0")
    vr_table.add_row("antithetic", f"{anti_price:.4f}", f"{anti_se:.4f}",
                     f"{mstd(rep['anti'])[0]:.4f}±{mstd(rep['anti'])[1]:.4f}",
                     f"{mstd(rep['anti_var'])[0]:.5f}±{mstd(rep['anti_var'])[1]:.5f}",
                     f"{vr_anti:.3f} ({mstd(rep['vr_anti'])[0]:.3f}±{mstd(rep['vr_anti'])[1]:.3f})")
    vr_table.add_row("control-variate (geo)", f"{cv_price:.4f}", f"{cv_se:.4f}",
                     f"{mstd(rep['cv'])[0]:.4f}±{mstd(rep['cv'])[1]:.4f}",
                     f"{mstd(rep['cv_var'])[0]:.5f}±{mstd(rep['cv_var'])[1]:.5f}",
                     f"{vr_cv:.2f} ({mstd(rep['vr_cv'])[0]:.2f}±{mstd(rep['vr_cv'])[1]:.2f})")
    console.print(vr_table)
    console.print(f"[green]CV beta (single run) = {beta:.4f};  geo-Asian closed form = "
                  f"{mc.geometric_asian_call_price(p.s0, p.k, p.r, p.sigma, p.t, p.m):.4f}[/green]")

    # ---- 4. Greeks -----------------------------------------------------------
    bs_delta, bs_vega = bs_greeks(p)
    z_greeks = mc.gbm_increments(n_paths, p.m, args.seed)
    paths_base = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, z_greeks)

    greeks = {"bs": {"delta": bs_delta, "vega": bs_vega}, "pathwise": {}, "fd_delta": {}, "fd_vega": {}}
    # pathwise (vanilla + asian only)
    for name in ["vanilla", "asian"]:
        delta, delta_se, vega, vega_se = pathwise_greeks(paths_base, z_greeks, p, name)
        greeks["pathwise"][name] = {"delta": delta, "delta_se": delta_se,
                                    "vega": vega, "vega_se": vega_se}
    # finite differences (all four payoffs)
    for name in PAYOFFS:
        greeks["fd_delta"][name] = {}
        greeks["fd_vega"][name] = {}
        for frac in DELTA_BUMPS:
            h = frac * p.s0
            est, se = fd_greek(name, z_greeks, p, h, "delta")
            greeks["fd_delta"][name][str(frac)] = {"est": est, "se": se}
        for eps in VEGA_BUMPS:
            est, se = fd_greek(name, z_greeks, p, eps, "vega")
            greeks["fd_vega"][name][str(eps)] = {"est": est, "se": se}

    g_table = Table(title="Greeks (vanilla benchmark: BS delta / vega)", header_style="bold")
    g_table.add_column("Method"); g_table.add_column("delta"); g_table.add_column("delta SE")
    g_table.add_column("vega"); g_table.add_column("vega SE")
    g_table.add_row("BS analytic", f"{bs_delta:.6f}", "-", f"{bs_vega:.6f}", "-")
    for name in ["vanilla", "asian"]:
        pw = greeks["pathwise"][name]
        g_table.add_row(f"pathwise ({name})", f"{pw['delta']:.6f}", f"{pw['delta_se']:.5f}",
                        f"{pw['vega']:.6f}", f"{pw['vega_se']:.5f}")
    for frac in DELTA_BUMPS:
        fd = greeks["fd_delta"]["vanilla"][str(frac)]
        g_table.add_row(f"FD delta h={frac}*S0 (vanilla)", f"{fd['est']:.6f}", f"{fd['se']:.5f}", "-", "-")
    for eps in VEGA_BUMPS:
        fd = greeks["fd_vega"]["vanilla"][str(eps)]
        g_table.add_row(f"FD vega eps={eps} (vanilla)", "-", "-", f"{fd['est']:.6f}", f"{fd['se']:.5f}")
    console.print(g_table)
    console.print(Panel(
        "[bold]Bias-variance discussion[/bold]\n"
        "Pathwise Greeks apply only to a.s. differentiable payoffs (vanilla, arithmetic Asian). The "
        "digital and up-and-out barrier payoffs are discontinuous in S (indicator / barrier step), so the "
        "naive pathwise (score-function) derivative is ill-defined at the discontinuity and the estimator "
        "degenerates to a boundary term; only finite differences are reported for those two payoffs.\n"
        f"Vanilla delta FD vs BS analytic: small-h FD is near-unbiased but noisy (large SE); large-h FD is "
        f"stable but biased. Pathwise vanilla delta/vega is unbiased with SE comparable to the price SE.\n"
        f"{CVAR_NOTE}."
    ))

    # ---- 5. convergence ------------------------------------------------------
    ref_asian, ref_asian_se = asian_reference(p)
    console.print(f"[yellow]Asian reference (chunked CV, N=2^21) = {ref_asian:.6f} (SE {ref_asian_se:.4f})[/yellow]")
    conv = convergence_experiment(p, ref_asian, rng_seed_base=2000, reps=args.reps)

    slopes = {}
    conv_table = Table(title="Convergence: fitted slope log10(RMSE) vs log10(N)", header_style="bold")
    conv_table.add_column("Estimator"); conv_table.add_column("slope"); conv_table.add_column("slope SE")
    conv_table.add_column("intercept"); conv_table.add_column("R^2")
    for key, label in [("asian_naive", "Asian naive"), ("asian_antithetic", "Asian antithetic"),
                       ("asian_cv", "Asian control-variate"), ("vanilla_naive", "Vanilla naive")]:
        lgN = np.log10(np.array(conv["N"], dtype=float))
        lgr = np.log10(np.array(conv[key], dtype=float))
        res = linregress(lgN, lgr)
        slopes[key] = {"slope": res.slope, "slope_se": res.stderr, "intercept": res.intercept, "r2": res.rvalue ** 2}
        conv_table.add_row(label, f"{res.slope:.4f}", f"{res.stderr:.4f}",
                           f"{res.intercept:.4f}", f"{res.rvalue**2:.4f}")
    console.print(conv_table)

    # ---- 6. performance ------------------------------------------------------
    # workload for loop vs vectorized (small enough for pure Python, same for both)
    bench_n = 1024
    rng_bench = np.random.default_rng(123)
    z_bench = rng_bench.standard_normal((bench_n, p.m))

    # quick correctness agreement between the two performance implementations
    pv = asian_vectorized(z_bench, p.s0, p.r, p.sigma, p.t, p.m, p.k)
    pl = asian_naive_loop(z_bench, p.s0, p.r, p.sigma, p.t, p.m, p.k)
    assert abs(pv - pl) < 1e-9 * max(1.0, abs(pv)), f"loop/vectorized mismatch: {pv} vs {pl}"

    t_naive = timeit.repeat(lambda: asian_naive_loop(z_bench, p.s0, p.r, p.sigma, p.t, p.m, p.k),
                            repeat=5, number=3)
    t_vec = timeit.repeat(lambda: asian_vectorized(z_bench, p.s0, p.r, p.sigma, p.t, p.m, p.k),
                          repeat=5, number=500)
    per_naive = min(t_naive) / 3
    per_vec = min(t_vec) / 500
    speedup = per_naive / per_vec

    # memory (tracemalloc) on the vectorized path generation at full N=2^17
    tracemalloc.start()
    _ = mc.gbm_paths(p.s0, p.r, p.sigma, p.t, p.m, z_val)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_mem / (1024 * 1024)

    perf_table = Table(title="Performance (Asian pricer, N=1024, m=252)", header_style="bold")
    perf_table.add_column("Implementation"); perf_table.add_column("per-call (min of 5)"); perf_table.add_column("notes")
    perf_table.add_row("naive Python loop", f"{per_naive*1e3:.2f} ms", "numpy only for RNG; single scalar loop")
    perf_table.add_row("vectorized NumPy", f"{per_vec*1e3:.4f} ms", "cumsum/exp/mean on (N,m)")
    perf_table.add_row("speedup", f"{speedup:,.1f}x", "")
    console.print(perf_table)
    console.print(f"[cyan]Vectorized full-size path generation (N=2^17) peak tracemalloc ≈ {peak_mb:.0f} MiB[/cyan]")

    # line_profiler on the vectorized hot loop
    try:
        from line_profiler import LineProfiler
        lp = LineProfiler()
        lp.add_function(asian_vectorized)
        wrapped = lp(asian_vectorized)
        for _ in range(3):
            wrapped(z_val, p.s0, p.r, p.sigma, p.t, p.m, p.k)
        buf = io.StringIO()
        lp.print_stats(stream=buf, output_unit=1e-3)
        profile_text = buf.getvalue()
    except Exception as exc:  # pragma: no cover
        profile_text = f"line_profiler unavailable: {exc}"
    console.print(Panel(profile_text.strip(), title="line_profiler (vectorized hot loop, ms)"))
    (outdir / "line_profile.txt").write_text(profile_text)

    # ---- 7. figures ----------------------------------------------------------
    # convergence
    fig = go.Figure()
    colors = {"asian_naive": "blue", "asian_antithetic": "orange", "asian_cv": "green", "vanilla_naive": "red"}
    labels = {"asian_naive": "Asian naive", "asian_antithetic": "Asian antithetic",
              "asian_cv": "Asian control-variate", "vanilla_naive": "Vanilla naive"}
    for key in colors:
        fig.add_trace(go.Scatter(x=np.log10(conv["N"]), y=np.log10(conv[key]),
                                 mode="markers+lines", name=labels[key],
                                 marker=dict(color=colors[key])))
    fig.update_layout(title="RMSE vs N (log-log); fitted slopes vs -1/2",
                      xaxis_title="log10(N)", yaxis_title="log10(RMSE)")
    fig.write_html(outdir / "convergence.html", include_plotlyjs=True)

    # variance reduction
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=["antithetic", "control-variate (geo)"],
                          y=[mstd(rep["vr_anti"])[0], mstd(rep["vr_cv"])[0]],
                          error_y=dict(type="data", array=[mstd(rep["vr_anti"])[1], mstd(rep["vr_cv"])[1]]),
                          name="VR factor"))
    fig2.update_layout(title=f"Variance-reduction factor vs naive (R={args.reps} replications, N=2^17)",
                       yaxis_title="VR factor (naive var / estimator var)")
    fig2.write_html(outdir / "var_reduction.html", include_plotlyjs=True)

    # greeks: FD bias vs bump (vanilla, benchmark BS)
    fig3 = go.Figure()
    db = [greeks["fd_delta"]["vanilla"][str(f)]["est"] - bs_delta for f in DELTA_BUMPS]
    vb = [greeks["fd_vega"]["vanilla"][str(e)]["est"] - bs_vega for e in VEGA_BUMPS]
    fig3.add_trace(go.Scatter(x=np.log10(DELTA_BUMPS), y=db, mode="markers+lines", name="FD delta bias vs h"))
    fig3.add_trace(go.Scatter(x=np.log10(VEGA_BUMPS), y=vb, mode="markers+lines", name="FD vega bias vs eps",
                              yaxis="y2"))
    fig3.update_layout(title="Finite-difference bias vs bump size (vanilla, vs BS analytic)",
                       xaxis_title="log10(bump)", yaxis_title="delta bias",
                       yaxis2=dict(title="vega bias", overlaying="y", side="right"))
    fig3.write_html(outdir / "greeks.html", include_plotlyjs=True)

    # ---- 8. metrics.json -----------------------------------------------------
    fit_seconds = time.perf_counter() - t_start
    metrics = {
        "validation": {
            "bs_price": bs, "mc_numpy": mc_np, "mc_numpy_se": se_np,
            "mc_torch": mc_torch, "mc_torch_se": se_torch,
            "z_score": z_score, "z_pass": bool(abs(z_score) <= 3),
            "numpy_torch_diff": diff_np_torch,
            "cross_check_pass": bool(cross_ok),
        },
        "payoffs": payoff_results,
        "variance_reduction": {
            "naive": {"price": naive_price, "se": naive_se, "var": naive_var},
            "antithetic": {"price": anti_price, "se": anti_se, "var": anti_var, "vr_vs_naive": vr_anti},
            "control_variate": {"beta": beta, "price": cv_price, "se": cv_se, "var": cv_var,
                                "vr_vs_naive": vr_cv,
                                "geo_asian_closed_form": mc.geometric_asian_call_price(p.s0, p.k, p.r, p.sigma, p.t, p.m)},
            "replications": {k: to_jsonable(v) for k, v in rep.items()},
            "replication_summary": {
                "naive_price": list(mstd(rep["naive"])), "anti_price": list(mstd(rep["anti"])),
                "cv_price": list(mstd(rep["cv"])), "cv_beta": list(mstd(rep["cv_beta"])),
                "vr_anti": list(mstd(rep["vr_anti"])), "vr_cv": list(mstd(rep["vr_cv"])),
            },
        },
        "greeks": greeks,
        "convergence": {k: v for k, v in conv.items() if k != "_raw"},
        "convergence_slopes": slopes,
        "performance": {
            "bench_n": bench_n, "naive_loop_per_call_s": per_naive,
            "vectorized_per_call_s": per_vec, "speedup": speedup,
            "vectorized_peak_mib": peak_mb,
        },
    }
    payload = {
        "params": {**p.to_dict(), "seed": args.seed, "torch_seed": args.torch_seed,
                   "n_paths": n_paths, "reps": args.reps, "delta_bumps_fraction": DELTA_BUMPS,
                   "vega_bumps_absolute": VEGA_BUMPS, "vega_bump_scaling_note": CVAR_NOTE},
        "metrics": to_jsonable(metrics),
        "fit_seconds": fit_seconds,
        "history": {
            "convergence_raw": to_jsonable(conv["_raw"]),
            "asian_reference": {"price": ref_asian, "se": ref_asian_se},
            "replication_seeds": rep_seeds,
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ---- 9. summary ----------------------------------------------------------
    summary = Table(title="Summary", header_style="bold")
    summary.add_column("Quantity"); summary.add_column("Value")
    summary.add_row("BS z-score (vanilla)", f"{z_score:+.3f}  ({'PASS' if abs(z_score) <= 3 else 'FAIL'})")
    summary.add_row("Asian naive price", f"{naive_price:.4f} ± {naive_se:.4f}")
    summary.add_row("Asian antithetic price", f"{anti_price:.4f} ± {anti_se:.4f}")
    summary.add_row("Asian CV price", f"{cv_price:.4f} ± {cv_se:.4f}")
    summary.add_row("VR antithetic (mean±std)", f"{mstd(rep['vr_anti'])[0]:.3f} ± {mstd(rep['vr_anti'])[1]:.3f}")
    summary.add_row("VR control-variate (mean±std)", f"{mstd(rep['vr_cv'])[0]:.2f} ± {mstd(rep['vr_cv'])[1]:.2f}")
    summary.add_row("Convergence slopes", "; ".join(f"{labels[k]}={slopes[k]['slope']:.3f}" for k in slopes))
    summary.add_row("Naive-loop per call", f"{per_naive*1e3:.2f} ms")
    summary.add_row("Vectorized per call", f"{per_vec*1e3:.4f} ms")
    summary.add_row("Speedup", f"{speedup:,.1f}x")
    summary.add_row("Total runtime", f"{fit_seconds:.1f} s")
    console.print(summary)
    console.print(f"[bold green]metrics.json ->[/bold green] {outdir / 'metrics.json'}")
    console.print(f"[bold green]figures    ->[/bold green] {outdir}/{{convergence,var_reduction,greeks}}.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
