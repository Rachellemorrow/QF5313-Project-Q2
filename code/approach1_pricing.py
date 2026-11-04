"""Question 2 — Option A: Monte Carlo engine, variance reduction, Greeks, performance.

Owned by the deepseek-numerical subagent.  Linear pipeline:

    load model -> simulate -> price (both engines) -> validate vs Black-Scholes
    -> variance reduction -> Greeks -> convergence -> profiling -> write outputs

Usage (from the repository root)::

    uv run python code/approach1_pricing.py --seed 42 --dataset if_main
"""

from __future__ import annotations

import argparse
import cProfile
import io
import math
import pstats
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mc_common as mc  # noqa: E402
from mc_common import (Model, PAYOFFS, bs_call_delta, bs_call_price,  # noqa: E402
                       bs_call_vega, geometric_asian_price, mc_price_np,
                       mc_price_torch, mc_rmse_slope,
                       pathwise_asian_delta,
                       pathwise_asian_vega, pathwise_vanilla_delta,
                       pathwise_vanilla_vega, simulate_gbm_loop, simulate_gbm_np,
                       variance_reduction_ratio, write_metrics, results_dir,
                       asian_cv_reference, estimate_control_beta, spawn_seed_grid)

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()

DELTA_BUMPS = [0.1, 0.01, 0.001]


# ---------------------------------------------------------------------------
# Greek helpers (pathwise and finite-difference on COMMON random numbers)
# ---------------------------------------------------------------------------

def _payoff_fn(name: str) -> Callable:
    def fn(S, model):
        if name == "barrier":
            return PAYOFFS[name](S, model.K, model.B)
        return PAYOFFS[name](S, model.K)
    return fn


def _mean_se(contrib: np.ndarray) -> Dict[str, float]:
    return {"estimate": float(contrib.mean()), "se": mc.se_of(contrib),
            "n": int(contrib.shape[0])}


def compute_greeks_shared(model: Model, n_paths: int, seed: int,
                          bumps: Sequence[float] = DELTA_BUMPS,
                          n_boot: int = 2000, boot_seed: int = 0) -> Dict:
    """Pathwise + finite-difference Greeks driven by the *same* shock matrix.

    Returns ``{pathwise, fd, paired_diff}`` where ``paired_diff`` holds
    paired-bootstrap standard errors of ``pathwise - finite_difference`` for the
    vanilla and Asian delta/vega at *every* bump size.
    """
    Z = np.random.default_rng(seed).standard_normal((n_paths, model.steps))
    S = mc.gbm_paths_from_z_np(model, Z)
    W = mc.brownian_paths_from_z(model, Z)
    disc = math.exp(-model.r * model.T)

    pw_contrib = {
        "vanilla_delta": pathwise_vanilla_delta(model, S),
        "vanilla_vega": pathwise_vanilla_vega(model, S, W[:, -1]),
        "asian_delta": pathwise_asian_delta(model, S),
        "asian_vega": pathwise_asian_vega(model, S, W),
    }

    # per-path finite-difference contributions, aligned with pathwise via Z
    fd_contrib: Dict[tuple, np.ndarray] = {}
    for name in PAYOFFS:
        payoff = _payoff_fn(name)
        for h in bumps:
            S_up = S * (1 + h)
            S_dn = S * (1 - h)
            fd_contrib[(name, "delta", h)] = disc * (
                payoff(S_up, model) - payoff(S_dn, model)) / (2 * h * model.S0)
            up = Model(**{**model.asdict(), "sigma": model.sigma * (1 + h)})
            dn = Model(**{**model.asdict(), "sigma": model.sigma * (1 - h)})
            S_up = mc.gbm_paths_from_z_np(up, Z)
            S_dn = mc.gbm_paths_from_z_np(dn, Z)
            fd_contrib[(name, "vega", h)] = (
                math.exp(-up.r * up.T) * payoff(S_up, up)
                - math.exp(-dn.r * dn.T) * payoff(S_dn, dn)) / (2 * h * model.sigma)

    pathwise: Dict[str, dict] = {}
    for key, arr in pw_contrib.items():
        payoff_name, greek = key.split("_", 1)
        pathwise.setdefault(payoff_name, {})[greek] = _mean_se(arr)

    fd: Dict[str, dict] = {}
    for name in PAYOFFS:
        fd[name] = {"delta": {}, "vega": {}}
        for h in bumps:
            fd[name]["delta"][str(h)] = _mean_se(fd_contrib[(name, "delta", h)])
            fd[name]["vega"][str(h)] = _mean_se(fd_contrib[(name, "vega", h)])

    paired: Dict[str, dict] = {}
    for payoff_name, fd_name, pw_delta, pw_vega in [
            ("vanilla", "european", pw_contrib["vanilla_delta"], pw_contrib["vanilla_vega"]),
            ("asian", "asian", pw_contrib["asian_delta"], pw_contrib["asian_vega"])]:
        for h in bumps:
            for greek, pw_arr in (("delta", pw_delta), ("vega", pw_vega)):
                md, se = mc.paired_bootstrap_se(pw_arr, fd_contrib[(fd_name, greek, h)],
                                                n_boot, boot_seed)
                paired[f"{payoff_name}_{greek}_h{h}"] = {
                    "mean_diff": md, "se": se, "h": h}
    return {"pathwise": pathwise, "fd": fd, "paired_diff": paired}


# ---------------------------------------------------------------------------
# Variance reduction (Asian payoff)
# ---------------------------------------------------------------------------

def asian_estimators(model: Model, n_paths: int, seed: int,
                     beta: float, geo_price: float) -> Dict[str, dict]:
    """Naive / antithetic / control-variate estimates of the Asian call.

    ``beta`` is the control coefficient from an independent pilot sample.
    ``geo_price`` is the closed-form discounted geometric-Asian price.
    Also returns the raw arithmetic/geometric samples for the A/G correlation.
    """
    disc = math.exp(-model.r * model.T)
    E_G_undisc = math.exp(model.r * model.T) * geo_price

    S_naive, _ = simulate_gbm_np(model, n_paths, seed, antithetic=False)
    A_naive = mc.payoff_asian_call(S_naive, model.K)
    G_naive = mc.payoff_geometric_asian(S_naive, model.K)
    cv = A_naive - beta * (G_naive - E_G_undisc)

    S_anti, _ = simulate_gbm_np(model, n_paths, seed + 1, antithetic=True)
    A_anti_full = mc.payoff_asian_call(S_anti, model.K)
    half = n_paths // 2
    pair_avg = (A_anti_full[:half] + A_anti_full[half:2 * half]) / 2.0

    naive = disc * A_naive
    anti = disc * pair_avg
    cvp = disc * cv

    return {
        "naive": {"price": float(naive.mean()), "se": mc.se_of(naive), "n": int(naive.shape[0])},
        "antithetic": {"price": float(anti.mean()), "se": mc.se_of(anti), "n": int(anti.shape[0])},
        "control_variate": {"price": float(cvp.mean()), "se": mc.se_of(cvp), "n": int(cvp.shape[0])},
        "a_g_correlation": float(np.corrcoef(A_naive, G_naive)[0, 1]),
        # raw (undiscounted) per-path samples for paired variance-ratio estimation
        "_naive_raw": naive,
        "_anti_pair": anti,
        "_cv_raw": cvp,
    }


# ---------------------------------------------------------------------------
# Convergence experiment
# ---------------------------------------------------------------------------

def _bootstrap_slope_ci(ns: Sequence[int], price_arrays: List[np.ndarray],
                        ref: float, n_boot: int, seed: int) -> Dict[str, float]:
    """Bootstrap CI of the OLS log-log RMSE slope by resampling repeats."""
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        rmse_b = []
        for arr in price_arrays:
            arr = np.asarray(arr, dtype=float)
            ii = rng.integers(0, arr.shape[0], size=arr.shape[0])
            rmse_b.append(math.sqrt(((arr[ii] - ref) ** 2).mean()))
        slopes[b] = mc_rmse_slope(ns, np.asarray(rmse_b))["slope"]
    return {"mean": float(slopes.mean()),
            "ci_low": float(np.percentile(slopes, 2.5)),
            "ci_high": float(np.percentile(slopes, 97.5)),
            "n_boot": n_boot}


def convergence_experiment(model: Model, ns: List[int], repeats: int, seed: int,
                           beta: float, geo_price: float, ref_price: float,
                           n_boot_slope: int = 500, boot_seed: int = 0) -> Dict:
    """RMSE vs N with disjoint per-(n, rep, method) RNG streams.

    Seeds are spawned with :func:`numpy.random.SeedSequence` so no two
    (n, rep, method) cells reuse an RNG stream.  Repeated price arrays are kept
    and the fitted log-log slope is reported with a bootstrap CI (the plain OLS
    standard error is unreliable under a shared/noisy reference).
    """
    disc = math.exp(-model.r * model.T)
    E_G_undisc = math.exp(model.r * model.T) * geo_price
    bs = bs_call_price(model.S0, model.K, model.r, model.sigma, model.T)

    keys = ["vanilla_naive", "asian_naive", "asian_anti", "asian_cv"]
    n_methods = 2  # 0 = naive (shared sample), 1 = antithetic
    grid = spawn_seed_grid(seed, len(ns), repeats, n_methods)

    prices: Dict[str, Dict[int, list]] = {k: {} for k in keys}
    rmse: Dict[str, list] = {k: [] for k in keys}
    se_mean: Dict[str, list] = {k: [] for k in keys}

    for i, n in enumerate(ns):
        err = {k: [] for k in keys}
        ses = {k: [] for k in keys}
        for r in range(repeats):
            # naive sample (also drives vanilla & CV estimators)
            S, _ = simulate_gbm_np(model, n, grid[i][r][0], antithetic=False)
            ev = disc * mc.payoff_european_call(S, model.K)
            prices["vanilla_naive"].setdefault(n, []).append(float(ev.mean()))
            err["vanilla_naive"].append((ev.mean() - bs) ** 2)
            ses["vanilla_naive"].append(mc.se_of(ev))
            A = mc.payoff_asian_call(S, model.K)
            prices["asian_naive"].setdefault(n, []).append(float(disc * A.mean()))
            err["asian_naive"].append((disc * A.mean() - ref_price) ** 2)
            ses["asian_naive"].append(disc * A.std(ddof=1) / math.sqrt(n))
            G = mc.payoff_geometric_asian(S, model.K)
            cv = disc * (A - beta * (G - E_G_undisc))
            prices["asian_cv"].setdefault(n, []).append(float(cv.mean()))
            err["asian_cv"].append((cv.mean() - ref_price) ** 2)
            ses["asian_cv"].append(mc.se_of(cv))
            # antithetic sample
            S_a, _ = simulate_gbm_np(model, n, grid[i][r][1], antithetic=True)
            A_a = mc.payoff_asian_call(S_a, model.K)
            half = n // 2
            pair = disc * (A_a[:half] + A_a[half:2 * half]) / 2.0
            prices["asian_anti"].setdefault(n, []).append(float(pair.mean()))
            err["asian_anti"].append((pair.mean() - ref_price) ** 2)
            ses["asian_anti"].append(mc.se_of(pair))
        for k in keys:
            rmse[k].append(float(math.sqrt(float(np.mean(err[k])))))
            se_mean[k].append(float(np.mean(ses[k])))

    out: Dict = {
        "ns": [int(n) for n in ns],
        "repeats": repeats,
        "reference_price": float(ref_price),
        "reference_source": "geometric control-variate reference (arithmetic Asian) "
                             "and closed-form BS (vanilla)",
        "rmse": rmse,
        "se_mean": se_mean,
        "prices": {k: {str(n): v for n, v in d.items()} for k, d in prices.items()},
    }
    for k in keys:
        out[f"slope_rmse_{k}"] = mc_rmse_slope(ns, np.asarray(rmse[k]))
        out[f"slope_se_{k}"] = mc_rmse_slope(ns, np.asarray(se_mean[k]))
        ref = bs if k == "vanilla_naive" else ref_price
        out[f"slope_rmse_{k}_bootstrap"] = _bootstrap_slope_ci(
            ns, [np.asarray(prices[k][n]) for n in ns], ref, n_boot_slope, boot_seed)
    return out


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_engines(model: Model, n_paths: int, seed: int) -> Dict:
    Z = np.random.default_rng(seed).standard_normal((n_paths, model.steps))
    kwargs = dict(S0=model.S0, r=model.r, sigma=model.sigma, dt=model.dt,
                  steps=model.steps)

    import timeit

    def run_loop():
        return simulate_gbm_loop(**kwargs, Z=Z)

    def run_vec():
        return mc.gbm_paths_from_z_np(model, Z)

    # raw repeat lists (not just the best-of)
    t_loop_repeats = timeit.repeat(run_loop, number=1, repeat=5)
    t_vec_repeats = [t / 5 for t in timeit.repeat(run_vec, number=5, repeat=5)]

    # memory
    tracemalloc.start()
    run_loop()
    _, mem_loop = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    run_vec()
    _, mem_vec = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # hot-loop profile (line_profiler with documented time unit, cProfile fallback)
    line_profile = None
    try:
        from line_profiler import LineProfiler
        lp = LineProfiler()
        lp.add_function(simulate_gbm_loop)
        lp.runcall(run_loop)
        stats = lp.get_stats()
        unit = float(getattr(lp, "timer_unit", 1e-9) or 1e-9)  # seconds per unit
        timings = {}
        for key, entries in stats.timings.items():
            timings[str(key)] = [
                [int(ln), int(hits), float(t * unit)] for (ln, hits, t) in entries]
        line_profile = {"timer_unit_seconds": unit, "timings": timings}
    except Exception:
        line_profile = None

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(2):
        run_loop()
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(8)
    profile_text = buf.getvalue()

    return {
        "n_paths": n_paths,
        "steps": model.steps,
        "naive_loop_seconds_repeats": [float(x) for x in t_loop_repeats],
        "vectorized_seconds_repeats": [float(x) for x in t_vec_repeats],
        "naive_loop_seconds": float(min(t_loop_repeats)),
        "vectorized_seconds": float(min(t_vec_repeats)),
        "speedup": float(min(t_loop_repeats) / min(t_vec_repeats)),
        "naive_loop_peak_memory_bytes": int(mem_loop),
        "vectorized_peak_memory_bytes": int(mem_vec),
        "cprofile_top": profile_text,
        "line_profiler": line_profile,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_html(fig) -> str:
    import plotly.io as pio
    return pio.to_html(fig, include_plotlyjs=True, full_html=True)


def convergence_figure(conv: Dict) -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    ns = conv["ns"]
    styles = {
        "vanilla_naive": ("Vanilla naive", "circle", "black"),
        "asian_naive": ("Asian naive", "square", "firebrick"),
        "asian_anti": ("Asian antithetic", "diamond", "seagreen"),
        "asian_cv": ("Asian control-variate", "triangle-up", "steelblue"),
    }
    for k, (label, sym, color) in styles.items():
        fig.add_trace(go.Scatter(
            x=ns, y=conv["rmse"][k], mode="lines+markers", name=label,
            marker=dict(symbol=sym, color=color)))
    x0, y0 = ns[-1], conv["rmse"]["vanilla_naive"][-1]
    xs = np.array(ns, dtype=float)
    fig.add_trace(go.Scatter(
        x=xs, y=y0 * (xs / x0) ** -0.5, mode="lines", name="slope -1/2",
        line=dict(dash="dash", color="grey")))
    fig.update_layout(title="RMSE vs path count (log-log)",
                      xaxis_type="log", yaxis_type="log",
                      xaxis_title="N paths", yaxis_title="RMSE")
    return _fig_html(fig)


def greeks_figure(greeks: Dict) -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    names = ["Vanilla delta", "Vanilla vega", "Asian delta", "Asian vega"]
    pw = [greeks["pathwise"]["vanilla"]["delta"]["estimate"],
          greeks["pathwise"]["vanilla"]["vega"]["estimate"],
          greeks["pathwise"]["asian"]["delta"]["estimate"],
          greeks["pathwise"]["asian"]["vega"]["estimate"]]
    pw_se = [greeks["pathwise"]["vanilla"]["delta"]["se"],
             greeks["pathwise"]["vanilla"]["vega"]["se"],
             greeks["pathwise"]["asian"]["delta"]["se"],
             greeks["pathwise"]["asian"]["vega"]["se"]]
    fd = [greeks["fd"]["european"]["delta"]["0.01"]["estimate"],
          greeks["fd"]["european"]["vega"]["0.01"]["estimate"],
          greeks["fd"]["asian"]["delta"]["0.01"]["estimate"],
          greeks["fd"]["asian"]["vega"]["0.01"]["estimate"]]
    fd_se = [greeks["fd"]["european"]["delta"]["0.01"]["se"],
             greeks["fd"]["european"]["vega"]["0.01"]["se"],
             greeks["fd"]["asian"]["delta"]["0.01"]["se"],
             greeks["fd"]["asian"]["vega"]["0.01"]["se"]]
    fig.add_trace(go.Bar(name="Pathwise", x=names, y=pw,
                         error_y=dict(type="data", array=pw_se)))
    fig.add_trace(go.Bar(name="Finite difference (h=0.01)", x=names, y=fd,
                         error_y=dict(type="data", array=fd_se)))
    fig.add_trace(go.Scatter(
        name="Analytic (vanilla)", x=names[:2],
        y=[greeks["analytic"]["delta"], greeks["analytic"]["vega"]],
        mode="markers", marker=dict(symbol="x", size=12, color="black")))
    fig.update_layout(title="Greeks: pathwise vs finite difference", barmode="group")
    return _fig_html(fig)


def profiling_figure(prof: Dict) -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Naive loop", "Vectorized NumPy"],
                         y=[prof["naive_loop_seconds"], prof["vectorized_seconds"]],
                         marker_color=["firebrick", "seagreen"]))
    fig.update_layout(title=f"Simulation runtime (N={prof['n_paths']}, "
                            f"steps={prof['steps']})",
                      yaxis_title="seconds (log)", yaxis_type="log")
    return _fig_html(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q2 Option A — Monte Carlo pricing engine")
    p.add_argument("--seed", type=int, default=42, help="global RNG seed")
    p.add_argument("--dataset", type=str, default="if_main")
    p.add_argument("--paths", type=int, default=1 << 17, help="main MC path count")
    p.add_argument("--ref-paths", type=int, default=1 << 22,
                   help="control-variate reference production path budget (arithmetic Asian)")
    p.add_argument("--ref-pilot", type=int, default=1 << 14,
                   help="reference control-variate pilot path count")
    p.add_argument("--pilot-paths", type=int, default=1 << 14,
                   help="main control-variate pilot path count")
    p.add_argument("--greek-paths", type=int, default=1 << 17)
    p.add_argument("--profile-paths", type=int, default=4000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-boot-slope", type=int, default=500)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--min-paths", type=int, default=512)
    p.add_argument("--max-paths", type=int, default=65536)
    return p.parse_args()


def _with_ci(d: Dict[str, float]) -> Dict[str, float]:
    """Attach an asymptotic 95% confidence interval to a price estimate."""
    d = dict(d)
    z = 1.959963984540054
    d["ci_low"] = d["price"] - z * d["se"]
    d["ci_high"] = d["price"] + z * d["se"]
    return d


def main() -> None:
    args = parse_args()
    model = mc.load_model()
    out_dir = results_dir("approach1", args.dataset)
    t_start = time.time()

    console.print(Panel(f"[bold]Q2 Option A — Monte Carlo engine[/bold]\n"
                        f"dataset={args.dataset} seed={args.seed}\n"
                        f"model={model.asdict()}"))

    # ---- 1. pricing (both engines) + validation ---------------------------
    prices_np = mc_price_np(model, args.paths, args.seed)
    prices_torch = mc_price_torch(model, args.paths, args.seed)
    bs = bs_call_price(model.S0, model.K, model.r, model.sigma, model.T)
    z = (prices_np["european"]["price"] - bs) / prices_np["european"]["se"]

    engine_agreement = {}
    for k in prices_np:
        engine_agreement[k] = {
            "numpy": prices_np[k]["price"], "torch": prices_torch[k]["price"],
            "abs_diff": abs(prices_np[k]["price"] - prices_torch[k]["price"]),
            "se_numpy": prices_np[k]["se"],
        }

    t_rich = Table(title="Prices (NumPy / PyTorch, identical shocks)")
    t_rich.add_column("Payoff"); t_rich.add_column("NumPy"); t_rich.add_column("Torch")
    t_rich.add_column("SE")
    for k in prices_np:
        t_rich.add_row(k, f"{prices_np[k]['price']:.5f}", f"{prices_torch[k]['price']:.5f}",
                       f"{prices_np[k]['se']:.5f}")
    console.print(t_rich)
    console.print(f"European validation: MC={prices_np['european']['price']:.5f} "
                  f"BS={bs:.5f} z={z:.3f} (|z|<=3 -> {'PASS' if abs(z) <= 3 else 'FAIL'})")

    # ---- 2. variance reduction (Asian) ------------------------------------
    geo_price, mu_G, var_G = geometric_asian_price(
        model.S0, model.K, model.r, model.sigma, model.T, model.steps)
    t0 = time.time()
    beta = estimate_control_beta(model, args.pilot_paths, args.seed + 1000)
    pilot_seconds = time.time() - t0
    asian = asian_estimators(model, args.paths, args.seed + 2000, beta, geo_price)

    vrr_cv = variance_reduction_ratio(asian["_naive_raw"], asian["_cv_raw"],
                                      args.n_boot, args.seed + 3000, paired=True)
    vrr_anti = variance_reduction_ratio(asian["_naive_raw"], asian["_anti_pair"],
                                        args.n_boot, args.seed + 3001, paired=False,
                                        scale=2.0)
    console.print(Panel(f"Asian variance reduction (pilot beta={beta:.4f}):\n"
                        f"  naive  price={asian['naive']['price']:.5f} se={asian['naive']['se']:.5f}\n"
                        f"  anti   price={asian['antithetic']['price']:.5f} se={asian['antithetic']['se']:.5f}\n"
                        f"  CV     price={asian['control_variate']['price']:.5f} se={asian['control_variate']['se']:.5f}\n"
                        f"  A/G corr={asian['a_g_correlation']:.6f}\n"
                        f"  VRR antithetic={vrr_anti['ratio']:.1f} [{vrr_anti['ci_low']:.1f},{vrr_anti['ci_high']:.1f}]\n"
                        f"  VRR CV={vrr_cv['ratio']:.1f} [{vrr_cv['ci_low']:.1f},{vrr_cv['ci_high']:.1f}]\n"
                        f"  CV pilot: n={args.pilot_paths} ({args.pilot_paths/args.paths:.3f}x main) "
                        f"time={pilot_seconds:.3f}s"))

    # ---- 3. Greeks (shared shocks, paired PW-FD) ---------------------------
    g = compute_greeks_shared(model, args.greek_paths, args.seed + 4000,
                              bumps=DELTA_BUMPS, n_boot=args.n_boot, boot_seed=args.seed + 6000)
    analytic = {
        "delta": bs_call_delta(model.S0, model.K, model.r, model.sigma, model.T),
        "vega": bs_call_vega(model.S0, model.K, model.r, model.sigma, model.T),
    }
    greeks = {"pathwise": g["pathwise"], "fd": g["fd"], "analytic": analytic,
              "paired_diff": g["paired_diff"]}

    t_g = Table(title="Greeks (vanilla)")
    t_g.add_column("Quantity"); t_g.add_column("Pathwise"); t_g.add_column("FD h=0.01")
    t_g.add_column("Analytic")
    t_g.add_row("delta", f"{g['pathwise']['vanilla']['delta']['estimate']:.5f}",
                f"{g['fd']['european']['delta']['0.01']['estimate']:.5f}", f"{analytic['delta']:.5f}")
    t_g.add_row("vega", f"{g['pathwise']['vanilla']['vega']['estimate']:.4f}",
                f"{g['fd']['european']['vega']['0.01']['estimate']:.4f}", f"{analytic['vega']:.4f}")
    console.print(t_g)
    for h in DELTA_BUMPS:
        pd = greeks["paired_diff"][f"vanilla_delta_h{h}"]
        pv = greeks["paired_diff"][f"vanilla_vega_h{h}"]
        console.print(f"  PW-FD h={h}: vanilla delta diff={pd['mean_diff']:+.5f} "
                      f"se={pd['se']:.5f} | vega diff={pv['mean_diff']:+.3f} se={pv['se']:.3f}")

    # ---- 4. convergence ----------------------------------------------------
    ref = asian_cv_reference(model, args.ref_paths, args.seed + 7000,
                             n_pilot=args.ref_pilot)
    ns = []
    n = args.min_paths
    while n <= args.max_paths:
        ns.append(n)
        n *= 2
    conv = convergence_experiment(model, ns, args.repeats, args.seed + 8000,
                                  beta, geo_price, ref["price"],
                                  n_boot_slope=args.n_boot_slope, boot_seed=args.seed + 9000)
    smallest_cv_se = conv["se_mean"]["asian_cv"][-1]
    console.print(Panel(f"Reference (Asian CV): price={ref['price']:.5f} se={ref['se']:.5f} "
                        f"(smallest CV SE={smallest_cv_se:.5f}; ratio={smallest_cv_se/ref['se']:.1f}x)\n"
                        f"Convergence slopes (theoretical -1/2):\n" +
                        "\n".join(
                            f"  {k}: rmse slope={conv[f'slope_rmse_{k}']['slope']:+.3f} "
                            f"se={conv[f'slope_rmse_{k}']['slope_se']:.3f} "
                            f"boot95=[{conv[f'slope_rmse_{k}_bootstrap']['ci_low']:+.3f},"
                            f"{conv[f'slope_rmse_{k}_bootstrap']['ci_high']:+.3f}] | "
                            f"SE slope={conv[f'slope_se_{k}']['slope']:+.3f}"
                            for k in ["vanilla_naive", "asian_naive", "asian_anti", "asian_cv"])))

    # ---- 5. profiling ------------------------------------------------------
    prof = profile_engines(model, args.profile_paths, args.seed + 10000)
    console.print(Panel(f"Profiling: naive loop={prof['naive_loop_seconds']:.4f}s, "
                        f"vectorized={prof['vectorized_seconds']:.6f}s, "
                        f"speedup={prof['speedup']:.1f}x, "
                        f"memory loop={prof['naive_loop_peak_memory_bytes']/1e6:.1f}MB, "
                        f"vec={prof['vectorized_peak_memory_bytes']/1e6:.1f}MB"))

    # ---- 6. write outputs --------------------------------------------------
    fit_seconds = time.time() - t_start
    fig_conv = convergence_figure(conv)
    fig_greeks = greeks_figure(greeks)
    fig_prof = profiling_figure(prof)
    (out_dir / "convergence.html").write_text(fig_conv, encoding="utf-8")
    (out_dir / "greeks.html").write_text(fig_greeks, encoding="utf-8")
    (out_dir / "profiling.html").write_text(fig_prof, encoding="utf-8")

    metrics = {
        "validation": {"european_mc": prices_np["european"]["price"],
                       "european_bs": bs, "z_score": z, "z_pass": bool(abs(z) <= 3)},
        "prices": {"numpy": {k: _with_ci(v) for k, v in prices_np.items()},
                   "torch": {k: _with_ci(v) for k, v in prices_torch.items()}},
        "engine_agreement": engine_agreement,
        "variance_reduction": {
            "asian": {k: v for k, v in asian.items() if not k.startswith("_")},
            "beta": beta, "geo_asian_price": geo_price,
            "a_g_correlation": asian["a_g_correlation"],
            "pilot_paths": args.pilot_paths,
            "pilot_fraction_of_main": args.pilot_paths / args.paths,
            "pilot_seconds": pilot_seconds,
            "vrr_cv": vrr_cv, "vrr_antithetic": vrr_anti,
        },
        "greeks": greeks,
        "convergence": {k: v for k, v in conv.items() if k != "prices"},
        "profiling": {k: v for k, v in prof.items()
                      if k not in ("cprofile_top", "line_profiler")},
    }
    history = {
        "cprofile_top": prof["cprofile_top"],
        "line_profiler": prof["line_profiler"],
        "convergence_prices": conv["prices"],
        "reference": ref,
    }
    params = {"model": model.asdict(), "seed": args.seed, "dataset": args.dataset,
              "paths": args.paths, "ref_paths": args.ref_paths,
              "ref_pilot": args.ref_pilot, "pilot_paths": args.pilot_paths,
              "greek_paths": args.greek_paths, "n_boot": args.n_boot,
              "n_boot_slope": args.n_boot_slope, "repeats": args.repeats,
              "convergence_ns": ns}
    write_metrics(out_dir / "metrics.json", params, metrics, fit_seconds, history)
    console.print(f"[green]Wrote[/green] {out_dir / 'metrics.json'} "
                  f"(fit_seconds={fit_seconds:.2f})")


if __name__ == "__main__":
    main()
