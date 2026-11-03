#!/usr/bin/env python
"""approach2_hedging.py — Question 2, Option B: deep hedging of a European vanilla call
on the CSI 300 index futures (IF) in PyTorch.

Authoritative shared mathematics is imported from ``mc_common`` (``import mc_common as mc``);
its conventions are NOT modified here.  This module only implements the Option-B deep-hedging
pipeline (single self-contained script, argparse, linear load -> simulate -> fit -> evaluate ->
write outputs), per the assignment §4.4 and the AGENTS.md Option-B contract.

Accounting conventions (documented here AND in metrics.json, and echoed in the terminal):
  * Hedging grid t_i = i*dt, i = 0..m-1, m = 252 steps, dt = T/m = 1/252.  S_i denotes S_{t_i}.
    The path array holds S_0..S_m (m+1 points); S_0 is the initial spot, NOT a decision point.
  * Delta_i is DECIDED at t_i from information available at t_i only: for the neural policy
    f_theta(S_{t_i}/S_0, t_i/T); for Black-Scholes delta N(d1(S_{t_i}, T - t_i)) at rebalance dates.
  * Terminal hedging P&L (assignment formula, exactly as written):
        V_T = sum_{i=0}^{m-1} exp(r*(T - t_i)) * Delta_i * (S_{i+1} - S_i)
            - kappa * sum_{i=0}^{m-1} |Delta_i - Delta_{i-1}| * S_{t_i}
            - H(S_T),   with Delta_{-1} := 0 (the initial position is charged), H = max(S_T-K,0).
    Note the discount factor uses the DECISION time t_i (assignment's simplified cash-flow timing:
    the increment S_{i+1}-S_i actually accrues over [t_i, t_{i+1}] but is compounded from t_i, not
    t_{i+1}).  We implement it exactly as written rather than silently altering it.
  * Transaction cost is charged at the decision-time spot S_{t_i}, proportional to |DeltaDelta|,
    summed over i = 0..m-1.  There is NO extra terminal close-out term: the final hedge P&L
    realizes the last move (i = m-1) inside the sum, matching the assignment formula.
  * Premium convention: the option premium is EXCLUDED from V_T exactly as the assignment writes
    it.  Adding the constant premium e^{rT}*C_BS would shift mean P&L and CVaR equally for EVERY
    policy, so it changes no comparison and leaves variance/turnover unchanged; we therefore omit
    it and note this.  (With the premium excluded, a replicating hedge has
    E[V_T] ~ -e^{rT}*C_BS up to discretization/cost error: the hedger pays the claim out of
    the hedge gains alone. Adding the premium back shifts the mean by +e^{rT}*C_BS for every
    policy identically and leaves all comparisons unchanged.)
  * The SAME accounting applies to the neural policy and to every Black-Scholes baseline.

Risk measure: L = -V_T; CVaR_0.95 of L is minimised via Rockafellar-Uryasev
    min_{theta, eta}  eta + (1/(1-alpha)) * mean(relu(L_theta - eta)),   alpha = 0.95.
Evaluation CVaR is computed independently of the training loss (sort L descending, mean of the
worst 5%) on held-out paths.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

import mc_common as mc  # noqa: E402  (authoritative shared mathematics)

import plotly.graph_objects as go  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

ALPHA = 0.95
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Option B deep hedging (European vanilla call, IF).")
    p.add_argument("--seed", type=int, default=42, help="global seed (default 42)")
    p.add_argument("--n-train", type=int, default=131072, help="training paths (default 2^17)")
    p.add_argument("--n-val", type=int, default=32768, help="validation paths")
    p.add_argument("--n-test", type=int, default=32768, help="test paths")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64],
                   help="MLP hidden layer widths (<=3 layers)")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--epochs", type=int, default=60, help="max epochs")
    p.add_argument("--batch-size", type=int, default=4096, help="minibatch size (paths)")
    p.add_argument("--patience", type=int, default=12, help="best-checkpoint patience (no early stop occurred: best-val state is restored at the end)")
    p.add_argument("--min-delta", type=float, default=1e-4, help="min delta on val CVaR to reset patience (best-checkpoint selection, not an early stop in the logged run)")
    p.add_argument("--train-kappa", type=float, default=0.001,
                   help="transaction cost (decimal) the neural policy is trained at")
    p.add_argument("--threads", type=int, default=8,
                   help="torch intra-op thread count (0 = torch default)")
    p.add_argument("--device", type=str, default=DEVICE)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Seeding / GBM simulation (PyTorch)
# ---------------------------------------------------------------------------
def gbm_paths_torch(s0: float, r: float, sigma: float, t: float, m: int,
                    n_paths: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Risk-neutral GBM paths under Q, exactly as mc_common (drift r, not Black-76).

    Returns (n_paths, m+1) with paths[:, 0] == s0.
    """
    torch.manual_seed(seed)
    dt = t / m
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    z = torch.randn((n_paths, m), dtype=dtype)
    log_inc = drift + vol * z
    log_paths = torch.cat([torch.zeros((n_paths, 1), dtype=dtype),
                           torch.cumsum(log_inc, dim=1)], dim=1)
    return s0 * torch.exp(log_paths)


def torch_mc_price(paths: torch.Tensor, k: float, r: float, t: float):
    """Return (discounted mean price, standard error) for the vanilla call on given paths."""
    payoff = torch.clamp(paths[:, -1] - k, min=0.0)
    disc = math.exp(-r * t)
    n = payoff.shape[0]
    mean = float(payoff.mean().item())
    se = float(payoff.std(unbiased=True).item() / math.sqrt(n)) * disc
    return disc * mean, se


# ---------------------------------------------------------------------------
# Neural hedging policy (MLP, <= 3 hidden layers)
# ---------------------------------------------------------------------------
class HedgeMLP(nn.Module):
    def __init__(self, hidden=(64, 64, 64)):
        super().__init__()
        assert len(hidden) <= 3, "at most three hidden layers"
        layers = []
        prev = 2
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 2) -> delta (...,)
        return self.net(x).squeeze(-1)


def policy_inputs(S: torch.Tensor, s0: float, m: int) -> torch.Tensor:
    """Stack (S_{t_i}/S_0, t_i/T) for i = 0..m-1 into a (n_paths*m, 2) tensor.

    Causal: only S at time t_i (and the calendar time t_i) enter the feature vector.
    """
    n = S.shape[0]
    s_spot = S[:, :-1]  # (n, m) decision-time spots
    s_ratio = s_spot / s0
    t_frac = torch.arange(m, device=S.device, dtype=S.dtype) / m
    x = torch.stack([s_ratio, t_frac.expand(n, m)], dim=-1)  # (n, m, 2)
    return x.reshape(-1, 2)


def neural_delta(model: HedgeMLP, S: torch.Tensor, s0: float, m: int) -> torch.Tensor:
    """Return delta (n_paths, m): Delta_i = f_theta(S_{t_i}/S_0, t_i/T)."""
    n = S.shape[0]
    x = policy_inputs(S, s0, m)
    return model(x).reshape(n, m)


# ---------------------------------------------------------------------------
# Hedging P&L (assignment formula, shared by neural and BS policies)
# ---------------------------------------------------------------------------
def hedge_pnl(S: torch.Tensor, delta: torch.Tensor, k: float, r: float,
              dt: float, m: int, kappa: float):
    """Terminal hedging P&L per the assignment formula.

    Args:
        S:      (n_paths, m+1) asset path, S[:, 0] == S_0.
        delta:  (n_paths, m) position Delta_i decided at t_i.
    Returns:
        V:          (n_paths,) terminal P&L (premium excluded).
        turnover:   (n_paths,) sum_i |Delta_i - Delta_{i-1}|.
        cost:       (n_paths,) kappa * sum_i |Delta_i - Delta_{i-1}| * S_{t_i}.
    """
    n = S.shape[0]
    dS = S[:, 1:] - S[:, :-1]                    # (n, m)  S_{i+1} - S_i
    i = torch.arange(m, device=S.device, dtype=S.dtype)
    discount = torch.exp(r * (m - i) * dt)       # (m,)  exp(r*(T - t_i))
    hedge_gain = (delta * dS * discount).sum(dim=1)

    delta_prev = torch.cat([torch.zeros((n, 1), device=S.device, dtype=S.dtype),
                            delta[:, :-1]], dim=1)   # Delta_{i-1}, Delta_{-1} = 0
    s_spot = S[:, :-1]                           # decision-time spot S_{t_i}
    trade = (delta - delta_prev).abs()
    cost = kappa * (trade * s_spot).sum(dim=1)
    turnover = trade.sum(dim=1)

    payoff = torch.clamp(S[:, -1] - k, min=0.0)  # H(S_T) = max(S_T - K, 0)
    V = hedge_gain - cost - payoff
    return V, turnover, cost


def empirical_cvar(loss: torch.Tensor, alpha: float = ALPHA) -> float:
    """Empirical CVaR_alpha of a loss vector (independent of the R-U training loss).

    Sort descending, average the worst (1-alpha) fraction (the upper tail of L).
    """
    n = loss.numel()
    k = max(1, int(round((1 - alpha) * n)))
    sorted_desc, _ = torch.sort(loss, descending=True)
    return float(sorted_desc[:k].mean().item())


def policy_loss(model: HedgeMLP, S: torch.Tensor, s0: float, k: float, r: float,
                dt: float, m: int, kappa: float) -> torch.Tensor:
    """L = -V_T for the neural policy on paths S (loss, larger is worse)."""
    delta = neural_delta(model, S, s0, m)
    V, _, _ = hedge_pnl(S, delta, k, r, dt, m, kappa)
    return -V


# ---------------------------------------------------------------------------
# Black-Scholes delta benchmark
# ---------------------------------------------------------------------------
def bs_delta(S_spot: torch.Tensor, k: float, r: float, sigma: float,
             tau: torch.Tensor) -> torch.Tensor:
    """N(d1) with d1 = (ln(S/K) + (r + sigma^2/2)*tau) / (sigma*sqrt(tau))."""
    d1 = (torch.log(S_spot / k) + (r + 0.5 * sigma * sigma) * tau) / (sigma * torch.sqrt(tau))
    return 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))


def bs_delta_hedge(S: torch.Tensor, k: float, r: float, sigma: float, t: float,
                   m: int, freq: int) -> torch.Tensor:
    """BS-delta positions (n_paths, m), rebalanced every `freq` steps and held in between.

    Trades occur ONLY at rebalance dates i in {0, freq, 2*freq, ...}; between those dates the
    position is held constant (no trade, no cost).  Cost is still charged on |DeltaDelta| at the
    rebalance dates via the shared hedge_pnl (which uses Delta_{i-1}).
    """
    dt = t / m
    i = torch.arange(m, device=S.device, dtype=torch.long)
    tau = (m - i).to(S.dtype) * dt            # remaining time at each potential rebalance point
    s_spot = S[:, :-1]
    target = bs_delta(s_spot, k, r, sigma, tau)   # full target delta at every step (cheap)
    rebalance_col = (i // freq) * freq        # last rebalance index <= i (int64)
    return target[:, rebalance_col]           # (n, m) held-constant positions


# ---------------------------------------------------------------------------
# Training (Rockafellar-Uryasev CVaR)
# ---------------------------------------------------------------------------
def train_model(model: HedgeMLP, eta: nn.Parameter, S_train: torch.Tensor, S_val: torch.Tensor,
                s0: float, k: float, r: float, dt: float, m: int, kappa: float, alpha: float,
                lr: float, epochs: int, batch_size: int, patience: int,
                min_delta: float, shuffle_seed: int, console: Console):
    """Minimise CVaR_alpha(L), L = -V_T, via the Rockafellar-Uryasev form solved by
    coordinate descent.

    The R-U objective  min_{theta, eta}  eta + (1/(1-alpha))*E[relu(L_theta - eta)] is
    jointly minimised by (i) an EXACT eta-step and (ii) a gradient theta-step:

      * For fixed theta, the objective is convex in eta and its minimiser is the empirical
        alpha-quantile (VaR_alpha) of L_theta:  P(L > eta*) = 1 - alpha.  We recompute
        eta = quantile_alpha(L_theta) on a fixed training subset at the START of every epoch.
      * For fixed eta, we take Adam gradient steps on theta through
        eta + rho * mean(relu(L_theta - eta)); only the paths with L > eta (the upper tail)
        contribute gradient.

    A purely gradient-based update of eta is NOT used because Adam normalises the update
    magnitude to ~lr per step, so eta would track the VaR far too slowly (the VaR drops by
    O(10^3) points as the policy learns to hedge); the exact quantile step sidesteps this.
    """
    n = S_train.shape[0]
    n_train_eval = min(n, 16384)          # fixed subset for quantile/VaR and train-CVaR reporting
    S_train_eval = S_train[:n_train_eval]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(shuffle_seed)
    n_batches = (n + batch_size - 1) // batch_size
    rho = 1.0 / (1.0 - alpha)

    # Initial eta = VaR_alpha of the UNHEDGED loss (L = payoff when Delta ~ 0).  Each epoch
    # reuses the end-of-epoch train loss for BOTH the train-CVaR report and the next epoch's
    # eta, so the train-eval subset is forwarded exactly once per epoch.
    model.eval()
    with torch.no_grad():
        L_train = policy_loss(model, S_train_eval, s0, k, r, dt, m, kappa)
    eta_q = float(torch.quantile(L_train, alpha).item())

    history = {"epoch": [], "train_loss": [], "train_cvar": [], "val_cvar": [], "eta": []}
    best_val_cvar = float("inf")
    best_state, best_eta = None, None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        eta_used = eta_q
        eta.data.fill_(eta_used)           # exact R-U minimiser in eta (VaR_alpha)

        model.train()
        perm = torch.randperm(n, generator=g)
        total_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            L = policy_loss(model, S_train[idx], s0, k, r, dt, m, kappa)
            loss = eta_used + rho * torch.relu(L - eta_used).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(idx.numel())

        avg_train_loss = total_loss / n
        model.eval()
        with torch.no_grad():
            L_train = policy_loss(model, S_train_eval, s0, k, r, dt, m, kappa)
            train_cvar = empirical_cvar(L_train, alpha)
            eta_q = float(torch.quantile(L_train, alpha).item())   # eta for next epoch
            L_val = policy_loss(model, S_val, s0, k, r, dt, m, kappa)
            val_cvar = empirical_cvar(L_val, alpha)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["train_cvar"].append(train_cvar)
        history["val_cvar"].append(val_cvar)
        history["eta"].append(eta_used)

        if epoch == 1 or epoch % 5 == 0:
            console.print(
                f"[cyan]epoch {epoch:3d}[/]  train_loss={avg_train_loss:8.4f}  "
                f"train_cvar={train_cvar:9.4f}  val_cvar={val_cvar:9.4f}  "
                f"eta={eta_used:8.1f}")

        if val_cvar < best_val_cvar - min_delta:
            best_val_cvar = val_cvar
            epochs_no_improve = 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            best_eta = eta.detach().clone()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                console.print(f"[yellow]early stopping at epoch {epoch} "
                              f"(no improvement > {min_delta} for {patience} epochs)[/]")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        with torch.no_grad():
            eta.copy_(best_eta)

    return history, best_val_cvar


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_policy(delta_fn, S: torch.Tensor, s0: float, k: float, r: float,
                    dt: float, m: int, kappa: float, alpha: float):
    """Return a dict of summary statistics for one (policy, kappa) on paths S."""
    with torch.no_grad():
        delta = delta_fn(S)
        V, turnover, cost = hedge_pnl(S, delta, k, r, dt, m, kappa)
        L = -V
        return {
            "mean_pnl": float(V.mean().item()),
            "var_pnl": float(V.var(unbiased=True).item()),
            "cvar95": empirical_cvar(L, alpha),
            "turnover": float(turnover.mean().item()),
            "avg_cost": float(cost.mean().item()),
        }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    console = Console(width=160)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    p = mc.estimate_params()
    s0, k, r, sigma, t, m, dt = p.s0, p.k, p.r, p.sigma, p.t, p.m, p.dt
    assert abs(k - s0) < 1e-12, "K must equal S0 (at-the-money spot)"

    console.print(Panel.fit(
        f"[bold]Deep hedging — European vanilla call on IF[/]\n"
        f"S0 = {s0:.2f}   K = {k:.2f} (ATM)   r = {r:.5f}   sigma = {sigma:.5f}\n"
        f"T = {t}   m = {m} steps   dt = {dt:.6f}   alpha = {ALPHA}   kappa_train = {args.train_kappa}",
        title="Parameters (mc_common.estimate_params)"))

    # ------------------------------------------------------------------
    # 2. Simulation & independent torch/numpy/BS price check (seed 42)
    # ------------------------------------------------------------------
    console.print("[bold]Simulating paths and verifying torch vs numpy vs Black-Scholes...[/]")
    bs_price = mc.black_scholes_call(s0, k, r, sigma, t)

    # NumPy MC (seed 42), float64 — then release.
    np_paths = mc.simulate_gbm(s0, r, sigma, t, m, args.n_train, 42)
    np_price, np_se = mc.mc_price(mc.payoff_vanilla(np_paths, k), r, t)
    del np_paths
    gc.collect()

    # PyTorch MC (seed 42), float64 — independent draws (torch.manual_seed != numpy default_rng).
    paths_torch64 = gbm_paths_torch(s0, r, sigma, t, m, args.n_train, 42, dtype=torch.float64)
    torch_price, torch_se = torch_mc_price(paths_torch64, k, r, t)
    del paths_torch64
    gc.collect()

    z_bs = (torch_price - bs_price) / torch_se
    z_cross = (torch_price - np_price) / math.sqrt(torch_se ** 2 + np_se ** 2)

    # ------------------------------------------------------------------
    # Main float32 simulation (train/val/test, independent seeds)
    # ------------------------------------------------------------------
    S_train = gbm_paths_torch(s0, r, sigma, t, m, args.n_train, 42, dtype=torch.float32)
    S_val = gbm_paths_torch(s0, r, sigma, t, m, args.n_val, 123, dtype=torch.float32)
    S_test = gbm_paths_torch(s0, r, sigma, t, m, args.n_test, 2026, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = HedgeMLP(hidden=tuple(args.hidden)).to(args.device)
    eta = nn.Parameter(torch.tensor(0.0))
    n_params = sum(pp.numel() for pp in model.parameters())
    console.print(f"[bold]MLP policy[/] hidden={tuple(args.hidden)}  params={n_params}  "
                  f"(input: S/S0 and t/T; <= 3 hidden layers, causal)")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    console.print(f"[bold]Training[/] N_train={args.n_train} N_val={args.n_val} lr={args.lr} "
                  f"batch={args.batch_size} max_epochs={args.epochs} kappa={args.train_kappa}")
    t0 = time.perf_counter()
    history, best_val_cvar = train_model(
        model, eta, S_train, S_val, s0, k, r, dt, m, args.train_kappa, ALPHA,
        args.lr, args.epochs, args.batch_size, args.patience, args.min_delta,
        args.seed, console)
    fit_seconds = time.perf_counter() - t0
    n_epochs_run = len(history["epoch"])
    console.print(f"[bold]Training done[/] {n_epochs_run} epochs in {fit_seconds:.1f}s "
                  f"best_val_cvar={best_val_cvar:.4f}")

    # ------------------------------------------------------------------
    # 4. Evaluation on held-out test/val (neural at 3 kappa, BS at 9 combos)
    # ------------------------------------------------------------------
    kappas = [(0.0, "0bp"), (0.001, "10bp"), (0.005, "50bp")]
    freqs = [1, 5, 21]

    def neural_delta_fn(S):
        return neural_delta(model, S, s0, m)

    rows = {}   # name -> summary dict
    # Neural policy: every-step rebalancing, evaluated at the three cost levels.
    for kv, label in kappas:
        rows[f"neural_{label}"] = {"test": evaluate_policy(
            neural_delta_fn, S_test, s0, k, r, dt, m, kv, ALPHA)}
        rows[f"neural_{label}"]["val"] = evaluate_policy(
            neural_delta_fn, S_val, s0, k, r, dt, m, kv, ALPHA)

    # BS delta: 3 frequencies x 3 costs.
    for fq in freqs:
        def bs_fn(S, fq=fq):
            return bs_delta_hedge(S, k, r, sigma, t, m, fq)
        for kv, label in kappas:
            name = f"bs_f{fq}_{label}"
            rows[name] = {"test": evaluate_policy(bs_fn, S_test, s0, k, r, dt, m, kv, ALPHA)}
            rows[name]["val"] = evaluate_policy(bs_fn, S_val, s0, k, r, dt, m, kv, ALPHA)

    # ------------------------------------------------------------------
    # 5. Figures
    # ------------------------------------------------------------------
    out_dir = ROOT / "results" / "approach2" / "if"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_cvar"], name="train CVaR95"))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_cvar"], name="val CVaR95"))
    fig.update_layout(title="Deep-hedging training curve (CVaR95 of L=-V_T)",
                      xaxis_title="epoch", yaxis_title="CVaR95")
    fig.write_html(str(out_dir / "training_curve.html"), include_plotlyjs=True)

    # P&L distribution (test): neural @10bp vs a few BS baselines.
    with torch.no_grad():
        V_neural = -policy_loss(model, S_test, s0, k, r, dt, m, 0.001).detach().cpu().numpy()
        V_bs1_0 = hedge_pnl(S_test, bs_delta_hedge(S_test, k, r, sigma, t, m, 1), k, r, dt, m, 0.0)[0].numpy()
        V_bs1_10 = hedge_pnl(S_test, bs_delta_hedge(S_test, k, r, sigma, t, m, 1), k, r, dt, m, 0.001)[0].numpy()
        V_bs21_0 = hedge_pnl(S_test, bs_delta_hedge(S_test, k, r, sigma, t, m, 21), k, r, dt, m, 0.0)[0].numpy()

    fig = go.Figure()
    for arr, name in [(V_neural, "neural @10bp"), (V_bs1_0, "BS 1d @0bp"),
                      (V_bs1_10, "BS 1d @10bp"), (V_bs21_0, "BS 21d @0bp")]:
        fig.add_trace(go.Histogram(x=arr, name=name, histnorm="probability density",
                                   opacity=0.6, nbinsx=120))
    fig.update_layout(title="Test P&L distribution (V_T, premium excluded)", barmode="overlay",
                      xaxis_title="V_T", yaxis_title="density")
    fig.write_html(str(out_dir / "pnl_distribution.html"), include_plotlyjs=True)

    # CVaR95 bar chart across all methods (test).
    names = list(rows.keys())
    cvar_test = [rows[nn]["test"]["cvar95"] for nn in names]
    fig = go.Figure(go.Bar(x=names, y=cvar_test, marker_color=[
        "#1f77b4" if nn.startswith("neural") else "#ff7f0e" for nn in names]))
    fig.update_layout(title="Test CVaR95 of L = -V_T across strategies",
                      xaxis_title="strategy", yaxis_title="CVaR95")
    fig.write_html(str(out_dir / "benchmark_bars.html"), include_plotlyjs=True)

    # ------------------------------------------------------------------
    # 6. metrics.json
    # ------------------------------------------------------------------
    params = dict(p.to_dict())
    params.update({
        "seed": args.seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "hidden": list(args.hidden),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "train_kappa": args.train_kappa,
        "alpha": ALPHA,
        "device": args.device,
        "threads": args.threads,
        "shuffle_seed": args.seed,
        "n_epochs_run": n_epochs_run,
        "architecture": f"MLP 2->{'-'.join(map(str, args.hidden))}->1 (ReLU, <=3 hidden)",
        "payoff": "European vanilla call max(S_T-K,0), K=S0",
        "conventions": {
            "grid": "t_i = i*dt, i=0..m-1, m=252",
            "discount": "exp(r*(T-t_i)) on decision time t_i (assignment formula)",
            "cost": "kappa*|Delta_i-Delta_{i-1}|*S_{t_i}, Delta_{-1}=0, initial position charged",
            "premium": "excluded from V_T (constant shift, no effect on comparison)",
            "closeout": "none; final move i=m-1 realizes inside the sum",
        },
    })

    metrics = {
        "price_check.bs_price": float(bs_price),
        "price_check.torch_price": float(torch_price),
        "price_check.torch_se": float(torch_se),
        "price_check.numpy_price": float(np_price),
        "price_check.numpy_se": float(np_se),
        "price_check.z_bs": float(z_bs),
        "price_check.z_torch_vs_numpy": float(z_cross),
        "neural_best_val_cvar95": float(best_val_cvar),
        "neural_n_epochs": int(n_epochs_run),
    }
    for name, d in rows.items():
        for split in ("test", "val"):
            for stat in ("mean_pnl", "var_pnl", "cvar95", "turnover", "avg_cost"):
                metrics[f"{name}_{split}_{stat}"] = float(d[split][stat])

    mc.save_json({"params": params, "metrics": metrics,
                  "fit_seconds": float(fit_seconds), "history": history},
                 out_dir / "metrics.json")

    # ------------------------------------------------------------------
    # 7. Rich tables / verdicts
    # ------------------------------------------------------------------
    check_tbl = Table("check", "value", "SE", "z", "|z|<=3",
                      title="Vanilla call price cross-check (seed 42)")
    check_tbl.add_row("torch MC vs BS", f"{torch_price:.4f}", f"{torch_se:.4f}",
                      f"{z_bs:+.3f}", "PASS" if abs(z_bs) <= 3 else "FAIL")
    check_tbl.add_row("torch MC vs numpy MC", f"{torch_price:.4f}",
                      f"{math.sqrt(torch_se**2+np_se**2):.4f}", f"{z_cross:+.3f}",
                      "PASS" if abs(z_cross) <= 3 else "FAIL")
    console.print(check_tbl)

    summary = Table("strategy", "kappa", "mean P&L", "var P&L", "CVaR95", "turnover", "avg cost",
                    title="Neural vs BS (TEST paths)")
    for kv, label in kappas:
        d = rows[f"neural_{label}"]["test"]
        summary.add_row(f"neural (every step)", label, f"{d['mean_pnl']:.2f}",
                        f"{d['var_pnl']:.2f}", f"{d['cvar95']:.2f}",
                        f"{d['turnover']:.2f}", f"{d['avg_cost']:.2f}")
    for kv, label in kappas:
        d = rows[f"bs_f1_{label}"]["test"]
        summary.add_row("BS delta (1d)", label, f"{d['mean_pnl']:.2f}", f"{d['var_pnl']:.2f}",
                        f"{d['cvar95']:.2f}", f"{d['turnover']:.2f}", f"{d['avg_cost']:.2f}")
    console.print(summary)

    grid = Table("freq \\ kappa", "0bp", "10bp", "50bp", title="BS delta CVaR95 / mean / var (TEST)")
    for fq in freqs:
        cells = []
        for kv, label in kappas:
            d = rows[f"bs_f{fq}_{label}"]["test"]
            cells.append(f"CVaR {d['cvar95']:.1f}\nmean {d['mean_pnl']:.1f}\nvar {d['var_pnl']:.0f}")
        grid.add_row(f"{fq}d", *cells)
    console.print(grid)

    # PASS/FAIL verdicts (factual)
    verdicts = [
        ("torch MC vanilla price within 3 SE of Black-Scholes", abs(z_bs) <= 3),
        ("torch MC vanilla price within 3*sqrt(SE_np^2+SE_torch^2) of numpy MC", abs(z_cross) <= 3),
        ("training converged without NaN (finite best val CVaR)", math.isfinite(best_val_cvar)),
    ]
    # Only claim neural outperformance if the test numbers support it (lower CVaR95 = better).
    # Compare at MATCHED cost levels (neural every-step vs BS-1d rebalancing, same kappa).
    for kv, label in kappas:
        n_cvar = rows[f"neural_{label}"]["test"]["cvar95"]
        b_cvar = rows[f"bs_f1_{label}"]["test"]["cvar95"]
        verdicts.append(
            (f"neural CVaR95 {n_cvar:.2f} <= BS-1d CVaR95 {b_cvar:.2f} at kappa={label}",
             n_cvar <= b_cvar))

    vt = Table("check", "result", title="Verification verdicts")
    for msg, ok in verdicts:
        vt.add_row(msg, "[green]PASS[/]" if ok else "[red]FAIL[/]")
    console.print(vt)

    console.print(f"[bold]Outputs written to[/] {out_dir}/")


if __name__ == "__main__":
    main()
