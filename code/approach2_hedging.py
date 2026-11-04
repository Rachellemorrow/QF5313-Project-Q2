"""Question 2 — Option B: Deep hedging in PyTorch.

Owned by the deepseek-numerical subagent.  Linear pipeline:

    load model -> simulate train/val/test paths -> train CVaR-optimal MLP policy
    (per transaction-cost level) -> evaluate against Black-Scholes delta hedging
    (frequencies 1/5/21, costs 0/10/50 bp) and a zero-hedge baseline
    -> paired bootstrap CIs -> write outputs

Usage (from the repository root)::

    uv run python code/approach2_hedging.py --seed 42 --dataset if_main

Conventions (identical across neural and baseline policies)
----------------------------------------------------------
* We are short one European call ``H(S) = max(S_T - K, 0)`` and hedge with the
  underlying.  The terminal hedging P&L is the assignment-specified::

      V_T = sum_t exp(r (T - t)) * Delta_t * (S_{t+dt} - S_t)
            - kappa * sum_t |Delta_t - Delta_{t-1}| * S_t - H(S)

* Initial holding is zero (``Delta_{-1} = 0``), no premium is received, costs are
  *not* interest-accumulated, and there is *no* terminal closeout (the final
  position is not unwound).  Rebalancing happens on a fixed grid of ``freq``
  days; the position chosen at the last rebalancing time is held to maturity.
* Loss ``L = -V_T``; we minimize the 95% CVaR of ``L`` with the
  Rockafellar-Uryasev objective ``eta + mean(relu(L - eta)) / (1 - alpha)``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mc_common as mc  # noqa: E402
from mc_common import (Model, bs_call_delta, bs_call_price,  # noqa: E402
                       gbm_paths_from_z_np, mc_price_np, mc_price_torch,
                       write_metrics, results_dir)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()

FREQS = [1, 5, 21]
COSTS = [0.0, 0.001, 0.005]
ALPHA = 0.95


# ---------------------------------------------------------------------------
# P&L machinery (assignment formula)
# ---------------------------------------------------------------------------

def rebal_indices(steps: int, freq: int) -> np.ndarray:
    """Rebalancing step indices ``0, freq, 2*freq, ... < steps``."""
    return np.arange(0, steps, freq)


def hedging_pnl_np(S: np.ndarray, delta: np.ndarray, rebal_idx: np.ndarray,
                   kappa: float, model: Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Terminal hedging P&L (NumPy, vectorized).

    Returns ``(V, turnover, total_cost)`` of shape ``(B,)``.
    """
    idx = np.asarray(rebal_idx, dtype=int)
    next_idx = np.empty_like(idx)
    next_idx[:-1] = idx[1:]
    next_idx[-1] = model.steps
    S_tau = S[:, idx]           # (B, n_rebal)
    S_next = S[:, next_idx]     # (B, n_rebal)
    t = idx * model.dt
    w = np.exp(model.r * (model.T - t))[None, :]  # (1, n_rebal)
    gains = (w * delta * (S_next - S_tau)).sum(axis=1)

    prev = np.concatenate([np.zeros((S.shape[0], 1)), delta[:, :-1]], axis=1)
    trade = delta - prev
    turnover = np.abs(trade).sum(axis=1)
    costs = kappa * (np.abs(trade) * S_tau).sum(axis=1)

    H = np.maximum(S[:, -1] - model.K, 0.0)
    V = gains - costs - H
    return V, turnover, costs


def hedging_pnl_torch(S: "torch.Tensor", delta: "torch.Tensor",
                      rebal_idx: np.ndarray, kappa: float,
                      model: Model) -> "torch.Tensor":
    """Terminal hedging P&L (PyTorch, vectorized & differentiable in ``delta``).

    Returns ``V`` of shape ``(B,)`` (loss ``L = -V`` is minimized).
    """
    idx = torch.as_tensor(np.asarray(rebal_idx, dtype=int), dtype=torch.long,
                          device=S.device)
    n = idx.shape[0]
    next_idx = torch.empty(n, dtype=torch.long, device=S.device)
    next_idx[:-1] = idx[1:]
    next_idx[-1] = model.steps
    S_tau = S[:, idx]           # (B, n)
    S_next = S[:, next_idx]     # (B, n)
    t = idx.to(S.dtype) * model.dt
    w = torch.exp(model.r * (model.T - t)).unsqueeze(0)  # (1, n)
    gains = (w * delta * (S_next - S_tau)).sum(dim=1)

    prev = torch.cat([torch.zeros(S.shape[0], 1, dtype=S.dtype, device=S.device),
                      delta[:, :-1]], dim=1)
    trade = delta - prev
    costs = kappa * (trade.abs() * S_tau).sum(dim=1)

    H = torch.clamp(S[:, -1] - model.K, min=0.0)
    return gains - costs - H


def bs_delta_schedule(S: np.ndarray, rebal_idx: np.ndarray, model: Model) -> np.ndarray:
    """Black-Scholes delta at each rebalancing time."""
    S_tau = S[:, rebal_idx]
    tau = rebal_idx * model.dt
    remaining = model.T - tau
    d1 = (np.log(S_tau / model.K) + (model.r + 0.5 * model.sigma ** 2) * remaining) \
        / (model.sigma * np.sqrt(remaining))
    return norm.cdf(d1)


def empirical_cvar(losses: np.ndarray, alpha: float = ALPHA) -> Tuple[float, float]:
    """Fractional empirical VaR and CVaR of the *upper* tail of ``losses``."""
    x = np.sort(np.asarray(losses, dtype=float))
    if not 0 < alpha < 1 or x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("CVaR requires nonempty finite losses and 0 < alpha < 1")
    n = x.shape[0]
    k = int(math.ceil(alpha * n))
    var_ = x[k - 1]
    cvar = ((k - alpha * n) * x[k - 1] + x[k:].sum()) / (n * (1.0 - alpha))
    return float(var_), float(cvar)


def loss_metrics(losses: np.ndarray) -> dict:
    """VaR/CVaR (upper tail), variance and expectation of ``-V`` losses."""
    var_, cvar = empirical_cvar(losses)
    return {
        "cvar95": cvar,
        "var95": var_,
        "pnl_variance": float(losses.var(ddof=1)),
        "expected_pnl": float(-losses.mean()),  # E[V_T] = -E[L]
        "expected_loss": float(losses.mean()),
    }


# ---------------------------------------------------------------------------
# Neural policy
# ---------------------------------------------------------------------------

class PolicyMLP(nn.Module):
    """Delta_t = f_theta(S_t / S_0, t/T) with two hidden layers (<= 32 units)."""

    def __init__(self, hidden: int = 32, init_delta: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(init_delta)
            self.net[-1].weight.fill_(0.0)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (..., 2) -> (...,)
        return self.net(x).squeeze(-1)


def _policy_inputs(S: "torch.Tensor", model: Model, rebal_idx: np.ndarray) -> "torch.Tensor":
    """(B, n_rebal, 2) features ``[S_t/S0, t/T]``."""
    t = torch.as_tensor(rebal_idx * model.dt / model.T, dtype=S.dtype, device=S.device)
    x1 = S[:, rebal_idx] / model.S0
    x2 = t.unsqueeze(0).expand(x1.shape[0], -1)
    return torch.stack([x1, x2], dim=-1)


def neural_delta(policy: PolicyMLP, S: "torch.Tensor", model: Model,
                 rebal_idx: np.ndarray) -> "torch.Tensor":
    return policy(_policy_inputs(S, model, rebal_idx))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def simulate_dataset(model: Model, n_paths: int, seed: int,
                     dtype: np.dtype = np.float32) -> np.ndarray:
    """Simulate ``(n_paths, steps+1)`` GBM paths with a chunked NumPy RNG."""
    S = np.empty((n_paths, model.steps + 1), dtype=dtype)
    rng = np.random.default_rng(seed)
    chunk = 1 << 15
    done = 0
    while done < n_paths:
        c = min(chunk, n_paths - done)
        Z = rng.standard_normal((c, model.steps))
        Sc = gbm_paths_from_z_np(model, Z)
        S[done:done + c] = Sc.astype(dtype)
        done += c
    return S


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_policy(model: Model, S_train: "torch.Tensor", S_val: "torch.Tensor",
                 kappa: float, seed: int, args) -> dict:
    torch.manual_seed(seed)
    policy = PolicyMLP(args.hidden, init_delta=bs_call_delta(
        model.S0, model.K, model.r, model.sigma, model.T))
    rebal_idx = rebal_indices(model.steps, 1)  # daily training grid

    # Initialise the quantile level eta at the empirical 95%-VaR of the initial
    # (constant-delta) policy so the Rockafellar-Uryasev objective targets the
    # tail from the first step rather than degenerating to expected loss.
    with torch.no_grad():
        S_init = S_train[:min(S_train.shape[0], 8192)]
        d_init = neural_delta(policy, S_init, model, rebal_idx)
        V_init = hedging_pnl_torch(S_init, d_init, rebal_idx, kappa, model)
        init_eta = float(np.quantile((-V_init).numpy(), ALPHA))
    eta = nn.Parameter(torch.tensor(init_eta, dtype=torch.float32))
    # separate optimizers: the quantile level eta needs a much larger learning
    # rate than the network weights to track the (unknown) VaR level.
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    optimizer_eta = torch.optim.Adam([eta], lr=args.lr_eta)

    n_train = S_train.shape[0]
    n_batch = args.batch_size
    alpha = ALPHA
    inv_tail = 1.0 / (1.0 - alpha)

    best_val = float("inf")
    best_state = None
    patience_left = args.patience
    history = {"train_loss": [], "val_cvar": [], "val_var": [], "eta": [],
               "grad_norm": []}

    for epoch in range(args.epochs):
        policy.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_steps = 0
        for i in range(0, n_train, n_batch):
            idx = perm[i:i + n_batch]
            S = S_train[idx]
            delta = neural_delta(policy, S, model, rebal_idx)
            V = hedging_pnl_torch(S, delta, rebal_idx, kappa, model)
            L = -V
            loss = eta + inv_tail * torch.relu(L - eta).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            optimizer.zero_grad()
            optimizer_eta.zero_grad()
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in [*policy.parameters(), eta]):
                raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            optimizer_eta.step()
            epoch_loss += loss.item() * idx.shape[0]
            n_steps += idx.shape[0]

        # gradient norm (policy parameters only)
        grad_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                grad_norm += float((p.grad ** 2).sum())
        grad_norm = math.sqrt(grad_norm)

        # validation (minibatched, independent empirical CVaR)
        val_losses = []
        with torch.no_grad():
            for i in range(0, S_val.shape[0], n_batch):
                Sv = S_val[i:i + n_batch]
                dv = neural_delta(policy, Sv, model, rebal_idx)
                Vv = hedging_pnl_torch(Sv, dv, rebal_idx, kappa, model)
                val_losses.append((-Vv).detach().numpy())
        val_losses = np.concatenate(val_losses)
        var_, cvar = empirical_cvar(val_losses, alpha)

        history["train_loss"].append(epoch_loss / n_steps)
        history["val_cvar"].append(cvar)
        history["val_var"].append(var_)
        history["eta"].append(float(eta.item()))
        history["grad_norm"].append(grad_norm)
        console.print(f"  k={kappa} epoch={epoch+1} train_RU={history['train_loss'][-1]:.5f} val_CVaR={cvar:.5f}")

        if cvar < best_val:
            best_val = cvar
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            best_eta = float(eta.item())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # restore best
    policy.load_state_dict(best_state)
    destination = results_dir("approach2", args.dataset)
    torch.save({"state_dict": best_state, "eta": best_eta, "model": model.asdict(),
                "hidden": args.hidden, "seed": seed}, destination / f"policy_k{kappa}.pt")
    (destination / f"history_k{kappa}.json").write_text(json.dumps(history, indent=2, allow_nan=False))
    return {
        "kappa": kappa,
        "epochs_run": len(history["train_loss"]),
        "best_val_cvar": best_val,
        "best_eta": best_eta,
        "history": history,
        "policy": policy,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bs(model: Model, S_test: np.ndarray) -> Dict:
    out: Dict[str, dict] = {}
    for f in FREQS:
        ridx = rebal_indices(model.steps, f)
        delta = bs_delta_schedule(S_test, ridx, model)
        out[str(f)] = {}
        for k in COSTS:
            V, turnover, costs = hedging_pnl_np(S_test, delta, ridx, k, model)
            m = loss_metrics(-V)
            m["turnover"] = float(turnover.mean())
            m["avg_cost"] = float(costs.mean())
            out[str(f)][str(k)] = m
    return out


def evaluate_neural(model: Model, policy: PolicyMLP, S_test: "torch.Tensor",
                    S_test_np: np.ndarray) -> Dict:
    out: Dict[str, dict] = {}
    with torch.no_grad():
        for f in FREQS:
            ridx = rebal_indices(model.steps, f)
            delta = neural_delta(policy, S_test, model, ridx)
            out[str(f)] = {}
            for k in COSTS:
                V, turnover, costsum = hedging_pnl_np(S_test_np, delta.detach().numpy(),
                                                      ridx, k, model)
                losses = -V
                m = loss_metrics(losses)
                m["turnover"] = float(turnover.mean())
                m["avg_cost"] = float(costsum.mean())
                out[str(f)][str(k)] = m
    return out


def paired_cvar_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, alpha: float,
                          n_boot: int, seed: int) -> dict:
    """Bootstrap CI of ``CVaR(loss_b) - CVaR(loss_a)`` on common paths."""
    n = loss_a.shape[0]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        ii = rng.integers(0, n, size=n)
        _, cva = empirical_cvar(loss_a[ii], alpha)
        _, cvb = empirical_cvar(loss_b[ii], alpha)
        diffs[b] = cvb - cva
    return {
        "point_diff": float(empirical_cvar(loss_b, alpha)[1]-empirical_cvar(loss_a, alpha)[1]),
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_html(fig) -> str:
    import plotly.io as pio
    return pio.to_html(fig, include_plotlyjs=True, full_html=True)


def training_figure(trainings: Dict[float, dict]) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=len(COSTS), cols=1, shared_xaxes=True,
                        subplot_titles=[f"kappa = {k}" for k in COSTS])
    for r, k in enumerate(COSTS, start=1):
        h = trainings[k]["history"]
        ep = list(range(1, len(h["train_loss"]) + 1))
        fig.add_trace(go.Scatter(x=ep, y=h["train_loss"], name=f"train loss (k={k})",
                                 mode="lines"), row=r, col=1)
        fig.add_trace(go.Scatter(x=ep, y=h["val_cvar"], name=f"val CVaR (k={k})",
                                 mode="lines"), row=r, col=1)
    fig.update_layout(title="Training: RU loss and validation CVaR")
    return _fig_html(fig)


def comparison_figure(bs: Dict, neural: Dict, zero: Dict) -> str:
    import plotly.graph_objects as go
    cats = []
    bs_c, nn_c, zero_c = [], [], []
    for f in FREQS:
        for k in COSTS:
            cats.append(f"f={f},k={k}")
            bs_c.append(bs[str(f)][str(k)]["cvar95"])
            # neural policy trained at k, evaluated at matched k
            nn_c.append(neural[str(k)][str(f)][str(k)]["cvar95"])
            zero_c.append(zero["cvar95"])
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Black-Scholes delta", x=cats, y=bs_c))
    fig.add_trace(go.Bar(name="Neural (CVaR)", x=cats, y=nn_c))
    fig.add_trace(go.Bar(name="Zero hedge", x=cats, y=zero_c))
    fig.update_layout(title="CVaR(95%) by rebalancing frequency and cost",
                      barmode="group")
    return _fig_html(fig)


def histogram_figure(bs_loss: np.ndarray, nn_loss: np.ndarray, kappa: float) -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bs_loss, name="BS delta", opacity=0.6,
                               histnorm="probability density"))
    fig.add_trace(go.Histogram(x=nn_loss, name="Neural", opacity=0.6,
                               histnorm="probability density"))
    fig.update_layout(title=f"Loss distribution (freq=1, kappa={kappa})")
    return _fig_html(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q2 Option B — Deep hedging (PyTorch)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, default="if_main")
    p.add_argument("--n-train", type=int, default=1 << 17)
    p.add_argument("--n-val", type=int, default=1 << 14)
    p.add_argument("--n-test", type=int, default=1 << 14)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-eta", type=float, default=1.0,
                   help="learning rate for the CVaR quantile level eta")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--threads", type=int, default=1, help="CPU threads; one avoids oversubscription for small MLPs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.n_train, args.n_val, args.n_test, args.batch_size, args.epochs, args.patience, args.n_boot) <= 0:
        raise ValueError("Path counts and training settings must be positive")
    torch.set_num_threads(args.threads)
    model = mc.load_model()
    out_dir = results_dir("approach2", args.dataset)
    t_start = time.time()

    console.print(Panel(f"[bold]Q2 Option B — Deep hedging[/bold]\n"
                        f"dataset={args.dataset} seed={args.seed} threads={args.threads}\n"
                        f"model={model.asdict()}"))

    # ---- 0. engine agreement check (vanilla call) --------------------------
    agree_np = mc_price_np(model, 1 << 15, args.seed)
    agree_torch = mc_price_torch(model, 1 << 15, args.seed)
    z_agree = (agree_np["european"]["price"] - agree_torch["european"]["price"]) \
        / agree_np["european"]["se"]
    console.print(f"NumPy/PyTorch agreement (identical shocks): "
                  f"numpy={agree_np['european']['price']:.5f} "
                  f"torch={agree_torch['european']['price']:.5f} z={z_agree:.3f}")

    # ---- 1. datasets -------------------------------------------------------
    console.print("Simulating datasets ...")
    S_train = torch.from_numpy(simulate_dataset(model, args.n_train, args.seed))
    S_val = torch.from_numpy(simulate_dataset(model, args.n_val, args.seed + 10))
    S_test_np = simulate_dataset(model, args.n_test, args.seed + 20, dtype=np.float64)
    S_test = torch.from_numpy(S_test_np.astype(np.float32))

    # ---- 2. train cost-specific policies -----------------------------------
    trainings: Dict[float, dict] = {}
    for i, k in enumerate(COSTS):
        console.print(f"Training policy for kappa={k} ...")
        res = train_policy(model, S_train, S_val, k, args.seed + 30 + i, args)
        trainings[k] = res
        console.print(f"  kappa={k}: epochs={res['epochs_run']} "
                      f"best_val_cvar={res['best_val_cvar']:.4f} eta={res['best_eta']:.4f}")

    # ---- 3. baselines ------------------------------------------------------
    bs = evaluate_bs(model, S_test_np)
    zero_loss = np.maximum(S_test_np[:, -1] - model.K, 0.0)
    zero = loss_metrics(zero_loss)
    zero["turnover"] = 0.0
    zero["avg_cost"] = 0.0

    # ---- 4. neural evaluation ----------------------------------------------
    neural: Dict[str, dict] = {}
    for k in COSTS:
        neural[str(k)] = evaluate_neural(model, trainings[k]["policy"], S_test, S_test_np)

    # ---- 5. paired bootstrap CIs (neural - BS) -----------------------------
    bootstrap_cis: Dict[str, dict] = {}
    for f in FREQS:
        for k in COSTS:
            bs_loss = _bs_losses(model, S_test_np, f, k)
            nn_loss = _neural_losses(model, trainings[k]["policy"], S_test, f, k, S_test_np)
            bootstrap_cis[f"f{f}_k{k}"] = paired_cvar_bootstrap(
                bs_loss, nn_loss, ALPHA, args.n_boot, args.seed + 1000 + f * 10 + int(k * 1000))

    # ---- 6. figures --------------------------------------------------------
    fig_train = training_figure(trainings)
    fig_comp = comparison_figure(bs, neural, zero)
    bs_loss_f1 = _bs_losses(model, S_test_np, 1, 0.001)
    nn_loss_f1 = _neural_losses(model, trainings[0.001]["policy"], S_test, 1, 0.001, S_test_np)
    fig_hist = histogram_figure(bs_loss_f1, nn_loss_f1, 0.001)
    (out_dir / "training.html").write_text(fig_train, encoding="utf-8")
    (out_dir / "comparison.html").write_text(fig_comp, encoding="utf-8")
    (out_dir / "loss_histogram.html").write_text(fig_hist, encoding="utf-8")

    # ---- 7. metrics --------------------------------------------------------
    fit_seconds = time.time() - t_start
    metrics = {
        "engine_agreement": {
            "numpy": agree_np["european"]["price"],
            "torch": agree_torch["european"]["price"],
            "z_score": z_agree,
        },
        "zero_hedge": zero,
        "bs": bs,
        "neural": neural,
        "bootstrap_cis_neural_minus_bs": bootstrap_cis,
        "training": {str(k): {kk: vv for kk, vv in v.items() if kk != "history"
                              and kk != "policy"}
                     for k, v in trainings.items()},
    }
    history = {
        "training_curves": {str(k): v["history"] for k, v in trainings.items()},
    }
    params = {"model": model.asdict(), "seed": args.seed, "dataset": args.dataset,
              "n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test,
              "batch_size": args.batch_size, "epochs": args.epochs,
              "patience": args.patience, "lr": args.lr, "lr_eta": args.lr_eta,
              "hidden": args.hidden,
              "alpha": ALPHA, "freqs": FREQS, "costs": COSTS,
               "threads": args.threads}
    params["seeds"] = {"train": args.seed, "validation": args.seed+10, "test": args.seed+20,
                       "initialization": [args.seed+30+i for i in range(len(COSTS))]}
    params["meets_minimum_training_paths"] = args.n_train >= 2**17
    params["numerics"] = "CPU float32 MLP/training, float64 test paths and common NumPy P&L for tables/bootstrap; CPU thread count recorded"
    params["architecture"] = [2, args.hidden, args.hidden, 1]
    params["accounting"] = "Assignment gains exp(r*(T-t)); costs unaccumulated; zero premium, initial holding zero, no terminal liquidation"
    write_metrics(out_dir / "metrics.json", params, metrics, fit_seconds, history)

    # ---- summary table -----------------------------------------------------
    t = Table(title="CVaR(95%) — BS vs Neural (matched cost) vs Zero")
    t.add_column("freq"); t.add_column("kappa")
    t.add_column("BS"); t.add_column("Neural"); t.add_column("Zero")
    for f in FREQS:
        for k in COSTS:
            t.add_row(str(f), str(k),
                      f"{bs[str(f)][str(k)]['cvar95']:.3f}",
                      f"{neural[str(k)][str(f)][str(k)]['cvar95']:.3f}",
                      f"{zero['cvar95']:.3f}")
    console.print(t)
    console.print(f"[green]Wrote[/green] {out_dir / 'metrics.json'} "
                  f"(fit_seconds={fit_seconds:.2f})")


def _bs_losses(model: Model, S_test: np.ndarray, f: int, k: float) -> np.ndarray:
    ridx = rebal_indices(model.steps, f)
    delta = bs_delta_schedule(S_test, ridx, model)
    V, _, _ = hedging_pnl_np(S_test, delta, ridx, k, model)
    return -V


def _neural_losses(model: Model, policy: PolicyMLP, S_test: "torch.Tensor",
                    f: int, k: float, S_test_np: np.ndarray = None) -> np.ndarray:
    ridx = rebal_indices(model.steps, f)
    with torch.no_grad():
        delta = neural_delta(policy, S_test, model, ridx)
        prices = S_test.numpy().astype(np.float64) if S_test_np is None else S_test_np
        V, _, _ = hedging_pnl_np(prices, delta.numpy(), ridx, k, model)
    return -V


if __name__ == "__main__":
    main()
