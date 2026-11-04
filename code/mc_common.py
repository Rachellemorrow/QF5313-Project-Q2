"""Shared Monte Carlo utilities for Question 2 (Monte Carlo pricing & hedging).

Owned by the deepseek-numerical subagent.  This module is imported by
``approach1_pricing.py``, ``approach2_hedging.py`` and ``tests/test_numerical.py``.

Conventions
-----------
* The model is the assignment-prescribed *risk-neutral* geometric Brownian motion
  with drift ``r``::

      S_{t_{i+1}} = S_{t_i} * exp[(r - 0.5*sigma^2) * dt + sigma*sqrt(dt) * Z_i]

  This is explicitly *not* the Black-76 (zero-drift) futures convention; it is
  the model requested by the assignment for the Black-Scholes validation.
* Times ``t_0=0, t_1=dt, ..., t_m=T`` with ``m = steps``.  All path-dependent
  payoffs monitor ``t_1..t_m`` (the initial point ``t_0`` is *excluded*).
* Discounting uses the continuous rate ``r``:  price = exp(-r T) * E[payoff].
* Every stochastic component is seeded; pass an explicit ``seed``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

try:  # torch is optional for NumPy-only consumers (tests, quick checks)
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is a hard dependency of the assignment
    torch = None
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Model & repository helpers
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Calibrated market parameters for one dataset.

    Defaults follow the shared lead/subagent contract: ``K = S0`` and
    ``B = 1.2 * S0``; ``T = 1`` year with ``steps = 252`` monitoring points.
    """

    S0: float = 100.0
    K: float = 100.0
    B: float = 120.0
    r: float = 0.03
    sigma: float = 0.20
    mu: float = 0.05
    T: float = 1.0
    steps: int = 252

    @property
    def dt(self) -> float:
        return self.T / self.steps

    @property
    def sqrt_dt(self) -> float:
        return math.sqrt(self.dt)

    def asdict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


DEFAULT_MODEL = Model()


def repo_root() -> Path:
    """Repository root (parent of the ``code`` directory)."""
    return Path(__file__).resolve().parents[1]


def calibration_path() -> Path:
    return repo_root() / "data" / "clean" / "calibration.json"


def validate_model(model: Model) -> Model:
    """Validate a :class:`Model`; raise :class:`ValueError` on non-physical input.

    Rejects non-positive ``S0/K/B``, non-positive ``sigma``/``T``, non-finite
    rates, and non-positive-integer ``steps``.  This is deliberately strict:
    degenerate ``T=0`` / ``sigma=0`` cases are *not* silently accepted.
    """
    def _pos(name: str, v: float) -> None:
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"Model.{name} must be finite and > 0, got {v!r}")
    _pos("S0", model.S0)
    _pos("K", model.K)
    _pos("B", model.B)
    _pos("sigma", model.sigma)
    _pos("T", model.T)
    if not math.isfinite(model.r):
        raise ValueError(f"Model.r must be finite, got {model.r!r}")
    if not math.isfinite(model.mu):
        raise ValueError(f"Model.mu must be finite, got {model.mu!r}")
    if isinstance(model.steps, bool) or not isinstance(model.steps, int) or model.steps < 1:
        raise ValueError(f"Model.steps must be a positive integer, got {model.steps!r}")
    return model


def load_model(path: Optional[os.PathLike] = None) -> Model:
    """Load and validate a :class:`Model` from ``data/clean/calibration.json``.

    The lead agent owns the calibration file with keys
    ``S0, K, B, r, sigma, mu, T, steps``.  Unlike earlier versions this *errors*
    rather than silently falling back to defaults when the file is absent,
    missing keys, or contains invalid parameters.
    """
    p = Path(path) if path is not None else calibration_path()
    if not p.exists():
        raise FileNotFoundError(
            f"calibration file not found: {p}. Provide data/clean/calibration.json "
            f"(keys: {sorted(f.name for f in fields(Model))}).")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration file {p} is not valid JSON: {exc}") from exc
    required = {f.name for f in fields(Model)}
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f"calibration file {p} missing required keys: {missing}")
    model = Model(**{k: data[k] for k in required})
    return validate_model(model)


def results_dir(approach: str, dataset: str) -> Path:
    """``results/<approach>/<dataset>/``, created on demand."""
    d = repo_root() / "results" / approach / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_jsonable(obj):
    """Recursively convert numpy/torch scalars & arrays to plain JSON types.

    Arrays are converted via ``tolist()`` *before* any ``item()`` attempt so
    that multi-element ndarrays are handled correctly (ndarray also exposes a
    size-1-only ``item()`` that would otherwise raise).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {_json_key(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):  # numpy scalar
        return _to_jsonable(obj.item())
    if hasattr(obj, "tolist"):  # torch tensors and other array-likes
        return _to_jsonable(obj.tolist())
    if hasattr(obj, "item"):  # torch scalar etc.
        try:
            return _to_jsonable(obj.item())
        except (ValueError, TypeError):
            return str(obj)
    return str(obj)


def _json_key(k):
    return k if isinstance(k, str) else str(k)


def write_metrics(path: Path, params: dict, metrics: dict, fit_seconds: float,
                  history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": _to_jsonable(params),
        "metrics": _to_jsonable(metrics),
        "fit_seconds": float(fit_seconds),
        "history": _to_jsonable(history),
    }
    # allow_nan=False -> the file is strict JSON and any NaN/Inf in the results
    # raises loudly instead of being silently serialized as invalid tokens.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Exact GBM simulation
# ---------------------------------------------------------------------------

def simulate_gbm_np(model: Model, n_paths: int, seed: int,
                    antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate GBM paths with NumPy.

    Returns ``(S, Z)`` where ``S`` has shape ``(n_paths, steps + 1)`` and ``Z``
    is the underlying standard-normal shock matrix ``(n_paths, steps)``.

    With ``antithetic=True`` the shocks are generated as ``n_paths/2`` pairs
    ``(Z, -Z)`` so the total number of simulated paths is still ``n_paths``
    (equal path budget).
    """
    rng = np.random.default_rng(seed)
    steps = model.steps
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError(
                f"antithetic simulation requires an even path count, got {n_paths}")
        half = n_paths // 2
        Z = rng.standard_normal((half, steps))
        Z = np.concatenate([Z, -Z], axis=0)
    else:
        Z = rng.standard_normal((n_paths, steps))
    S = gbm_paths_from_z_np(model, Z)
    return S, Z


def gbm_paths_from_z_np(model: Model, Z: np.ndarray) -> np.ndarray:
    """Vectorized GBM path construction from a shock matrix ``Z`` (n, steps)."""
    validate_model(model)
    Z = np.asarray(Z, dtype=float)
    if Z.shape[1] != model.steps:
        raise ValueError(
            f"Z has {Z.shape[1]} steps but model.steps = {model.steps}")
    drift = (model.r - 0.5 * model.sigma ** 2) * model.dt
    vol = model.sigma * model.sqrt_dt
    log_inc = drift + vol * Z
    cum = np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(log_inc, axis=1)], axis=1)
    log_S = math.log(model.S0) + cum
    return np.exp(log_S)


def gbm_paths_from_z_torch(model: Model, Z: np.ndarray) -> "torch.Tensor":
    """GBM path construction in PyTorch from the *same* shock matrix ``Z``.

    Using an identical ``Z`` (produced by the NumPy RNG) makes the two engines
    agree up to floating-point precision rather than merely up to statistical
    error.
    """
    assert torch is not None, "torch is required for the PyTorch engine"
    Zt = torch.as_tensor(np.asarray(Z, dtype=np.float64), dtype=torch.float64)
    drift = (model.r - 0.5 * model.sigma ** 2) * model.dt
    vol = model.sigma * model.sqrt_dt
    log_inc = drift + vol * Zt
    cum = torch.cat([torch.zeros((Zt.shape[0], 1), dtype=torch.float64),
                     torch.cumsum(log_inc, dim=1)], dim=1)
    log_S = math.log(model.S0) + cum
    return torch.exp(log_S)


def simulate_gbm_loop(S0: float, r: float, sigma: float, dt: float,
                      steps: int, Z: np.ndarray) -> np.ndarray:
    """Naive Python-loop GBM simulation (used only for the profiling baseline)."""
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    S = np.empty((n, steps + 1))
    for i in range(n):
        s = S0
        S[i, 0] = s
        row = Z[i]
        for j in range(steps):
            s = s * math.exp(drift + vol * row[j])
            S[i, j + 1] = s
    return S


# ---------------------------------------------------------------------------
# Payoffs (all monitor t_1 .. t_m, excluding the initial point)
# ---------------------------------------------------------------------------

def _monitored(S) -> np.ndarray:
    """Slice of S restricted to monitored times t_1..t_m."""
    return S[:, 1:] if S.ndim == 2 else S[1:]


def payoff_european_call(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S[:, -1] - K, 0.0)


def payoff_asian_call(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(_monitored(S).mean(axis=1) - K, 0.0)


def payoff_geometric_asian(S: np.ndarray, K: float) -> np.ndarray:
    g = np.exp(np.mean(np.log(_monitored(S)), axis=1))
    return np.maximum(g - K, 0.0)


def payoff_up_and_out(S: np.ndarray, K: float, B: float) -> np.ndarray:
    hit = _monitored(S).max(axis=1) >= B
    payoff = np.maximum(S[:, -1] - K, 0.0)
    return np.where(hit, 0.0, payoff)


def payoff_digital(S: np.ndarray, K: float) -> np.ndarray:
    return (S[:, -1] > K).astype(float)


def _torch_monitored(S: "torch.Tensor") -> "torch.Tensor":
    return S[:, 1:] if S.dim() == 2 else S[1:]


def payoff_european_call_torch(S: "torch.Tensor", K: float) -> "torch.Tensor":
    return torch.clamp(S[:, -1] - K, min=0.0)


def payoff_asian_call_torch(S: "torch.Tensor", K: float) -> "torch.Tensor":
    return torch.clamp(_torch_monitored(S).mean(dim=1) - K, min=0.0)


def payoff_geometric_asian_torch(S: "torch.Tensor", K: float) -> "torch.Tensor":
    g = torch.exp(torch.mean(torch.log(_torch_monitored(S)), dim=1))
    return torch.clamp(g - K, min=0.0)


def payoff_up_and_out_torch(S: "torch.Tensor", K: float, B: float) -> "torch.Tensor":
    hit = _torch_monitored(S).max(dim=1).values >= B
    payoff = torch.clamp(S[:, -1] - K, min=0.0)
    return torch.where(hit, torch.zeros_like(payoff), payoff)


def payoff_digital_torch(S: "torch.Tensor", K: float) -> "torch.Tensor":
    return (S[:, -1] > K).to(torch.float64)


PAYOFFS = {
    "european": payoff_european_call,
    "asian": payoff_asian_call,
    "barrier": payoff_up_and_out,
    "digital": payoff_digital,
}

PAYOFFS_TORCH = {
    "european": payoff_european_call_torch,
    "asian": payoff_asian_call_torch,
    "barrier": payoff_up_and_out_torch,
    "digital": payoff_digital_torch,
}


def mc_price_np(model: Model, n_paths: int, seed: int,
                antithetic: bool = False) -> Dict[str, dict]:
    """Price all four payoffs with the NumPy engine.

    Returns a dict ``name -> {"price", "se", "n"}``.  ``antithetic`` prices use
    the same total path budget as the naive run (``n_paths`` paths).
    """
    S, _ = simulate_gbm_np(model, n_paths, seed, antithetic=antithetic)
    disc = math.exp(-model.r * model.T)
    out: Dict[str, dict] = {}
    n_eff = S.shape[0] if not antithetic else S.shape[0] // 2
    for name, fn in PAYOFFS.items():
        if name == "barrier":
            vals = fn(S, model.K, model.B)
        else:
            vals = fn(S, model.K)
        if antithetic:
            vals = (vals[:n_eff] + vals[n_eff:2 * n_eff]) / 2.0
        out[name] = {
            "price": float(disc * vals.mean()),
            "se": float(disc * vals.std(ddof=1) / math.sqrt(vals.shape[0])),
            "n": int(vals.shape[0]),
        }
    return out


def mc_price_torch(model: Model, n_paths: int, seed: int,
                   antithetic: bool = False) -> Dict[str, dict]:
    """Price all four payoffs with the PyTorch engine (identical shocks)."""
    _, Z = simulate_gbm_np(model, n_paths, seed, antithetic=antithetic)
    S = gbm_paths_from_z_torch(model, Z)
    disc = math.exp(-model.r * model.T)
    out: Dict[str, dict] = {}
    n_eff = S.shape[0] if not antithetic else S.shape[0] // 2
    for name, fn in PAYOFFS_TORCH.items():
        if name == "barrier":
            vals = fn(S, model.K, model.B)
        else:
            vals = fn(S, model.K)
        if antithetic:
            vals = (vals[:n_eff] + vals[n_eff:2 * n_eff]) / 2.0
        arr = vals.detach().numpy()
        out[name] = {
            "price": float(disc * arr.mean()),
            "se": float(disc * arr.std(ddof=1) / math.sqrt(arr.shape[0])),
            "n": int(arr.shape[0]),
        }
    return out


# ---------------------------------------------------------------------------
# Black-Scholes references (European vanilla call)
# ---------------------------------------------------------------------------

def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    if sigma <= 0:
        raise ValueError(f"Black-Scholes requires sigma > 0, got {sigma}")
    if T <= 0:
        raise ValueError(f"Black-Scholes requires T > 0, got {T}")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    return float(norm.cdf(d1))


def bs_call_vega(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    return float(S0 * math.sqrt(T) * norm.pdf(d1))


def bs_call_delta_at(S_t: float, K: float, r: float, sigma: float, tau: float) -> float:
    """Black-Scholes delta of a vanilla call with time-to-maturity ``tau``."""
    if tau <= 0:
        return float(S_t > K)
    d1 = (math.log(S_t / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))
    return float(norm.cdf(d1))


# ---------------------------------------------------------------------------
# Discrete geometric-Asian closed form
# ---------------------------------------------------------------------------

def geometric_asian_price(S0: float, K: float, r: float, sigma: float,
                          T: float, steps: int) -> Tuple[float, float, float]:
    """Discounted price of a geometric-Asian call for the *discrete* schedule
    ``t_1..t_m`` (m = steps, equally spaced, initial point excluded).

    Returns ``(price, mu_G, var_G)`` where ``mu_G = E[ln G]`` and
    ``var_G = Var[ln G]``.
    """
    if sigma <= 0:
        raise ValueError(f"geometric-Asian closed form requires sigma > 0, got {sigma}")
    if T <= 0:
        raise ValueError(f"geometric-Asian closed form requires T > 0, got {T}")
    if steps < 1:
        raise ValueError(f"geometric-Asian closed form requires steps >= 1, got {steps}")
    m = steps
    mu_G = math.log(S0) + (r - 0.5 * sigma ** 2) * T * (m + 1) / (2 * m)
    var_G = sigma ** 2 * T * (m + 1) * (2 * m + 1) / (6 * m * m)
    s_G = math.sqrt(var_G)
    E_G = math.exp(mu_G + 0.5 * var_G)
    d1 = (mu_G + var_G - math.log(K)) / s_G
    d2 = d1 - s_G
    price = math.exp(-r * T) * (E_G * norm.cdf(d1) - K * norm.cdf(d2))
    return float(price), float(mu_G), float(var_G)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def se_of(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.shape[0] < 2:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(x.shape[0]))


def variance_reduction_ratio(naive: np.ndarray, reduced: np.ndarray,
                             n_boot: int = 2000, seed: int = 0,
                             paired: bool = True, scale: float = 1.0) -> dict:
    """Variance-reduction ratio ``Var(naive) / (scale * Var(reduced))`` with a
    bootstrap CI.

    ``scale`` accounts for estimators that consume a different path budget.  For
    a control variate, ``naive`` and ``reduced`` use the same N paths and are
    aligned per-path (``paired=True``, ``scale=1``).  For antithetic sampling
    each reduced value is an average of a ``(Z, -Z)`` pair that consumes two
    paths, so the equal-path-budget ratio uses ``scale=2`` and the bootstrap
    resamples the two arrays independently (``paired=False``).
    """
    naive = np.asarray(naive, dtype=float)
    reduced = np.asarray(reduced, dtype=float)
    vn = naive.var(ddof=1)
    vr = reduced.var(ddof=1)
    if not (vn > 0 and vr > 0):
        raise ValueError(
            f"variance-reduction ratio needs positive variances, got naive={vn}, reduced={vr}")
    ratio = vn / (scale * vr)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        if paired:
            ii = rng.integers(0, naive.shape[0], size=naive.shape[0])
            vnb = naive[ii].var(ddof=1)
            vrb = reduced[ii].var(ddof=1)
        else:
            vnb = naive[rng.integers(0, naive.shape[0], naive.shape[0])].var(ddof=1)
            vrb = reduced[rng.integers(0, reduced.shape[0], reduced.shape[0])].var(ddof=1)
        boots[b] = vnb / (scale * vrb) if vrb > 0 else np.nan
    boots = boots[np.isfinite(boots)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots.size else (float("nan"), float("nan"))
    return {
        "ratio": float(ratio),
        "naive_var": float(vn),
        "reduced_var": float(vr),
        "scale": float(scale),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
    }


def paired_bootstrap_se(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                        seed: int = 0) -> Tuple[float, float]:
    """Paired-bootstrap standard error of ``mean(a) - mean(b)``.

    ``a`` and ``b`` are aligned per-path contributions.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    boots = np.empty(n_boot)
    for k in range(n_boot):
        ii = rng.integers(0, n, size=n)
        boots[k] = diff[ii].mean()
    return float(diff.mean()), float(boots.std(ddof=1))


def spawn_seed_grid(base_seed: int, n_levels: int, repeats: int,
                    n_methods: int) -> List[List[List[object]]]:
    """Spawn disjoint RNG seeds via ``numpy.random.SeedSequence``.

    Returns ``grid[i][r][m]`` = a :class:`~numpy.random.SeedSequence` keyed by
    ``(level i, repeat r, method m)`` so that no two (n, rep, method) cells ever
    share an RNG stream.  Each cell is directly passable to
    :func:`numpy.random.default_rng`.
    """
    top = np.random.SeedSequence(base_seed).spawn(n_levels)
    grid: List[List[List[object]]] = []
    for i in range(n_levels):
        mid = top[i].spawn(repeats)
        row: List[List[object]] = []
        for r in range(repeats):
            low = mid[r].spawn(n_methods)
            row.append([low[m] for m in range(n_methods)])
        grid.append(row)
    return grid


def mc_rmse_slope(ns: Sequence[int], rmse: np.ndarray) -> dict:
    """OLS slope of ``log10(rmse)`` vs ``log10(n)`` plus its standard error."""
    x = np.log10(np.asarray(ns, dtype=float))
    y = np.log10(np.asarray(rmse, dtype=float))
    A = np.vstack([np.ones_like(x), x]).T
    (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([intercept, slope])
    dof = x.shape[0] - 2
    s2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    cov = s2 * np.linalg.inv(A.T @ A)
    return {
        "slope": float(slope),
        "slope_se": float(math.sqrt(cov[1, 1])),
        "intercept": float(intercept),
    }


# ---------------------------------------------------------------------------
# Pathwise Greeks
# ---------------------------------------------------------------------------

def pathwise_vanilla_delta(model: Model, S: np.ndarray) -> np.ndarray:
    disc = math.exp(-model.r * model.T)
    ind = (S[:, -1] > model.K).astype(float)
    return disc * ind * (S[:, -1] / model.S0)


def pathwise_vanilla_vega(model: Model, S: np.ndarray, W_T: np.ndarray) -> np.ndarray:
    disc = math.exp(-model.r * model.T)
    ind = (S[:, -1] > model.K).astype(float)
    return disc * ind * S[:, -1] * (W_T - model.sigma * model.T)


def brownian_paths_from_z(model: Model, Z: np.ndarray) -> np.ndarray:
    """W(t_i) for i=0..steps from the shock matrix ``Z`` (W_0 = 0)."""
    inc = model.sqrt_dt * Z
    return np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)


def pathwise_asian_delta(model: Model, S: np.ndarray) -> np.ndarray:
    disc = math.exp(-model.r * model.T)
    mean_S = _monitored(S).mean(axis=1)
    ind = (mean_S > model.K).astype(float)
    return disc * ind * (mean_S / model.S0)


def pathwise_asian_vega(model: Model, S: np.ndarray, W: np.ndarray) -> np.ndarray:
    disc = math.exp(-model.r * model.T)
    t = model.dt * np.arange(1, model.steps + 1)  # monitored times t_1..t_m
    Sm = _monitored(S)                       # (n, m)
    Wm = W[:, 1:]                             # (n, m)
    dS_dsigma = Sm * (Wm - model.sigma * t)   # (n, m)
    mean_deriv = dS_dsigma.mean(axis=1)
    ind = (Sm.mean(axis=1) > model.K).astype(float)
    return disc * ind * mean_deriv


def finite_difference_greek(model: Model, n_paths: int, seed: int, payoff: Callable,
                            param: str, h: float) -> dict:
    """Central finite-difference Greek with common random numbers.

    ``param`` is ``"S0"`` (delta) or ``"sigma"`` (vega); ``h`` is the fractional
    bump (bump = ``h * S0`` or ``h * sigma``).
    """
    Z = np.random.default_rng(seed).standard_normal((n_paths, model.steps))
    up = Model(**{**model.asdict(), param: getattr(model, param) * (1 + h)})
    dn = Model(**{**model.asdict(), param: getattr(model, param) * (1 - h)})
    S_up = gbm_paths_from_z_np(up, Z)
    S_dn = gbm_paths_from_z_np(dn, Z)
    disc_up = math.exp(-up.r * up.T)
    disc_dn = math.exp(-dn.r * dn.T)
    p_up = disc_up * payoff(S_up, up.K) if param == "S0" else disc_up * payoff(S_up, up.K)
    p_dn = disc_dn * payoff(S_dn, dn.K)
    scale = getattr(model, param)
    contrib = (p_up - p_dn) / (2 * h * scale)
    return {
        "estimate": float(contrib.mean()),
        "se": se_of(contrib),
        "contrib": contrib,
        "n": n_paths,
        "h": h,
    }


def fd_vanilla_greeks(model: Model, n_paths: int, seed: int, h: float) -> dict:
    """Central-difference delta & vega for the vanilla call with CRN."""
    delta = finite_difference_greek(model, n_paths, seed, payoff_european_call,
                                    "S0", h)
    vega = finite_difference_greek(model, n_paths, seed, payoff_european_call,
                                   "sigma", h)
    return {"delta": delta, "vega": vega}


# ---------------------------------------------------------------------------
# Chunked large-reference estimator (geometric control variate)
# ---------------------------------------------------------------------------

def estimate_control_beta(model: Model, n_pilot: int, seed: int) -> float:
    """Estimate the arithmetic-on-geometric control coefficient on an
    *independent* pilot sample."""
    S, _ = simulate_gbm_np(model, n_pilot, seed, antithetic=False)
    A = payoff_asian_call(S, model.K)
    G = payoff_geometric_asian(S, model.K)
    return float(np.cov(A, G, ddof=1)[0, 1] / np.var(G, ddof=1))


def asian_cv_reference(model: Model, n_paths: int, seed: int,
                       n_pilot: int = 1 << 14, chunk: int = 1 << 16,
                       antithetic: bool = True) -> dict:
    """High-precision arithmetic-Asian reference via the geometric control
    variate.

    A control coefficient ``beta`` is estimated on an *independent* pilot sample
    (distinct seed) and applied to a chunked antithetic production run, so the
    reference standard error is ~10^3x smaller than the naive-MC floor.  Each
    chunk uses a *distinct* RNG seed (no reuse), and the number of actual
    simulated paths is returned.  The remaining reference SE is reported as a
    documented floor rather than being forced below any threshold.
    """
    beta = estimate_control_beta(model, n_pilot, seed)
    geo_price, _, _ = geometric_asian_price(
        model.S0, model.K, model.r, model.sigma, model.T, model.steps)
    E_G = math.exp(model.r * model.T) * geo_price
    disc = math.exp(-model.r * model.T)
    n = 0
    s = 0.0
    s2 = 0.0
    done = 0
    chunk_id = 0
    while done < n_paths:
        c = min(chunk, n_paths - done)
        if antithetic:
            c -= c % 2
            if c == 0:
                break
        S, _ = simulate_gbm_np(model, c, seed + 1 + chunk_id, antithetic=antithetic)
        A = payoff_asian_call(S, model.K)
        G = payoff_geometric_asian(S, model.K)
        if antithetic:
            half = c // 2
            A = (A[:half] + A[half:2 * half]) / 2.0
            G = (G[:half] + G[half:2 * half]) / 2.0
        cv = A - beta * (G - E_G)
        s += float(cv.sum())
        s2 += float((cv ** 2).sum())
        n += int(cv.shape[0])
        done += c
        chunk_id += 1
    if n < 2:
        raise ValueError("asian_cv_reference produced fewer than 2 effective paths")
    mean = s / n
    var = (s2 - n * mean ** 2) / (n - 1)
    return {
        "price": float(disc * mean),
        "se": float(disc * math.sqrt(var / n)),
        "n": int(n),
        "n_simulated_paths": int(done),
        "beta": float(beta),
        "n_pilot": int(n_pilot),
        "antithetic": bool(antithetic),
        "method": "geometric control-variate (independent pilot beta, "
                  "chunked antithetic production, distinct chunk seeds)",
    }
