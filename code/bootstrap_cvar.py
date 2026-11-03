#!/usr/bin/env python
"""Paired bootstrap of out-of-sample CVaR differences: Neural vs BS-delta hedging.

Reads the per-path TEST losses dumped by approach2_hedging.py
(`results/approach2/if/test_losses.npz`, losses L = -V_T on the SAME 2^15 test paths,
seed 2026) and computes, for each transaction cost level κ ∈ {0, 10, 50} bp:

    ΔCVaR = CVaR_0.95(L_neural) − CVaR_0.95(L_bs_fX_κ)

where X ∈ {1, 5, 21} days (BS-1d is the primary comparator). The bootstrap resamples
paths with replacement (B replicates, seed 42) and recomputes both CVaRs and their
difference on each resample ("paired" — both policies are always evaluated on the same
resampled paths). Reported: point estimate, bootstrap mean, 2.5/97.5 percentiles,
fraction of resamples with ΔCVaR < 0 (neural better), and the input SEs.

Usage:  uv run python code/bootstrap_cvar.py [--boot 10000] [--seed 42] [--alpha 0.95]
Output: results/approach2/if/bootstrap_cvar.json + rich tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
LOSSES = ROOT / "results" / "approach2" / "if" / "test_losses.npz"
OUT = ROOT / "results" / "approach2" / "if" / "bootstrap_cvar.json"


def cvar(loss: np.ndarray, alpha: float) -> float:
    n = loss.shape[0]
    k = max(1, int(round((1.0 - alpha) * n)))
    return float(np.sort(loss)[::-1][:k].mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.95)
    args = ap.parse_args()

    if not LOSSES.exists():
        print(f"FAIL: {LOSSES} not found; run code/approach2_hedging.py first.")
        return 1

    data = np.load(LOSSES)
    losses = {k: data[k] for k in data.files}
    n = len(losses["neural_0bp"])
    console = Console()
    console.print(f"[bold]Paired bootstrap[/] n_test={n}  B={args.boot}  seed={args.seed}  alpha={args.alpha}")

    rng = np.random.default_rng(args.seed)
    results = {}
    for kappa in ["0bp", "10bp", "50bp"]:
        neural = losses[f"neural_{kappa}"]
        for freq in [1, 5, 21]:
            bs = losses[f"bs_f{freq}_{kappa}"]
            point = cvar(neural, args.alpha) - cvar(bs, args.alpha)
            boots = np.empty(args.boot)
            for b in range(args.boot):
                idx = rng.integers(0, n, n)
                boots[b] = cvar(neural[idx], args.alpha) - cvar(bs[idx], args.alpha)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            frac_neg = float((boots < 0.0).mean())
            key = f"neural_vs_bs_f{freq}_{kappa}"
            results[key] = {
                "cvar_neural": cvar(neural, args.alpha),
                "cvar_bs": cvar(bs, args.alpha),
                "delta_cvar": point,
                "boot_mean": float(boots.mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "frac_boot_neural_better": frac_neg,
                "corr_paths": float(np.corrcoef(neural, bs)[0, 1]),
            }
            table = Table(title=f"ΔCVaR95 = neural − BS-{freq}d   (κ = {kappa})")
            table.add_column("metric"); table.add_column("value", justify="right")
            table.add_row("CVaR neural", f"{results[key]['cvar_neural']:.3f}")
            table.add_row("CVaR BS", f"{results[key]['cvar_bs']:.3f}")
            table.add_row("ΔCVaR (point)", f"{point:+.3f}")
            table.add_row("bootstrap mean ± sd", f"{boots.mean():+.3f} ± {boots.std(ddof=1):.3f}")
            table.add_row("95% percentile CI", f"[{lo:+.3f}, {hi:+.3f}]")
            table.add_row("P(neural better) bootstrap", f"{frac_neg:.3f}")
            table.add_row("corr(L_neural, L_bs) across paths", f"{results[key]['corr_paths']:.3f}")
            console.print(table)

    payload = {"params": {"boot": args.boot, "seed": args.seed, "alpha": args.alpha,
                          "n_test": n, "losses_file": str(LOSSES.relative_to(ROOT))},
               "metrics": results,
               "fit_seconds": None,
               "history": {}}
    OUT.write_text(json.dumps(payload, indent=2))
    # note: fit_seconds is deliberately null — this is a post-processing bootstrap, not a fit.
    console.print(f"[green]Wrote {OUT}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
