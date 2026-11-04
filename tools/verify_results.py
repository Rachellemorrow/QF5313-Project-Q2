"""Lead acceptance checks and independent replay of held-out hedge accounting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from mc_common import Model
from approach2_hedging import PolicyMLP, simulate_dataset, neural_delta


def tail_mean(loss, alpha=.95):
    # RU objective evaluated at an empirical quantile: independent of sorting-tail implementation.
    eta = np.quantile(loss,alpha,method="inverted_cdf")
    return float(eta+np.maximum(loss-eta,0).mean()/(1-alpha))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hedging-dataset",default="if_main")
    args = parser.parse_args()
    a = json.loads((ROOT/"results/approach1/if_main/metrics.json").read_text())
    base = ROOT/"results/approach2"/args.hedging_dataset
    b = json.loads((base/"metrics.json").read_text())
    if not a["metrics"]["validation"]["z_pass"] or b["params"]["n_train"] < 2**17:
        raise ValueError("Analytical validation or mandatory training count failed")
    model = Model(**b["params"]["model"])
    seed = b["params"]["seed"]
    if b["params"]["seeds"]["test"] != seed+20:
        raise ValueError("Unrecognized test seed")
    prices = simulate_dataset(model,b["params"]["n_test"],seed+20,dtype=np.float64)
    tensor = torch.tensor(prices,dtype=torch.float32)
    torch.set_num_threads(1)
    replay = {}
    for cost in (0.,.001,.005):
        checkpoint = torch.load(base/f"policy_k{cost}.pt",weights_only=False,map_location="cpu")
        policy = PolicyMLP(checkpoint["hidden"])
        policy.load_state_dict(checkpoint["state_dict"])
        curve = b["history"]["training_curves"][str(cost)]
        if not curve["val_cvar"] or not np.isfinite(np.array(list(curve.values()))).all():
            raise ValueError("Invalid training history")
        for frequency in (1,5,21):
            idx = np.arange(0,model.steps,frequency)
            with torch.no_grad():
                delta = neural_delta(policy,tensor,model,idx).numpy().astype(float)
            pnl = -np.maximum(prices[:,-1]-model.K,0)
            previous = np.zeros(len(prices))
            costs = np.zeros(len(prices))
            for j,t in enumerate(idx):
                next_t = idx[j+1] if j+1<len(idx) else model.steps
                pnl += np.exp(model.r*(model.T-t*model.dt))*delta[:,j]*(prices[:,next_t]-prices[:,t])
                charge = cost*abs(delta[:,j]-previous)*prices[:,t]
                costs += charge
                pnl -= charge
                previous = delta[:,j]
            actual = dict(cvar95=tail_mean(-pnl),expected_pnl=float(pnl.mean()),pnl_variance=float(pnl.var(ddof=1)),avg_cost=float(costs.mean()))
            recorded = b["metrics"]["neural"][str(cost)][str(frequency)][str(cost)]
            for key,value in actual.items():
                if not np.isclose(value,recorded[key],rtol=2e-6,atol=2e-5):
                    raise ValueError(f"Independent replay mismatch {cost}/{frequency}/{key}: {value} vs {recorded[key]}")
            replay[f"{cost}/{frequency}"] = actual
    expected = [ROOT/"results/approach1/if_main"/f"{name}.html" for name in ("convergence","greeks","profiling")]
    expected += [base/f"{name}.html" for name in ("training","comparison","loss_histogram")]
    for path in expected:
        text = path.read_text()
        if "plotly.js" not in text or '<script src="https://cdn.plot.ly' in text:
            raise ValueError(f"Missing self-contained plot: {path}")
    out = ROOT/"results/verification"
    out.mkdir(exist_ok=True)
    (out/f"{args.hedging_dataset}.json").write_text(json.dumps({"passed": True,"independent_hedging_replay": replay,
        "source_metrics_sha256": hashlib.sha256((base/"metrics.json").read_bytes()).hexdigest()},indent=2,allow_nan=False)+"\n")
    print("PASS independent hedging replay, training counts/history, pricing validation, inline figures")


if __name__ == "__main__":
    main()
