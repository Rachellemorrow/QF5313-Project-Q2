"""Render factual report tables directly from executed, full-size metrics."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows):
    render = lambda row: "| " + " | ".join(f"{x:.8g}" if isinstance(x,float) else str(x) for x in row) + " |"
    return "\n".join([render(headers),render(["---"]*len(headers)),*[render(row) for row in rows]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hedging-dataset",default="if_main")
    args = parser.parse_args()
    ap = "results/approach1/if_main/metrics.json"
    bp = f"results/approach2/{args.hedging_dataset}/metrics.json"
    a,b = [json.loads((ROOT/path).read_text()) for path in (ap,bp)]
    if b["params"]["n_train"] < 2**17:
        raise ValueError("Cannot report smoke run as full training")
    am,bm = a["metrics"],b["metrics"]
    lines = ["<!-- GENERATED_RESULTS_START -->",f"The following tables are generated from `{ap}` and `{bp}`. Values are index-point payoffs/P&L; the digital pays one unit.",
             "### Executed pricing and validation"]
    lines.append(table(["Payoff","NumPy price","SE","95% lower","95% upper","PyTorch price"],[
        [name,v["price"],v["se"],v["price"]-1.96*v["se"],v["price"]+1.96*v["se"],am["prices"]["torch"][name]["price"]]
        for name,v in am["prices"]["numpy"].items()]))
    val = am["validation"]
    lines.append(f"European BS reference: {val['european_bs']:.10f}; MC z-score: {val['z_score']:.6f}; criterion passed: {val['z_pass']}. Engines use identical shocks; their agreement checks arithmetic implementation, not independence of RNG libraries.")
    vr = am["variance_reduction"]
    lines += ["### Asian variance reduction",table(["Estimator","Price","SE","Variance of estimate","95% lower","95% upper"],[
        [name,v["price"],v["se"],v["se"]**2,v["price"]-1.96*v["se"],v["price"]+1.96*v["se"]] for name,v in vr["asian"].items() if isinstance(v,dict)]),
        f"Independent-pilot coefficient beta: {vr['beta']:.9f}; A/G correlation {vr['a_g_correlation']:.9f}. Production path budgets are matched; pilot overhead is {vr['pilot_paths']} additional paths ({vr['pilot_seconds']:.4f}s).",
        table(["Method","Variance reduction","Bootstrap lower","Bootstrap upper"],[[name,vr[key]["ratio"],vr[key]["ci_low"],vr[key]["ci_high"]] for name,key in [("Antithetic","vrr_antithetic"),("Control variate","vrr_cv")]])]
    g = am["greeks"]
    rows = []
    for payoff,quantities in g["fd"].items():
        for quantity,bumps in quantities.items():
            pw = g["pathwise"].get("vanilla" if payoff=="european" else payoff,{}).get(quantity,{})
            for bump,v in bumps.items():
                rows.append([payoff,quantity,bump,v["estimate"],v["se"],pw.get("estimate","invalid pathwise"),pw.get("se","—")])
    lines += ["### Greeks",table(["Payoff","Greek","Fractional bump","FD","FD SE","Pathwise","Pathwise SE"],rows),
              table(["Paired PW minus FD","Mean difference","Paired SE","Difference/SE"],[[k,v["mean_diff"],v["se"],v["mean_diff"]/v["se"] if v["se"] else "undefined"] for k,v in g["paired_diff"].items()]),
              f"At the 10% spot bump, absolute paired delta difference/SE values are {abs(g['paired_diff']['vanilla_delta_h0.1']['mean_diff']/g['paired_diff']['vanilla_delta_h0.1']['se']):.3f} (vanilla) and {abs(g['paired_diff']['asian_delta_h0.1']['mean_diff']/g['paired_diff']['asian_delta_h0.1']['se']):.3f} (Asian). Thus agreement must be assessed bump by bump rather than asserted universally. Large spot bumps introduce smoothing/truncation bias. Discontinuous payoffs can have rising finite-difference variance as bumps shrink; a sample SE is not an estimate of truncation bias.",
              "### Convergence",table(["Estimator","RMSE slope","OLS slope SE","Bootstrap lower","Bootstrap upper","SE-versus-N slope"],[
                  [key,am["convergence"][f"slope_rmse_{key}"]["slope"],am["convergence"][f"slope_rmse_{key}"]["slope_se"],am["convergence"][f"slope_rmse_{key}_bootstrap"]["ci_low"],am["convergence"][f"slope_rmse_{key}_bootstrap"]["ci_high"],am["convergence"][f"slope_se_{key}"]["slope"]]
                  for key in ("vanilla_naive","asian_naive","asian_anti","asian_cv")]),
              f"Independent Asian reference: `{json.dumps(a['history']['reference'],sort_keys=True)}`. Bootstrap slope intervals condition on this estimated reference and pilot; they do not integrate reference uncertainty. The original noisier reference produced a flat CV slope and is retained under `results/archive/initial_option_a`; the redesigned reference reduces, but does not eliminate, that uncertainty."]
    pr = am["profiling"]
    lines += ["### Profiling",table(["Loop seconds","NumPy seconds","Speedup","Loop peak bytes","NumPy peak bytes"],[[pr[k] for k in ("naive_loop_seconds","vectorized_seconds","speedup","naive_loop_peak_memory_bytes","vectorized_peak_memory_bytes")]]),
              "These are matched supplied-shock workloads. Memory values are traced allocations, not total process RSS. The raw hot-loop profile is stored in the metrics history. CPU contention can affect wall-clock measurements.",
              "### Full-size neural training",
              f"Architecture: {b['params']['architecture']}, SiLU hidden activations, unconstrained scalar output. Training paths: {b['params']['n_train']}; validation: {b['params']['n_val']}; test: {b['params']['n_test']}. Seeds: `{json.dumps(b['params']['seeds'])}`. Adam learning rates: weights {b['params']['lr']}, eta {b['params']['lr_eta']}; batch size {b['params']['batch_size']}; maximum epochs {b['params']['epochs']}; validation patience {b['params']['patience']}; CPU threads {b['params']['threads']}.",
              table(["Training cost","Epochs","Best validation CVaR","Restored eta"],[[k,v["epochs_run"],v["best_val_cvar"],v["best_eta"]] for k,v in bm["training"].items()]),
              "Policies train on the daily grid. The 5/21-day results evaluate those same policies at fewer decision times; they are not separately optimized frequency-specific networks. Restored models minimize validation CVaR among observed epochs; convergence to the global CVaR optimum is not established.",
              "### All held-out frequency/cost benchmarks"]
    rows=[]
    for freq in (1,5,21):
        for cost in (0.,.001,.005):
            for policy,v in [("BS",bm["bs"][str(freq)][str(cost)]),("Neural",bm["neural"][str(cost)][str(freq)][str(cost)]),("Zero",bm["zero_hedge"])]:
                rows.append([freq,cost,policy,*[v[k] for k in ("cvar95","expected_pnl","pnl_variance","avg_cost","turnover")]])
    lines += [table(["Days","Cost","Policy","CVaR95","Mean P&L","P&L variance","Mean cost","Turnover (units)"],rows),
              "### Paired bootstrap uncertainty",table(["Combination","Observed neural minus BS CVaR","95% lower","95% upper"],[[k,v.get("point_diff",v["mean_diff"]),v["ci_low"],v["ci_high"]] for k,v in bm["bootstrap_cis_neural_minus_bs"].items()]),
              "Negative differences favor neural hedging; intervals crossing zero are inconclusive. These are marginal paired-path bootstrap intervals conditional on each fitted model, without multiplicity adjustment or independent retraining. No general outperformance claim follows from this single training seed.",
              f"Recorded approach wall times: A {a['fit_seconds']:.3f}s, B {b['fit_seconds']:.3f}s. Source figures: `results/approach1/if_main/{{convergence,greeks,profiling}}.html` and `results/approach2/{args.hedging_dataset}/{{training,comparison,loss_histogram}}.html`.",
              "<!-- GENERATED_RESULTS_END -->"]
    text = "\n\n".join(lines)
    path = ROOT/"report.md"
    report = path.read_text()
    start,end = "<!-- GENERATED_RESULTS_START -->","<!-- GENERATED_RESULTS_END -->"
    if start in report:
        report = report[:report.index(start)]+text+report[report.index(end)+len(end):]
    else:
        report = report.replace("Pending numerical implementation, full execution, integration checks and Kimi's\nfinal independent audit. No pricing, variance-reduction, convergence, speedup or\nneural-performance claim is made at this stage.",text)
    path.write_text(report)
    print("Generated factual report tables from full-size metrics")


if __name__ == "__main__":
    main()
