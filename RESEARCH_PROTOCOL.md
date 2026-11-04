# Independent rebuild protocol

This implementation is written afresh from assignment.md and AGENTS.md. The user
confirmed the existing unstaged deletions were intentional. No historical source,
data snapshots, experimental outputs or reports are restored or used as targets.
Existing Git history is preserved. The AGENTS.md history begins at e7bc8d8;
this alone does not establish the time of every earlier agent session.

## Models and authentic invocation evidence

OpenCode CLI: v2.1.4. Lead: openai/gpt-6-astra (current harness).
DeepSeek: deepseek/deepseek-v4-pro, child ses_e5d898135ffesfJ6YyK8ysLrvi.
Kimi: moonshotai/kimi-k3.1-pro-highspeed, child ses_e5d893b81ffef5eNqrveRk54r2.
Both child invocations successfully read the specification. Kimi is read-only.
The installed V2 CLI does not support `opencode agent list`; the actual subagent
catalog and successful calls establish discovery. No provider secrets are recorded.

## Interfaces and methodology (draft decisions for group review)

Data calibration: data/clean/calibration.json with S0,K,B,r,sigma,mu,T,steps.
T=1 year, steps=252, K=S0, B=1.2*S0. Daily log returns, 252-day annualization.
Use assignment drift r; explain its distinction from martingale futures dynamics.
Asian average and strict up-and-out monitoring use t1,...,tm, not t0.
Digital pays one unit iff terminal value exceeds K. Prices discounted exp(-rT).
Antithetic uncertainty uses independent pairs; CV coefficient uses an independent
pilot and the exact geometric expectation on the same discrete grid.
FD uses common shocks and spot bumps {0.1,0.01,0.001}*S0; volatility bumps use
the same fractions of sigma. Pathwise estimators only for continuous payoffs.

Hedging uses the instructor's literal P&L, exp(r*(T-t)) gain weights, unaccumulated
costs, initial position zero, no premium and no terminal liquidation. European
call throughout matched comparisons. MLP inputs current S/S0 and calendar time;
RU upper-tail loss CVaR, alpha=.95, >=131072 training paths, independent validation
and test seeds. All three costs and all three rebalancing frequencies evaluated.
Bootstrap comparisons use paired held-out paths; they do not measure retraining
uncertainty. No neural outperformance is presumed.

Lead owns data, dependencies, comparison, independent integration checks and report.
DeepSeek owns mc_common.py, approach1_pricing.py, approach2_hedging.py and its tests.
All outputs are newly executed. Numerical repairs, independent replay and full
clean-clone reproduction have completed; final session/group acceptance remains pending.

## Integration protocol update

Initial Option A results exposed a high-noise Asian reference that obscured CV
RMSE convergence; DeepSeek is repairing the reference and repetition seeds before
acceptance. The initial Option B full job (50 epochs, 8 CPU threads) is retained
as an exploratory run. The integrated reproducible default uses 20 maximum epochs,
five-epoch validation patience, and one CPU thread, retaining all 131072 training
paths, 16384 validation/test paths, and the same unmodified RU objective. This
computational budget is fixed before inspecting integrated held-out outcomes;
nonconvergence and training variability must be disclosed. Checkpoints, full curves,
finite-gradient checks and matched float64 test accounting are now mandatory.
