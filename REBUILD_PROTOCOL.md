# Independent Q2 rebuild protocol

This is a fresh implementation authorized by the user on 2026-11-04. Deleted
historical implementation, snapshots and outputs are not restoration sources.
The historical `submission` tag remains at `0a27070` and `main` is unchanged.

## Agents and ownership

- OpenAI lead: `openai/gpt-6-astra` (current Build primary); owns data, environment,
  comparison, integration tests, adjudication, reports and Git.
- Numerical implementer: `deepseek/deepseek-v4-pro`, child
  `ses_e5d966a28ffewhBw2uMDwpVvfn`; owns numerical modules and numerical tests.
- Read-only auditor: `moonshotai/kimi-k3.1-pro`, child
  `ses_e5d94143fffe3mOwD3AaUKaFGN`; no edits or shell permission.
- OpenCode v2.1.4 uses project `.opencode/agents/*.md`, V2 permission lists.
  `opencode agent list` is unsupported; the location-scoped `/api/agent` endpoint
  verifies loaded agents. No global configuration or authentication was changed.

## Fixed scientific choices

1. Reacquire IF0 vendor main continuous close, CSI300 index and all SHIBOR tenors.
   Calibration uses available observations in 2017-01-01 through 2026-11-03.
   Vendor rolls are not identifiable from the close-only series; do not claim
   that roll contamination is removed. No trimming/winsorization or imputation.
2. Sample daily log-return volatility, annualization 252. Historical drift is
   annual mean log return plus half annual variance, never the pricing drift.
3. SHIBOR simple deposit quotes are converted by log(1+q*tau)/tau (ACT/360),
   then continuous zero rates interpolated. This is a deposit-curve proxy, not
   an OIS curve or claim of arbitrage-free bootstrapping.
4. Assignment GBM drift r; T=1, 252 steps, K=S0, B=1.2*S0. Average and monitor
   t1,...,tm (not t0). This is the stock-like assignment model calibrated with
   futures returns, distinct from a zero-drift futures Black-76 model.
5. Antithetic sampling unit is a pair. Equal-total-path VRF is
   var(single)/(2*var(pairmean)); pooling negatively correlated paths as IID
   overestimates, not underestimates, the mean's standard error.
6. Asian control uses an independent pilot beta and the exact discrete schedule.
   Report pilot cost and conditional standard errors. FD uses common shocks,
   spot bumps {0.1,0.01,0.001}S0 and vol bumps {0.1,0.01,0.001}sigma.
7. Hedging is on European call, no premium, initial holding zero, no terminal
   liquidation charge, left-time gain accumulation exactly as the assignment,
   unaccumulated proportional costs. All strategies use the same conventions.
8. CVaR is upper-tail loss=-P&L, alpha=.95, fractional empirical boundary weight.
   At least 131072 training paths; independent validation/test RNG streams.
   Best model selected only on validation. Bootstrap uses paired test indices
   and is conditional on fitted models, not training-seed uncertainty.

These choices are agent-assisted proposals requiring student-group review.
All reported numerical claims must come from new executed artifacts.
