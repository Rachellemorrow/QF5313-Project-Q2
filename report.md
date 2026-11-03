# Question 2: Monte Carlo pricing and CVaR hedging — independent rebuild

**Agent-assisted English draft; student-group mathematical review and final
authorship are outstanding.** This rebuild starts from the instructor's
specification, not the removed historical implementation. Historical results
are not evidence for the experiments below.

## 1. Objective and financial interpretation

We price European, arithmetic-Asian, discretely monitored up-and-out, and
unit-cash digital calls, then hedge a European call under transaction costs.
The assignment requires
\(S_{i+1}=S_i\exp[(r-\sigma^2/2)\Delta t+\sigma\sqrt{\Delta t}Z_i]\).
This is exact GBM at the observation times, not Euler discretization. We use
252 steps over one year, independent standard normal shocks, strike S0 and
barrier 1.2 S0. The averaging and barrier sets exclude t0. Knockout occurs at
equality with the barrier; the digital pays only for strictly ST>K.

Although calibrated with IF0 futures returns, this is the assignment's
stock-like drift-r model. An actual futures-price risk-neutral model normally
has zero drift and uses Black-76 pricing. These are different economic models;
our CVaR is a model-implied risk-neutral loss measure, not a physical-market
risk estimate. Historical drift is estimated but never substituted for r.

## 2. Data and calibration

Fresh AKShare retrieval uses Sina IF0 main continuous futures, Sina CSI300
index history and Jin10 SHIBOR. `data/metadata.json` records sources, retrieval
timestamp, schemas, gaps and SHA-256 hashes. `data/fetch.py --offline` verifies
snapshots and recomputes calibration. The fixed cutoff is 2026-11-03.

IF0 contains 2,375 observations from 2017-01-17 to 2026-11-03 (2,374 returns).
The latest level is 3,978.8. Log-return volatility is 0.2007613271 annually,
annual mean log return is 0.0193619632, and the GBM historical drift estimate
is 0.0395145184. Annualization is 252 and variance uses ddof=1. No observations
are trimmed, filled or roll-adjusted. Seventeen absolute log returns above
5% are disclosed in `data/clean/calibration.json`.

Vendor constituent identifiers are unavailable: exact rolls and their
artificial jumps cannot be separately verified. Thus volatility may contain
roll contamination. Weekday gaps include exchange holidays and are not proof
of missing trading observations. Index data are a separate reference snapshot,
not mixed into futures volatility estimation.

A direct cross-check against index trading dates identifies **2017-10-09** as
absent from IF0 inside its coverage window. It is not imputed. The next
observed return spans that missing session; treating all successive-observation
returns as one trading day slightly approximates the annualization. This
known gap is retained explicitly rather than silently changing the sample.

SHIBOR is a deposit-curve proxy, not an OIS curve. Percentage simple quotes
are converted to continuous zero rates as log(1+q tau)/tau, with ACT/360 tenor
approximations, then linearly interpolated. At T=1 the 1Y node is used directly:
the 2026-11-03 quote of 1.439% becomes r=0.0142874466081. No claim is made of
arbitrage-free multi-instrument curve bootstrapping.

## 3. Monte Carlo estimators

Discount every payoff by exp(-rT). For IID discounted payoffs X, the mean's
standard error is s(X)/sqrt(N). For antithetic draws, independent units are
Y=(h(Z)+h(-Z))/2, and SE=s(Y)/sqrt(N/2) for N total paths. The equal-total-path
variance reduction ratio is s(X)^2/[2s(Y)^2], not s(X)^2/s(Y)^2. Pilot and
evaluation workloads must be stated separately for the control estimator.

Let G be the geometric mean of the same m monitored levels. Its log mean is
log(S0)+(r-sigma²/2)T(m+1)/(2m), and its log variance is
sigma² T(m+1)(2m+1)/(6m²). The latter is linear, not quadratic, in T.
The geometric-call control expectation follows the lognormal call formula,
discounted at T. An independent pilot estimates beta=Cov(A,Gcall)/Var(Gcall);
evaluation averages A-beta(Gcall-E[Gcall]). Conditional standard errors are
valid given that pilot coefficient. Monte Carlo uncertainty is not parameter
estimation uncertainty.

## 4. Greeks, convergence and computation

For European calls, pathwise delta is exp(-rT)1(ST>K)ST/S0; vega is
exp(-rT)1(ST>K)ST(WT-sigma T). For Asian calls replace the terminal derivatives
by the average derivatives of monitored levels and gate on mean(S)>K.
Digital and barrier indicators have moving-boundary contributions missed by
naive pathwise differentiation; use common-random-number finite differences.
Spot bumps are {0.1,0.01,0.001}S0; volatility bumps are the corresponding
fractions of sigma, explicitly different units. Large-bump truncation bias
must not be confused with simulation noise.

Convergence requires independent replications at each path count, measured
RMSE and fitted log-log slopes with uncertainty. A high-precision independent
Asian control estimate is a reference with its own nonzero SE, not a known
analytic arithmetic-Asian price. Loop/vectorization timings must use the same
workload and actual repeated measurements; vectorization allocates full arrays.

## 5. Deep hedging and accounting

The policy receives only current S/S0 and time and has at most three hidden
layers. Let delta_-1=0. Over left endpoints ti<T,

\[
V_T=\sum_i e^{r(T-t_i)}\Delta_i(S_{i+1}-S_i)
-\kappa\sum_i |\Delta_i-\Delta_{i-1}|S_i-(S_T-K)^+.
\]

There is no premium, initial inventory, terminal liquidation charge or extra
financing term. Costs are not accumulated. The assignment's simplified
left-time gain accumulation is implemented literally, rather than silently
replacing it with a self-financing stock account. Neural, BS-delta (1/5/21
days) and zero hedge share the same payoff, paths and accounting.

Loss is -VT. At alpha=.95, optimize the Rockafellar–Uryasev expression
eta+mean(relu(L-eta))/(1-alpha). Empirical CVaR integrates the upper loss tail,
including fractional weight at the order-statistic boundary; a conditional
mean above an interpolated quantile need not equal this finite-sample value.
Independent validation chooses checkpoints; test paths must never influence
training. Bootstrap paired held-out paths to estimate CVaR differences
(neural minus daily BS), conditional on fitted models. This does not estimate
training-seed uncertainty, and CVaR improvement does not imply variance
improvement or dominance across transaction costs.

## 6. Executed results and independent audit

Pending new numerical execution and artifact verification. No historical
experiment results are asserted here as rebuilt results.

## 7. Reproduction and statement of work

The new branch is `q2-rebuild-20261104`; the old submission tag is preserved.
OpenAI (`openai/gpt-6-astra`) designs the protocol, acquires and verifies data,
integrates, runs independent checks and prepares this draft. DeepSeek
(`deepseek/deepseek-v4-pro`) implements numerical modules. Moonshot Kimi
(`moonshotai/kimi-k3.1-pro`) independently audits without edit permission.
Genuine OpenCode child-session IDs are recorded in `REBUILD_PROTOCOL.md`.
The student group must verify mathematical decisions, limitations and the
final report; no group approval has been inferred or asserted.
