# Monte Carlo pricing and CVaR hedging calibrated to Chinese futures

**English draft for student-group mathematical review. Experimental integration and final acceptance are pending.**

## 1. Objective and interpretation

We implement Question 2, Options A and B, of the instructor's assignment.
The historical IF continuous series supplies a price level and volatility calibration.
The required simulation is nevertheless a spot-style risk-neutral GBM,
\(dS_t=rS_tdt+\sigma S_tdW_t\), rather than the usual zero-drift risk-neutral
futures-price model. Consequently the mandatory analytical reference is Black–Scholes,
not Black-76. These experiments are model-implied pricing and loss calculations;
they are not physical-measure forecasts of futures trading risk.

## 2. Data and preprocessing

All observations were newly fetched in this rebuild. Sources, retrieval timestamp,
units, date ranges, diagnostics and SHA-256 hashes are in `data/metadata.json`.
`data/fetch.py --offline` verifies hashes and independently recalculates parameters.
The calibration uses a trailing six-calendar-year window ending at the latest common
date of IF, CSI300 and SHIBOR observations. Daily log returns are
\(x_i=\log(S_i/S_{i-1})\), with
\(\hat\sigma=\sqrt{252\,s_x^2}\) and
\(\hat\mu=252\bar x+\hat\sigma^2/2\). Historical drift is reported separately
and is not used as the risk-neutral simulation drift.

Sina IF0 is the provider's main continuous series. No independent back adjustment
is applied. Constituent contract IDs and exact roll dates are unavailable in this
endpoint; the largest changes in IF-minus-index basis are recorded as diagnostics,
not certified roll identifications. Calibration therefore includes possible roll
discontinuities. Weekday gaps include exchange holidays and are not automatically
classified as missing observations; comparison against index trading dates is also
reported. Duplicate dates and missing numerical values cause retrieval to fail.

SHIBOR tenor quotes, expressed in percent per annum, are approximated as simple
fixed-tenor-fraction deposit rates (days/365 for weeks, months/12 for months).
We form \(z(u)=\log(1+q(u)u)/u\), linearly interpolate
zero yields and discount by \(e^{-z(T)T}\). At the principal one-year maturity this
uses the observed one-year node. This is an explicit deposit-curve approximation,
not a bootstrapped arbitrage-free market zero curve.

## 3. Simulation, contracts and analytical references

We take \(T=1\), \(m=252\), \(K=S_0\), and \(B=1.2S_0\), with
\(t_i=iT/m\). The simulator applies
\[
S_{i+1}=S_i\exp\{(r-\sigma^2/2)T/m+\sigma\sqrt{T/m}Z_i\}.
\]
The initial point is excluded from Asian averaging and barrier monitoring.
The arithmetic Asian payoff is \((m^{-1}\sum_{i=1}^m S_i-K)^+\);
the barrier payoff is \((S_m-K)^+1\{\max_{1\le i\le m}S_i<B\}\);
the digital pays \(1\{S_m>K\}\). European validation uses \((S_m-K)^+\).
All price observations are multiplied by \(e^{-rT}\).

For the European call,
\(C=S_0\Phi(d_1)-Ke^{-rT}\Phi(d_2)\),
\(d_1=[\log(S_0/K)+(r+\sigma^2/2)T]/(\sigma\sqrt T)\),
\(d_2=d_1-\sigma\sqrt T\).
We evaluate \(z=(\widehat C-C)/\widehat{SE}\) and report actual violations
of the assignment's \(|z|\le3\) criterion. Independent lead references use
one-dimensional quadrature without importing the numerical implementation
(`tools/independent_reference.py`).

## 4. Variance reduction and uncertainty

Naive standard errors use independent discounted payoffs. For antithetic sampling,
the independent observation is the pair average \([Y(Z)+Y(-Z)]/2\), not each
individual payoff. Comparisons must account for the two paths in each pair.
The Asian control variate is the discounted geometric-Asian call, with coefficient
estimated on an independent pilot and then held fixed for production samples.

Writing \(G=\exp(m^{-1}\sum_i\log S_i)\), its log mean and variance are
\[
a=\log S_0+(r-\sigma^2/2)\bar t,\qquad
b=\sigma^2m^{-2}\sum_{i,j}\min(t_i,t_j).
\]
Thus its discounted call expectation is
\[
e^{-rT}\{e^{a+b/2}\Phi[(a+b-\log K)/\sqrt b]
-K\Phi[(a-\log K)/\sqrt b]\}.
\]
The full covariance-matrix expression is evaluated independently in the reference
script. Variance reduction and confidence intervals must be read from executed
artifacts; the indicative factor-five target is not imposed on outputs.

## 5. Greeks and convergence

For continuous vanilla and Asian payoffs, differentiate paths with respect to spot
and volatility: \(\partial S_t/\partial S_0=S_t/S_0\) and
\(\partial S_t/\partial\sigma=S_t(W_t-\sigma t)\).
Call payoff derivatives multiply these quantities (or their arithmetic mean) by
the corresponding in-the-money indicator and discount factor. Vanilla benchmarks
are \(\Delta=\Phi(d_1)\), \(\nu=S_0\phi(d_1)\sqrt T\).
Central differences share random shocks, with spot bumps
\(\{.1,.01,.001\}S_0\) and volatility bumps
\(\{.1,.01,.001\}\sigma\); vega is per unit absolute volatility.
Naive pathwise differentiation of digital and knockout indicators omits boundary
contributions and is invalid. Their finite-difference uncertainty and bump dependence
must therefore be examined explicitly.

Convergence is assessed through independent repetitions at multiple path counts.
Log–log RMSE slopes are estimated rather than prescribed. A numerical Asian reference
has uncertainty of its own, which may generate an RMSE floor and must be disclosed.
Finite-difference truncation bias similarly need not disappear by increasing paths.

## 6. Computational profiling

The Python-loop and vectorized implementations are benchmarked at matched workloads
using repeated timings and an actual hot-loop profile. Vectorization requires
path-by-time arrays and temporaries; speedup is not interpreted as a memory saving.
Timing results and hardware/runtime details will be inserted after execution.

## 7. Deep hedging and CVaR

The policy uses only current normalized price and time, with at most three hidden
layers. At rebalancing times \(t_j\), the principal European-call P&L is literally
the assignment's convention:
\[
V_T=\sum_j e^{r(T-t_j)}\Delta_j(S_{t_{j+1}}-S_{t_j})
-\kappa\sum_j|\Delta_j-\Delta_{j-1}|S_{t_j}-(S_T-K)^+.
\]
Initial position is zero, no premium is added, and no terminal liquidation cost is
charged. Costs are not accumulated to maturity. The gain accumulation factor uses
the beginning of each holding interval, as specified, rather than silently replacing
it with another self-financing cash-account convention. Neural, BS and zero-hedge
policies share the same paths, payoff and accounting.

Loss is \(L=-V_T\). Training minimizes
\(\eta+(1-.95)^{-1}n^{-1}\sum_i(L_i-\eta)^+\) jointly over the MLP and
\(\eta\), using Adam and minibatches. At least 131072 training paths and separate
validation/test seeds are required. Held-out CVaR is evaluated independently using
the empirical upper tail, including fractional tail mass when required.
The full benchmark crosses intervals 1, 5 and 21 trading days with costs 0,
10 and 50 basis points. Paired bootstrap intervals compare held-out strategies;
these quantify test-path uncertainty, not variability from retraining the model.

## 8. Results and independent audit

Pending numerical implementation, full execution, integration checks and Kimi's
final independent audit. No pricing, variance-reduction, convergence, speedup or
neural-performance claim is made at this stage.

## 9. Reproducibility and limitations

See the ten-line README for frozen installation and offline execution. The source
data snapshots are committed; regenerable results are ignored. Full reproduction
must execute training rather than merely load checkpoints. The live session export
must be refreshed after the final response. Submission tagging remains pending.

Limitations include continuous-contract roll contamination, constant volatility,
the spot-versus-futures convention, simplified cash-flow timing, model-implied rather
than physical loss tails, discrete barrier monitoring, and single-seed training
variability unless additional full independent fits are actually executed.

## 10. Statement of work

OpenAI established the shared mathematical conventions, independently retrieved and
verified data, constructed quadrature references and infrastructure, and drafts this
report. DeepSeek is assigned the numerical engines, estimator experiments and neural
training. Kimi performs independent read-only review. Actual provider identifiers and
child session IDs are recorded in RESEARCH_PROTOCOL.md and authentic session exports.
The student group must review the assumptions, mathematical decisions, empirical
verification and final English report; this draft is not a record of group approval.
