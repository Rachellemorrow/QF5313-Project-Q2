# Monte Carlo pricing and CVaR hedging calibrated to Chinese futures

**English draft for student-group mathematical review. Full experiments, independent replay and clean-clone reproduction have passed; final session export and group approval remain pending.**

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
The daily mean is equivalently the intercept estimate in an intercept-only log-return
regression. Metadata report both iid and Bartlett HAC standard errors for the
annualized log drift; neither removes unknown continuous-contract roll effects.

Sina IF0 is the provider's main continuous series. No independent back adjustment
is applied. Constituent contract IDs and exact roll dates are unavailable in this
endpoint; the largest changes in IF-minus-index basis are recorded as diagnostics,
not certified roll identifications. Calibration therefore includes possible roll
discontinuities. Weekday gaps include exchange holidays and are not automatically
classified as missing observations; comparison against index trading dates is also
reported. Duplicate dates and missing numerical values cause retrieval to fail.

The executed snapshot contains 1,899 IF observations and 1,899 CSI300 observations
from 2019-01-02 to 2026-11-03, plus 1,947 SHIBOR observations. The calibration window
is 2020-11-03 to 2026-11-03 (1,455 prices, 1,454 log returns). There are no missing IF
observations relative to the index's trading dates in this window. These counts and
the following calibration values are recorded in `data/metadata.json` and
`data/clean/calibration.json`:

| Parameter | Executed value |
|---|---:|
| Initial level / strike | 3978.8 |
| Barrier | 4774.56 |
| Annualized volatility | 0.190647912849 |
| Historical arithmetic GBM drift estimate | -0.012779861736 |
| Annualized mean log return | -0.030953175072 |
| One-year continuous discount rate | 0.014287446608 |

Historical plots are `results/data/calibration.html` and `results/data/curve.html`.

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

The newly executed independent reference artifact `results/independent/reference.json`
gives a BS call value of 329.077643012334 and quadrature value of 329.077643012334
(rounded), delta 0.567599324304 and vega 1564.469149519032 per unit volatility.
The digital reference is 0.484891562413. The geometric-Asian quadrature reference is
181.285332694516, independently agreeing with the discrete lognormal closed form.
These are deterministic analytical checks, not Monte Carlo validation results.

As a separately labelled model-convention check, the same initial level interpreted
as a martingale futures price gives the Black-76 value 297.873559517241. Independent
zero-drift lognormal quadrature agrees (`secondary_black76_zero_drift_*` fields in
the reference artifact). This secondary calculation is not substituted for the
assignment's drift-r Monte Carlo validation or hedging model.

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
For the unit digital call the analytical references are
\(\Delta_D=e^{-rT}\phi(d_2)/(S_0\sigma\sqrt T)\) and
\(\nu_D=-e^{-rT}\phi(d_2)d_1/\sigma\), independently computed in the reference
artifact and compared with all finite-difference bumps below.
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
The recorded environment is macOS 26.6.2 on arm64, with 10 logical CPU cores,
Python 3.13.13, NumPy 2.2.6 and PyTorch 2.7.1 (`results/environment.json`). The
integrated hedging run uses one CPU thread and does not use the available MPS device.
Executed timing and memory results appear below; they are machine-specific.

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
Each cost-specific neural model was trained on the daily rebalancing grid only.
The same fitted model was subsequently evaluated at 5-day and 21-day rebalancing
frequencies without frequency-specific retraining; these comparisons therefore
measure transfer of the daily-trained policy to coarser decision grids.

## 8. Results and independent audit

<!-- GENERATED_RESULTS_START -->

The following tables are generated from `results/approach1/if_main/metrics.json` and `results/approach2/if_main/metrics.json`. Values are index-point payoffs/P&L; the digital pays one unit.

### Executed pricing and validation

| Payoff | NumPy price | SE | 95% lower | 95% upper | PyTorch price |
| --- | --- | --- | --- | --- | --- |
| asian | 188.5397 | 0.79075672 | 186.98981 | 190.08958 | 188.5397 |
| barrier | 55.364736 | 0.38565681 | 54.608849 | 56.120623 | 55.364736 |
| digital | 0.48586726 | 0.0013613426 | 0.48319903 | 0.48853549 | 0.48586726 |
| european | 329.97462 | 1.4302484 | 327.17134 | 332.77791 | 329.97462 |

European BS reference: 329.0776430123; MC z-score: 0.627151; criterion passed: True. Engines use identical shocks; their agreement checks arithmetic implementation, not independence of RNG libraries.

### Asian variance reduction

| Estimator | Price | SE | Variance of estimate | 95% lower | 95% upper |
| --- | --- | --- | --- | --- | --- |
| antithetic | 187.78394 | 0.58903504 | 0.34696227 | 186.62943 | 188.93844 |
| control_variate | 187.89186 | 0.019935527 | 0.00039742525 | 187.85278 | 187.93093 |
| naive | 187.60691 | 0.7883828 | 0.62154744 | 186.06168 | 189.15214 |

Independent-pilot coefficient beta: 1.031544394; A/G correlation 0.999680345. Production path budgets are matched; pilot overhead is 16384 additional paths (0.0838s).

| Method | Variance reduction | Bootstrap lower | Bootstrap upper |
| --- | --- | --- | --- |
| Antithetic | 1.7913978 | 1.755625 | 1.8280754 |
| Control variate | 1563.9355 | 1520.8636 | 1607.499 |

### Greeks

| Payoff | Greek | Fractional bump | FD | FD SE | Pathwise | Pathwise SE |
| --- | --- | --- | --- | --- | --- | --- |
| asian | delta | 0.001 | 0.53878253 | 0.0014968085 | 0.53875382 | 0.0014998857 |
| asian | delta | 0.01 | 0.53841263 | 0.0014697506 | 0.53875382 | 0.0014998857 |
| asian | delta | 0.1 | 0.53034474 | 0.001176386 | 0.53875382 | 0.0014998857 |
| asian | vega | 0.001 | 907.84191 | 4.3445256 | 907.84487 | 4.3445212 |
| asian | vega | 0.01 | 907.83779 | 4.3445209 | 907.84487 | 4.3445212 |
| asian | vega | 0.1 | 907.81291 | 4.3444716 | 907.84487 | 4.3445212 |
| barrier | delta | 0.001 | -0.0010273044 | 0.010712753 | invalid pathwise | — |
| barrier | delta | 0.01 | -0.0091585773 | 0.0034095292 | invalid pathwise | — |
| barrier | delta | 0.1 | -0.0066953026 | 0.00063950748 | invalid pathwise | — |
| barrier | vega | 0.001 | -577.83185 | 89.751207 | invalid pathwise | — |
| barrier | vega | 0.01 | -530.94718 | 27.682497 | invalid pathwise | — |
| barrier | vega | 0.1 | -556.47517 | 8.3146384 | invalid pathwise | — |
| digital | delta | 0.001 | 0.00051132882 | 2.1938411e-05 | invalid pathwise | — |
| digital | delta | 0.01 | 0.00051416429 | 6.8249471e-06 | invalid pathwise | — |
| digital | delta | 0.1 | 0.00049802104 | 1.6777401e-06 | invalid pathwise | — |
| digital | vega | 0.001 | -0.35505495 | 0.083681826 | invalid pathwise | — |
| digital | vega | 0.01 | -0.36491759 | 0.026810436 | invalid pathwise | — |
| digital | vega | 0.1 | -0.34045824 | 0.0081407919 | invalid pathwise | — |
| european | delta | 0.001 | 0.56787016 | 0.0016145232 | 0.5679659 | 0.0016160987 |
| european | delta | 0.01 | 0.56780407 | 0.0016002422 | 0.5679659 | 0.0016160987 |
| european | delta | 0.1 | 0.56142402 | 0.0014440091 | 0.5679659 | 0.0016160987 |
| european | vega | 0.001 | 1562.7005 | 8.0854666 | 1562.6918 | 8.0854865 |
| european | vega | 0.01 | 1562.7234 | 8.085378 | 1562.6918 | 8.0854865 |
| european | vega | 0.1 | 1562.6886 | 8.0852148 | 1562.6918 | 8.0854865 |

| Paired PW minus FD | Mean difference | Paired SE | Difference/SE |
| --- | --- | --- | --- |
| asian_delta_h0.001 | -2.8706518e-05 | 6.6839153e-05 | -0.42948657 |
| asian_delta_h0.01 | 0.00034118666 | 0.00020915945 | 1.6312276 |
| asian_delta_h0.1 | 0.00840908 | 0.00066605013 | 12.625296 |
| asian_vega_h0.001 | 0.0029532928 | 0.0021257535 | 1.3892922 |
| asian_vega_h0.01 | 0.0070731765 | 0.00791959 | 0.89312408 |
| asian_vega_h0.1 | 0.031953307 | 0.024675745 | 1.2949277 |
| vanilla_delta_h0.001 | 9.5736625e-05 | 5.1592548e-05 | 1.8556289 |
| vanilla_delta_h0.01 | 0.00016182885 | 0.00015541125 | 1.0412943 |
| vanilla_delta_h0.1 | 0.0065418769 | 0.00050966005 | 12.835766 |
| vanilla_vega_h0.001 | -0.0086934883 | 0.0060679512 | -1.4326892 |
| vanilla_vega_h0.01 | -0.031646866 | 0.021756684 | -1.4545814 |
| vanilla_vega_h0.1 | 0.0031177806 | 0.059401797 | 0.052486301 |

At the 10% spot bump, absolute paired delta difference/SE values are 12.836 (vanilla) and 12.625 (Asian). Thus agreement must be assessed bump by bump rather than asserted universally. Large spot bumps introduce smoothing/truncation bias. Discontinuous payoffs can have rising finite-difference variance as bumps shrink; a sample SE is not an estimate of truncation bias.

| Digital Greek | Bump fraction | FD | Analytical | (FD-analytical)/SE |
| --- | --- | --- | --- | --- |
| delta | 0.001 | 0.00051132882 | 0.0005183591 | -0.32045528 |
| delta | 0.01 | 0.00051416429 | 0.0005183591 | -0.61462963 |
| delta | 0.1 | 0.00049802104 | 0.0005183591 | -12.122296 |
| vega | 0.001 | -0.35505495 | -0.35116358 | -0.046501953 |
| vega | 0.01 | -0.36491759 | -0.35116358 | -0.5130094 |
| vega | 0.1 | -0.34045824 | -0.35116358 | 1.3150238 |

### Convergence

| Estimator | RMSE slope | OLS slope SE | Bootstrap lower | Bootstrap upper | SE-versus-N slope |
| --- | --- | --- | --- | --- | --- |
| vanilla_naive | -0.52434448 | 0.033794341 | -0.5953035 | -0.46284215 | -0.49967175 |
| asian_naive | -0.54901151 | 0.046514527 | -0.60536454 | -0.48312048 | -0.50059834 |
| asian_anti | -0.4928473 | 0.02837916 | -0.55032679 | -0.43404393 | -0.50146489 |
| asian_cv | -0.51292858 | 0.022532822 | -0.57810601 | -0.44086546 | -0.49732015 |

Independent Asian reference: `{"antithetic": true, "beta": 1.0320540353027041, "method": "geometric control-variate (independent pilot beta, chunked antithetic production, distinct chunk seeds)", "n": 2097152, "n_pilot": 16384, "n_simulated_paths": 4194304, "price": 187.90663391055116, "se": 0.003498590570753974}`. Bootstrap slope intervals condition on this estimated reference and pilot; they do not integrate reference uncertainty. The original noisier reference produced a flat CV slope and is retained under `results/archive/initial_option_a`; the redesigned reference reduces, but does not eliminate, that uncertainty.

### Profiling

| Loop seconds | NumPy seconds | Speedup | Loop peak bytes | NumPy peak bytes |
| --- | --- | --- | --- | --- |
| 0.17263342 | 0.0075886168 | 22.748996 | 8096512 | 32352555 |

These are matched supplied-shock workloads. Memory values are traced allocations, not total process RSS. The raw hot-loop profile is stored in the metrics history. CPU contention can affect wall-clock measurements.

### Full-size neural training

Architecture: [2, 32, 32, 1], SiLU hidden activations, unconstrained scalar output. Training paths: 131072; validation: 16384; test: 16384. Seeds: `{"initialization": [72, 73, 74], "test": 62, "train": 42, "validation": 52}`. Adam learning rates: weights 0.001, eta 1.0; batch size 2048; maximum epochs 20; validation patience 5; CPU threads 1.

| Training cost | Epochs | Best validation CVaR | Restored eta |
| --- | --- | --- | --- |
| 0.0 | 20 | 402.849 | 393.25513 |
| 0.001 | 20 | 422.00179 | 411.99515 |
| 0.005 | 20 | 525.2261 | 515.07263 |

Policies train on the daily grid. The 5/21-day results evaluate those same policies at fewer decision times; they are not separately optimized frequency-specific networks. Restored models minimize validation CVaR among observed epochs; convergence to the global CVaR optimum is not established.

### All held-out frequency/cost benchmarks

| Days | Cost | Policy | CVaR95 | Mean P&L | P&L variance | Mean cost | Turnover (units) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | BS | 346.7973 | -301.19415 | 565.81746 | 0 | 5.5961944 |
| 1 | 0 | Neural | 402.12687 | -299.42217 | 14703.596 | 0 | 6.4519798 |
| 1 | 0 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |
| 1 | 0.001 | BS | 375.49166 | -323.39668 | 650.53601 | 22.202533 | 5.5961944 |
| 1 | 0.001 | Neural | 421.32619 | -324.72748 | 7058.6419 | 23.568522 | 5.8893197 |
| 1 | 0.001 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |
| 1 | 0.005 | BS | 509.48447 | -412.20682 | 2118.0748 | 111.01267 | 5.5961944 |
| 1 | 0.005 | Neural | 524.87472 | -404.18682 | 9456.3986 | 104.35372 | 5.1440906 |
| 1 | 0.005 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |
| 5 | 0 | BS | 387.91154 | -300.66865 | 1574.3012 | 0 | 2.8261092 |
| 5 | 0 | Neural | 437.19559 | -299.25037 | 15269.47 | 0 | 3.1954183 |
| 5 | 0 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |
| 5 | 0.001 | BS | 402.68967 | -311.88286 | 1616.6152 | 11.2142 | 2.8261092 |
| 5 | 0.001 | Neural | 440.08242 | -312.66623 | 7877.2423 | 11.725376 | 2.9314974 |
| 5 | 0.001 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |
| 5 | 0.005 | BS | 465.7312 | -356.73966 | 2012.1613 | 56.071002 | 2.8261092 |
| 5 | 0.005 | Neural | 502.18964 | -352.8236 | 10730.237 | 53.142765 | 2.6264402 |
| 5 | 0.005 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |
| 21 | 0 | BS | 473.55041 | -300.75766 | 5600.503 | 0 | 1.6070304 |
| 21 | 0 | Neural | 525.3505 | -299.46951 | 17576.002 | 0 | 1.7625975 |
| 21 | 0 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |
| 21 | 0.001 | BS | 481.74766 | -307.13436 | 5662.9586 | 6.3766981 | 1.6070304 |
| 21 | 0.001 | Neural | 511.89303 | -307.66795 | 10958.943 | 6.5153507 | 1.6306682 |
| 21 | 0.001 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |
| 21 | 0.005 | BS | 514.93045 | -332.64115 | 5951.2701 | 31.883491 | 1.6070304 |
| 21 | 0.005 | Neural | 545.03467 | -330.34811 | 13924.078 | 30.575493 | 1.5192124 |
| 21 | 0.005 | Zero | 1908.7772 | -336.24283 | 273370.05 | 0 | 0 |

### Paired bootstrap uncertainty

| Combination | Observed neural minus BS CVaR | 95% lower | 95% upper |
| --- | --- | --- | --- |
| f1_k0.0 | 55.32957 | 54.232812 | 56.438022 |
| f1_k0.001 | 45.834537 | 44.251592 | 47.344723 |
| f1_k0.005 | 15.390245 | 12.981025 | 17.429697 |
| f21_k0.0 | 51.800092 | 47.029582 | 57.043398 |
| f21_k0.001 | 30.145367 | 25.613589 | 33.87183 |
| f21_k0.005 | 30.104227 | 25.420468 | 35.072887 |
| f5_k0.0 | 49.284053 | 46.62826 | 51.768726 |
| f5_k0.001 | 37.392757 | 34.818548 | 40.181876 |
| f5_k0.005 | 36.458432 | 32.939597 | 39.483282 |

Negative differences favor neural hedging; intervals crossing zero are inconclusive. These are marginal paired-path bootstrap intervals conditional on each fitted model, without multiplicity adjustment or independent retraining. No general outperformance claim follows from one fitted network per cost.

Recorded approach wall times: A 61.236s, B 426.358s. Source figures: `results/approach1/if_main/{convergence,greeks,profiling}.html` and `results/approach2/if_main/{training,comparison,loss_histogram}.html`.

<!-- GENERATED_RESULTS_END -->

## 9. Reproducibility and limitations

See the ten-line README for frozen installation and offline execution. The source
data snapshots are committed; regenerable results are ignored. Full reproduction
must execute training rather than merely load checkpoints. The live session export
must be refreshed after the final response. Submission tagging remains pending.
Exact post-session export and submission steps are in `FINALIZE.md`.
The inherited `submission` tag points to old commit `0a27070` and has deliberately
not been moved; it is not evidence of acceptance of this independent rebuild.

Full clean-clone reproduction executed from commit
`4c6c8aff36aa8652be7214b9eda49b5c943dbf36`, with a clean source tree, a new virtual
environment and `uv sync --frozen --offline`. The Python/uv package cache was already
populated inside the original repository; this is not a claim of cold-machine
offline package installation. No live market-data retrieval was required.
All eight pipeline stages returned zero, including full neural retraining (about
427.5 seconds for the B process), independent held-out replay and comparison output.
Evidence is in `results/clean_clone/20261104T090738342929/evidence.json` and the clone's
timestamped `results/reproduction/` manifest. Across original and reproduced runs,
325 pricing numerical fields and 631 hedging/history numerical fields agreed within
the explicitly checked tolerances; timing fields were not treated as deterministic.
The numerical source files and lockfile hashes were identical. The reproduced B
artifacts were promoted to the canonical `results/approach2/if_main/` location after
archiving the earlier exploratory full run. `results/reproduction_comparison.json`
records these comparisons. Later local-pytest configuration/report edits do not
change those numerical source files or dependencies.

An earlier all-tests run encountered 17 failures from unexpected untracked tests
using a different API. With explicit user authorization, those files were preserved
under `.cache/unattributed-tests/` without reuse; the current independent suite then
passed all 31 tests. Earlier direct pytest runs did not explicitly set a local
temporary base directory; their write-locality compliance is therefore not verified.
The configuration now enforces `.cache/pytest-tmp`. This historical uncertainty is
retained rather than represented as a clean compliance pass.

Limitations include continuous-contract roll contamination, constant volatility,
the spot-versus-futures convention, simplified cash-flow timing, model-implied rather
than physical loss tails, discrete barrier monitoring, and single-seed training
variability unless additional full independent fits are actually executed.
In this executed 20-epoch experiment, all nine matched neural-minus-BS CVaR point
differences are positive, and all nine marginal paired-bootstrap intervals exclude
zero on the positive side. Thus these particular trained policies underperform the
BS benchmark on held-out CVaR. This finding is conditional on the model, architecture,
training budget and single set of fitted networks; it is not a general impossibility
result for deep hedging. All policies reached the 20-epoch budget, so the experiment
does not establish optimization convergence.

## 10. Conclusion

Under the assignment-prescribed GBM, the European Monte Carlo price passed the
Black–Scholes validation with a z-score of 0.627151. The discrete geometric-Asian
control variate reduced the Asian estimator's variance by approximately 1563.94
times at matched production path budgets, with additional pilot cost reported
separately. These results support the numerical implementation within the stated
model and monitoring conventions.

Across all nine matched cost–frequency combinations, the three fitted neural
policies had higher held-out CVaR than BS delta hedging; all corresponding marginal
paired-bootstrap intervals for neural-minus-BS CVaR were positive. Less frequent
rebalancing reduced turnover and, at positive cost rates, transaction costs, but
did not uniformly reduce tail loss. For BS hedging, daily rebalancing gave the lowest
observed CVaR at zero and 10 bp costs, whereas 5-day rebalancing gave the lowest at
50 bp; moving further
to 21 days increased CVaR at every cost level relative to 5 days.

These findings are conditional on the IF0 calibration, possible continuous-contract
roll effects, the prescribed spot-style GBM and accounting conventions, and the
20-epoch training budget. There is one fitted network per cost, trained on the daily
grid; the coarser-grid results do not represent frequency-specific retraining.
The paired-bootstrap intervals describe test-path uncertainty conditional on those
fits, not training variability or general superiority of either strategy.

## 11. Statement of work

OpenAI established the shared mathematical conventions, independently retrieved and
verified data, constructed quadrature references and infrastructure, and drafts this
report. DeepSeek implemented the numerical engines, estimator experiments and initial
neural training code, then repaired confirmed pricing defects. OpenAI integrated the
hedging precision/checkpoint repairs and executed full-size training, independent replay
and clean-clone reproduction. Kimi performed independent read-only reviews of the data
and integrated algorithms. Actual provider identifiers and
child session IDs are recorded in RESEARCH_PROTOCOL.md and authentic session exports.
The student group must review the assumptions, mathematical decisions, empirical
verification and final English report; this draft is not a record of group approval.
