# Question 2 — Monte Carlo Pricing and CVaR Hedging on Chinese Futures

## 1. Scope and model convention

This study addresses both options of Question 2. Option A develops Monte Carlo estimators for European, arithmetic Asian, up-and-out barrier and cash-or-nothing digital calls, with analytical checks, variance reduction, Greek estimation and computational benchmarks. Option B trains a neural hedging policy by minimizing the 95% conditional value-at-risk (CVaR) of the terminal hedging loss and compares it with Black–Scholes (BS) delta hedging and an unhedged position.

The IF0 continuous futures series is used for historical calibration. Following the assignment's specified simulation convention, however, the pricing and hedging experiments use a *spot-style* risk-neutral geometric Brownian motion (GBM):

$$
 dS_t=rS_t\,dt+\sigma S_t\,dW_t.
$$

This convention makes the BS call the primary analytical reference. A zero-drift futures-price model and the Black–76 formula represent a different modelling convention; we report a separate check but do not mix the two. All reported hedging losses are simulated under the stated model and should not be interpreted as forecasts of actual futures-trading losses.

## 2. Market data and calibration

### 2.1 Sources and preprocessing

The historical inputs comprise the Sina IF0 continuous futures series, the CSI 300 index and SHIBOR tenor quotations. The saved dataset contains 1,899 IF observations and 1,899 CSI 300 observations dated 2 January 2019–3 November 2026, and 1,947 SHIBOR observations. Data provenance, retrieval times, units, quality checks and hashes are recorded in `data/metadata.json`.

The volatility calibration uses the six-year window from 3 November 2020 to 3 November 2026: 1,455 prices and 1,454 daily log returns. No IF observations are missing relative to the CSI 300 trading dates in that window. Returns and annualized estimates are defined by

$$
 x_i=\log(S_i/S_{i-1}),\qquad
 \widehat\sigma=\sqrt{252\,s_x^2},\qquad
 \widehat\mu=252\bar x+\tfrac12\widehat\sigma^2.
$$

The historical drift is reported for reference only; the risk-neutral simulation uses $r$. The data check distinguishes exchange holidays from actual missing observations, rejects duplicate dates and missing numerical values, and compares the futures and index calendars. Annualized mean log-return uncertainty is also assessed with conventional and Bartlett heteroskedasticity-and-autocorrelation-consistent (HAC) standard errors. Neither method corrects potential contract-roll effects.

The IF0 series is not independently back-adjusted. Exact constituent contracts and roll dates are unavailable from the used endpoint; unusually large changes in the IF–index basis are diagnostic observations, not confirmed roll dates. Consequently, the estimated volatility may contain continuous-contract roll effects.

### 2.2 Parameter estimates

| Quantity | Estimate |
|---|---:|
| Initial price $S_0$ and strike $K$ | 3978.80 |
| Knockout barrier $B=1.2S_0$ | 4774.56 |
| Annualized volatility $\widehat\sigma$ | 0.190647912849 |
| Annualized historical GBM drift $\widehat\mu$ | -0.012779861736 |
| Annualized mean log return | -0.030953175072 |
| One-year continuously compounded rate $r$ | 0.014287446608 |

The detailed calibration is in `data/clean/calibration.json`. SHIBOR quotes are treated as simple deposit rates at fixed tenor fractions (days/365 for weekly tenors and months/12 for monthly tenors). The implied zero yield is approximated by $z(u)=\log[1+q(u)u]/u$, with linear interpolation in zero yields and discount factor $e^{-z(T)T}$. The one-year experiment uses the observed one-year node. This construction is a practical deposit-curve approximation, not an arbitrage-free bootstrapped term structure.

## 3. Option A — Monte Carlo pricing

### 3.1 Simulation and payoffs

Set $T=1$ year, $m=252$ monitoring dates, $\Delta t=T/m$, $K=S_0$ and $B=1.2S_0$. With independent standard normal innovations $Z_i$,

$$
 S_{i+1}=S_i\exp\left[\left(r-\frac{\sigma^2}{2}\right)\Delta t
 +\sigma\sqrt{\Delta t}\,Z_i\right].
$$

The four payoff definitions are

$$
\begin{aligned}
H_{\mathrm{Euro}}&=(S_T-K)^+,\\
H_{\mathrm{Asian}}&=\left(\frac1m\sum_{i=1}^m S_{t_i}-K\right)^+,\\
H_{\mathrm{barrier}}&=(S_T-K)^+\mathbf1\!\left\{\max_{1\le i\le m}S_{t_i}<B\right\},\\
H_{\mathrm{digital}}&=\mathbf1\{S_T>K\}.
\end{aligned}
$$

The arithmetic average and barrier maximum are based on the 252 future monitoring points; $S_0$ is excluded. The barrier is monitored discretely. Every simulated price is the sample mean of $e^{-rT}H$. NumPy and PyTorch use the same innovations for cross-engine arithmetic verification; matching outputs therefore do not constitute independent random-number replication.

For a European call the analytical reference is

$$
 C_{\mathrm{BS}}=S_0\Phi(d_1)-Ke^{-rT}\Phi(d_2),\qquad
 d_1=\frac{\log(S_0/K)+(r+\sigma^2/2)T}{\sigma\sqrt T},\quad
 d_2=d_1-\sigma\sqrt T.
$$

Independent quadrature reproduces the BS price $329.0776430$, delta $0.5675993$ and vega $1564.4691495$ (vega per unit absolute volatility). The unit digital analytical price is $0.48489156$. As a separate check under a zero-drift futures-price process, the Black–76 call price is $297.8735595$; it is not the reference used to validate the principal simulation.

### 3.2 Prices and validation

The main pricing run gives the following results (prices in index points, except the unit digital):

| Payoff | NumPy price | SE | 95% lower | 95% upper | PyTorch price |
| --- | --- | --- | --- | --- | --- |
| asian | 188.5397 | 0.79075672 | 186.98981 | 190.08958 | 188.5397 |
| barrier | 55.364736 | 0.38565681 | 54.608849 | 56.120623 | 55.364736 |
| digital | 0.48586726 | 0.0013613426 | 0.48319903 | 0.48853549 | 0.48586726 |
| european | 329.97462 | 1.4302484 | 327.17134 | 332.77791 | 329.97462 |

The European Monte Carlo estimate is $329.97462$, with standard error $1.43025$. Relative to the BS reference, $z=(329.97462-329.0776430)/1.43025=0.62715$, satisfying the assignment's $|z|\leq3$ validation criterion. The observed agreement between engines verifies their consistent evaluation of the specified payoffs.

### 3.3 Variance reduction for the arithmetic Asian call

Two variance-reduction estimators are compared with independent discounted-payoff sampling. In the antithetic estimator, the independent sampling unit is the average of the outcomes generated by $(Z,-Z)$; uncertainty is calculated at the *pair* level. For the control variate, the correlated variable is the discounted discrete geometric-Asian call. Its coefficient is fitted on a separate pilot sample and then fixed in the production run.

Let $G=\exp[m^{-1}\sum_i\log S_{t_i}]$. Since $\log G$ is normal, define

$$
 a=\log S_0+(r-\sigma^2/2)\bar t,\qquad
 b=\frac{\sigma^2}{m^2}\sum_{i=1}^m\sum_{j=1}^m\min(t_i,t_j).
$$

Then the known expectation of the discounted geometric-Asian call is

$$
 \mathbb E[e^{-rT}(G-K)^+]
 =e^{-rT}\left[e^{a+b/2}\Phi\!\left(\frac{a+b-\log K}{\sqrt b}\right)
 -K\Phi\!\left(\frac{a-\log K}{\sqrt b}\right)\right].
$$

The production estimates are:

| Estimator | Price | SE | Variance of estimate | 95% lower | 95% upper |
| --- | --- | --- | --- | --- | --- |
| antithetic | 187.78394 | 0.58903504 | 0.34696227 | 186.62943 | 188.93844 |
| control_variate | 187.89186 | 0.019935527 | 0.00039742525 | 187.85278 | 187.93093 |
| naive | 187.60691 | 0.7883828 | 0.62154744 | 186.06168 | 189.15214 |

These figures come from a separate variance-reduction experiment and therefore need not equal the naive Asian estimate in the main pricing table. The fitted pilot coefficient is $\widehat\beta=1.0315444$ and the arithmetic/geometric payoff correlation is $0.9996803$. Production path budgets are matched; the control variate uses another 16,384 pilot paths (approximately 0.0838 seconds).

| Method | Variance reduction | Bootstrap lower | Bootstrap upper |
| --- | --- | --- | --- |
| Antithetic | 1.7913978 | 1.755625 | 1.8280754 |
| Control variate | 1563.9355 | 1520.8636 | 1607.499 |

The control variate reduces the estimated sampling variance by a factor of **1563.94**, with a bootstrap interval of **[1520.86, 1607.50]** for the variance-reduction ratio. The antithetic improvement is more modest, at approximately 1.79. Variance reduction is evaluated using the estimator's independent sampling units and does not imply an identical runtime or total path budget once pilot sampling is included.

### 3.4 Greeks and estimator behaviour

For the European and arithmetic Asian calls, pathwise derivatives follow from

$$
 \frac{\partial S_t}{\partial S_0}=\frac{S_t}{S_0},\qquad
 \frac{\partial S_t}{\partial\sigma}=S_t(W_t-\sigma t).
$$

The payoff derivatives incorporate the appropriate in-the-money indicator and discount factor. For European calls, $\Delta_{\mathrm{BS}}=\Phi(d_1)$ and $\nu_{\mathrm{BS}}=S_0\phi(d_1)\sqrt T$. Central finite differences (FD) use common random numbers at spot and volatility bump fractions $0.001$, $0.01$ and $0.1$. For discontinuous digital and barrier payoffs, naive pathwise differentiation omits boundary terms and is invalid; FD is reported with its sampling uncertainty instead.

At the smallest bump, the European FD delta is $0.567870$ (SE $0.001615$), compared with pathwise delta $0.567966$ (SE $0.001616$). For the Asian call, the corresponding figures are $0.538783$ and $0.538754$. The European FD vega is $1562.701$ (SE $8.085$), compared with pathwise $1562.692$. These agree within their observed sampling variability at the small bump.

A large spot bump gives a different conclusion: the paired pathwise-minus-FD discrepancy is $12.836$ standard errors for the European delta and $12.625$ for the Asian delta at the $10\%$ bump. This is consistent with material finite-difference truncation bias. For the unit digital, the analytical delta is $0.0005183591$; the FD delta at the $10\%$ bump is $0.0004980210$, a discrepancy of $-12.122$ FD standard errors. The full bump-by-bump results are retained in Appendix A. Decreasing the bump is not a universal remedy, because finite-difference variance for discontinuous payoffs can increase as the perturbation shrinks.

### 3.5 Convergence and computational performance

Monte Carlo root mean squared error (RMSE) is assessed using independent repetitions at increasing simulation sizes. The estimated log–log slopes are:

| Estimator | RMSE slope | OLS slope SE | Bootstrap lower | Bootstrap upper | SE-versus-N slope |
| --- | --- | --- | --- | --- | --- |
| vanilla_naive | -0.52434448 | 0.033794341 | -0.5953035 | -0.46284215 | -0.49967175 |
| asian_naive | -0.54901151 | 0.046514527 | -0.60536454 | -0.48312048 | -0.50059834 |
| asian_anti | -0.4928473 | 0.02837916 | -0.55032679 | -0.43404393 | -0.50146489 |
| asian_cv | -0.51292858 | 0.022532822 | -0.57810601 | -0.44086546 | -0.49732015 |

The results are broadly consistent with the theoretical $N^{-1/2}$ sampling rate. The independent Asian numerical reference, computed with antithetic geometric control variates and $4,194,304$ simulated paths, is $187.906634$ with SE $0.003499$. The slope intervals condition on this estimated reference and do not include its uncertainty; a noisy numerical benchmark can introduce an apparent RMSE floor.

A matched-workload comparison of Python loops and NumPy vectorization produced:

| Loop seconds | NumPy seconds | Speedup | Loop peak bytes | NumPy peak bytes |
| --- | --- | --- | --- | --- |
| 0.17263342 | 0.0075886168 | 22.748996 | 8096512 | 32352555 |

NumPy was approximately **22.75 times faster** in this test, with higher traced peak memory (32.35 MB versus 8.10 MB). These values are trace-level allocations and machine-specific; they should not be interpreted as total resident memory usage. The recorded software environment is macOS on arm64 with Python 3.13.13, NumPy 2.2.6 and PyTorch 2.7.1.

## 4. Option B — Deep hedging

### 4.1 Trading P&L and risk criterion

The hedging target is the one-year European call. A policy $\Delta_j$ is selected at each rebalancing date $t_j$. The implemented terminal P&L is

$$
 V_T=\sum_j e^{r(T-t_j)}\Delta_j(S_{t_{j+1}}-S_{t_j})
 -\kappa\sum_j|\Delta_j-\Delta_{j-1}|S_{t_j}
 -(S_T-K)^+.
$$

Here $\Delta_{-1}=0$, $\kappa$ is the proportional transaction-cost rate, and $L=-V_T$ is terminal loss. This follows the assignment's explicit accounting convention: no initial option premium, no terminal liquidation charge, and no maturity accumulation on transaction costs. The same paths, payoffs and cash-flow convention are used for the neural policy, BS delta hedge and zero hedge.

For tail probability $1-\alpha=0.05$, training minimizes the Rockafellar–Uryasev representation

$$
 \operatorname{CVaR}_{0.95}(L)
 =\min_{\eta}\left\{\eta+\frac{1}{0.05}\,\mathbb E[(L-\eta)^+]\right\}.
$$

The neural weights and the auxiliary threshold $\eta$ are optimized jointly. Reported held-out CVaR is computed independently from the test-loss tail, including fractional tail mass if needed, rather than reporting the training objective as a test score.

### 4.2 Architecture and experimental design

The network has dimensions **2–32–32–1**, two SiLU hidden layers and an unrestricted scalar hedge output. Its inputs are normalized current price and time; no future price information is supplied. Adam is used with weight learning rate $0.001$, threshold learning rate $1.0$, minibatch size $2,048$, validation patience of five epochs and a maximum of 20 epochs. Each transaction-cost level has its own fitted network. The experiment uses 131,072 training paths, 16,384 validation paths and 16,384 held-out test paths. Train/validation/test seeds are 42/52/62; initialization seeds are 72, 73 and 74 for the three cost-specific fits.

The networks are trained with **daily** rebalancing. Each fitted network is evaluated on daily, 5-day and 21-day decision grids without retraining for the coarser grids. Consequently, those tests assess how a daily-trained policy transfers to less frequent execution; they are not separate optimal policies for each frequency.

| Training cost | Epochs | Best validation CVaR | Restored eta |
| --- | --- | --- | --- |
| 0.0 | 20 | 402.849 | 393.25513 |
| 0.001 | 20 | 422.00179 | 411.99515 |
| 0.005 | 20 | 525.2261 | 515.07263 |

All three fits reached the 20-epoch budget. The restored checkpoints had the best validation CVaR among the observed epochs, but this does not establish convergence to a global or even a stationary CVaR optimum.

### 4.3 Held-out hedging comparison

Rebalancing intervals are 1, 5 and 21 trading days; transaction-cost rates are $0$, $0.001$ (10 basis points) and $0.005$ (50 basis points). The comparison below reports the empirical 95% CVaR of terminal loss. Smaller CVaR indicates better tail-loss performance.

| Rebalance (days) | Cost (bp) | BS CVaR$_{95}$ | Neural CVaR$_{95}$ | Neural − BS | 95% paired bootstrap CI |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 346.797 | 402.127 | +55.330 | [54.233, 56.438] |
| 1 | 10 | 375.492 | 421.326 | +45.835 | [44.252, 47.345] |
| 1 | 50 | 509.484 | 524.875 | +15.390 | [12.981, 17.430] |
| 5 | 0 | 387.912 | 437.196 | +49.284 | [46.628, 51.769] |
| 5 | 10 | 402.690 | 440.082 | +37.393 | [34.819, 40.182] |
| 5 | 50 | 465.731 | 502.190 | +36.458 | [32.940, 39.483] |
| 21 | 0 | 473.550 | 525.351 | +51.800 | [47.030, 57.043] |
| 21 | 10 | 481.748 | 511.893 | +30.145 | [25.614, 33.872] |
| 21 | 50 | 514.930 | 545.035 | +30.104 | [25.420, 35.073] |

The zero-hedge CVaR is **1908.777** under every cost/frequency combination because the zero policy never trades. Both BS and neural strategies substantially reduce tail exposure relative to remaining unhedged. However, the fitted neural policy has **higher** held-out CVaR than BS delta hedging in **all nine** comparisons. The observed neural-minus-BS differences range from 15.390 to 55.330 index points. All nine marginal paired-bootstrap intervals are strictly positive.

The intervals are constructed by resampling matched held-out paths with the two policies evaluated on the same shocks. They describe test-set sampling variability **conditional on the trained policies**. They do not account for parameter re-estimation, multiple-comparison adjustments or variability across independent training runs. Thus the experiment establishes underperformance for these particular trained policies and settings, not a general failure of deep hedging.

Transaction costs create a clear trade-off between rebalancing frequency and hedge error. For BS hedging, the lowest observed CVaR occurs with daily rebalancing at zero cost and 10 bp, but with **5-day** rebalancing at 50 bp (465.731, compared with 509.484 for daily). Rebalancing every 21 days increases BS tail loss relative to every five days at each cost. The full mean-P&L, P&L variance, transaction-cost and turnover results are presented in Appendix B.

## 5. Reproducibility and independent verification

The project uses a pinned Python 3.13 environment (`pyproject.toml`, `uv.lock`), committed data snapshots and an offline data-validation route in `data/fetch.py`. The `README.md` gives the repository-specific execution sequence. Main numerical outputs are written to `results/approach1/if_main/metrics.json` and `results/approach2/if_main/metrics.json`; figures are generated in `results/data/`, `results/approach1/if_main/` and `results/approach2/if_main/`. These `results/` files are regenerated by the pipeline and are not assumed to be committed to Git.

Independent analytical references and review are documented in `tools/independent_reference.py`, `AUDIT.md` and `RESEARCH_PROTOCOL.md`. A clean-clone execution using the pinned dependencies returned **PASS**, including full neural retraining and held-out evaluation, rather than merely reloading a saved model. In the documented independent reproduction, 325 numerical pricing fields and 631 numerical hedging/history fields agreed with the reference outputs within the prescribed numerical tolerances; machine-dependent timings were excluded from equality checks. A further full clean-clone run on 5 November 2026 also returned PASS.

The final OpenCode parent and two child sessions were exported into `SESSION.md` with `"final": true`, including native message and tool records. The source repository preserves the development commits. The final GitHub merge, group report approval and repository-wide submission tag are separate submission steps; these are not treated as completed numerical experiments.

Earlier direct `pytest` runs did not explicitly enforce a repository-local temporary directory, so their historical write-locality compliance could not be verified retroactively. The configuration now points temporary test output to `.cache/pytest-tmp`; the unresolved historical uncertainty remains disclosed in `AUDIT.md`.

## 6. Discussion and limitations

The principal pricing check is consistent with the prescribed GBM: the European Monte Carlo estimate is within three estimated standard errors of the BS value. The very high arithmetic/geometric Asian payoff correlation explains the unusually large control-variate gain. For Greek estimation, common random numbers improve paired comparisons, but large finite-difference bumps generate material bias and discontinuous payoffs require particular care.

The hedge comparison depends on several modelling choices. IF0 may contain contract-roll discontinuities; volatility and the discount curve are held fixed; the hedge dynamics use a spot-style drift-$r$ GBM rather than a futures martingale; the barrier is discretely monitored; and the P&L cash flows follow the assignment's simplified convention. The simulation assesses risk under that model, not under an empirically estimated physical distribution of future returns.

The neural results are limited by one network architecture, one fitted network per cost level and a 20-epoch training budget. In particular, the five- and 21-day evaluations do not retrain their policies for those frequencies. Paired bootstrap confidence intervals assess held-out-path uncertainty, but cannot rule out better outcomes under new initializations, longer optimization, richer policy states or alternative risk specifications. None of those unexecuted alternatives is claimed as an empirical result.

## 7. Conclusion

The Monte Carlo pricing implementation passes the European BS validation criterion ($z=0.627$). The discrete geometric-Asian control variate reduces the variance of the arithmetic Asian estimator by approximately **1564 times**, while convergence estimates remain close to the expected square-root Monte Carlo rate. Vectorization substantially reduces runtime at the cost of additional memory allocation.

In the hedging experiment, both tested policies improve markedly on the unhedged tail-loss benchmark. BS delta hedging nevertheless has a lower 95% CVaR than each of the three trained neural policies at every evaluated transaction-cost and rebalancing-frequency combination. When transaction costs are high, more frequent rebalancing is not automatically preferable: the lowest BS tail loss at 50 bp occurs on the 5-day grid. These conclusions apply to the calibrated model, specified accounting rules and executed training budget.

## 8. Statement of work

The implementation used OpenCode with an OpenAI lead agent, a DeepSeek numerical subagent and a Kimi independent-audit subagent. OpenAI established model conventions, assembled and checked market data, produced independent analytical references, integrated the code and ran the full reproduction checks. DeepSeek developed and tested the numerical pricing and initial neural-hedging components and repaired identified defects. Kimi examined the data and algorithms independently and recorded review findings. The genuine session history is preserved in `SESSION.md`, with supporting roles and identifiers in `RESEARCH_PROTOCOL.md`.

The student group is responsible for reviewing and approving the mathematical assumptions, quantitative interpretation, report text and final submission. Agent-generated calculations and wording are treated as material to be verified against the executed outputs, not as substitutes for that responsibility.

## Appendix A. Detailed Greek sensitivity results

The bump fraction is relative to $S_0$ for delta and to $\sigma$ for vega. FD and pathwise standard errors are reported separately. A dash indicates that naive pathwise differentiation is invalid for the discontinuous payoff.

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

The paired pathwise-minus-FD differences below use the same simulation shocks and therefore have smaller standard errors than differences between independent estimates:

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

Analytical digital checks are:

| Digital Greek | Bump fraction | FD | Analytical | (FD-analytical)/SE |
| --- | --- | --- | --- | --- |
| delta | 0.001 | 0.00051132882 | 0.0005183591 | -0.32045528 |
| delta | 0.01 | 0.00051416429 | 0.0005183591 | -0.61462963 |
| delta | 0.1 | 0.00049802104 | 0.0005183591 | -12.122296 |
| vega | 0.001 | -0.35505495 | -0.35116358 | -0.046501953 |
| vega | 0.01 | -0.36491759 | -0.35116358 | -0.5130094 |
| vega | 0.1 | -0.34045824 | -0.35116358 | 1.3150238 |

## Appendix B. Additional held-out hedging diagnostics

Reported transaction costs are in the same index-point convention as the hedging P&L. Turnover is the sum of absolute changes in hedge positions. The zero hedge has mean P&L $-336.243$, P&L variance $273370.05$, zero transaction cost and zero turnover throughout.

| Days | Cost (bp) | Policy | Mean P&L | P&L variance | Mean transaction cost | Turnover |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0 | BS | -301.194 | 565.817 | 0.000 | 5.596 |
| 1 | 0 | Neural | -299.422 | 14703.596 | 0.000 | 6.452 |
| 1 | 10 | BS | -323.397 | 650.536 | 22.203 | 5.596 |
| 1 | 10 | Neural | -324.727 | 7058.642 | 23.569 | 5.889 |
| 1 | 50 | BS | -412.207 | 2118.075 | 111.013 | 5.596 |
| 1 | 50 | Neural | -404.187 | 9456.399 | 104.354 | 5.144 |
| 5 | 0 | BS | -300.669 | 1574.301 | 0.000 | 2.826 |
| 5 | 0 | Neural | -299.250 | 15269.470 | 0.000 | 3.195 |
| 5 | 10 | BS | -311.883 | 1616.615 | 11.214 | 2.826 |
| 5 | 10 | Neural | -312.666 | 7877.242 | 11.725 | 2.931 |
| 5 | 50 | BS | -356.740 | 2012.161 | 56.071 | 2.826 |
| 5 | 50 | Neural | -352.824 | 10730.237 | 53.143 | 2.626 |
| 21 | 0 | BS | -300.758 | 5600.503 | 0.000 | 1.607 |
| 21 | 0 | Neural | -299.470 | 17576.002 | 0.000 | 1.763 |
| 21 | 10 | BS | -307.134 | 5662.959 | 6.377 | 1.607 |
| 21 | 10 | Neural | -307.668 | 10958.943 | 6.515 | 1.631 |
| 21 | 50 | BS | -332.641 | 5951.270 | 31.883 | 1.607 |
| 21 | 50 | Neural | -330.348 | 13924.078 | 30.575 | 1.519 |

## Appendix C. Output and figure locations

| Output | Location |
|---|---|
| Input metadata and calibration | `data/metadata.json`, `data/clean/calibration.json` |
| Independent analytical reference | `results/independent/reference.json` |
| Pricing metrics | `results/approach1/if_main/metrics.json` |
| Hedging metrics | `results/approach2/if_main/metrics.json` |
| Data and curve figures | `results/data/calibration.html`, `results/data/curve.html` |
| Pricing figures | `results/approach1/if_main/convergence.html`, `greeks.html`, `profiling.html` |
| Hedging figures | `results/approach2/if_main/training.html`, `comparison.html`, `loss_histogram.html` |
| Independent review and execution guidance | `AUDIT.md`, `RESEARCH_PROTOCOL.md`, `README.md` |

The metrics and HTML figures under `results/` are reproducible outputs. If the project is submitted without generated artifacts, they must be regenerated using the repository instructions before inspecting those files.
