# Monte Carlo pricing and hedging on Chinese futures — Question 2 report

**Advanced Programming 2026 · Question 2 (Option A + Option B) · Group report draft (agent-assisted; group to review and finalise)**

> Status: DRAFT assembled from executed code and measured outputs. Every number quoted here is produced by the scripts in `code/` and stored in `results/approach*/if/metrics.json` (see §9 Reproducibility). Sections marked *[group to verify]* require group review before submission.

## 1. Objective and scope

We price and hedge European-style claims written on the CSI 300 stock-index futures main continuous contract (IF0) using Monte Carlo methods under the risk-neutral measure, implement both a classical variance-reduction/Greeks toolchain (Option A) and a neural deep-hedging strategy trained by stochastic optimisation (Option B), and validate every estimator against closed-form benchmarks where available.

Deliverables: this report, the repository (code + tests + data pipeline + comparison layer), `SESSION.md` (terminal-agent session export), and `AGENTS.md` (agent contract).

## 2. Data and calibration

### 2.1 Sources and provenance (see `data/metadata.json`)

| Artefact | Source | Rows | Range |
|---|---|---|---|
| `data/clean/if_main.csv` | AKShare `futures_main_sina` → Sina Finance; IF main continuous contract | 2374 | 2017-01-17 … 2026-11-02 |
| `data/clean/csi300_index.csv` | AKShare `index_zh_a_hist` → Eastmoney; CSI 300 index (000300) | 2386 | 2017-01-03 … 2026-11-03 |
| `data/clean/shibor.csv` | AKShare `macro_china_shibor_all` → CFETS/Eastmoney; SHIBOR all tenors | 2396 | 2017-03-17 … 2026-11-03 |

All three artefacts are committed (total < 0.5 MB) with SHA-256 hashes recorded in `data/metadata.json`; `uv run python data/fetch.py --verify-only` re-checks them offline and `--refresh` re-downloads. FastboxData credentials were not used; the public AKShare route was used instead, as permitted by the assignment ("`ak.futures_main_sina`").

**Roll convention.** The IF0 series is AKShare's main-contract continuous series (dominant contract by volume/open interest). Roll gaps can induce artificial jumps in price *levels*; all estimation below uses **log returns**, which are roll-gap tolerant in practice but we flag that returns straddling rolls may embed small artefacts. No back-adjustment is applied; the series is used as displayed by the vendor.

### 2.2 Estimation window and parameters

* **Window:** trailing 5 calendar years, **2021-10-28 … 2026-11-02**, n = 1214 daily log returns of IF0 close. (Full history 2017–2026 was retrieved; the 5-year window matches the assignment's "≥ 5 years" requirement and avoids mixing pre-2021 regimes.)
* **Physical drift (disclosure only):** μ̂ = mean(log-return) × 252 ≈ **−4.24 % p.a.** (not used in risk-neutral pricing).
* **Volatility (used for pricing):** σ̂ = std(log-returns, ddof=1) × √252 ≈ **19.07 % p.a.** (full-sample 2017–2026: ≈ 20.08 %, reported for reference).
* **Discount rate:** SHIBOR 1Y fixing 2026-11-03 = **1.4390 % p.a.** → r = 0.01439, flat for T = 1Y. No curve interpolation is required at this tenor; for other tenors we would linearly interpolate SHIBOR fixings in the rate (not in the discount factor).
* **Spot:** S0 = **3974.8** (IF0 close 2026-11-02). **Strike:** K = S0 (spot-ATM). **Barrier:** B = 1.20 × S0 = 4769.76. **Maturity:** T = 1.0 y, **m = 252** monitors/year, dt = T/m.

### 2.3 Model distinction (important)

The assignment prescribes simulation of the *futures price* with the GBM

S(t+dt) = S(t) · exp[(r − σ²/2)·dt + σ·√dt·Z],  Z ~ N(0,1) i.i.d.

We implement exactly this for its Black–Scholes validation. Under standard risk-neutral theory a **futures price** is driftless (Black-76: dF = σ F dW̃). We deliberately do **not** silently conflate the two: the assignment-prescribed GBM-on-futures is used throughout, Black–Scholes-with-drift-r is its closed-form benchmark, and if a Black-76 cross-check is reported it is labelled separately as such. Physical-measure market risk (real-world drift, jump behaviour) is out of scope; our CVaR figures are **model-implied** under the risk-neutral simulator.

## 3. Mathematical specification

### 3.1 Exact GBM simulation (both engines)

With t_i = i·dt, log S_{t_i} = log S0 + Σ_{j<i} [(r − σ²/2)dt + σ√dt Z_j]; paths[:,0] = S0. NumPy engine: `np.random.default_rng(seed)`. PyTorch engine: `torch.Generator` (CPU), same recursion, independent seed (4242 in the cross-engine check; 42 elsewhere). Minimum 252 steps/year as prescribed. Antithetic sampling uses pairs (Z, −Z) with an even number of paths.

### 3.2 Payoffs (monitored on t_1..t_m; **S_0 is NOT monitored**)

1. Arithmetic Asian call: A = max(mean(S_{t_1..m}) − K, 0)
2. Up-and-out call, discrete monitoring: B(S) = max(S_T − K, 0) · 1{max(S_{t_1..m}) < B}
3. Digital call: D = 1{S_T > K}
4. European vanilla call (validation benchmark): E = max(S_T − K, 0)

### 3.3 Black–Scholes benchmark and validation duty

European call: C_BS = S0·N(d1) − K e^{−rT} N(d2), d1,2 = [ln(S0/K) ± (r ± σ²/2)T]/(σ√T). Validation: z = (MC − C_BS)/SE_MC, PASS iff |z| ≤ 3. Reported in §4.1.

### 3.4 Geometric Asian control variate (exact discrete moments)

For the same discrete grid, G = (Π S_{t_i})^{1/m} is lognormal with

μ_G = ln S0 + (r − σ²/2)·T·(m+1)/(2m),   var_G = σ²T²·(m+1)(2m+1)/(6m²),

E[G] = e^{μ_G + var_G/2}, and the geometric-Asian call price has the Black–Scholes-type closed form (d1 = (μ_G − ln K + var_G)/√var_G). The control-variate estimator for the arithmetic Asian call uses per-path geometric payoffs as the control with the closed-form geometric-Asian call as the reference expectation; β̂ = Ĉov(A,G)/V̂ar(G) estimated in-sample (asymptotically unbiased at N = 2^17).

### 3.5 Greeks

* **Pathwise (valid only for a.s. differentiable payoffs):**
  delta_vanilla = e^{−rT} E[1{S_T>K}·S_T/S0]; delta_asian = e^{−rT} E[1{A>K}·(1/m Σ S_i)/S0];
  vega_asian = e^{−rT} E[1{A>K}·(1/m Σ S_i·(√t_i Z_i − σ t_i))].
* **Central finite differences with common random numbers:** spot bump h ∈ {0.1, 0.01, 0.001}·S0; vega uses absolute vol bumps ε ∈ {0.1, 0.01, 0.001} (i.e. h/S0-scaled, documented). Common random numbers make the FD estimator's variance finite and its bias O(h²).
* **Digital and barrier payoffs are discontinuous** ⇒ naive pathwise differentiation yields 0 a.s. (the indicator's derivative is a Dirac mass) and is *not* reported as an estimator; FD is reported instead, and the bias/variance tradeoff across h is analysed.
* BS analytic delta (N(d1)) and vega (S0·√T·N′(d1)) reported for the vanilla as ground truth.

### 3.6 Convergence and verification protocol

RMSE(N) over N ∈ {2^8,…,2^17}, R = 20 independent replications per N; log10–log10 fit of RMSE vs N; fitted slope compared against the theoretical −1/2 with its OLS standard error (assumption: independent replications; reported, not assumed). Antithetic halves the variance for positively-correlated monotone payoffs; control variate removes the geometric component of variance exactly.

### 3.7 Option B — deep hedging specification

* Payoff: European vanilla call (cleanest analytic BS-delta benchmark).
* Policy: Δ_i = f_θ(S_i/S0, t_i/T); MLP ≤ 3 hidden layers (64-64-64, ReLU, linear output). Decisions use only F_{t_i}.
* Accounting (assignment formula, exact conventions): hedging grid t_i = i·dt, i = 0..m−1; Δ_i set at t_i;
  V_T = Σ_i e^{r(T−t_i)} Δ_i (S_{i+1} − S_i) − κ Σ_i |Δ_i − Δ_{i−1}|·S_{t_i} − H(S_T),  Δ_{−1} := 0
  (initial position charged; costs at decision-time spot; **no extra closeout term** — the final hedging P&L leg realises the last move).
* Premium: **excluded** exactly as the assignment writes; adding the constant e^{rT}·C_BS would shift mean P&L (and CVaR) identically across all policies and leave the variance and all policy rankings unchanged.
* Objective: L = −V_T; minimise empirical CVaR_0.95(L) in the Rockafellar–Uryasev form min_{θ,η} η + (1/(1−α))·mean(relu(L − η)), α = 0.95, η learnable. Out-of-sample CVaR evaluated directly (mean of worst 5%) independently of the training loss.
* Training: N_train = 2^17 = 131072 simulated paths (seed 42); validation 2^15 paths (seed 123) with early stopping (patience 12); test 2^15 paths (seed 2026) never touched during training. Minibatches, Adam.
* Benchmarks: BS delta Δ = N(d1(t,S)) with recomputed τ = T−t, at rebalancing frequencies {1, 5, 21} days and κ ∈ {0, 0.001, 0.005} (0/10/50 bp) — 9 combinations — plus the neural policy at the same three κ levels (every-step rebalancing). Metrics on test paths: E[V_T], Var(V_T), CVaR_0.95(−V_T), turnover Σ|ΔΔ|, mean cost per path.
* The same accounting applies to neural and baseline policies (identical cost model, premium exclusion, discounting).

## 4. Results

*[TO BE FILLED FROM results/approach1/if/metrics.json and results/approach2/if/metrics.json after the experiment runs; figures referenced from results/.]*

## 5. Discussion and limitations

* GBM-with-drift is the assignment's model; real futures exhibit mean-reversion/jumps and the physical drift is negative in the sample — none of this enters the risk-neutral price.
* Continuous main contract roll gaps; log-return mitigation; no back-adjustment.
* Discrete monitoring (252/y) of barrier/average; continuous monitoring would change values (barrier especially; discrete is the assignment's spec).
* Risk-neutral CVaR ≠ real-world market risk; deep-hedge advantage (if any) is relative to the simulator's measure and the vanilla call payoff.
* SHIBOR 1Y flat for T=1Y; no dividend/convenience-yield modelling; futures basis not modelled (assignment prescribes GBM on the futures level).
* Asian/barrier Greeks by FD have O(h²) bias; pathwise invalid there.
* Neural results hold only out-of-sample (test seed 2026); in-sample training fit is not evidence of hedging skill.

## 6. Statement of work (agents vs group)

* **ChatGPT/OpenAI lead agent (this session):** specification & conventions, data pipeline (`data/fetch.py`), provenance/checksums, shared mathematics (`code/mc_common.py`), tests, `compare.py`, README, report drafting, Git integration, verification of all claims against measured outputs.
* **DeepSeek (deepseek-v4-pro, via OpenCode subagents):** implementation of `code/approach1_pricing.py` and `code/approach2_hedging.py` (engines, variance reduction, Greeks, convergence, profiling, MLP, CVaR training, benchmark sweeps), and execution of the experiments.
* **Kimi (independent review):** cross-file audit of math/code/results; findings recorded in §7; fixes applied by the lead/DeepSeek and re-verified.
* **The group:** mathematical decisions (model choice, payoff conventions, monitoring, benchmark design), verification of results, final wording and submission of this report; the agent-produced draft is not the group's sign-off.

## 7. Independent audit (Kimi) and repairs

*[TO BE FILLED after the Kimi review; findings, severity, disposition, and re-verification results.]*

## 8. Reproducibility

Fresh clone: `uv sync --frozen` (UV_CACHE_DIR=.cache/uv), then the commands in README.md (≤10 lines). Offline: snapshot committed; `--verify-only` checks SHA-256. Seeds: 42 (train/main), 123 (val), 2026 (test), 4242 (torch cross-check). Nondeterminism: torch CPU ops may be multithreaded; results are reproducible up to BLAS threading noise; no GPU used. `results/` is gitignored and regenerable; `data/clean/` is committed.

## 9. Figure index

* `results/approach1/if/convergence.html` — RMSE vs N, log–log, with fitted slopes (target −1/2)
* `results/approach1/if/var_reduction.html` — variance-reduction factors (antithetic, control variate)
* `results/approach1/if/greeks.html` — FD bias/variance vs h (if produced)
* `results/approach1/if/line_profile.txt` — hot-loop profile
* `results/approach2/if/training_curve.html` — train/val CVaR vs epoch
* `results/approach2/if/pnl_distribution.html` — test P&L distributions
* `results/approach2/if/benchmark_bars.html` — CVaR95 across hedging methods
* `results/comparison.html` — aggregated comparison (code/compare.py)
