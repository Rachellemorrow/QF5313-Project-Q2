# Advanced Programming 2026 — Project Examination

Master of Quantitative Finance — Advanced Programming. Take-home project realized with a terminal coding agent.

- Format: 4 questions; each group selects **three** of the four. Within each selected question, **both** options (A and B) are realized.
- Due date: 2026-11-08. Work in groups of 3–5 students. Language: English.
- Permitted aid: a terminal coding agent is **mandatory** (see §1). The mathematical content, the verification, and the written report remain the group's responsibility.

---

## 0. Deliverables (all four, per project)

1. **AGENTS.md** — the agent contract per §2.4, committed before the first agent session.
2. **Code** — a git repository structured per §2, containing all data pipelines, models, and the comparison layer.
3. **Report** — `report.md` at the repo root: mathematical specification of every estimator/model, experimental protocol, results tables, figure references, discussion of findings and limitations. Written by the group.
4. **Session export** — `SESSION.md` at the repo root: the complete markdown export of the terminal-agent session(s) that produced the repository (all prompts, agent responses, tool calls). Exported from the agent (opencode/openchamber session export). Redact secrets; never commit credentials.

Missing any of the four components caps the grade at the code component alone.

---

## 1. Terminal coding agent

- Approved agents: `opencode` or `openchamber`. Any OpenCode-compatible terminal agent is acceptable.
- The repo must contain `AGENTS.md` (contract for the agent, see §2.4) **before** the first agent session.
- The agent may write code, run commands, and manage the repo. The group directs methodology, reviews output, and owns all mathematical decisions. A statement-of-work division (what the agent did vs. what the group decided) is included in `report.md`.
- Session hygiene rules (non-negotiable, enforced by grading):
  - no writes outside the repository root; tool caches live in-repo and gitignored;
  - destructive or irreversible actions require explicit user approval inside the session transcript;
  - every run of every script must be reproducible from a clean clone with the documented commands.

---

## 2. Common project specification

### 2.1 Environment (uniform across all questions)

Python 3.13 or 3.14 (3.15 is not fully supported by PyTorch), managed with `uv`; dependencies pinned.

```bash
uv venv --python 3.13            # never `uv venv python=3.13` (creates a literal directory)
source .venv/bin/activate
uv pip install numpy scipy pandas scikit-learn plotly rich networkx akshare
uv pip install torch              # the PyPI name is `torch`, not `pytorch`
# optional per question: python-igraph (Q4)
```

### 2.2 Repository layout (mandatory)

```
repo/
  AGENTS.md            # deliverable 1; §2.4 contract, written first
  .gitignore           # .venv/, caches, results/ per policy, secrets, *.ipynb
  README.md            # 10-line max: what, how to run, one-command reproduction
  pyproject.toml       # pinned deps (uv)
  data/
    fetch.py           # retrieval + snapshot fallback + checksums
    raw/               # as fetched (gitignored if regenerable)
    clean/             # analysis-ready (committed if < 50 MB, else snapshot rule)
    metadata.json      # source, retrieval date, schema, SHA-256 of each artifact
  code/
    approach1_*.py     # one self-contained script per approach
    approach2_*.py
    compare.py         # aggregates results/*/metrics.json → tables + figures
  results/             # figures (plotly HTML, include_plotlyjs inline), metrics.json
  report.md            # deliverable 3
  SESSION.md           # deliverable 4
```

### 2.3 Data policy

- No credentials in the repository.
- Datasets > 50 MB: commit the snapshot only if not regenerable; otherwise commit the fetch script plus checksums.

### 2.4 `AGENTS.md` contract (minimum content)

1. One-paragraph goal and the mathematical objective.
2. Environment facts (§2.1) and exact run commands.
3. Conventions: seeds, splits, metrics schema, figure policy.
4. Agent rules: nothing outside the repo root; in-repo caches; propose-then-act on destructive operations; commit after each milestone.

### 2.5 Reproducibility and reporting conventions

- Global seed fixed (`--seed 42` default, CLI-overridable); every stochastic step seeded; any residual nondeterminism documented.
- Every `approachN_*.py`:
  - CLI (`argparse`), single purpose, linear pipeline: load → verify → split → fit/train → evaluate → write `results/approachN/<dataset>/metrics.json` + figures;
  - `metrics.json` schema: `{params: {...}, metrics: {...}, fit_seconds, history: {...}}`;
  - terminal output via `rich` (panel, metrics table, confusion/summary matrix); figures via `plotly` written as self-contained HTML (`include_plotlyjs=True`).
- `compare.py` reads all `metrics.json`, prints a ranked table, writes `results/comparison.html`.

### 2.6 Git discipline

`git init` and initial commit **before** any code; commit after each milestone (conventional messages: `data:`, `feat:`, `exp:`, `report:`); final tag `submission`. Grading inspects history: a single terminal commit is a reproducibility failure.

### 2.7 Grading

| Component | Weight |
|---|---|
| Data provisioning, verification, provenance | 15 |
| Mathematical correctness of models/estimators | 30 |
| Results: experiments, verification checks, presentation | 20 |
| Code quality, repo organization, reproducibility | 15 |
| Report: clarity, discussion, limitations | 10 |
| Session integrity (SESSION.md consistent with repo history) | 10 |

---

## 3. Question 1 — Implied volatility surface of China ETF options

Course mapping: Sessions 1–2 (Black–Scholes, simulation), Session 4 (models, calibration). Skills: root finding, constrained least squares, no-arbitrage analysis, optimization.

### 3.1 Data (public, verified)

| Item | Source |
|---|---|
| Option chains: strikes, expiries, call/put, contract unit (SSE 50ETF `510050`, 300ETF `510300`, 500ETF, 科创50) | `ak.option_current_day_sse()` |
| Per-contract daily OHLCV, CFFEX CSI 300 index options (`io`), historical | `ak.option_cffex_hs300_daily_sina(symbol=...)`, contract list via `ak.option_cffex_hs300_list_sina()` |
| Underlying index / ETF daily history | `ak.index_zh_a_hist`, `ak.fund_etf_hist_em` |
| Discount curve: SHIBOR (all tenors), LPR, FR007 swap | `ak.macro_china_shibor_all()`, `ak.macro_china_lpr()`, `ak.macro_china_swap_rate()` |

Single-currency market: the only rate required is domestic; interpolate the SHIBOR zero curve to option tenors. ETF dividend yield: estimate or set to zero, justify. Snapshot: one month of daily chains for ≥ 4 expiries, ≥ 5 sample dates spread over ≥ 2 years.

### 3.2 Common requirements

- Put–call parity residuals $C - P - (S_0 - K e^{-rT})$ per strike; quote filtering rule stated and justified (volume, spread, moneyness band).
- Implied volatility by inverting Black–Scholes:
  $d_\pm = \frac{\ln(F/K) \pm \frac12\sigma^2 T}{\sigma\sqrt T}$, $\ C_{BS}(\sigma) = e^{-rT}(F N(d_+) - K N(d_-))$;
  solve $C_{BS}(\sigma_{IV}) = C_{mkt}$ by bracketed Brent on $\sigma \in [10^{-4}, 5]$ (both quotes per strike; report divergence cases).
- Deliverable surface: $\sigma_{IV}(K, T)$ on a fixed log-moneyness grid $k = \ln(K/F_T)$ per expiry.

### 3.3 Option A — Market pipeline (classical)

1. Per expiry $T_j$, fit **SVI** in total variance $w(k) = \sigma_{IV}^2 T$:
   $w(k) = a + b\big(\rho(k - m) + \sqrt{(k-m)^2 + \eta^2}\big)$, parameters constrained $b \ge 0$, $|\rho| < 1$, $\eta > 0$; least squares over liquid strikes; report initial values and solver.
2. No-arbitrage verification, reported per expiry and per date:
   butterfly (Gatheral): $g(k) = \Big(1 - \frac{k\,w'(k)}{2w(k)}\Big)^2 - \frac{w'(k)^2}{4}\Big(\frac{1}{w(k)} + \frac14\Big) \ge 0$;
   calendar: $w_{T_2}(k) \ge w_{T_1}(k)$ for $T_2 > T_1$; violations counted and located, not silently dropped.
3. Term structure: interpolate $w(k, T)$ linearly in total variance; report any negative forward variance.
4. Repricing consistency: $\max_k |C_{BS}(\sigma_{fit}(k)) - C_{mkt}| / C_{mkt}$ on liquid strikes; median and 95th percentile across dates.

### 3.4 Option B — Differentiable calibration (PyTorch)

1. SVI calibration by gradient descent in `torch` (Adam/L-BFGS), same constraints via reparametrization; convergence profiles vs. `scipy.optimize.least_squares` (iterations to tolerance, wall time, objective value).
2. Neural surface: $\hat\sigma_\theta(k, T)$ MLP ($\le 4$ layers), loss = squared pricing error in **option price space** $|C_{BS}(\hat\sigma) - C_{mkt}|^2$ plus arbitrage penalties $\lambda_1 \max(0, -g(k))^2 + \lambda_2 \max(0, -\partial_T w)^2$; ablation on $(\lambda_1, \lambda_2)$.
3. Robustness study: randomly drop 30% of quotes; compare SVI refit vs. neural surface extrapolation error.
4. Same verification duties as §3.3.2.

### 3.5 Benchmarks (indicative)

Median liquid-strike repricing relative error < 2%; zero butterfly violations on interior grid after fitting; calibration table across all sample dates.

---

## 4. Question 2 — Monte Carlo pricing and hedging on Chinese futures

Course mapping: Sessions 1–2 (BS, Monte Carlo, RNG), Session 4 (profiling). Skills: exact simulation, variance reduction, Greeks, stochastic optimization.

### 4.1 Data

| Item | Source |
|---|---|
| CSI 300 futures (`IF`) and commodity futures daily series (e.g. rebar `RB`, copper `CU`, crude `SC`) | FastboxData read-only (credentials via env) or `ak.futures_main_sina(symbol=...)`, `ak.futures_zh_daily_sina` |
| Underlying index daily history | `ak.index_zh_a_hist` |
| Risk-free curve | SHIBOR (`ak.macro_china_shibor_all`) |

Calibration set: ≥ 5 years of daily data; estimate $\hat\mu, \hat\sigma$ (realized vol, log-return regression) per underlying; state the estimation window. Discounting at the interpolated SHIBOR curve.

### 4.2 Common requirements

Exact GBM simulation under $\mathbb{Q}$: $S_{t_{i+1}} = S_{t_i}\exp\big((r - \tfrac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\, Z_i\big)$, $Z_i \sim \mathcal{N}(0,1)$ i.i.d., seeded generator; $m \ge 252$ steps/year. Validation duty: European MC price within $3\times$ standard error of the Black–Scholes closed form (report $z$-score).

Payoffs (all priced for both MC engines): arithmetic Asian call $A_T = \frac1m \sum_i S_{t_i}$; up-and-out call (discrete monitoring, barrier $B$); digital call.

### 4.3 Option A — Engine, variance reduction, Greeks, performance

1. Naive MC; **antithetic** ($Z, -Z$ pairs; variance ratio reported); **control variate** with the geometric-Asian closed-form price:
   $\hat\mu_{CV} = \hat\mu_{A} - \hat\beta(\hat\mu_{G} - \mathbb{E}[G])$, $\hat\beta = \widehat{\mathrm{Cov}}(A, G)/\widehat{\mathrm{Var}}(G)$.
2. Greeks: delta and vega by (i) pathwise estimator (e.g. $\delta = e^{-rT}\,\mathbb{E}[\mathbf{1}_{S_T > K} S_T / S_0]$ for vanilla), (ii) central finite differences with bump $h \in \{10^{-1}, 10^{-2}, 10^{-3}\} S_0$; report bias–variance tradeoff.
3. Convergence: RMSE vs. $N$ on log–log, all estimators; verify $O(N^{-1/2})$ slope.
4. Performance: vectorized `numpy` engine vs. naive loop implementation; `timeit` before/after; profiling of the hot loop (line_profiler or equivalent) included in the report.

### 4.4 Option B — Deep hedging (PyTorch)

1. Simulator as §4.2 with proportional transaction cost $\kappa$.
2. Hedging strategy: $\Delta_t^{\theta} = f_{\theta}(S_t/S_{t_0}, t)$, MLP ($\le 3$ hidden layers); terminal P&L
   $V_T = \sum_{t} e^{r(T-t)}\Delta_{t}(S_{t+\Delta t} - S_t) - \kappa\sum_t |\Delta_t - \Delta_{t-1}| S_t - \text{option payoff}$.
3. Objective: minimize CVaR$_\alpha$ of $-V_T$ (empirical $\alpha = 0.95$ quantile-based loss), train on simulated paths (≥ $2^{17}$), validate on independent seeds.
4. Benchmarks: Black–Scholes delta hedging at rebalancing frequencies $\{1, 5, 21\}$ days and cost levels $\kappa \in \{0, 10, 50\}$ bp; report CVaR$_\alpha$, variance, expected P&L tables; sensitivity of the learned strategy to $\kappa$.

### 4.5 Benchmarks (indicative)

Naive-vs-control-variate variance reduction factor ≥ 5 on the Asian payoff; pathwise/finite-difference delta agreement within reported standard errors.

---

## 5. Question 3 — China macro panel: regimes, nowcasting, indicator clustering

Course mapping: Sessions 2–3 (pandas, datetime, clustering, SQL), Session 3 KMeans. Skills: panel construction with publication lags, factor models, classification, hierarchical clustering, honest temporal validation.

### 5.1 Data (public, verified)

| Item | Source |
|---|---|
| 170 `macro_china_*` interfaces: GDP, CPI, PPI, PMI (NBS & Caixin), M0/M1/M2, 社融 (TSF), LPR, Shibor, retail sales, industrial output, industrial profits, exports/imports YoY, trade balance, consumer confidence, electricity output, … | `akshare` `macro_china` family |
| Offline fallback | committed snapshot `data/clean/macro_panel.csv` |

Build a monthly panel $x_{i,t}$, $i = 1,\dots,d \ge 20$, ≥ 10 years, with per-series publication lag $L_i$ documented in `metadata.json`. Information set discipline:
$\mathcal{F}_t = \{x_{i,s} : s + L_i \le t\}$; any use of $x_{i,s}$ with $s + L_i > t$ is leakage and a grading fault.

### 5.2 Common requirements

- Panel hygiene: missing-value report; outlier rule (e.g. MAD) stated; transformations (logs, YoY, Sa) documented per series; standardized panel $\tilde x_{i,t} = (x_{i,t} - \mu_i)/s_i$ with statistics from training windows only.
- Regime label: contraction indicator $d_t = \mathbf{1}\{\text{PMI}_{t} < 50 \text{ and } \text{PMI}_{t-1} < 50\}$ (state and justify the labeling; sensitivity to the 49.5 threshold reported).
- Validation: expanding-window walk-forward; metrics: Brier score, AUC, accuracy; naive prior-frequency baseline reported alongside.

### 5.3 Option A — Classical

1. Factor model: PCA on $\tilde x_t$; choose rank $r$ by cross-validated nowcast error; $\tilde x_t = \Lambda f_t + \varepsilon_t$.
2. Nowcasting bridge: $\hat y_{t} = \beta_0 + \beta^\top f_t$ for quarterly GDP growth (or monthly industrial output), estimated on $\mathcal{F}_{t}$ only; nowcast revision path within the quarter reported.
3. Regime model: probit $\mathbb{P}(d_{t+1} = 1) = \Phi(\beta^\top z_t)$, $z_t$ = factors + yield/spread controls; coefficient table with standard errors.
4. Indicator clustering: distance $d_{ij} = \sqrt{2(1 - \rho_{ij})}$ on rolling 3-year correlations, Ward linkage; dendrogram; cluster stability via adjusted Rand index across adjacent windows; relate clusters to lead/lag cross-correlations with GDP.

### 5.4 Option B — Sequence learning (PyTorch)

1. Sequence model (LSTM/GRU, ≤ 2 layers, dropout reported): input = window of $\tilde x_{t-h:t}$, output $\mathbb{P}(d_{t+1}=1)$; class imbalance handled explicitly (weighting or threshold selection on validation).
2. Latent macro states: hidden-state embeddings $h_t$; $k$-means on $\{h_t\}$; compare clusters with the Option-A correlation clusters (ARI/NMI) and with the regime labels.
3. Comparison: Brier/AUC of (i) probit baseline, (ii) LSTM, (iii) LSTM without lag discipline (intentional leakage demo, reported as the leakage premium — this ablation is mandatory).
4. Training discipline: early stopping on walk-forward validation; seeds × 3, mean ± std reported.

### 5.5 Benchmarks (indicative)

Regime AUC ≥ 0.75 walk-forward (classical baseline); nowcast error vs. a random-walk-with-drift benchmark; leakage ablation quantified.

---

## 6. Question 4 — China financial and trade networks

Course mapping: Session 3 (KMeans, SQL, imbalance). Skills: graph construction, community detection, centrality, node embeddings, link prediction.

### 6.1 Data (public, verified)

| Network | Source |
|---|---|
| A-share shareholding: top-10 (circulating) shareholders per stock per quarter | `akshare` `stock_gdfx` interfaces (46 endpoints, Eastmoney) |
| Fund holdings: fund–stock quarterly portfolios | `ak.fund_portfolio_hold_em(symbol=..., date=...)` |
| China trade partners: annual bilateral flows (HS, partner-level) | UN Comtrade public API, keyless: `comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=156&period=...` |
| Industry classification (ground truth) | `ak.stock_board_industry_*` / SW industry |

Primary network (choose one, or both as A/B contrast): ownership network or trade network. Snapshot: ≥ 8 quarters (ownership) or ≥ 10 years (trade).

### 6.2 Common requirements

- Construction rules stated as mathematics: e.g. correlation network $w_{ij} = \sqrt{2(1-\rho_{ij})}$ (price networks), bipartite holdings $B \in \{0,1\}^{|F| \times |S|}$ with weighted projection $W = B^\top B$ (fund–stock), directed weighted trade graph $T_{ij}$ (value of exports $i \to j$).
- Graph statistics table per period: $|V|$, $|E|$, density, degree distribution (log–log fit), assortativity, connected components.
- Verification duty: assert no self-loops, symmetric treatment where required, isolated nodes reported.

### 6.3 Option A — Network analytics (networkx)

1. Minimum spanning tree (Kruskal) of the metric distance network; evolution of tree diameter across periods.
2. Community detection: Louvain, modularity
   $Q = \frac{1}{2m}\sum_{ij}\Big[A_{ij} - \frac{k_i k_j}{2m}\Big]\,\delta(c_i, c_j)$;
   compare communities with industry classification via ARI/NMI; resolution-sensitivity sweep.
3. Centrality: degree, betweenness, PageRank (iterative solution of $p = \alpha P^\top p + (1-\alpha)\mathbf{1}/n$, $\alpha = 0.85$); ranked tables; interpretation (systemic firms / dominant trade partners).
4. Temporal analysis: community migration matrix between consecutive periods; identify structurally central nodes and structural breaks.

### 6.4 Option B — Node embeddings and link prediction (PyTorch)

1. node2vec (own implementation of the biased-walk skip-gram): objective $\max_\theta \sum_v \sum_{u \in \mathcal{N}(v)} \log p_\theta(u \mid z_v)$; walk parameters $(p, q)$ swept.
2. Link prediction: score $s(u,v) = \sigma(z_u^\top W z_v)$; edges split 80/10/10 with equal negative sampling; metrics AUC, average precision; baselines: common-neighbors, Adamic–Adar, preferential attachment.
3. Task framing (choose per network): recover held-out holdings edges; predict industry membership from $z_v$ (linear probe); predict future trade-link formation $\Delta T_{ij} > 0$.
4. Clustering the embedding space ($k$-means on $z_v$) vs. Louvain communities vs. industry labels (ARI/NMI matrix); embedding dimension swept $\{8, 16, 32, 64\}$.

### 6.5 Benchmarks (indicative)

Link-prediction AUC ≥ 0.75 vs. ≥ 0.65 for the best heuristic baseline; ARI(embedding clusters, industry) reported against ARI(Louvain, industry).

---

## Appendix — Data source registry (verified 2026-10)

| Source | Access | Status |
|---|---|---|
| akshare (MIT, 23k★, active) | `pip install akshare`, aliyun mirror | public |
| SSE option chains (`option_current_day_sse`, daily stats) | akshare → sse.com.cn | public |
| CFFEX index option contract lists & daily OHLCV | akshare → Sina Finance | public |
| `macro_china_*` (170 series incl. Shibor/LPR/TSF/PMI) | akshare → Eastmoney/NBS mirrors | public |
| `stock_gdfx` shareholder interfaces (46), `fund_portfolio_hold_em` | akshare → Eastmoney | public |
| UN Comtrade preview API | keyless REST | public |
| CNINFO disclosure API (annual-report PDFs, unauthenticated POST `hisAnnouncement/query` + `static.cninfo.com.cn`) | REST | public |
| DuXiaoman FinCorpus (zh finance announcements/news, Apache-2.0) | `hf-mirror.com/datasets/Duxiaoman-DI/FinCorpus` | public |
| CFSD labeled Chinese financial sentiment (Bian et al. 2019); Chinese financial sentiment dictionary | GitHub `seanxlwang/Bianetal2019CFSD`, `MengLingchao/Chinese_financial_sentiment_dictionary` | public |
| FastboxData (SZSE L2 order book, CTP futures, Tushare daily) | read-only server, credentials via env | private, course-provided |
| FX quotations SQLite (6 pairs, 2003–2020; interest rates CHF-only — caveat) | bundled on request | optional comparison dataset |

General fallback rule: every external interface used by `fetch.py` must have a committed snapshot; grading runs offline.
