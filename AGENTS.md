# Advanced Programming 2026 — Question 2 Agent Contract

## 1. Scope and ownership

Implement **Question 2: Monte Carlo pricing and hedging on Chinese futures** in full, including **both Option A and Option B**, using the instructor's assignment (`assignment.md`) as the authoritative specification. Do not implement other questions in this repository unless explicitly requested.

The agent may implement code, acquire data, run experiments, create figures, draft technical explanations, and manage Git. **The student group remains responsible for mathematical decisions, verification of results, and the final English report.** Distinguish verified output from assumptions, drafts, and unresolved issues. Never fabricate empirical results, test results, commit history, or agent session logs.

**Before beginning:** Read this file and `assignment.md` completely. If `assignment.md` is missing, request the original assignment before implementing specification-dependent work. Check existing files, Git history, available agents, and permissions. Preserve any existing work.

## 2. Environment and reproducibility

- **Python:** 3.13 (use `uv`); maintain `pyproject.toml` and a committed `uv.lock` with pinned dependency versions.
- **Libraries:** NumPy, SciPy, pandas, scikit-learn where useful, PyTorch (`torch`), Plotly, Rich, AKShare, and test/profiling tools as needed.
- **Caches and artifacts:** All writes, caches, temporary files, and model checkpoints must remain **within the repository root**. Set `UV_CACHE_DIR=.cache/uv` and configure other tools similarly. Ignore `.venv/`, `.cache/`, `__pycache__/`, `results/`, regenerable `data/raw/`, credentials, and `*.ipynb` in `.gitignore`.
- **Randomness:** Global CLI seed defaults to `42`, with `--seed` override. Seed each stochastic component, use independent train/validation/test seeds, and disclose nondeterminism.
- **Scripts:** Each approach script must support `argparse` and run as a linear pipeline: load → verify → simulate/fit → evaluate → write outputs. Print readable output using Rich.
- **Metrics:** Write `results/approachN/<dataset>/metrics.json` with at least `{"params": {...}, "metrics": {...}, "fit_seconds": ..., "history": {...}}`. Ensure JSON values are valid and documented.
- **Figures:** Plotly standalone HTML with `include_plotlyjs=True`. `code/compare.py` must aggregate approach results, print ranked/comparison tables, and generate `results/comparison.html`.
- **Offline grading:** Every remote data source must have a reproducible retrieval script and a committed usable offline snapshot per the assignment's size policy; record provenance, retrieval date, schema, and SHA-256 hashes in `data/metadata.json`. A fresh clone must be able to run without remote data access.

### Target run commands

Make the following commands work from the repository root after implementation; document any required flags and an offline option in `README.md` (maximum 10 lines):

```bash
export UV_CACHE_DIR=.cache/uv
uv sync --frozen
uv run python data/fetch.py --help
uv run python code/approach1_pricing.py --seed 42
uv run python code/approach2_hedging.py --seed 42
uv run python code/compare.py
```

The fetch script must document the real-data retrieval command and also support reproduction from an existing offline snapshot. Do not treat `--help` as a data-fetch test.

## 3. Data and mathematical specification

- Obtain **at least five years of daily Chinese futures prices**, preferably CSI 300 futures (`IF`), using authorized FastboxData or available AKShare interfaces. Retrieve underlying index history as relevant and SHIBOR term structure data for discounting.
- Document the contract series used, continuous-contract rolls, any artificial jumps, missing observations, data transformations, and the estimation window.
- Estimate historical drift and annualized volatility from documented return calculations. Keep historical drift estimation separate from risk-neutral pricing.
- Implement the **assignment-prescribed** exact GBM simulation under the risk-neutral measure:

  \[
  S_{t_{i+1}}=S_{t_i}\exp\!\left[(r-\tfrac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}\,Z_i\right],
  \qquad Z_i\overset{iid}{\sim}N(0,1).
  \]

- Use at least **252 simulation steps per year**, with stated monitoring and averaging conventions.
- **Model distinction:** A futures *price* under standard risk-neutral assumptions is typically modeled with zero drift (Black-76 convention), while the assignment explicitly requests a GBM with drift `r`. Implement the required assignment model for its Black–Scholes validation, but explicitly distinguish it from a futures-price model; never silently conflate the two. If a secondary Black-76 check is feasible, label it separately.
- Price **all three required payoffs using both NumPy and PyTorch MC engines**:
  1. Arithmetic Asian call: `max(mean(S[t_1],...,S[t_m]) - K, 0)`.
  2. Discretely monitored up-and-out call: `max(S_T - K, 0) * 1{max(S[t_1],...,S[t_m]) < B}` (state whether the initial point is monitored).
  3. Digital call: `1{S_T > K}`.
- Include a European vanilla call to validate simulation against Black–Scholes: report `z = (MC_price - BS_price) / MC_standard_error` and check `|z| <= 3`. If a validation fails, investigate; do not conceal it.

## 4. Option A — Monte Carlo engine, Greeks, and performance

Implement and evaluate **all** required components:

1. **Naive MC**, **antithetic variates** `(Z,-Z)`, and **geometric-Asian control variate**. Use the correct geometric-Asian expectation for the same **discrete observation schedule**, estimate the control coefficient consistently, and report variance-reduction ratios with valid uncertainty estimates. The indicative target is variance reduction factor ≥ 5 for Asian payoff; never force or fabricate a pass.
2. **Delta and vega** using (i) valid pathwise estimators and (ii) central finite differences with bump sizes `h in {0.1, 0.01, 0.001} * S0` as required by the assignment (state how vega bumps are scaled). Use common random numbers in finite differences. Explain why naive pathwise differentiation can fail for discontinuous digital and barrier payoffs. Check analytical benchmarks where available and compare within reported standard errors.
3. **Convergence experiments:** repeated independent runs across multiple path counts; RMSE versus N in log–log coordinates for all estimators, and fitted slope compared with theoretical `-1/2`. Report fitted values and uncertainty, not assumed success.
4. **Computational profiling:** naive Python loops versus vectorized NumPy at the same workload, repeated `timeit` measurements, and an actual hot-loop profile using `line_profiler` or equivalent. Report runtime and speedup without hiding memory costs.

## 5. Option B — Deep hedging in PyTorch

1. Use the same simulation and payoff conventions as above; independently check agreement of NumPy and PyTorch prices within statistical error.
2. Implement a trading policy `Delta_theta(t) = f_theta(S_t / S_0, t)` with an MLP having **at most three hidden layers**. Ensure decisions use only information available at time `t`.
3. Implement the **assignment-specified** terminal hedging P&L with proportional costs:

   \[
   V_T=\sum_t e^{r(T-t)}\Delta_t(S_{t+\Delta t}-S_t)
       -\kappa\sum_t|\Delta_t-\Delta_{t-1}|S_t-H(S).
   \]

   Specify time indices, initial holding, premium convention, cost accumulation convention, and terminal closeout treatment. Apply the **same convention** to neural and baseline policies. If the assignment's stated cash-flow timing is simplified, explain rather than silently alter it.
4. Define `L = -V_T` and minimize empirical **CVaR at alpha = 0.95**. A suitable training objective is the Rockafellar–Uryasev form `min_{theta, eta} eta + (1/(1-alpha)) mean(relu(L_theta - eta))`. Verify quantile/CVaR evaluation independently from training loss.
5. Use **at least `2^17 = 131072` simulated training paths**. Use minibatches, independent validation/test paths and seeds, monitored training history, and defensible early stopping.
6. Compare with **Black–Scholes delta hedging** at rebalancing intervals **1, 5, and 21 trading days** and transaction costs **0, 10, and 50 basis points** (`0`, `0.001`, `0.005`). Report **CVaR(95%)**, **P&L variance**, and **expected P&L** for each required combination; additionally report trading turnover and average cost if feasible.
7. Use the European vanilla call for the cleanest analytic BS-delta benchmark. If deep hedging other payoffs, use a payoff-consistent delta benchmark or clearly label a proxy. Do not compare option strategies on mismatched payoffs, paths, premiums, or accounting conventions.
8. Evaluate on held-out paths; do not claim neural outperformance unless supported by out-of-sample results. A risk-neutral simulator's CVaR is a model-implied loss measure, not automatically a physical-measure market-risk result.

## 6. Required repository structure

```text
repo/
  AGENTS.md
  .gitignore
  README.md
  assignment.md              # source specification, if permitted to include
  pyproject.toml
  uv.lock
  data/
    fetch.py
    raw/                     # gitignored when regenerable
    clean/                   # committed offline fallback, subject to size rules
    metadata.json
  code/
    approach1_pricing.py
    approach2_hedging.py
    compare.py
  tests/
  results/                   # reproducible outputs; normally gitignored
  report.md
  SESSION.md
```

Maintain additional modules only when useful; required entry points and outputs must remain discoverable. Do not store secrets, account credentials, API keys, or tokens in the repository.

## 7. Multi-agent orchestration: ChatGPT + DeepSeek + Kimi

**Approved execution environment:** Use **OpenCode or an approved OpenCode-compatible terminal agent** for all assignment-producing sessions. ChatGPT/OpenAI, DeepSeek, and Kimi refer to **models/providers inside OpenCode**, not standalone apps used instead of OpenCode. Do **not** invoke Gemini or depend on Gemini credentials.

### 7.1 Roles and file ownership

**ChatGPT / OpenAI — lead agent (`chatgpt-lead`, primary)**
- Own the mathematical specification, risk-neutral convention, payoff definitions, shared interfaces, parameters, and experiment protocol. Make and document methodological decisions for group review.
- Coordinate the other agents, integrate and debug all implementations, manage Git commits and reproducibility, run cross-engine tests, and prepare the **draft** English report and statement of work.
- Own `AGENTS.md`, `README.md`, `pyproject.toml`, `data/fetch.py`, `data/metadata.json`, `code/compare.py`, `tests/integration_*`, and `report.md`; coordinate rather than concurrently edit subagent-owned files.
- Independently check numerical claims, especially exact GBM moments, European BS z-scores, payoff conventions, risk measures, train/test segregation, and whether experimental conclusions follow from actual results.

**DeepSeek — numerical engineering (`deepseek-numerical`, subagent)**
- Primary coding responsibility for `code/approach1_pricing.py` and `code/approach2_hedging.py` and associated isolated numerical helper modules/tests.
- Implement reproducible NumPy/PyTorch engines, Asian/barrier/digital payoffs, antithetic and geometric-Asian control variates, Greeks, RMSE experiments, profiling, MLP, P&L, CVaR training, and benchmark sweeps.
- Produce runnable code and actual diagnostic output, numerical timings, memory/use-of-seed disclosures. Explain unstable gradients, failed training, and computational bottlenecks; never assert benchmarks passed without evidence.
- Before integration, send the lead explicit contracts for function signatures, units, file paths, metric keys, seeds, and required runtime/dependencies.

**Kimi — independent long-context audit (`kimi-review`, read-only subagent)**
- Review `assignment.md`, `AGENTS.md`, source files, tests, metrics, figures, report and Git milestones **together** for cross-file consistency and coverage. Use Kimi as an independent reviewer, not as an authority whose assertions substitute for test results.
- Audit derivations and numerical edge cases independently: GBM drift versus futures/Black-76 conventions; exact discrete geometric-Asian expectation; antithetic sampling units; pathwise Greek validity; finite-difference common random numbers; CVaR tail orientation; option premium and transaction cost timing; train/test leakage; 3x3 BS hedge benchmarks; data provenance; reproducibility.
- Provide a prioritized issue list with precise file/line references, why each issue matters, and a reproducible test or correction proposal. **Do not edit implementation files or silently fix code.** The lead reviews findings and delegates confirmed fixes to DeepSeek or integrates them directly.
- During final acceptance, compare every material report claim with actual generated metrics/figures and verify no required deliverable is missing.

### 7.2 Dispatch protocol

1. The ChatGPT lead inspects `opencode agent list`, `opencode models`, available credentials, the actual repository and Git history. Model/agent IDs are installation-specific; **do not invent a provider/model identifier**. If a required provider is not available, mark it BLOCKED and report it rather than pretending to use it.
2. The lead publishes a compact shared interface and experiment plan: model interpretation, parameters, CLI, output schema, seeds, fixed payoffs, timing and accounting conventions. Then delegates independent work.
3. DeepSeek implements the numerical modules; ChatGPT simultaneously owns infrastructure/data retrieval and tests **in separate files**. Kimi can review the specification immediately; after coding, it reviews the integrated snapshot without editing.
4. ChatGPT runs all executable tests and simulations, dispatches confirmed numerical defects to DeepSeek, checks Kimi's findings, and repeats testing until actual PASS or a clearly documented unresolved failure.
5. Each milestone generates a separate real Git commit. Maintain authentic OpenCode parent and child-agent session evidence for the eventual `SESSION.md` export. Never fabricate subagent calls, commits, benchmark numbers, or logs.
6. If a subagent cannot run within OpenCode, disclose the specific blocker and any approved substitution. Do not silently switch to standalone apps that violate the assignment's terminal-agent rule.

### 7.3 Independent verification checklist

Run automated and numerical checks for: GBM moments; BS MC z-score; payoff boundary cases; no barrier-overshoot counting error; discrete geometric-Asian expectation; covariance and antithetic standard errors; pathwise/finite-difference Greeks; convergence; NumPy/PyTorch agreement; CVaR upper-tail orientation; transaction cost sign/timing; absence of foresight; independence of train/test paths; complete benchmark coverage; offline reproduction.

Kimi's independent review must contain specific findings supported by reproducible evidence. ChatGPT must independently adjudicate those findings and ensure confirmed defects are repaired and the affected experiments rerun. Preserve failures and explain unresolved limitations.

## 8. Git, permissions, and session integrity

- **`AGENTS.md` must exist and have been committed before the first coding-agent session**, as required by the assignment. Inspect the actual history; if this condition was not met, report the discrepancy truthfully. Never backdate or fabricate commits.
- Preserve existing repository history. Commit after each meaningful milestone with prefixes such as `chore:`, `data:`, `feat:`, `test:`, `exp:`, and `report:`. Do not squash everything into a single commit.
- Never write outside the repository root; never force-push, rewrite history, delete data, or perform other destructive/irreversible actions without explicit user approval recorded in the genuine transcript.
- Keep a **complete authentic** terminal-agent session export in root `SESSION.md`, including prompts, agent responses, tool calls, and child-agent sessions as supported. Export from the actual terminal agent; redact credentials but **do not fabricate transcripts**. If the export is finalized after the live session ends, provide exact finalization instructions and verify it before final submission.
- Create the final `submission` Git tag **only after** the repository, report, session export, and clean-clone reproduction have passed verification. If this repository is only one question of a shared group project, coordinate the tag at the group-repository level.

## 9. Report and acceptance criteria

Prepare a factual **English draft** of `report.md` covering mathematical specifications for every estimator/model, data and calibration windows, protocols, pricing and validation tables, variance reduction, Greeks, RMSE convergence, profiling, training, all hedging benchmarks, transaction-cost sensitivity, limitations, figure references, reproducibility, and a **statement of work** describing what agents did versus what the group decided. The group must review and finalize the report itself.

Do not mark the work complete until:

1. Five-year real-data coverage and metadata/offline fallback are verified.
2. All three required option payoffs are priced in **both** engines, and analytical checks are reported.
3. All Option A requirements, numerical tables, and figures have actually run.
4. All Option B training/test and mandatory cost/frequency combinations have actually run.
5. Independent verification findings and repairs are recorded.
6. A clean-clone offline reproduction and comparison output are verified.
7. The required deliverables (`AGENTS.md`, code, `report.md`, `SESSION.md`) and milestone Git history exist.
8. No empirical result, test, subagent invocation, or session record is invented.

Proceed autonomously through safe, reversible implementation steps after initialization. Ask for approval only when unavoidable (e.g., credentials, destructive operations, external authorization). At each milestone, report actual PASS/FAIL and precise remaining blockers; do not replace execution with a plan.
