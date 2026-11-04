# Q2 acceptance ledger

This ledger distinguishes executable research completion from submission finalization.

| Requirement | Status | Evidence / limitation |
|---|---|---|
| Read authoritative assignment and contract | PASS | Complete reads in parent session |
| OpenAI / DeepSeek / Kimi genuine providers | PASS | Three native session exports; RESEARCH_PROTOCOL.md |
| Initial AGENTS commit before current rebuild session | PASS | e7bc8d8 committed 2026-11-03 07:32:09 UTC; current parent created 17:29:56 UTC |
| AGENTS before every historical first coding session | NOT VERIFIED | Current rebuild evidence does not establish all earlier session timing |
| Independent new implementation and observations | PASS | No old implementation/result restored; user-authorized isolation of unattributed tests |
| Python 3.13, pinned uv environment | PASS | Python 3.13.13; uv.lock; fresh offline dependency installation |
| Five-year futures / index / SHIBOR snapshots and hashes | PASS | metadata.json, offline verifier and independent data test |
| Complete contract constituent/roll identification | NOT VERIFIED | Provider IF0 lacks constituent flags; basis/return diagnostics and contamination disclosed |
| GBM, payoffs, discounting, both engines | PASS | Current tests and approach1 metrics |
| European Black–Scholes 3-SE validation | PASS | Actual z-score approximately 0.627 |
| Naive / antithetic / discrete geometric CV | PASS | Independent pilot, pair-level SE, bootstrap variance ratios |
| Pathwise / CRN finite-difference Greeks | PASS | All bumps, valid continuous pathwise estimators, paired differences |
| Repeated convergence and slope uncertainty | PASS | Repaired independent reference and seed grid, bootstrap slopes; initial failure retained |
| Matched profiling and memory evidence | PASS | timeit repeats, line_profiler/cProfile, traced memory |
| At least 131072 training paths | PASS | Three cost-specific full-size fits; independent validation/test seeds |
| Causal MLP, exact prescribed P&L, CVaR | PASS | Independent P&L replay, causality/gradient/RU checks |
| 3×3 classical and neural benchmark grid plus zero hedge | PASS | Full metrics, report tables, comparison output |
| Paired bootstrap uncertainty | PASS | Observed differences, 400 resamples, seed and intervals recorded |
| Neural outperformance | FAIL | All nine matched CVaR evaluations of the three fitted policies favor BS; this is an empirical result, not a missing implementation |
| Independent retraining variability | NOT VERIFIED | One fitted network per cost; conditional path uncertainty only |
| Automated tests | PASS | 31 current tests; initial 17 incompatible external-to-scope failures recorded |
| Figures and comparison | PASS | Standalone HTML and independent output checks |
| Complete clean-clone A+B training reproduction | PASS | Commit 4c6c8af, fresh venv, cached offline dependencies, all eight stages successful; 325 pricing and 631 hedging/history numeric fields reproduced |
| Final Kimi report-to-artifact audit | NOT VERIFIED | Review in progress |
| All historical temporary writes inside repository | NOT VERIFIED | Earlier direct pytest tmp_path locality unverified; fixed local basetemp now enforced |
| English report and statement of work | PASS | Draft report.md; student approval remains pending |
| Authentic complete final session export | NOT VERIFIED | Genuine provisional parent/children export exists; final response must be exported after session ends |
| Submission tag | NOT VERIFIED | Deferred until final export, reproduction and group scope approval |

The final session cannot contain its own future export-confirmation response. Follow
FINALIZE.md after the session becomes idle. No submission-ready claim is made while
the final export and group approval remain outstanding.
