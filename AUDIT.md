# Independent review and lead adjudication

## M1: Kimi read-only audit

Native child session: `ses_e5d893b81ffef5eNqrveRk54r2`, model
`moonshotai/kimi-k3.1-pro-highspeed`. This file summarizes findings; the authentic
transcript is exported separately. Kimi read the scoped source/snapshots/reference
output but did not run shell commands or tests.

1. **Medium: unknown IF0 roll contamination. Accepted limitation.** Provider data
   lack constituent IDs. We do not delete extreme market moves or label basis
   jumps as certified rolls. Added dates and unknown roll classifications for the
   largest returns, and Bartlett HAC alongside explicitly labelled iid drift SE.
   A verified contract-level adjusted sensitivity series is still unavailable.
2. **Medium: offline verifier discarded recalculated quality metadata. Confirmed,
   repaired.** It now compares regenerated nested diagnostics and row/date metadata.
   Independent tests additionally verify calibration-window span/count, spot,
   strike/barrier, rate date and all curve nodes. Verification wording now says
   deterministic recalibration; independence refers to the separate test formulas.
3. **Low: ACT/365 wording inaccurate for months. Confirmed, repaired.** Documented
   the actual fixed fractions (months/12); the one-year rate is unchanged.
4. **Low: quadrature error naming and geometric check. Accepted, strengthened.**
   Error fields now identify finite-interval error estimates, disclose Gaussian-tail
   truncation, and compare the full covariance-matrix geometric variance with its
   closed-form schedule expression and the option quadrature with the lognormal
   closed form.

Lead reran `data/fetch.py --offline --refresh-diagnostics`, the independent reference
script, and `pytest tests/integration_data.py -q`: all completed successfully (one
data test). These repairs do not alter calibration parameters or snapshot prices.
Final integrated numerical/report audit is pending.

## M1 repair verification and infrastructure review

Kimi confirmed the four data repairs by inspection in the same authentic child
session. Remaining infrastructure findings were independently accepted by the lead:

- Session export remains provisional until all parent/child sessions are idle.
  The exporter now records role/model, message boundaries/counts, tool-part counts,
  export timestamp and embedded-content hashes; obvious credential patterns are
  redacted and counted. `--final` checks idle state and referenced session coverage.
  Replacement requires an explicit flag. Manual secret review is still required.
- Clean-clone verification now requires a clean source tree and records dirty-tree
  failures. Every setup stage writes a distinct log and return code, including
  failures before dependency installation. Lockfile identity is recorded.
- Offline dependency installation uses a shared pre-populated in-repository uv
  cache. This is explicitly recorded and does not claim cold-machine installation
  without packages or a Python interpreter. Offline data reproduction is separate.
- Reproduction logs use unique timestamped directories. The manifest records
  revision, Python, seed, lock hash and freshly generated required output hashes.
  Detailed benchmark-grid verification remains part of numerical integration.
- Production data and quadrature validation use explicit exceptions, not optimized-
  away assertions. The independent reference calculates its Gaussian tail bound.

Lead executed both the offline data checker and reference script with `python -O`;
both passed. Native exporter verification succeeded with a completed Kimi session.
That child-only export is a test artifact, not final session-completeness evidence.

A separate fresh clone at `.cache/m1-offline-preflight` installed the frozen lock
offline into a fresh virtual environment and passed the offline data checker and
independent data test. This is **M1-only** evidence, not complete Option A/B reproduction.

## Integrated numerical audit (Kimi, same child session)

Kimi identified reference-limited CV convergence, overlapping convergence seeds,
missing PW–FD paired uncertainty, silent calibration defaults and permissive NaN
serialization. The lead independently confirmed these and delegated the Option A/
shared-module repairs to DeepSeek. Original results are retained as exploratory
evidence, not overwritten as if the first convergence experiment had passed.

For Option B, Kimi confirmed table/bootstrap precision mismatch and absent saved
checkpoints/gradient guards. Lead repairs now use a float64 copy of each trained
policy and identical float64 paths/NumPy P&L for both tables and bootstrap, with an
explicit equality check. Training remains float32. Checkpoints and histories are
persisted; nonfinite loss/gradient and CVaR inputs fail explicitly; best epoch is
recorded. Official entry points now reject fewer than 131072 training paths.
Bootstrap output includes the observed difference, resample count and seed.

Independent lead tests verify literal irregular-interval P&L, NumPy/Torch agreement,
transaction cost sign, fractional CVaR against independently minimized RU objective,
future-path perturbation causality, autograd finite differences, and exact equality
of table/bootstrap evaluation metrics. Six integration/data tests passed after the
repairs. Numerical full-run acceptance remains pending.

**Accepted uncertainty limitation:** different cost-specific model initialization
seeds mean the cross-cost neural sensitivity does not isolate costs from training
randomness. Paired-path intervals are conditional on fitted policies, not retraining
uncertainty. No claim of a causal cost effect or universal neural superiority is made.

An already-running exploratory 50-epoch job predates these repairs. The separate
full-size integrated run uses dataset label `if_main_verified`, so its results cannot
be confused with the old smoke/full outputs. The evaluation-only checkpoint replay
will apply the final precision corrections without changing trained weights.

## Executed integration acceptance

DeepSeek archived the initial Option A results and ran the repaired full experiment.
The lead then executed the complete test collection. Its first run found 31 passing
current tests and 17 failures in an unexpected untracked `tests/test_numerics.py`
using incompatible `MarketParams` APIs. An additional untracked `tests 2/` directory
also appeared. The user explicitly authorized preserving both paths under
`.cache/unattributed-tests/`; neither was reused or deleted. Re-running the complete
current suite gave **31 passed**. This earlier collection failure is not concealed.

The full integrated B run completed with 131072/16384/16384 train/validation/test
paths. All three cost-specific policies ran 20 epochs. Final float64 checkpoint
evaluation and paired bootstrap completed successfully; `tools/verify_results.py
--hedging-dataset if_main_verified` independently replayed all nine neural P&L
combinations, CVaR, means, variances and costs successfully. It also verified training
counts, finite histories, pricing validation and self-contained figures. Generated
report/comparison tables use these verified metrics, not exploratory smoke outputs.

## Temporary-file hygiene correction

Final infrastructure inspection found that direct pytest invocations had no explicit
`--basetemp`, while several new tests use `tmp_path`. Those earlier direct runs may
therefore have used pytest's system temporary directory, contrary to the repository
write-locality contract. This cannot be retroactively marked compliant. Pytest now
sets `--basetemp=.cache/pytest-tmp` in pyproject.toml for all subsequent direct runs.
The full clean-clone driver already sets TMPDIR to the clone's `.cache/tmp` before
tests, so that run's temporary artifacts are repository-local. No external files
were deleted to conceal or repair the earlier risk.

## Full clean-clone reproduction

The clean-clone run from clean commit `4c6c8af` returned PASS, with frozen offline
dependency installation into a new venv and all eight reproduction stages returning
zero. Full B training ran again rather than loading checkpoints. Evidence lives in
`results/clean_clone/20261104T090738342929/evidence.json`. Lead compared 325 pricing
numeric fields and 631 hedging/history numeric fields across runs with no discrepancies
at tolerances 1e-11 and 1e-10 respectively (timing excluded). Numerical source and
lockfile hashes match. The canonical B outputs now come from this independently
reproduced run; prior exploratory output was preserved in results/archive/.
The canonical output checker and comparison/report generators then passed again.
