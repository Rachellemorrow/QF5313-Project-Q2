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
