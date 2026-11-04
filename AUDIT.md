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
