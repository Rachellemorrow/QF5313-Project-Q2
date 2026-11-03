"""Lead's independent quadrature and covariance references, no MC-core imports."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/verification/independent_reference.json"))
    args = parser.parse_args()
    c = json.loads(Path("data/clean/calibration.json").read_text())
    s, k, r, v, t = (c[x] for x in ("S0", "K", "r", "sigma", "T"))
    disc = np.exp(-r*t)
    density = lambda z: np.exp(-z*z/2) / np.sqrt(2*np.pi)
    zstrike = (np.log(k/s)-(r-v*v/2)*t)/(v*np.sqrt(t))
    integral, error = quad(lambda z: disc*(s*np.exp((r-v*v/2)*t+v*np.sqrt(t)*z)-k)*density(z), zstrike, 12, epsabs=1e-9)
    d1 = (np.log(s/k)+(r+v*v/2)*t)/(v*np.sqrt(t))
    bs = s*ndtr(d1)-k*disc*ndtr(d1-v*np.sqrt(t))
    # A secondary economically distinct reference, not assignment validation.
    b76d1 = (np.log(s/k)+v*v*t/2)/(v*np.sqrt(t))
    black76 = disc*(s*ndtr(b76d1)-k*ndtr(b76d1-v*np.sqrt(t)))
    geo = []
    for maturity, m in ((.25,63), (1.,252), (2.5,630)):
        times = np.arange(1,m+1)*maturity/m
        cov_variance = float(v*v*np.minimum.outer(times,times).mean())
        closed_variance = float(v*v*maturity*(m+1)*(2*m+1)/(6*m*m))
        geo.append({"T":maturity,"steps":m,"covariance_sum_variance":cov_variance,
                    "closed_form_variance":closed_variance,"difference":cov_variance-closed_variance})
    result = {"calibration": c, "european_quadrature":integral,"quadrature_error":error,
              "european_bs":float(bs),"bs_minus_quadrature":float(bs-integral),
              "digital_analytic":float(disc*ndtr(d1-v*np.sqrt(t))),
              "delta_analytic":float(ndtr(d1)),
              "vega_analytic":float(s*np.sqrt(t)*density(d1)),
              "black76_separate_zero_drift_model":float(black76),"geometric_covariance_checks":geo}
    assert abs(bs-integral) < 1e-8
    assert all(abs(g["difference"]) < 1e-14 for g in geo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key!="calibration"},indent=2))


if __name__ == "__main__":
    main()
