"""Independent lead calculations, with no imports from numerical implementation."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]


def calculate():
    p = json.loads((ROOT/"data/clean/calibration.json").read_text())
    s, k, r, v, T, m = (p[x] for x in ("S0", "K", "r", "sigma", "T", "steps"))
    disc = np.exp(-r*T)
    d1 = (np.log(s/k)+(r+v*v/2)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    bs = s*ndtr(d1)-k*disc*ndtr(d2)
    terminal = lambda z: s*np.exp((r-v*v/2)*T+v*np.sqrt(T)*z)
    density = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
    strike_z = (np.log(k/s)-(r-v*v/2)*T)/(v*np.sqrt(T))
    quadrature, error = quad(lambda z: disc*(terminal(z)-k)*density(z), strike_z, 12, epsabs=1e-9)
    times = np.arange(1,m+1)*T/m
    covariance = v*v*np.minimum.outer(times,times)
    meanlog = np.log(s)+(r-v*v/2)*times.mean()
    varlog = covariance.mean()
    g_z = (np.log(k)-meanlog)/np.sqrt(varlog)
    geo, geo_error = quad(lambda z: disc*(np.exp(meanlog+np.sqrt(varlog)*z)-k)*density(z), g_z, 12, epsabs=1e-9)
    assert abs(bs-quadrature) < 1e-8
    return {"params": p, "metrics": {"bs_call": float(bs), "call_quadrature": quadrature,
        "quadrature_absolute_error": error, "bs_delta": float(ndtr(d1)),
        "bs_vega": float(s*np.sqrt(T)*density(d1)), "digital": float(disc*ndtr(d2)),
        "geometric_asian_quadrature": geo, "geometric_quadrature_error": geo_error,
        "geometric_log_variance_covariance_matrix": float(varlog),
        "terminal_mean": float(s*np.exp(r*T)),
        "terminal_variance": float(s*s*np.exp(2*r*T)*np.expm1(v*v*T))},
        "fit_seconds": 0, "history": {"method": "Independent quadrature and full min(t_i,t_j) covariance matrix; no numerical-engine imports"}}


if __name__ == "__main__":
    result = calculate()
    out = ROOT/"results/independent"
    out.mkdir(parents=True, exist_ok=True)
    (out/"reference.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["metrics"], indent=2))
