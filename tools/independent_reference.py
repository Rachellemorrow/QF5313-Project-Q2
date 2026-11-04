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
    closed_variance = v*v*T*(m+1)*(2*m+1)/(6*m*m)
    if not np.isclose(varlog, closed_variance, rtol=1e-14):
        raise ValueError("Geometric variance mismatch")
    geometric_closed = disc*(np.exp(meanlog+varlog/2)*ndtr(-g_z+np.sqrt(varlog))-k*ndtr(-g_z))
    if abs(geo-geometric_closed) >= 1e-8 or abs(bs-quadrature) >= 1e-8:
        raise ValueError("Quadrature versus closed form mismatch")
    tail_bound = float(s*ndtr(v*np.sqrt(T)-12))
    if tail_bound >= 1e-9:
        raise ValueError("Quadrature tail exceeds tolerance; increase integration limit")
    black_d1 = (np.log(s/k)+v*v*T/2)/(v*np.sqrt(T))
    black76 = disc*(s*ndtr(black_d1)-k*ndtr(black_d1-v*np.sqrt(T)))
    black_z = (np.log(k/s)+v*v*T/2)/(v*np.sqrt(T))
    black_quad, _ = quad(lambda z: disc*(s*np.exp(-v*v*T/2+v*np.sqrt(T)*z)-k)*density(z),black_z,12)
    if abs(black76-black_quad) >= 1e-8:
        raise ValueError("Secondary Black-76 deterministic validation failed")
    digital_price = lambda spot, vol: disc*ndtr((np.log(spot/k)+(r-vol*vol/2)*T)/(vol*np.sqrt(T)))
    digital_delta = float(disc*density(d2)/(s*v*np.sqrt(T)))
    digital_vega = float(-disc*density(d2)*d1/v)
    fd_delta = (digital_price(s*(1+1e-5),v)-digital_price(s*(1-1e-5),v))/(2e-5*s)
    fd_vega = (digital_price(s,v*(1+1e-5))-digital_price(s,v*(1-1e-5)))/(2e-5*v)
    if not np.allclose([fd_delta,fd_vega],[digital_delta,digital_vega],rtol=1e-7,atol=1e-10):
        raise ValueError("Digital analytical Greeks versus deterministic differences failed")
    return {"params": p, "metrics": {"bs_call": float(bs), "call_quadrature": quadrature,
        "quadrature_finite_interval_error_estimate": error, "bs_delta": float(ndtr(d1)),
        "bs_vega": float(s*np.sqrt(T)*density(d1)), "digital": float(disc*ndtr(d2)),
        "digital_delta": digital_delta,
        "digital_vega": digital_vega,
        "digital_deterministic_fd_delta": float(fd_delta),
        "digital_deterministic_fd_vega": float(fd_vega),
        "vanilla_quadrature_tail_upper_bound": tail_bound,
        "secondary_black76_zero_drift_call": float(black76),
        "secondary_black76_zero_drift_quadrature": float(black_quad),
        "geometric_asian_quadrature": geo, "geometric_quadrature_finite_interval_error_estimate": geo_error,
        "geometric_log_variance_from_covariance_matrix": float(varlog),
        "geometric_asian_closed_form": float(geometric_closed),
        "terminal_mean": float(s*np.exp(r*T)),
        "terminal_variance": float(s*s*np.exp(2*r*T)*np.expm1(v*v*T))},
        "fit_seconds": 0, "history": {"method": "Independent quadrature and full min(t_i,t_j) covariance matrix; no numerical-engine imports", "quadrature_truncation": "Upper Gaussian limit 12; omitted positive payoff tail bounded by discounted E[S] * Phi(sigma*sqrt(T)-12), below 1e-27 here; quadrature error estimates exclude this tail."}}


if __name__ == "__main__":
    result = calculate()
    out = ROOT/"results/independent"
    out.mkdir(parents=True, exist_ok=True)
    (out/"reference.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["metrics"], indent=2))
