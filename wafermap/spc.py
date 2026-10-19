"""Statistical Process Control (SPC) for fab process parameters.

Every tool in a fab is watched with control charts: if a measurement such as
film thickness or critical dimension starts behaving unusually, the tool is
stopped *before* it scraps wafers. This module implements:

* Individuals / Moving-Range (I-MR) control limits
* Western Electric run rules (+ the Nelson trend rule)
* EWMA chart, which catches small sustained shifts much sooner than rule 1
* Process capability indices Cp, Cpk, Pp, Ppk
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D2_N2 = 1.128  # bias constant: E[moving range of 2 points] = 1.128 * sigma
D4_N2 = 3.267  # upper limit factor for the moving-range chart

RULES = {
    "rule1": "1 point beyond 3 sigma",
    "rule2": "2 of 3 points beyond 2 sigma (same side)",
    "rule3": "4 of 5 points beyond 1 sigma (same side)",
    "rule4": "8 points in a row on one side of the centre line",
    "trend": "6 points in a row steadily increasing or decreasing",
}


@dataclass
class IMRLimits:
    center: float
    sigma: float
    ucl: float
    lcl: float
    mr_bar: float
    mr_ucl: float


def imr_limits(x: np.ndarray) -> IMRLimits:
    """Control limits from in-control (Phase I) data.

    Sigma is estimated from the average moving range, not the plain standard
    deviation, so a slow drift in the baseline data does not inflate it.
    """
    x = np.asarray(x, dtype=float)
    mr_bar = np.abs(np.diff(x)).mean()
    sigma = mr_bar / D2_N2
    center = x.mean()
    return IMRLimits(center, sigma, center + 3 * sigma, center - 3 * sigma, mr_bar, D4_N2 * mr_bar)


def _trailing_count(cond: np.ndarray, window: int) -> np.ndarray:
    """How many of the last `window` points (ending at each index) are True.

    Windows that would start before the first point return -1 so they never
    satisfy a rule. Works on 1-D series or on 2-D (simulations x time).
    """
    c = np.cumsum(cond, axis=-1, dtype=int)
    out = c.copy()
    out[..., window:] = c[..., window:] - c[..., :-window]
    out[..., : window - 1] = -1
    return out


def western_electric_flags(x: np.ndarray, center: float, sigma: float) -> dict[str, np.ndarray]:
    """Boolean array per rule: True where the rule fires at that point.

    Vectorised so it can run on thousands of simulated series at once
    (x may be 1-D, or 2-D with time on the last axis).
    """
    x = np.asarray(x, dtype=float)
    z = (x - center) / sigma
    flags = {"rule1": np.abs(z) > 3}
    for rule, window, k, limit in (("rule2", 3, 2, 2.0), ("rule3", 5, 4, 1.0)):
        hi, lo = z > limit, z < -limit
        flags[rule] = (hi & (_trailing_count(hi, window) >= k)) | (lo & (_trailing_count(lo, window) >= k))
    flags["rule4"] = (_trailing_count(z > 0, 8) == 8) | (_trailing_count(z < 0, 8) == 8)
    d = np.diff(x, axis=-1)
    up = _trailing_count(d > 0, 5) == 5  # 5 rising steps = 6 rising points
    down = _trailing_count(d < 0, 5) == 5
    pad = np.zeros(x.shape[:-1] + (1,), dtype=bool)
    flags["trend"] = np.concatenate([pad, up | down], axis=-1)
    return flags


def western_electric(x: np.ndarray, center: float, sigma: float) -> dict[str, list[int]]:
    """Indices where each run rule fires (index = the point that completes it)."""
    return {rule: np.flatnonzero(f).tolist() for rule, f in western_electric_flags(x, center, sigma).items()}


def ewma(x: np.ndarray, center: float, sigma: float, lam: float = 0.2, L: float = 3.0):
    """Exponentially weighted moving average chart.

    z_i = lam * x_i + (1 - lam) * z_{i-1}. Averaging over recent points shrinks
    the noise, so a small shift (~1 sigma) stands out within a few points.
    Returns (z, ucl, lcl) - the limits widen over the first few points.
    x may be 1-D or 2-D (simulations x time).
    """
    x = np.asarray(x, dtype=float)
    z = np.empty_like(x)
    prev = np.full(x.shape[:-1], center, dtype=float)
    for i in range(x.shape[-1]):
        prev = lam * x[..., i] + (1 - lam) * prev
        z[..., i] = prev
    i = np.arange(1, x.shape[-1] + 1)
    width = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
    return z, center + width, center - width


def capability(x: np.ndarray, lsl: float, usl: float, sigma_within: float | None = None) -> dict[str, float]:
    """Cp/Cpk use short-term (within) sigma, Pp/Ppk the overall sigma.

    Cp asks "could the process fit in the spec if it were centred?", Cpk asks
    "does it, given where it is actually centred?". 1.33 is a common minimum.
    """
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    s_overall = x.std(ddof=1)
    if sigma_within is None:
        sigma_within = np.abs(np.diff(x)).mean() / D2_N2
    return {
        "mean": float(mu),
        "sigma_within": float(sigma_within),
        "sigma_overall": float(s_overall),
        "Cp": float((usl - lsl) / (6 * sigma_within)),
        "Cpk": float(min(usl - mu, mu - lsl) / (3 * sigma_within)),
        "Pp": float((usl - lsl) / (6 * s_overall)),
        "Ppk": float(min(usl - mu, mu - lsl) / (3 * s_overall)),
    }


def simulate_film_thickness(
    n: int = 150,
    target_nm: float = 50.0,
    sigma_nm: float = 0.35,
    spike_at: int = 75,
    shift_at: int = 100,
    shift_sigmas: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Deposition film thickness per run with two planted problems.

    * spike_at: one bad measurement (e.g. a metrology glitch) -> rule 1
    * shift_at: a small, permanent +1 sigma shift (e.g. after a chamber part
      swap) - too small for rule 1 to catch quickly, but EWMA and the run rules
      should.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(target_nm, sigma_nm, n)
    x[spike_at] += 4.5 * sigma_nm
    x[shift_at:] += shift_sigmas * sigma_nm
    return x


def average_run_length(
    shift_sigmas: float, n_sims: int = 2000, horizon: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Average number of points until the first alarm (ARL), by simulation.

    With shift_sigmas = 0 this is the in-control ARL0: how often the chart
    cries wolf (bigger is better). With a shift it is ARL1: how fast a real
    problem is caught (smaller is better). Choosing a chart is a trade-off
    between the two. Series with no alarm inside `horizon` count as `horizon`,
    so very large ARL0 values are slight underestimates.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(shift_sigmas, 1.0, size=(n_sims, horizon))
    flags = western_electric_flags(x, 0.0, 1.0)
    z, ucl, lcl = ewma(x, 0.0, 1.0)
    alarms = {
        "rule1_only": flags["rule1"],
        "all_western_electric": np.logical_or.reduce(list(flags.values())),
        "ewma": (z > ucl) | (z < lcl),
    }

    def arl(a: np.ndarray) -> float:
        first = np.where(a.any(axis=1), a.argmax(axis=1) + 1, horizon)
        return float(first.mean())

    return {name: round(arl(a), 1) for name, a in alarms.items()}


def first_alarm(indices: list[int] | np.ndarray, start: int) -> int | None:
    """First alarm index at or after `start` (None if never)."""
    after = [i for i in indices if i >= start]
    return min(after) if after else None
