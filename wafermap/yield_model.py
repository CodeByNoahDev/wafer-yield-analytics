"""Die-yield models and a Monte Carlo defect simulator.

Die yield is the fraction of dies on a wafer that work. If killer defects
land on the wafer at an average density D0 (defects per cm^2), a die of area A
"catches" on average A*D0 defects. How yield depends on A*D0 depends on how
the defects are spread out:

* Poisson          Y = exp(-A*D0)               defects fully random
* Murphy           Y = ((1 - exp(-A*D0)) / (A*D0))^2
* Seeds            Y = 1 / (1 + A*D0)
* Negative binomial Y = (1 + A*D0/alpha)^-alpha   defects clustered;
                                                  alpha -> inf gives Poisson

Real defect density is not constant: it varies from wafer to wafer (and from
region to region) because tools drift, chambers get dirty and get cleaned.
Stapper showed that if the density D follows a Gamma distribution with mean D0
and shape alpha, averaging the Poisson yield exp(-A*D) over D gives exactly
the negative binomial model - which is why the industry uses it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def poisson_yield(area_cm2: np.ndarray, d0: float) -> np.ndarray:
    return np.exp(-np.asarray(area_cm2) * d0)


def murphy_yield(area_cm2: np.ndarray, d0: float) -> np.ndarray:
    ad = np.asarray(area_cm2, dtype=float) * d0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ad > 0, -np.expm1(-ad) / np.where(ad > 0, ad, 1.0), 1.0)
    return ratio**2


def seeds_yield(area_cm2: np.ndarray, d0: float) -> np.ndarray:
    return 1.0 / (1.0 + np.asarray(area_cm2) * d0)


def neg_binomial_yield(area_cm2: np.ndarray, d0: float, alpha: float) -> np.ndarray:
    return (1.0 + np.asarray(area_cm2) * d0 / alpha) ** (-alpha)


YIELD_MODELS = {
    "poisson": poisson_yield,
    "murphy": murphy_yield,
    "seeds": seeds_yield,
    "negative_binomial": neg_binomial_yield,
}


def gross_dies_per_wafer(die_area_mm2: float, wafer_diameter_mm: float = 300.0, edge_exclusion_mm: float = 3.0) -> float:
    """Classic estimate of how many whole dies fit on a round wafer.

    DPW = pi*r^2/A - pi*d/sqrt(2A); the second term removes the partial dies
    lost around the circular edge.
    """
    d = wafer_diameter_mm - 2 * edge_exclusion_mm
    return np.pi * (d / 2) ** 2 / die_area_mm2 - np.pi * d / np.sqrt(2 * die_area_mm2)


def _uniform_in_disk(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    # sqrt() makes points uniform per unit *area*, not bunched at the centre.
    r = radius * np.sqrt(rng.random(n))
    t = 2 * np.pi * rng.random(n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def simulate_defects(
    rng: np.random.Generator,
    d0_per_cm2: float = 0.15,
    alpha: float | None = 2.0,
    cluster_fraction: float = 0.0,
    mean_cluster_size: float = 6.0,
    cluster_sigma_mm: float = 4.0,
    wafer_diameter_mm: float = 300.0,
) -> np.ndarray:
    """Random killer-defect positions (mm) on one wafer.

    alpha: this wafer's defect density is drawn from Gamma(alpha, d0/alpha),
        so densities vary wafer to wafer (smaller alpha = more variation).
        None means every wafer has exactly d0 (pure Poisson).
    cluster_fraction: optional share of defects that arrive in tight Gaussian
        clusters around random "parent" points (a Thomas cluster process),
        mimicking local particle showers. The rest are spread uniformly.
    """
    radius = wafer_diameter_mm / 2
    density = d0_per_cm2 if alpha is None else rng.gamma(alpha, d0_per_cm2 / alpha)
    expected = density * np.pi * (radius / 10) ** 2  # mm -> cm
    random_pts = _uniform_in_disk(rng, rng.poisson((1 - cluster_fraction) * expected), radius)

    n_parents = rng.poisson(cluster_fraction * expected / mean_cluster_size)
    parents = _uniform_in_disk(rng, n_parents, radius)
    children = [
        p + rng.normal(0, cluster_sigma_mm, size=(rng.poisson(mean_cluster_size), 2)) for p in parents
    ]
    pts = np.vstack([random_pts, *children]) if children else random_pts
    return pts[np.hypot(pts[:, 0], pts[:, 1]) <= radius]


def die_grid_yield(
    defects_mm: np.ndarray,
    die_w_mm: float,
    die_h_mm: float,
    wafer_diameter_mm: float = 300.0,
    edge_exclusion_mm: float = 3.0,
) -> tuple[int, int]:
    """Lay a die grid over the wafer and count (good dies, total dies).

    A die counts only if all four corners sit inside the usable radius; it is
    bad if at least one killer defect lands on it.
    """
    usable = wafer_diameter_mm / 2 - edge_exclusion_mm
    nx = int(np.ceil(wafer_diameter_mm / die_w_mm)) + 1
    ny = int(np.ceil(wafer_diameter_mm / die_h_mm)) + 1
    x0 = np.arange(-nx, nx) * die_w_mm  # lower-left corner of each die
    y0 = np.arange(-ny, ny) * die_h_mm
    X, Y = np.meshgrid(x0, y0)
    corners_inside = [
        np.hypot(X + dx, Y + dy) <= usable for dx in (0, die_w_mm) for dy in (0, die_h_mm)
    ]
    valid = np.logical_and.reduce(corners_inside)

    bad = np.zeros_like(valid)
    if len(defects_mm):
        ix = np.floor(defects_mm[:, 0] / die_w_mm).astype(int) + nx
        iy = np.floor(defects_mm[:, 1] / die_h_mm).astype(int) + ny
        keep = (ix >= 0) & (ix < 2 * nx) & (iy >= 0) & (iy < 2 * ny)
        bad[iy[keep], ix[keep]] = True
    return int((valid & ~bad).sum()), int(valid.sum())


def yield_vs_die_area(
    die_sides_mm: np.ndarray,
    n_wafers: int = 40,
    seed: int = 0,
    **defect_kwargs,
) -> dict[str, np.ndarray]:
    """Monte Carlo die yield for square dies of each side length.

    The same simulated wafers are reused for every die size, so differences
    between sizes come from die area alone (common random numbers).
    """
    rng = np.random.default_rng(seed)
    wafers = [simulate_defects(rng, **defect_kwargs) for _ in range(n_wafers)]
    yields, dies_per_wafer = [], []
    for side in die_sides_mm:
        good = total = 0
        for defects in wafers:
            g, t = die_grid_yield(defects, side, side)
            good += g
            total += t
        yields.append(good / total)
        dies_per_wafer.append(total / n_wafers)
    return {
        "die_side_mm": np.asarray(die_sides_mm, dtype=float),
        "area_cm2": np.asarray(die_sides_mm, dtype=float) ** 2 / 100,
        "yield": np.array(yields),
        "dies_per_wafer": np.array(dies_per_wafer),
    }


def fit_yield_models(area_cm2: np.ndarray, observed_yield: np.ndarray) -> dict[str, dict]:
    """Least-squares fit of every yield model; returns parameters and RMSE."""
    fits = {}
    for name, fn in YIELD_MODELS.items():
        if name == "negative_binomial":
            p0, bounds = [0.1, 2.0], ([1e-6, 0.05], [10.0, 1e4])
        else:
            p0, bounds = [0.1], ([1e-6], [10.0])
        params, _ = curve_fit(fn, area_cm2, observed_yield, p0=p0, bounds=bounds, maxfev=20000)
        residual = observed_yield - fn(area_cm2, *params)
        fits[name] = {
            "params": [float(p) for p in params],
            "rmse": float(np.sqrt(np.mean(residual**2))),
        }
    return fits
