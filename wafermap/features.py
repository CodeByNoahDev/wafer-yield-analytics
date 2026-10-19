"""Feature extraction for wafer-map pattern classification.

The features follow the ideas in Wu, Jang & Chen (2015), "Wafer Map Failure
Pattern Recognition and Similarity Ranking for Large-Scale Data Sets"
(IEEE Trans. Semiconductor Manufacturing) - the paper that released WM-811K:

* density features   - fail rate in radial bands and wafer zones
* Radon features     - projections of the fail map at many angles
* geometry features  - shape of the biggest connected cluster of fails

Every feature is designed to be (nearly) independent of wafer size and of
wafer rotation, because a scratch is a scratch whichever way the wafer was
loaded into the tool.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

RADIAL_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0001])
RADON_ANGLES = np.arange(0, 360, 10)
RADON_POINTS = 20


def _feature_names() -> list[str]:
    names = ["fail_rate"]
    names += [f"radial_{lo:.2f}_{hi:.2f}" for lo, hi in zip(RADIAL_EDGES[:-1], RADIAL_EDGES[1:])]
    names += ["zone_center"]
    names += [f"zone_mid_rank{i}" for i in range(4)]
    names += [f"zone_edge_rank{i}" for i in range(8)]
    names += [f"radon_mean_{i}" for i in range(RADON_POINTS)]
    names += [f"radon_std_{i}" for i in range(RADON_POINTS)]
    names += [
        "cluster_area_frac",
        "cluster_count_per_100",
        "cluster_mean_r",
        "cluster_r_std",
        "cluster_elongation",
        "cluster_angular_coverage",
    ]
    return names


FEATURE_NAMES: list[str] = _feature_names()


def polar_grid(wafer: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, r, theta) using the wafer's own centre and radius.

    Works for any map size, so it also handles real WM-811K maps.
    """
    mask = wafer > 0
    yy, xx = np.indices(wafer.shape, dtype=float)
    cy, cx = yy[mask].mean(), xx[mask].mean()
    dist = np.hypot(yy - cy, xx - cx)
    radius = dist[mask].max() + 0.5
    return mask, dist / radius, np.arctan2(yy - cy, xx - cx)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _sector_index(theta: np.ndarray, n: int) -> np.ndarray:
    """Which of n equal angular sectors each die is in.

    The tiny offset pushes dies that sit exactly on a sector boundary the same
    way every time, so rotating the wafer by 90 degrees just relabels sectors.
    """
    return np.floor((theta + np.pi + 1e-7) / (2 * np.pi) * n).astype(int) % n


def density_features(fails: np.ndarray, mask: np.ndarray, r: np.ndarray, theta: np.ndarray) -> list[float]:
    """Fail rate in radial bands and in 13 wafer zones (centre, 4 middle, 8 edge)."""
    feats = [_mean_or_zero(fails[mask])]

    for lo, hi in zip(RADIAL_EDGES[:-1], RADIAL_EDGES[1:]):
        band = mask & (r >= lo) & (r < hi)
        feats.append(_mean_or_zero(fails[band]))

    feats.append(_mean_or_zero(fails[mask & (r < 0.3)]))

    # Zone densities are sorted high-to-low so that the feature says "one zone
    # is bad" without caring *which* zone - that keeps it rotation-invariant.
    mid = mask & (r >= 0.3) & (r < 0.7)
    mid_idx = _sector_index(theta, 4)
    mid_rates = [_mean_or_zero(fails[mid & (mid_idx == k)]) for k in range(4)]
    feats += sorted(mid_rates, reverse=True)

    edge = mask & (r >= 0.7)
    edge_idx = _sector_index(theta, 8)
    edge_rates = [_mean_or_zero(fails[edge & (edge_idx == k)]) for k in range(8)]
    feats += sorted(edge_rates, reverse=True)
    return feats


def radon_transform(image: np.ndarray, angles: np.ndarray = RADON_ANGLES) -> np.ndarray:
    """Radon transform: sum the image along parallel lines at each angle.

    Returns a "sinogram" of shape (image width, number of angles). A straight
    scratch produces one sharp peak at the angle it lines up with, while a
    ring produces the same profile at every angle.
    """
    sinogram = np.empty((image.shape[1], len(angles)))
    for i, angle in enumerate(angles):
        rotated = ndimage.rotate(image, angle, reshape=False, order=1, mode="constant")
        sinogram[:, i] = rotated.sum(axis=0)
    return sinogram


def radon_features(fails: np.ndarray) -> list[float]:
    """Mean and std (across angles) of the Radon projections, resampled to 20 points.

    The mean profile captures *where* fails sit radially; the std captures how
    *directional* they are (high for scratches, ~0 for rings).
    """
    size = fails.shape[1]
    sinogram = radon_transform(fails.astype(float)) / size
    x_old = np.linspace(0, 1, size)
    x_new = np.linspace(0, 1, RADON_POINTS)
    mean_profile = np.interp(x_new, x_old, sinogram.mean(axis=1))
    std_profile = np.interp(x_new, x_old, sinogram.std(axis=1))
    return list(mean_profile) + list(std_profile)


def cluster_features(fails: np.ndarray, mask: np.ndarray, r: np.ndarray, theta: np.ndarray) -> list[float]:
    """Shape descriptors of the largest connected group of failing dies."""
    labels, n_regions = ndimage.label(fails, structure=np.ones((3, 3)))
    n_dies = mask.sum()
    if n_regions == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    sizes = np.bincount(labels.ravel())[1:]
    biggest = labels == (np.argmax(sizes) + 1)
    area_frac = biggest.sum() / n_dies
    count_per_100 = 100 * n_regions / n_dies

    rr = r[biggest]
    ys, xs = np.nonzero(biggest)
    if len(ys) >= 3:
        eig = np.linalg.eigvalsh(np.cov(np.stack([ys, xs])))
        # 0 for a round blob, approaching 1 for a thin line.
        elongation = 1 - (max(eig[0], 0.0) + 1e-9) / (eig[1] + 1e-9)
    else:
        elongation = 0.0

    # How much of the full circle the cluster spans (edge ring ~1, edge-loc small).
    coverage = len(np.unique(_sector_index(theta[biggest], 36))) / 36

    return [area_frac, count_per_100, rr.mean(), rr.std(), elongation, coverage]


def extract_features(wafer: np.ndarray) -> np.ndarray:
    """Turn one wafer map into a fixed-length feature vector (see FEATURE_NAMES)."""
    mask, r, theta = polar_grid(wafer)
    fails = (wafer == 2)
    feats = (
        density_features(fails, mask, r, theta)
        + radon_features(fails)
        + cluster_features(fails, mask, r, theta)
    )
    return np.asarray(feats, dtype=float)


def extract_features_batch(wafers: list[np.ndarray]) -> np.ndarray:
    return np.vstack([extract_features(w) for w in wafers])
