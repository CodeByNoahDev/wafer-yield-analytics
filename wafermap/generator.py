"""Synthetic wafer-map generator.

Wafer maps use the same encoding as the public WM-811K dataset so the rest of
the pipeline works on real fab data too:

    0 = no die (outside the wafer)
    1 = die passed electrical test
    2 = die failed electrical test

Each defect *pattern* is a spatial signature that points to a different root
cause in the fab (e.g. an edge ring often comes from etch/deposition
non-uniformity at the wafer edge, a scratch from wafer handling or CMP).
"""
from __future__ import annotations

import numpy as np

PATTERNS: tuple[str, ...] = (
    "none",
    "center",
    "donut",
    "edge_loc",
    "edge_ring",
    "loc",
    "random",
    "scratch",
    "near_full",
)

# Rough class mix for the demo dataset. Real fabs are heavily imbalanced:
# most wafers have no systematic pattern, and near-full failures are rare.
DEFAULT_CLASS_COUNTS: dict[str, int] = {
    "none": 900,
    "center": 220,
    "donut": 160,
    "edge_loc": 220,
    "edge_ring": 220,
    "loc": 200,
    "random": 140,
    "scratch": 180,
    "near_full": 60,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def wafer_coords(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, r, theta) for a square grid holding a circular wafer.

    r is the distance from the wafer centre normalised so the edge is 1.0;
    theta is the polar angle in radians.
    """
    yy, xx = np.mgrid[:size, :size].astype(float)
    c = (size - 1) / 2
    r = np.hypot(yy - c, xx - c) / (size / 2)
    theta = np.arctan2(yy - c, xx - c)
    return r <= 1.0, r, theta


def _angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest absolute difference between angles, in radians."""
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def _scratch_probability(size: int, rng: np.random.Generator) -> np.ndarray:
    """Probability map for a thin, slightly curved scratch (quadratic Bezier)."""
    yy, xx = np.mgrid[:size, :size].astype(float)
    c = (size - 1) / 2
    # Two end points somewhere on the wafer plus a control point that bends it.
    length = rng.uniform(0.5, 1.4) * size / 2
    angle = rng.uniform(0, np.pi)
    mid = np.array([c, c]) + rng.uniform(-0.35, 0.35, size=2) * size / 2
    direction = np.array([np.sin(angle), np.cos(angle)])
    p0 = mid - direction * length / 2
    p2 = mid + direction * length / 2
    normal = np.array([-direction[1], direction[0]])
    p1 = mid + normal * rng.uniform(-0.25, 0.25) * length

    t = np.linspace(0, 1, 200)
    # Real scratches are often broken: cut up to two gaps out of the curve.
    keep = np.ones_like(t, dtype=bool)
    for _ in range(rng.integers(0, 3)):
        start = rng.uniform(0.1, 0.8)
        keep &= ~((t > start) & (t < start + rng.uniform(0.05, 0.15)))
    t = t[keep][:, None]
    curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2  # (n_points, 2)

    pts = np.stack([yy.ravel(), xx.ravel()], axis=1)  # (size*size, 2)
    # Distance from every die to its nearest point on the curve.
    d = np.min(np.linalg.norm(pts[:, None, :] - curve[None, :, :], axis=2), axis=1)
    width = rng.uniform(0.6, 1.2)
    return (rng.uniform(0.5, 0.9) * _sigmoid((width - d) * 4)).reshape(size, size)


def pattern_probability(
    pattern: str, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-die failure probability for one defect pattern (before background)."""
    mask, r, theta = wafer_coords(size)
    p = np.zeros((size, size))
    # Pattern strength varies a lot in real fabs: some signatures are faint.
    strength = rng.uniform(0.35, 0.95)

    if pattern == "none":
        pass
    elif pattern == "center":
        # Slightly off-centre blob: centre-of-wafer issues (e.g. gas flow, spin coat).
        off = rng.normal(0, 0.06, size=2) * size / 2
        c = (size - 1) / 2
        yy, xx = np.mgrid[:size, :size].astype(float)
        rc = np.hypot(yy - c - off[0], xx - c - off[1]) / (size / 2)
        r0 = rng.uniform(0.12, 0.42)
        p = strength * _sigmoid((r0 - rc) / 0.03)
    elif pattern == "donut":
        r1 = rng.uniform(0.2, 0.45)
        r2 = r1 + rng.uniform(0.12, 0.3)
        p = strength * _sigmoid((r - r1) / 0.03) * _sigmoid((r2 - r) / 0.03)
    elif pattern == "edge_ring":
        # Rings are often incomplete, which makes them look like a wide edge-loc.
        r0 = rng.uniform(0.78, 0.92)
        coverage = rng.uniform(0.7, 1.0)
        gap_centre = rng.uniform(-np.pi, np.pi)
        gap = _sigmoid((_angle_diff(theta, gap_centre) - (1 - coverage) * np.pi) / 0.08)
        p = strength * _sigmoid((r - r0) / 0.025) * gap
    elif pattern == "edge_loc":
        r0 = rng.uniform(0.68, 0.88)
        centre_angle = rng.uniform(-np.pi, np.pi)
        half_width = np.deg2rad(rng.uniform(15, 70))
        in_arc = _sigmoid((half_width - _angle_diff(theta, centre_angle)) / 0.08)
        p = strength * _sigmoid((r - r0) / 0.025) * in_arc
    elif pattern == "loc":
        # Can sit near the edge, where it is easy to confuse with edge-loc.
        rc = rng.uniform(0.1, 0.8)
        ac = rng.uniform(-np.pi, np.pi)
        c = (size - 1) / 2
        cy = c + rc * np.sin(ac) * size / 2
        cx = c + rc * np.cos(ac) * size / 2
        yy, xx = np.mgrid[:size, :size].astype(float)
        d = np.hypot(yy - cy, xx - cx) / (size / 2)
        radius = rng.uniform(0.07, 0.22)
        p = strength * _sigmoid((radius - d) / 0.025)
    elif pattern == "random":
        p = np.full((size, size), rng.uniform(0.1, 0.35))
    elif pattern == "scratch":
        p = _scratch_probability(size, rng)
    elif pattern == "near_full":
        p = np.full((size, size), rng.uniform(0.6, 0.97))
    else:
        raise ValueError(f"Unknown pattern {pattern!r}; expected one of {PATTERNS}")

    return np.where(mask, p, 0.0)


def generate_wafer(
    pattern: str, size: int | None = None, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate one wafer map (0 = off-wafer, 1 = pass, 2 = fail).

    Every pattern also gets a low level of random background defects, because
    real wafers always have some random particle-driven fails on top of any
    systematic signature.
    """
    rng = rng if rng is not None else np.random.default_rng()
    size = size if size is not None else int(rng.integers(30, 57))
    mask, _, _ = wafer_coords(size)

    background = rng.uniform(0.0, 0.08)
    p_pattern = pattern_probability(pattern, size, rng)
    # Combine independent failure mechanisms: P(fail) = 1 - P(survive both).
    p_fail = 1 - (1 - p_pattern) * (1 - background)

    fails = (rng.random((size, size)) < p_fail) & mask
    wafer = np.where(mask, 1, 0).astype(np.int8)
    wafer[fails] = 2
    return wafer


def make_dataset(
    class_counts: dict[str, int] | int = DEFAULT_CLASS_COUNTS, seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate a labelled dataset of wafer maps.

    Wafer sizes vary (like real products with different die sizes), so maps are
    returned as a list rather than one stacked array.
    """
    if isinstance(class_counts, int):
        class_counts = {p: class_counts for p in PATTERNS}
    rng = np.random.default_rng(seed)
    wafers: list[np.ndarray] = []
    labels: list[str] = []
    for pattern, n in class_counts.items():
        for _ in range(n):
            wafers.append(generate_wafer(pattern, rng=rng))
            labels.append(pattern)
    return wafers, np.array(labels)
