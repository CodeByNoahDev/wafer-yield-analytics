import numpy as np
import pytest

from wafermap.features import FEATURE_NAMES, extract_features, radon_transform
from wafermap.generator import PATTERNS, generate_wafer, make_dataset, wafer_coords


@pytest.mark.parametrize("pattern", PATTERNS)
def test_wafer_encoding(pattern):
    wafer = generate_wafer(pattern, size=40, rng=np.random.default_rng(0))
    mask, _, _ = wafer_coords(40)
    assert set(np.unique(wafer)) <= {0, 1, 2}
    # Off-wafer positions are exactly the 0s.
    assert np.array_equal(wafer > 0, mask)


def _mean_fail_rate(pattern, rng, n=20):
    rates = []
    for _ in range(n):
        wafer = generate_wafer(pattern, rng=rng)
        rates.append((wafer == 2).sum() / (wafer > 0).sum())
    return np.mean(rates)


def test_pattern_fail_rates_are_sensible():
    rng = np.random.default_rng(1)
    assert _mean_fail_rate("none", rng) < 0.06  # background only: U(0, 0.08)
    assert _mean_fail_rate("near_full", rng) > 0.7


def test_dataset_is_reproducible():
    w1, y1 = make_dataset(3, seed=42)
    w2, y2 = make_dataset(3, seed=42)
    assert np.array_equal(y1, y2)
    assert all(np.array_equal(a, b) for a, b in zip(w1, w2))
    assert len(w1) == 3 * len(PATTERNS)


def test_feature_vector_shape_and_finite():
    wafer = generate_wafer("scratch", size=48, rng=np.random.default_rng(3))
    f = extract_features(wafer)
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_features_are_rotation_invariant():
    wafer = generate_wafer("edge_loc", size=45, rng=np.random.default_rng(4))
    f0 = extract_features(wafer)
    f90 = extract_features(np.rot90(wafer))
    assert np.allclose(f0, f90, atol=1e-6)


def test_radon_of_centred_point_peaks_in_the_middle():
    # A fail at the exact centre projects to the same spot at every angle.
    img = np.zeros((21, 21))
    img[10, 10] = 1.0
    assert (radon_transform(img).argmax(axis=0) == 10).all()


def test_radon_conserves_mass_for_extended_shapes():
    # Interpolation smears single pixels, but for real-sized blobs the total
    # projected "fail mass" is (almost) the same at every angle.
    yy, xx = np.mgrid[:41, :41]
    disk = (np.hypot(yy - 20, xx - 20) <= 8).astype(float)
    totals = radon_transform(disk).sum(axis=0)
    assert totals.std() / totals.mean() < 0.02
    assert totals.mean() == pytest.approx(disk.sum(), rel=0.03)


def test_radon_std_separates_scratch_from_ring():
    yy, xx = np.mgrid[:41, :41]
    r = np.hypot(yy - 20, xx - 20)
    ring = ((r > 15) & (r < 18)).astype(float)
    line = np.zeros((41, 41))
    line[20, 5:36] = 1.0
    assert radon_transform(line).std(axis=1).max() > 5 * radon_transform(ring).std(axis=1).max()


def test_empty_wafer_has_zero_cluster_features():
    wafer = generate_wafer("none", size=30, rng=np.random.default_rng(5))
    wafer[wafer == 2] = 1
    f = extract_features(wafer)
    assert f[0] == 0.0
    assert np.all(f[-6:] == 0.0)
