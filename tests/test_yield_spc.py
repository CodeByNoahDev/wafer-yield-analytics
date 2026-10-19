import numpy as np
import pytest

from wafermap.spc import average_run_length, capability, ewma, imr_limits, western_electric
from wafermap.yield_model import (
    die_grid_yield,
    fit_yield_models,
    gross_dies_per_wafer,
    murphy_yield,
    neg_binomial_yield,
    poisson_yield,
    seeds_yield,
    yield_vs_die_area,
)


def test_yield_models_basic_properties():
    a = np.array([0.0, 0.5, 1.0, 2.0])
    for fn in (poisson_yield, murphy_yield, seeds_yield):
        assert fn(a, 0.3)[0] == pytest.approx(1.0)
    # Poisson is the most pessimistic and Seeds the most optimistic.
    p, m, s = poisson_yield(a[1:], 0.5), murphy_yield(a[1:], 0.5), seeds_yield(a[1:], 0.5)
    assert np.all(p < m) and np.all(m < s)
    # Negative binomial tends to Poisson as clustering disappears (alpha -> inf).
    assert np.allclose(neg_binomial_yield(a, 0.5, 1e6), poisson_yield(a, 0.5), atol=1e-5)


def test_gross_dies_per_wafer_formula():
    # 300 mm wafer, no edge exclusion, 100 mm^2 die: 706.9 - 66.6 = 640.2
    assert gross_dies_per_wafer(100, 300, 0) == pytest.approx(640.2, abs=0.5)


def test_die_grid_counts_and_defect_kills_die():
    good, total = die_grid_yield(np.empty((0, 2)), 10, 10)
    assert good == total
    assert abs(total - gross_dies_per_wafer(100)) / total < 0.1
    good2, _ = die_grid_yield(np.array([[5.0, 5.0], [5.5, 5.5]]), 10, 10)
    assert good2 == good - 1  # two defects in the same die kill only that die


def test_poisson_fit_recovers_true_density_for_random_defects():
    sim = yield_vs_die_area(np.linspace(4, 24, 8), n_wafers=40, seed=0, d0_per_cm2=0.2, alpha=None)
    d0 = fit_yield_models(sim["area_cm2"], sim["yield"])["poisson"]["params"][0]
    assert d0 == pytest.approx(0.2, rel=0.15)


def test_negative_binomial_recovers_density_and_alpha():
    sim = yield_vs_die_area(np.linspace(4, 28, 12), n_wafers=400, seed=1, d0_per_cm2=0.15, alpha=2.0)
    fits = fit_yield_models(sim["area_cm2"], sim["yield"])
    d0, alpha = fits["negative_binomial"]["params"]
    assert d0 == pytest.approx(0.15, rel=0.15)
    assert 1.2 < alpha < 3.5
    assert fits["negative_binomial"]["rmse"] < fits["poisson"]["rmse"]


def test_rule1_and_rule4_fire():
    x = np.zeros(30)
    x[5] = 5.0
    x[20:28] = 0.5
    hits = western_electric(x, center=0.0, sigma=1.0)
    assert 5 in hits["rule1"]
    assert 27 in hits["rule4"]


def test_rule1_false_alarm_rate_is_low_in_control():
    x = np.random.default_rng(0).normal(0, 1, 20000)
    hits = western_electric(x, 0.0, 1.0)
    assert len(hits["rule1"]) / len(x) < 0.005  # theory: 0.27 %


def test_imr_sigma_estimate():
    x = np.random.default_rng(1).normal(10, 2, 5000)
    assert imr_limits(x).sigma == pytest.approx(2.0, rel=0.05)


def test_ewma_detects_small_shift_faster_than_rule1_on_average():
    # One run can be lucky either way, so compare the average delay over many
    # simulated shifts (the "average run length" used to design SPC charts).
    rng = np.random.default_rng(2)
    ewma_delays, rule1_delays = [], []
    for _ in range(200):
        x = np.concatenate([rng.normal(0, 1, 50), rng.normal(1.0, 1, 300)])
        z, ucl, _ = ewma(x, 0.0, 1.0)
        ewma_delays.append(np.argmax(z[50:] > ucl[50:]))
        rule1_delays.append(np.argmax(x[50:] > 3))
    assert np.mean(ewma_delays) < 0.5 * np.mean(rule1_delays)


def test_average_run_length_matches_theory():
    in_control = average_run_length(0.0, n_sims=2000, horizon=3000, seed=1)
    assert 320 < in_control["rule1_only"] < 420  # theory: 1 / 0.0027 = 370
    # Extra run rules buy speed with many more false alarms.
    assert in_control["all_western_electric"] < 0.4 * in_control["rule1_only"]
    shifted = average_run_length(1.0, n_sims=2000, horizon=3000, seed=1)
    assert 35 < shifted["rule1_only"] < 52  # theory: ~44
    assert shifted["ewma"] < 0.35 * shifted["rule1_only"]


def test_vectorised_rules_match_on_2d_input():
    x = np.random.default_rng(5).normal(size=(3, 200))
    from wafermap.spc import western_electric_flags
    flags_2d = western_electric_flags(x, 0.0, 1.0)
    for row in range(3):
        flags_1d = western_electric_flags(x[row], 0.0, 1.0)
        for rule in flags_1d:
            assert np.array_equal(flags_1d[rule], flags_2d[rule][row])


def test_capability_indices():
    z = np.random.default_rng(3).normal(size=2000)
    x = 10 + (z - z.mean()) / z.std(ddof=1)  # exactly mean 10, sd 1
    cap = capability(x, 7, 13)
    assert cap["Pp"] == pytest.approx(1.0)
    assert cap["Ppk"] == pytest.approx(1.0)
    assert cap["Cp"] == pytest.approx(1.0, rel=0.05)
