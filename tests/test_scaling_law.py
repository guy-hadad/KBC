"""Tests for the power-law fit behind the scaling-law figures."""

import numpy as np
import pytest

from eventfm.benchmark.scaling import (
    MIN_R_SQUARED,
    fit_scaling_law,
    to_error,
)

SIZES = [64, 128, 256, 512, 1024, 2048]


def _power_law(sizes, coefficient, exponent):
    return [(size, coefficient * size ** (-exponent)) for size in sizes]


def test_fit_recovers_a_known_exponent():
    points = _power_law(SIZES, coefficient=3.0, exponent=0.25)

    fit = fit_scaling_law(points)

    assert fit.exponent == pytest.approx(0.25, abs=1e-6)
    assert fit.coefficient == pytest.approx(3.0, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.is_reliable


def test_fit_recovers_a_steeper_exponent():
    fit = fit_scaling_law(_power_law(SIZES, coefficient=1.0, exponent=0.6))

    assert fit.exponent == pytest.approx(0.6, abs=1e-6)


def test_a_flat_curve_yields_a_near_zero_exponent():
    """A method that stops improving should not look data-hungry."""

    points = [(size, 0.4) for size in SIZES]

    fit = fit_scaling_law(points)

    assert fit.exponent == pytest.approx(0.0, abs=1e-9)


def test_noisy_curve_is_flagged_as_unreliable():
    rng = np.random.default_rng(0)
    points = [(size, float(rng.uniform(0.2, 0.5))) for size in SIZES]

    fit = fit_scaling_law(points)

    assert fit.r_squared < MIN_R_SQUARED
    assert not fit.is_reliable


def test_fit_needs_at_least_three_points():
    assert fit_scaling_law([(64, 0.4), (128, 0.3)]) is None


def test_fit_rejects_a_single_distinct_size():
    assert fit_scaling_law([(64, 0.4), (64, 0.3), (64, 0.2)]) is None


def test_fit_drops_non_positive_errors():
    """A perfect score gives error 0, which has no logarithm."""

    points = _power_law(SIZES, coefficient=2.0, exponent=0.3) + [(4096, 0.0)]

    fit = fit_scaling_law(points)

    assert fit.num_points == len(SIZES)
    assert fit.exponent == pytest.approx(0.3, abs=1e-6)


def test_predict_follows_the_fitted_law():
    fit = fit_scaling_law(_power_law(SIZES, coefficient=2.0, exponent=0.4))

    predicted = fit.predict([256.0])

    assert float(predicted[0]) == pytest.approx(2.0 * 256.0**-0.4, rel=1e-6)


def test_complement_error_inverts_a_higher_is_better_metric():
    assert to_error(0.75, "complement") == pytest.approx(0.25)
    # A perfect metric is clamped, not zero, so the log-fit stays defined.
    assert to_error(1.0, "complement") > 0.0


def test_direct_error_passes_an_error_metric_through():
    assert to_error(1.5, "direct") == pytest.approx(1.5)
    assert to_error(0.0, "direct") is None


def test_missing_values_are_dropped():
    assert to_error(None, "complement") is None
    assert to_error(float("nan"), "complement") is None
