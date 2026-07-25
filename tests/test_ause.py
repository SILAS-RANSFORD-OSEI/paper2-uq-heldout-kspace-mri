import numpy as np
import pytest

from paper2_uq_mri.metrics.ause import (
    DEFAULT_REMOVAL_FRACTIONS,
    normalized_ause,
    sparsification_error_curve,
)


def test_default_fraction_grid_is_frozen():
    expected = np.arange(
        0.0,
        1.0,
        0.01,
        dtype=np.float64,
    )

    assert np.array_equal(
        DEFAULT_REMOVAL_FRACTIONS,
        expected,
    )

    assert expected.size == 100
    assert expected[0] == 0.0
    assert expected[-1] == 0.99


def test_oracle_uncertainty_has_zero_normalized_ause():
    errors = np.asarray(
        [0.1, 0.2, 0.4, 0.8, 1.6],
        dtype=np.float64,
    )

    result = normalized_ause(
        errors,
        errors,
    )

    assert result.normalized_ause == pytest.approx(
        0.0,
        abs=1.0e-12,
    )

    assert np.allclose(
        result.method_curve,
        result.oracle_curve,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_constant_uncertainty_is_random_equivalent():
    errors = np.asarray(
        [0.1, 0.3, 0.9, 1.4, 2.0],
        dtype=np.float64,
    )

    uncertainty = np.ones_like(
        errors
    )

    result = normalized_ause(
        errors,
        uncertainty,
    )

    assert result.normalized_ause == pytest.approx(
        1.0,
        abs=1.0e-12,
    )

    assert np.allclose(
        result.method_curve,
        result.random_curve,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_reversed_ranking_can_be_worse_than_random():
    errors = np.asarray(
        [0.1, 0.2, 0.4, 0.8, 1.6],
        dtype=np.float64,
    )

    uncertainty = -errors

    result = normalized_ause(
        errors,
        uncertainty,
    )

    assert result.normalized_ause > 1.0


def test_tied_uncertainty_is_permutation_invariant():
    errors = np.asarray(
        [0.2, 1.5, 0.4, 2.0, 0.8, 1.1],
        dtype=np.float64,
    )

    uncertainty = np.asarray(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        dtype=np.float64,
    )

    weights = np.asarray(
        [1.0, 2.0, 1.5, 0.5, 3.0, 1.0],
        dtype=np.float64,
    )

    first = normalized_ause(
        errors,
        uncertainty,
        weights=weights,
    )

    permutation = np.asarray(
        [1, 0, 3, 2, 5, 4],
        dtype=np.int64,
    )

    second = normalized_ause(
        errors[permutation],
        uncertainty[permutation],
        weights=weights[permutation],
    )

    assert first.normalized_ause == pytest.approx(
        second.normalized_ause,
        abs=1.0e-12,
    )

    assert np.allclose(
        first.method_curve,
        second.method_curve,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_weight_scaling_does_not_change_result():
    errors = np.asarray(
        [0.2, 0.6, 1.0, 1.7, 2.5],
        dtype=np.float64,
    )

    uncertainty = np.asarray(
        [0.5, 0.2, 0.8, 0.4, 0.9],
        dtype=np.float64,
    )

    weights = np.asarray(
        [0.4, 0.8, 1.0, 0.6, 0.3],
        dtype=np.float64,
    )

    first = normalized_ause(
        errors,
        uncertainty,
        weights=weights,
    )

    second = normalized_ause(
        errors,
        uncertainty,
        weights=weights * 100.0,
    )

    assert first.normalized_ause == pytest.approx(
        second.normalized_ause,
        abs=1.0e-12,
    )

    assert np.allclose(
        first.method_curve,
        second.method_curve,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_uniform_weights_match_unweighted_result():
    errors = np.asarray(
        [0.2, 0.5, 0.9, 1.3, 2.1],
        dtype=np.float64,
    )

    uncertainty = np.asarray(
        [0.3, 0.8, 0.2, 1.0, 0.5],
        dtype=np.float64,
    )

    first = normalized_ause(
        errors,
        uncertainty,
    )

    second = normalized_ause(
        errors,
        uncertainty,
        weights=np.ones_like(errors),
    )

    assert first.normalized_ause == pytest.approx(
        second.normalized_ause,
        abs=1.0e-12,
    )


def test_direct_curve_matches_result_curve():
    errors = np.asarray(
        [0.1, 0.4, 0.8, 1.2, 1.9],
        dtype=np.float64,
    )

    uncertainty = np.asarray(
        [0.3, 0.6, 0.2, 0.9, 0.7],
        dtype=np.float64,
    )

    direct = sparsification_error_curve(
        errors,
        uncertainty,
    )

    result = normalized_ause(
        errors,
        uncertainty,
    )

    assert np.array_equal(
        direct,
        result.method_curve,
    )


def test_negative_errors_are_rejected():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        normalized_ause(
            np.asarray(
                [0.1, -0.2, 0.4]
            ),
            np.asarray(
                [0.3, 0.2, 0.1]
            ),
        )


def test_nonfinite_uncertainty_is_rejected():
    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        normalized_ause(
            np.asarray(
                [0.1, 0.2, 0.4]
            ),
            np.asarray(
                [0.3, np.nan, 0.1]
            ),
        )


def test_zero_weight_observations_are_excluded():
    errors = np.asarray(
        [0.1, 100.0, 0.5, 1.0],
        dtype=np.float64,
    )

    uncertainty = np.asarray(
        [0.2, -100.0, 0.4, 0.8],
        dtype=np.float64,
    )

    weights = np.asarray(
        [1.0, 0.0, 1.0, 1.0],
        dtype=np.float64,
    )

    with_zero_weight = normalized_ause(
        errors,
        uncertainty,
        weights=weights,
    )

    without_zero_weight = normalized_ause(
        errors[
            [0, 2, 3]
        ],
        uncertainty[
            [0, 2, 3]
        ],
    )

    assert (
        with_zero_weight.normalized_ause
        == pytest.approx(
            without_zero_weight.normalized_ause,
            abs=1.0e-12,
        )
    )
