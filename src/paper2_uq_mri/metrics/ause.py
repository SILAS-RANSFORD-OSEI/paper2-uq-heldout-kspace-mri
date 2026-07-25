"""Tie-aware weighted sparsification metrics for Paper 2.

Primary Task E quantity
-----------------------
The normalized area under the sparsification error curve
compares a method's uncertainty ranking against an oracle
ranking and a random-removal reference.

Interpretation:
    0.0  oracle-equivalent ranking
    1.0  random-equivalent ranking
    >1.0 worse than random

Lower values are better.

Fractions represent removed evaluation mass. Evaluation
mass is defined by the supplied non-negative pixel weights.
Paper 2 uses the frozen soft support values as these weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_REMOVAL_FRACTIONS = np.arange(
    0.0,
    1.0,
    0.01,
    dtype=np.float64,
)


@dataclass(frozen=True)
class AUSEResult:
    """Complete normalized-AUSE result for one volume."""

    removal_fractions: np.ndarray
    method_curve: np.ndarray
    oracle_curve: np.ndarray
    random_curve: np.ndarray
    raw_ause: float
    random_oracle_area: float
    normalized_ause: float
    weighted_mean_error: float
    positive_weight_observations: int
    total_evaluation_weight: float


def _as_vector(
    values: Iterable[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; "
            f"received shape {array.shape}."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array


def validate_removal_fractions(
    fractions: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Validate the frozen sparsification grid."""

    array = _as_vector(
        fractions,
        name="removal_fractions",
    )

    if np.any(array < 0.0):
        raise ValueError(
            "Removal fractions must be non-negative."
        )

    if np.any(array >= 1.0):
        raise ValueError(
            "Removal fractions must be strictly below 1."
        )

    if np.any(np.diff(array) <= 0.0):
        raise ValueError(
            "Removal fractions must be strictly increasing."
        )

    return array


def _validate_inputs(
    errors: Iterable[float] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    weights: Iterable[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error_array = _as_vector(
        errors,
        name="errors",
    )

    score_array = _as_vector(
        scores,
        name="scores",
    )

    if error_array.shape != score_array.shape:
        raise ValueError(
            "errors and scores must have identical shapes."
        )

    if np.any(error_array < 0.0):
        raise ValueError(
            "Absolute prediction errors must be non-negative."
        )

    if weights is None:
        weight_array = np.ones_like(
            error_array,
            dtype=np.float64,
        )

    else:
        weight_array = _as_vector(
            weights,
            name="weights",
        )

        if weight_array.shape != error_array.shape:
            raise ValueError(
                "weights must have the same shape as errors."
            )

        if np.any(weight_array < 0.0):
            raise ValueError(
                "weights must be non-negative."
            )

    positive = weight_array > 0.0

    if np.count_nonzero(positive) < 2:
        raise ValueError(
            "At least two observations must have "
            "positive evaluation weight."
        )

    return (
        error_array[positive],
        score_array[positive],
        weight_array[positive],
    )


def _grouped_retained_error_curve(
    errors: np.ndarray,
    ranking_scores: np.ndarray,
    weights: np.ndarray,
    removal_fractions: np.ndarray,
) -> np.ndarray:
    """Compute a tie-invariant retained-error curve.

    Pixels are ordered from lowest to highest ranking score.
    When a requested retained mass falls inside a tied-score
    group, a fractional amount of that entire group is used.
    Its contribution equals the group's weighted mean error,
    making the result invariant to ordering within ties.
    """

    order = np.argsort(
        ranking_scores,
        kind="stable",
    )

    sorted_scores = ranking_scores[
        order
    ]

    sorted_errors = errors[
        order
    ]

    sorted_weights = weights[
        order
    ]

    group_starts = np.concatenate(
        (
            np.asarray(
                [0],
                dtype=np.int64,
            ),
            np.flatnonzero(
                sorted_scores[1:]
                != sorted_scores[:-1]
            )
            + 1,
        )
    )

    group_weight = np.add.reduceat(
        sorted_weights,
        group_starts,
    )

    group_weighted_error = np.add.reduceat(
        sorted_weights
        * sorted_errors,
        group_starts,
    )

    cumulative_weight = np.cumsum(
        group_weight,
        dtype=np.float64,
    )

    cumulative_weighted_error = np.cumsum(
        group_weighted_error,
        dtype=np.float64,
    )

    total_weight = float(
        cumulative_weight[-1]
    )

    retained_mass = (
        1.0 - removal_fractions
    ) * total_weight

    curve = np.empty_like(
        retained_mass,
        dtype=np.float64,
    )

    for index, target_mass in enumerate(
        retained_mass
    ):
        group_index = int(
            np.searchsorted(
                cumulative_weight,
                target_mass,
                side="left",
            )
        )

        if group_index >= group_weight.size:
            group_index = (
                group_weight.size - 1
            )

        if group_index == 0:
            prior_weight = 0.0
            prior_weighted_error = 0.0

        else:
            prior_weight = float(
                cumulative_weight[
                    group_index - 1
                ]
            )

            prior_weighted_error = float(
                cumulative_weighted_error[
                    group_index - 1
                ]
            )

        required_from_group = (
            float(target_mass)
            - prior_weight
        )

        current_group_weight = float(
            group_weight[
                group_index
            ]
        )

        if current_group_weight <= 0.0:
            raise RuntimeError(
                "Encountered a non-positive tied-group weight."
            )

        current_group_mean_error = float(
            group_weighted_error[
                group_index
            ]
            / current_group_weight
        )

        retained_weighted_error = (
            prior_weighted_error
            + required_from_group
            * current_group_mean_error
        )

        curve[index] = (
            retained_weighted_error
            / float(target_mass)
        )

    return curve


def sparsification_error_curve(
    errors: Iterable[float] | np.ndarray,
    uncertainty: Iterable[float] | np.ndarray,
    *,
    weights: Iterable[float] | np.ndarray | None = None,
    removal_fractions: Iterable[float] | np.ndarray = (
        DEFAULT_REMOVAL_FRACTIONS
    ),
) -> np.ndarray:
    """Return the method sparsification error curve.

    Lower uncertainty is interpreted as greater confidence,
    so the retained set contains the lowest-uncertainty mass.
    """

    fractions = validate_removal_fractions(
        removal_fractions
    )

    (
        error_array,
        uncertainty_array,
        weight_array,
    ) = _validate_inputs(
        errors,
        uncertainty,
        weights,
    )

    return _grouped_retained_error_curve(
        errors=error_array,
        ranking_scores=uncertainty_array,
        weights=weight_array,
        removal_fractions=fractions,
    )


def normalized_ause(
    errors: Iterable[float] | np.ndarray,
    uncertainty: Iterable[float] | np.ndarray,
    *,
    weights: Iterable[float] | np.ndarray | None = None,
    removal_fractions: Iterable[float] | np.ndarray = (
        DEFAULT_REMOVAL_FRACTIONS
    ),
) -> AUSEResult:
    """Compute the frozen per-volume normalized AUSE."""

    fractions = validate_removal_fractions(
        removal_fractions
    )

    (
        error_array,
        uncertainty_array,
        weight_array,
    ) = _validate_inputs(
        errors,
        uncertainty,
        weights,
    )

    method_curve = (
        _grouped_retained_error_curve(
            errors=error_array,
            ranking_scores=uncertainty_array,
            weights=weight_array,
            removal_fractions=fractions,
        )
    )

    oracle_curve = (
        _grouped_retained_error_curve(
            errors=error_array,
            ranking_scores=error_array,
            weights=weight_array,
            removal_fractions=fractions,
        )
    )

    total_weight = float(
        np.sum(
            weight_array,
            dtype=np.float64,
        )
    )

    weighted_mean_error = float(
        np.sum(
            weight_array
            * error_array,
            dtype=np.float64,
        )
        / total_weight
    )

    random_curve = np.full_like(
        fractions,
        weighted_mean_error,
        dtype=np.float64,
    )

    integration = getattr(
        np,
        "trapezoid",
        np.trapz,
    )

    raw_ause = float(
        integration(
            method_curve
            - oracle_curve,
            x=fractions,
        )
    )

    random_oracle_area = float(
        integration(
            random_curve
            - oracle_curve,
            x=fractions,
        )
    )

    numerical_tolerance = 1.0e-12

    if raw_ause < -numerical_tolerance:
        raise RuntimeError(
            "Method curve was unexpectedly better than "
            "the oracle curve beyond numerical tolerance."
        )

    if raw_ause < 0.0:
        raw_ause = 0.0

    if random_oracle_area <= numerical_tolerance:
        raise ValueError(
            "Normalized AUSE is undefined because the "
            "random and oracle reference curves have "
            "negligible separation."
        )

    normalized_value = float(
        raw_ause
        / random_oracle_area
    )

    return AUSEResult(
        removal_fractions=fractions.copy(),
        method_curve=method_curve,
        oracle_curve=oracle_curve,
        random_curve=random_curve,
        raw_ause=raw_ause,
        random_oracle_area=random_oracle_area,
        normalized_ause=normalized_value,
        weighted_mean_error=weighted_mean_error,
        positive_weight_observations=int(
            error_array.size
        ),
        total_evaluation_weight=total_weight,
    )
