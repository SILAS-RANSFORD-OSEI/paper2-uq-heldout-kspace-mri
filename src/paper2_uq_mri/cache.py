"""Semantic boundary for the legacy Paper 1 reliability cache.

The NPZ storage keys ``x``, ``y``, and ``y_raw`` are legacy
implementation details. They are translated immediately into
Paper 2 semantic names:

- predictor_input -> C_v
- u_risk -> u_{risk,v}
- z_risk -> z_{s,v} with s = risk
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_STORAGE_KEYS = (
    "x",
    "y",
    "y_raw",
)

FULL_CHANNEL_NAMES = (
    "reconstruction_magnitude_normalized",
    "zero_filled_magnitude_normalized",
    "intervention_magnitude_normalized",
    "support_mask",
    "analytical_psf",
    "psf_gain_descriptor",
)

A4_CHANNEL_INDICES = (
    0,
    1,
    2,
)

ALPHA_LOG = 10.0
TARGET_QUANTILE = 0.99
EPSILON_NUM = 1.0e-8


@dataclass(frozen=True)
class TargetTransformDiagnostics:
    """Diagnostics for the stored nonlinear target transform."""

    reconstructed_quantile: float
    maximum_absolute_error: float
    percentile_99_absolute_error: float
    mean_absolute_error: float
    pearson_correlation: float
    passed: bool


@dataclass(frozen=True)
class ReliabilityCacheSample:
    """Semantically named Paper 2 view of one legacy NPZ sample."""

    cache_input_6ch: np.ndarray
    predictor_input: np.ndarray
    reconstruction_magnitude_normalized: np.ndarray
    zero_filled_magnitude_normalized: np.ndarray
    intervention_magnitude_normalized: np.ndarray
    support_mask: np.ndarray
    analytical_psf: np.ndarray
    psf_gain_descriptor: np.ndarray
    u_risk: np.ndarray
    z_risk: np.ndarray
    transform_diagnostics: TargetTransformDiagnostics


def _require_numeric_finite(
    name: str,
    array: np.ndarray,
) -> None:
    """Require a non-object, finite numerical array."""
    if array.dtype.hasobject:
        raise TypeError(
            f"{name} must not contain Python objects."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(
            f"{name} must be numerical, got {array.dtype}."
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )


def reconstruct_u_risk(
    z_risk: np.ndarray,
    alpha_log: float = ALPHA_LOG,
    quantile: float = TARGET_QUANTILE,
    epsilon_num: float = EPSILON_NUM,
) -> tuple[np.ndarray, float]:
    """Reconstruct the stored nonlinear risk-learning target."""
    z_risk = np.asarray(
        z_risk,
        dtype=np.float64,
    )

    target_quantile = float(
        np.quantile(
            z_risk,
            quantile,
        )
    )

    reconstructed = np.log1p(
        float(alpha_log)
        * z_risk
        /
        (
            target_quantile
            + float(
                epsilon_num
            )
        )
    )

    return (
        reconstructed.astype(
            np.float32
        ),
        target_quantile,
    )


def evaluate_target_transform(
    u_risk: np.ndarray,
    z_risk: np.ndarray,
    maximum_error_tolerance: float = 0.05,
    percentile_99_error_tolerance: float = 0.02,
    correlation_tolerance: float = 0.999,
) -> TargetTransformDiagnostics:
    """
    Compare the stored float16 target with its reconstructed form.

    The tolerances account for:
    - float32 computation before storage;
    - independent float16 quantization of u_risk and z_risk;
    - recomputation of Q_0.99 from quantized z_risk.
    """
    u_risk = np.asarray(
        u_risk,
        dtype=np.float32,
    )

    reconstructed, target_quantile = reconstruct_u_risk(
        z_risk
    )

    absolute_error = np.abs(
        u_risk
        - reconstructed
    )

    maximum_error = float(
        absolute_error.max()
    )

    percentile_99_error = float(
        np.quantile(
            absolute_error,
            0.99,
        )
    )

    mean_error = float(
        absolute_error.mean()
    )

    first = u_risk.reshape(
        -1
    ).astype(
        np.float64
    )

    second = reconstructed.reshape(
        -1
    ).astype(
        np.float64
    )

    if (
        np.std(first) == 0.0
        or np.std(second) == 0.0
    ):
        correlation = (
            1.0
            if np.allclose(
                first,
                second,
            )
            else 0.0
        )

    else:
        correlation = float(
            np.corrcoef(
                first,
                second,
            )[0, 1]
        )

    passed = bool(
        maximum_error
        <= maximum_error_tolerance
        and percentile_99_error
        <= percentile_99_error_tolerance
        and correlation
        >= correlation_tolerance
    )

    return TargetTransformDiagnostics(
        reconstructed_quantile=target_quantile,
        maximum_absolute_error=maximum_error,
        percentile_99_absolute_error=percentile_99_error,
        mean_absolute_error=mean_error,
        pearson_correlation=correlation,
        passed=passed,
    )


def load_reliability_cache_sample(
    path: Path | str,
    verify_transform: bool = True,
) -> ReliabilityCacheSample:
    """Load one NPZ archive and translate legacy fields."""
    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Cache sample does not exist: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        observed_keys = tuple(
            sorted(
                archive.files
            )
        )

        expected_keys = tuple(
            sorted(
                LEGACY_STORAGE_KEYS
            )
        )

        if observed_keys != expected_keys:
            raise KeyError(
                "Unexpected cache key set. "
                f"Expected {expected_keys}, "
                f"observed {observed_keys}."
            )

        cache_input_6ch = np.asarray(
            archive["x"],
            dtype=np.float32,
        )

        u_risk = np.asarray(
            archive["y"],
            dtype=np.float32,
        )

        z_risk = np.asarray(
            archive["y_raw"],
            dtype=np.float32,
        )

    _require_numeric_finite(
        "cache_input_6ch",
        cache_input_6ch,
    )

    _require_numeric_finite(
        "u_risk",
        u_risk,
    )

    _require_numeric_finite(
        "z_risk",
        z_risk,
    )

    if cache_input_6ch.ndim != 3:
        raise ValueError(
            "cache_input_6ch must have shape (6,H,W), "
            f"got {cache_input_6ch.shape}."
        )

    if cache_input_6ch.shape[0] != 6:
        raise ValueError(
            "cache_input_6ch must contain six channels, "
            f"got {cache_input_6ch.shape[0]}."
        )

    spatial_shape = (
        cache_input_6ch.shape[-2:]
    )

    if u_risk.shape != spatial_shape:
        raise ValueError(
            "u_risk spatial shape does not match the "
            f"cache input: {u_risk.shape} vs {spatial_shape}."
        )

    if z_risk.shape != spatial_shape:
        raise ValueError(
            "z_risk spatial shape does not match the "
            f"cache input: {z_risk.shape} vs {spatial_shape}."
        )

    support_mask = (
        cache_input_6ch[3]
    )

    if (
        float(
            support_mask.min()
        )
        < -1.0e-5
        or float(
            support_mask.max()
        )
        > 1.0 + 1.0e-5
    ):
        raise ValueError(
            "support_mask must remain within [0,1]."
        )

    predictor_input = np.ascontiguousarray(
        cache_input_6ch[
            list(
                A4_CHANNEL_INDICES
            )
        ]
    )

    diagnostics = evaluate_target_transform(
        u_risk=u_risk,
        z_risk=z_risk,
    )

    if (
        verify_transform
        and not diagnostics.passed
    ):
        raise ValueError(
            "The stored u_risk target does not satisfy "
            "the frozen nonlinear target transformation. "
            f"Diagnostics: {diagnostics}"
        )

    return ReliabilityCacheSample(
        cache_input_6ch=np.ascontiguousarray(
            cache_input_6ch
        ),

        predictor_input=predictor_input,

        reconstruction_magnitude_normalized=np.ascontiguousarray(
            cache_input_6ch[0]
        ),

        zero_filled_magnitude_normalized=np.ascontiguousarray(
            cache_input_6ch[1]
        ),

        intervention_magnitude_normalized=np.ascontiguousarray(
            cache_input_6ch[2]
        ),

        support_mask=np.ascontiguousarray(
            support_mask
        ),

        analytical_psf=np.ascontiguousarray(
            cache_input_6ch[4]
        ),

        psf_gain_descriptor=np.ascontiguousarray(
            cache_input_6ch[5]
        ),

        u_risk=np.ascontiguousarray(
            u_risk
        ),

        z_risk=np.ascontiguousarray(
            z_risk
        ),

        transform_diagnostics=diagnostics,
    )
