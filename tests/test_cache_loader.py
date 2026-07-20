"""Tests for semantic loading of legacy Paper 1 NPZ files."""

from pathlib import Path

import numpy as np

from paper2_uq_mri.cache import (
    ALPHA_LOG,
    EPSILON_NUM,
    load_reliability_cache_sample,
)


def test_loader_translates_legacy_fields(
    tmp_path: Path,
) -> None:
    height = 24
    width = 20

    generator = np.random.default_rng(
        20260720
    )

    cache_input = generator.random(
        (
            6,
            height,
            width,
        ),
        dtype=np.float32,
    )

    cache_input[
        3
    ] = np.clip(
        cache_input[
            3
        ],
        0.0,
        1.0,
    )

    z_risk = generator.random(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    q99 = np.quantile(
        z_risk,
        0.99,
    )

    u_risk = np.log1p(
        ALPHA_LOG
        * z_risk
        /
        (
            q99
            + EPSILON_NUM
        )
    )

    path = (
        tmp_path
        / "sample.npz"
    )

    np.savez_compressed(
        path,
        x=cache_input.astype(
            np.float16
        ),
        y=u_risk.astype(
            np.float16
        ),
        y_raw=z_risk.astype(
            np.float16
        ),
    )

    sample = load_reliability_cache_sample(
        path
    )

    assert sample.cache_input_6ch.shape == (
        6,
        height,
        width,
    )

    assert sample.predictor_input.shape == (
        3,
        height,
        width,
    )

    np.testing.assert_array_equal(
        sample.predictor_input,
        sample.cache_input_6ch[
            :3
        ],
    )

    np.testing.assert_array_equal(
        sample.support_mask,
        sample.cache_input_6ch[
            3
        ],
    )

    np.testing.assert_array_equal(
        sample.analytical_psf,
        sample.cache_input_6ch[
            4
        ],
    )

    np.testing.assert_array_equal(
        sample.psf_gain_descriptor,
        sample.cache_input_6ch[
            5
        ],
    )

    assert (
        sample.transform_diagnostics.passed
        is True
    )


def test_loader_rejects_unexpected_key(
    tmp_path: Path,
) -> None:
    array = np.zeros(
        (
            8,
            8,
        ),
        dtype=np.float16,
    )

    path = (
        tmp_path
        / "bad.npz"
    )

    np.savez_compressed(
        path,
        x=np.zeros(
            (
                6,
                8,
                8,
            ),
            dtype=np.float16,
        ),
        y=array,
        y_raw=array,
        extra=array,
    )

    try:
        load_reliability_cache_sample(
            path
        )

    except KeyError:
        pass

    else:
        raise AssertionError(
            "Unexpected NPZ keys were accepted."
        )
