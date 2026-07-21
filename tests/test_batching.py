"""Tests for tensor conversion and shape-safe batching."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from paper2_uq_mri.batching import (
    BatchingContractError,
    PredictorTensorSample,
    ShapeBucketBatchSampler,
    collate_predictor_batch,
    tensorize_semantic_sample,
)


def make_sample(
    *,
    height: int,
    width: int,
    sample_id: str = "sample",
    volume_id: str = "volume",
) -> PredictorTensorSample:
    C_v = torch.zeros(
        (
            3,
            height,
            width,
        ),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (
            1,
            height,
            width,
        ),
        dtype=torch.float32,
    )

    return PredictorTensorSample(
        sample_id=sample_id,
        volume_id=volume_id,
        paper2_role="D_fit",
        purpose="gradient_fit",
        C_v=C_v,
        u_risk_v=target,
        cache_shape_6ch=(
            6,
            height,
            width,
        ),
        spatial_shape=(
            height,
            width,
        ),
    )


def test_tensorizer_uses_only_first_three_channels() -> None:
    cache = np.arange(
        6 * 8 * 7,
        dtype=np.float16,
    ).reshape(
        6,
        8,
        7,
    )

    semantic = SimpleNamespace(
        cache_input_6ch=cache,
        predictor_input=cache[
            :3
        ].copy(),
        u_risk=np.ones(
            (
                8,
                7,
            ),
            dtype=np.float16,
        ),
    )

    sample = tensorize_semantic_sample(
        semantic,
        sample_id="sample",
        volume_id="volume",
        paper2_role="D_fit",
        purpose="gradient_fit",
    )

    assert sample.C_v.shape == (
        3,
        8,
        7,
    )

    assert sample.u_risk_v.shape == (
        1,
        8,
        7,
    )

    assert sample.C_v.dtype == torch.float32
    assert sample.u_risk_v.dtype == torch.float32

    assert torch.equal(
        sample.C_v,
        torch.from_numpy(
            cache[
                :3
            ].astype(
                np.float32
            )
        ),
    )


def test_tensorizer_rejects_modified_predictor_input() -> None:
    cache = np.zeros(
        (
            6,
            5,
            4,
        ),
        dtype=np.float16,
    )

    predictor = cache[
        :3
    ].copy()

    predictor[
        0,
        0,
        0,
    ] = 1

    semantic = SimpleNamespace(
        cache_input_6ch=cache,
        predictor_input=predictor,
        u_risk=np.zeros(
            (
                5,
                4,
            ),
            dtype=np.float16,
        ),
    )

    with pytest.raises(
        BatchingContractError,
        match="not exactly",
    ):
        tensorize_semantic_sample(
            semantic,
            sample_id="sample",
            volume_id="volume",
            paper2_role="D_fit",
            purpose="gradient_fit",
        )


def test_collate_stacks_homogeneous_shapes() -> None:
    batch = collate_predictor_batch(
        [
            make_sample(
                height=8,
                width=7,
                sample_id="a",
            ),
            make_sample(
                height=8,
                width=7,
                sample_id="b",
            ),
        ]
    )

    assert batch[
        "C_v"
    ].shape == (
        2,
        3,
        8,
        7,
    )

    assert batch[
        "u_risk_v"
    ].shape == (
        2,
        1,
        8,
        7,
    )

    assert batch[
        "predictor_channels"
    ] == 3

    assert batch[
        "source_cache_channels"
    ] == 6


def test_collate_rejects_mixed_shapes() -> None:
    with pytest.raises(
        BatchingContractError,
        match="cannot mix",
    ):
        collate_predictor_batch(
            [
                make_sample(
                    height=8,
                    width=7,
                ),
                make_sample(
                    height=9,
                    width=7,
                ),
            ]
        )


def test_shape_bucket_sampler_is_homogeneous() -> None:
    rows = pd.DataFrame(
        {
            "height": [
                8,
                8,
                9,
                9,
                9,
            ],
            "width": [
                7,
                7,
                6,
                6,
                6,
            ],
        }
    )

    sampler = ShapeBucketBatchSampler(
        rows,
        batch_size=2,
        seed=20260720,
        shuffle=True,
    )

    observed_indices = []

    for batch_indices in sampler:
        observed_indices.extend(
            batch_indices
        )

        shapes = {
            (
                int(
                    rows.iloc[
                        index
                    ][
                        "height"
                    ]
                ),
                int(
                    rows.iloc[
                        index
                    ][
                        "width"
                    ]
                ),
            )
            for index in batch_indices
        }

        assert len(
            shapes
        ) == 1

    assert sorted(
        observed_indices
    ) == list(
        range(
            len(
                rows
            )
        )
    )


def test_shape_bucket_sampler_is_reproducible() -> None:
    rows = pd.DataFrame(
        {
            "height": [
                8,
                8,
                8,
                9,
                9,
            ],
            "width": [
                7,
                7,
                7,
                6,
                6,
            ],
        }
    )

    first = ShapeBucketBatchSampler(
        rows,
        batch_size=2,
        seed=20260720,
        shuffle=True,
    )

    second = ShapeBucketBatchSampler(
        rows,
        batch_size=2,
        seed=20260720,
        shuffle=True,
    )

    assert list(
        first
    ) == list(
        second
    )


def test_one_batch_per_shape_limit() -> None:
    rows = pd.DataFrame(
        {
            "height": [
                8,
                8,
                8,
                8,
                9,
                9,
                9,
            ],
            "width": [
                7,
                7,
                7,
                7,
                6,
                6,
                6,
            ],
        }
    )

    sampler = ShapeBucketBatchSampler(
        rows,
        batch_size=2,
        seed=20260720,
        shuffle=False,
        max_batches_per_shape=1,
    )

    batches = list(
        sampler
    )

    assert len(
        batches
    ) == 2

    assert all(
        len(
            batch
        )
        == 2
        for batch in batches
    )



def test_confirmed_semantic_u_risk_mapping() -> None:
    """Map semantic u_risk to scientific tensor u_risk_v."""
    cache = np.zeros(
        (
            6,
            5,
            4,
        ),
        dtype=np.float16,
    )

    semantic = SimpleNamespace(
        cache_input_6ch=cache,
        predictor_input=cache[
            :3
        ].copy(),
        u_risk=np.ones(
            (
                5,
                4,
            ),
            dtype=np.float16,
        ),
    )

    sample = tensorize_semantic_sample(
        semantic,
        sample_id="sample",
        volume_id="volume",
        paper2_role="D_fit",
        purpose="gradient_fit",
    )

    assert sample.C_v.shape == (
        3,
        5,
        4,
    )

    assert sample.u_risk_v.shape == (
        1,
        5,
        4,
    )

    assert torch.all(
        sample.u_risk_v == 1
    )
