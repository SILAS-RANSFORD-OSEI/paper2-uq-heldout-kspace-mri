"""Tensor conversion and shape-safe minibatching.

The trainable Paper 2 predictors receive only the three-channel
predictor input:

    C_v = [
        |x_hat_v|,
        |x_0,v|,
        |x_hat_v - x_0,v|,
    ]

The stored six-channel cache tensor is never passed directly
to the predictor.

Heterogeneous matrix sizes are handled by grouping samples into
homogeneous spatial-shape buckets before stacking.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
)

from paper2_uq_mri.split_access import (
    FinalTestBarrier,
    RoleAwareReliabilityDataset,
)


class BatchingContractError(ValueError):
    """Raised when tensor or minibatch contracts are violated."""


@dataclass(frozen=True)
class PredictorTensorSample:
    """One tensorized reliability-prediction sample."""

    sample_id: str
    volume_id: str
    paper2_role: str
    purpose: str
    C_v: torch.Tensor
    u_risk_v: torch.Tensor
    cache_shape_6ch: tuple[int, int, int]
    spatial_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.C_v.ndim != 3:
            raise BatchingContractError(
                "C_v must have shape (3, H, W)."
            )

        if self.C_v.shape[0] != 3:
            raise BatchingContractError(
                "C_v must contain exactly three channels."
            )

        if self.u_risk_v.ndim != 3:
            raise BatchingContractError(
                "u_risk_v must have shape (1, H, W)."
            )

        if self.u_risk_v.shape[0] != 1:
            raise BatchingContractError(
                "u_risk_v must contain one target channel."
            )

        predictor_spatial_shape = tuple(
            int(value)
            for value in self.C_v.shape[-2:]
        )

        target_spatial_shape = tuple(
            int(value)
            for value in self.u_risk_v.shape[-2:]
        )

        if predictor_spatial_shape != target_spatial_shape:
            raise BatchingContractError(
                "C_v and u_risk_v spatial shapes do not match."
            )

        if predictor_spatial_shape != self.spatial_shape:
            raise BatchingContractError(
                "Recorded spatial shape does not match C_v."
            )

        if self.cache_shape_6ch[0] != 6:
            raise BatchingContractError(
                "The source cache tensor must contain six channels."
            )

        if not torch.isfinite(
            self.C_v
        ).all():
            raise BatchingContractError(
                "C_v contains non-finite values."
            )

        if not torch.isfinite(
            self.u_risk_v
        ).all():
            raise BatchingContractError(
                "u_risk_v contains non-finite values."
            )


def _as_numpy_array(
    value: Any,
    *,
    name: str,
) -> np.ndarray:
    """Convert a semantic cache field into a NumPy array."""
    array = np.asarray(
        value
    )

    if array.dtype == object:
        raise BatchingContractError(
            f"{name} cannot use object dtype."
        )

    return array


def tensorize_semantic_sample(
    semantic_sample: Any,
    *,
    sample_id: str,
    volume_id: str,
    paper2_role: str,
    purpose: str,
) -> PredictorTensorSample:
    """Translate one semantic cache sample into model tensors."""
    if not hasattr(
        semantic_sample,
        "cache_input_6ch",
    ):
        raise BatchingContractError(
            "Semantic sample does not expose cache_input_6ch."
        )

    if not hasattr(
        semantic_sample,
        "predictor_input",
    ):
        raise BatchingContractError(
            "Semantic sample does not expose predictor_input."
        )

    if not hasattr(
                semantic_sample,
                "u_risk",
            ):
        raise BatchingContractError(
            "Semantic sample does not expose the frozen transformed target attribute u_risk."
        )

    cache_input_6ch = _as_numpy_array(
        semantic_sample.cache_input_6ch,
        name="cache_input_6ch",
    )

    predictor_input = _as_numpy_array(
        semantic_sample.predictor_input,
        name="predictor_input",
    )

    target = _as_numpy_array(
        semantic_sample.u_risk,
        name="u_risk_v",
    )

    if cache_input_6ch.ndim != 3:
        raise BatchingContractError(
            "cache_input_6ch must have shape (6, H, W)."
        )

    if cache_input_6ch.shape[0] != 6:
        raise BatchingContractError(
            "cache_input_6ch must contain six channels."
        )

    if predictor_input.ndim != 3:
        raise BatchingContractError(
            "predictor_input must have shape (3, H, W)."
        )

    if predictor_input.shape[0] != 3:
        raise BatchingContractError(
            "predictor_input must contain exactly three channels."
        )

    if predictor_input.shape != cache_input_6ch[
        :3
    ].shape:
        raise BatchingContractError(
            "predictor_input shape does not match cache channels 0-2."
        )

    if not np.array_equal(
        predictor_input,
        cache_input_6ch[
            :3
        ],
    ):
        raise BatchingContractError(
            "predictor_input is not exactly cache_input_6ch[0:3]."
        )

    if target.ndim == 2:
        target = target[
            None,
            ...,
        ]

    elif (
        target.ndim == 3
        and target.shape[0] == 1
    ):
        pass

    else:
        raise BatchingContractError(
            "u_risk_v must have shape (H, W) or (1, H, W)."
        )

    if tuple(
        predictor_input.shape[
            -2:
        ]
    ) != tuple(
        target.shape[
            -2:
        ]
    ):
        raise BatchingContractError(
            "Predictor and target spatial shapes do not match."
        )

    C_v = torch.from_numpy(
        np.ascontiguousarray(
            predictor_input,
            dtype=np.float32,
        )
    )

    u_risk_v = torch.from_numpy(
        np.ascontiguousarray(
            target,
            dtype=np.float32,
        )
    )

    spatial_shape = tuple(
        int(value)
        for value in C_v.shape[
            -2:
        ]
    )

    return PredictorTensorSample(
        sample_id=str(
            sample_id
        ),
        volume_id=str(
            volume_id
        ),
        paper2_role=str(
            paper2_role
        ),
        purpose=str(
            purpose
        ),
        C_v=C_v,
        u_risk_v=u_risk_v,
        cache_shape_6ch=tuple(
            int(value)
            for value in cache_input_6ch.shape
        ),
        spatial_shape=spatial_shape,
    )


class RoleAwareTensorDataset(Dataset):
    """Tensorized wrapper around RoleAwareReliabilityDataset."""

    def __init__(
        self,
        base_dataset: RoleAwareReliabilityDataset,
    ) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index: int,
    ) -> PredictorTensorSample:
        record = self.base_dataset[
            index
        ]

        return tensorize_semantic_sample(
            record[
                "semantic_sample"
            ],
            sample_id=record[
                "sample_id"
            ],
            volume_id=record[
                "volume_id"
            ],
            paper2_role=record[
                "paper2_role"
            ],
            purpose=record[
                "purpose"
            ],
        )

    @property
    def rows(self) -> pd.DataFrame:
        return self.base_dataset.rows

    @property
    def paper2_role(self) -> str:
        return self.base_dataset.paper2_role

    @property
    def purpose(self) -> str:
        return self.base_dataset.purpose


def _resolve_shape_columns(
    rows: pd.DataFrame,
) -> tuple[str, str]:
    """Resolve height and width metadata columns."""
    height_candidates = (
        "height",
        "cache_height",
        "matrix_height",
    )

    width_candidates = (
        "width",
        "cache_width",
        "matrix_width",
    )

    height_column = next(
        (
            column
            for column in height_candidates
            if column in rows.columns
        ),
        None,
    )

    width_column = next(
        (
            column
            for column in width_candidates
            if column in rows.columns
        ),
        None,
    )

    if height_column is None or width_column is None:
        raise BatchingContractError(
            "Rows must contain recognizable height and width columns."
        )

    return (
        height_column,
        width_column,
    )


class ShapeBucketBatchSampler(
    Sampler[list[int]]
):
    """Yield batches containing one spatial shape only."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool = False,
        max_batches_per_shape: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise BatchingContractError(
                "batch_size must be at least one."
            )

        if (
            max_batches_per_shape is not None
            and max_batches_per_shape < 1
        ):
            raise BatchingContractError(
                "max_batches_per_shape must be positive."
            )

        height_column, width_column = (
            _resolve_shape_columns(
                rows
            )
        )

        heights = pd.to_numeric(
            rows[
                height_column
            ],
            errors="raise",
        ).astype(int)

        widths = pd.to_numeric(
            rows[
                width_column
            ],
            errors="raise",
        ).astype(int)

        buckets: dict[
            tuple[int, int],
            list[int],
        ] = defaultdict(
            list
        )

        for index, (
            height,
            width,
        ) in enumerate(
            zip(
                heights,
                widths,
            )
        ):
            buckets[
                (
                    int(
                        height
                    ),
                    int(
                        width
                    ),
                )
            ].append(
                int(
                    index
                )
            )

        if not buckets:
            raise BatchingContractError(
                "No rows were available for shape bucketing."
            )

        self.batch_size = int(
            batch_size
        )

        self.seed = int(
            seed
        )

        self.shuffle = bool(
            shuffle
        )

        self.drop_last = bool(
            drop_last
        )

        self.max_batches_per_shape = (
            max_batches_per_shape
        )

        self._buckets = {
            shape:
                tuple(
                    indices
                )
            for shape, indices
            in sorted(
                buckets.items()
            )
        }

    @property
    def shape_counts(
        self,
    ) -> Mapping[
        tuple[int, int],
        int,
    ]:
        return {
            shape:
                len(
                    indices
                )
            for shape, indices
            in self._buckets.items()
        }

    def __iter__(
        self,
    ) -> Iterator[
        list[int]
    ]:
        rng = np.random.default_rng(
            self.seed
        )

        shapes = list(
            self._buckets.keys()
        )

        if self.shuffle:
            rng.shuffle(
                shapes
            )

        for shape in shapes:
            indices = np.asarray(
                self._buckets[
                    shape
                ],
                dtype=int,
            ).copy()

            if self.shuffle:
                rng.shuffle(
                    indices
                )

            emitted_for_shape = 0

            for start in range(
                0,
                len(
                    indices
                ),
                self.batch_size,
            ):
                batch = indices[
                    start:
                    start
                    + self.batch_size
                ].tolist()

                if (
                    self.drop_last
                    and len(
                        batch
                    )
                    < self.batch_size
                ):
                    continue

                yield [
                    int(
                        index
                    )
                    for index in batch
                ]

                emitted_for_shape += 1

                if (
                    self.max_batches_per_shape
                    is not None
                    and emitted_for_shape
                    >= self.max_batches_per_shape
                ):
                    break

    def __len__(
        self,
    ) -> int:
        total = 0

        for count in self.shape_counts.values():
            if self.drop_last:
                batches = (
                    count
                    // self.batch_size
                )

            else:
                batches = (
                    count
                    + self.batch_size
                    - 1
                ) // self.batch_size

            if self.max_batches_per_shape is not None:
                batches = min(
                    batches,
                    self.max_batches_per_shape,
                )

            total += batches

        return total


def collate_predictor_batch(
    samples: Sequence[
        PredictorTensorSample
    ],
) -> dict[str, Any]:
    """Stack one homogeneous-shape predictor minibatch."""
    if not samples:
        raise BatchingContractError(
            "Cannot collate an empty minibatch."
        )

    spatial_shapes = {
        sample.spatial_shape
        for sample in samples
    }

    if len(
        spatial_shapes
    ) != 1:
        raise BatchingContractError(
            "A minibatch cannot mix spatial matrix sizes."
        )

    roles = {
        sample.paper2_role
        for sample in samples
    }

    if len(
        roles
    ) != 1:
        raise BatchingContractError(
            "A minibatch cannot mix Paper 2 data roles."
        )

    purposes = {
        sample.purpose
        for sample in samples
    }

    if len(
        purposes
    ) != 1:
        raise BatchingContractError(
            "A minibatch cannot mix access purposes."
        )

    C_v = torch.stack(
        [
            sample.C_v
            for sample in samples
        ],
        dim=0,
    )

    u_risk_v = torch.stack(
        [
            sample.u_risk_v
            for sample in samples
        ],
        dim=0,
    )

    if C_v.ndim != 4:
        raise BatchingContractError(
            "Batched C_v must have shape (B, 3, H, W)."
        )

    if C_v.shape[1] != 3:
        raise BatchingContractError(
            "Batched C_v must contain exactly three channels."
        )

    if u_risk_v.ndim != 4:
        raise BatchingContractError(
            "Batched u_risk_v must have shape (B, 1, H, W)."
        )

    if u_risk_v.shape[1] != 1:
        raise BatchingContractError(
            "Batched target must contain one channel."
        )

    if tuple(
        C_v.shape[
            -2:
        ]
    ) != tuple(
        u_risk_v.shape[
            -2:
        ]
    ):
        raise BatchingContractError(
            "Batched predictor and target shapes do not match."
        )

    return {
        "sample_id":
            tuple(
                sample.sample_id
                for sample in samples
            ),

        "volume_id":
            tuple(
                sample.volume_id
                for sample in samples
            ),

        "paper2_role":
            next(
                iter(
                    roles
                )
            ),

        "purpose":
            next(
                iter(
                    purposes
                )
            ),

        "spatial_shape":
            next(
                iter(
                    spatial_shapes
                )
            ),

        "C_v":
            C_v,

        "u_risk_v":
            u_risk_v,

        "source_cache_channels":
            6,

        "predictor_channels":
            3,
    }


def build_shape_safe_loader(
    *,
    cache_manifest_path: str,
    split_path: str,
    purpose: str,
    barrier: FinalTestBarrier,
    batch_size: int,
    seed: int,
    shuffle: bool,
    drop_last: bool = False,
    max_batches_per_shape: int | None = None,
    verify_transform: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a lazy, role-aware, shape-homogeneous loader."""
    base_dataset = RoleAwareReliabilityDataset.from_files(
        cache_manifest_path=cache_manifest_path,
        split_path=split_path,
        purpose=purpose,
        barrier=barrier,
        verify_transform=verify_transform,
        require_existing_paths=True,
    )

    tensor_dataset = RoleAwareTensorDataset(
        base_dataset
    )

    batch_sampler = ShapeBucketBatchSampler(
        tensor_dataset.rows,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
        max_batches_per_shape=max_batches_per_shape,
    )

    return DataLoader(
        tensor_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_predictor_batch,
        num_workers=num_workers,
        pin_memory=False,
    )
