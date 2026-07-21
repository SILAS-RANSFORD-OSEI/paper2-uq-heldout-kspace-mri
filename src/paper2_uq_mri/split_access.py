"""Role-aware cache indexing and leakage barriers.

This module enforces the frozen Paper 2 data-role contract:

- gradient_fit    -> D_fit
- model_selection -> D_dev
- calibration     -> D_cal
- final_evaluation-> D_test

D_test access is denied while the final-test barrier is CLOSED.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import pandas as pd


ROLE_D_FIT: Final[str] = "D_fit"
ROLE_D_DEV: Final[str] = "D_dev"
ROLE_D_CAL: Final[str] = "D_cal"
ROLE_D_TEST: Final[str] = "D_test"

VALID_ROLES: Final[tuple[str, ...]] = (
    ROLE_D_FIT,
    ROLE_D_DEV,
    ROLE_D_CAL,
    ROLE_D_TEST,
)

PURPOSE_GRADIENT_FIT: Final[str] = "gradient_fit"
PURPOSE_MODEL_SELECTION: Final[str] = "model_selection"
PURPOSE_CALIBRATION: Final[str] = "calibration"
PURPOSE_FINAL_EVALUATION: Final[str] = "final_evaluation"

VALID_PURPOSES: Final[tuple[str, ...]] = (
    PURPOSE_GRADIENT_FIT,
    PURPOSE_MODEL_SELECTION,
    PURPOSE_CALIBRATION,
    PURPOSE_FINAL_EVALUATION,
)

PURPOSE_TO_ROLE: Final[Mapping[str, str]] = {
    PURPOSE_GRADIENT_FIT:
        ROLE_D_FIT,

    PURPOSE_MODEL_SELECTION:
        ROLE_D_DEV,

    PURPOSE_CALIBRATION:
        ROLE_D_CAL,

    PURPOSE_FINAL_EVALUATION:
        ROLE_D_TEST,
}

BARRIER_CLOSED: Final[str] = "CLOSED"
BARRIER_OPEN: Final[str] = "OPEN"

REQUIRED_SPLIT_COLUMNS: Final[tuple[str, ...]] = (
    "volume_id",
    "paper1_split",
    "paper2_role",
    "assignment_seed",
    "assignment_algorithm",
    "locked",
    "test_designation",
)

PATH_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "_resolved_absolute_path",
    "resolved_absolute_path",
    "cache_path",
    "filepath",
    "path",
)


class SplitAccessError(RuntimeError):
    """Raised when a split role is accessed for a forbidden purpose."""


class SplitContractError(ValueError):
    """Raised when a split or cache manifest violates the contract."""


@dataclass(frozen=True)
class FinalTestBarrier:
    """Explicit state of the final-test access barrier."""

    status: str = BARRIER_CLOSED

    def __post_init__(self) -> None:
        normalized = str(
            self.status
        ).strip().upper()

        if normalized not in {
            BARRIER_CLOSED,
            BARRIER_OPEN,
        }:
            raise SplitContractError(
                "Final-test barrier status must be "
                "'CLOSED' or 'OPEN'."
            )

        object.__setattr__(
            self,
            "status",
            normalized,
        )

    @property
    def is_open(self) -> bool:
        return self.status == BARRIER_OPEN


def _normalize_boolean(
    value: Any,
) -> bool:
    """Parse a strict boolean stored in a CSV column."""
    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise SplitContractError(
        f"Cannot interpret boolean value: {value!r}"
    )


def _resolve_path_column(
    columns: Sequence[str],
) -> str:
    """Resolve the absolute cache-path column."""
    for candidate in PATH_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate

    raise SplitContractError(
        "No recognized cache-path column was found. "
        f"Expected one of {PATH_COLUMN_CANDIDATES}."
    )


def load_locked_split(
    split_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the frozen 281-volume Paper 2 split."""
    path = Path(
        split_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    split = pd.read_csv(
        path,
        dtype={
            "volume_id":
                str,

            "paper1_split":
                str,

            "paper2_role":
                str,

            "assignment_algorithm":
                str,

            "test_designation":
                str,
        },
    )

    missing_columns = sorted(
        set(
            REQUIRED_SPLIT_COLUMNS
        )
        - set(
            split.columns
        )
    )

    if missing_columns:
        raise SplitContractError(
            "Split file is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    split = split.copy()

    split[
        "volume_id"
    ] = (
        split[
            "volume_id"
        ]
        .astype(str)
        .str.strip()
    )

    split[
        "paper1_split"
    ] = (
        split[
            "paper1_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    split[
        "paper2_role"
    ] = (
        split[
            "paper2_role"
        ]
        .astype(str)
        .str.strip()
    )

    if len(
        split
    ) != 281:
        raise SplitContractError(
            "The frozen split must contain exactly "
            "281 volume rows."
        )

    if (
        split[
            "volume_id"
        ].nunique()
        != 281
    ):
        raise SplitContractError(
            "The frozen split contains duplicate volume IDs."
        )

    role_counts = (
        split[
            "paper2_role"
        ]
        .value_counts()
        .to_dict()
    )

    expected_counts = {
        ROLE_D_FIT:
            181,

        ROLE_D_DEV:
            20,

        ROLE_D_CAL:
            40,

        ROLE_D_TEST:
            40,
    }

    if role_counts != expected_counts:
        raise SplitContractError(
            "Unexpected Paper 2 role counts: "
            f"{role_counts}"
        )

    if not split[
        "locked"
    ].map(
        _normalize_boolean
    ).all():
        raise SplitContractError(
            "Every frozen split row must have locked=True."
        )

    fit_dev_sources = set(
        split.loc[
            split[
                "paper2_role"
            ].isin(
                [
                    ROLE_D_FIT,
                    ROLE_D_DEV,
                ]
            ),
            "paper1_split",
        ]
    )

    if fit_dev_sources != {
        "train",
    }:
        raise SplitContractError(
            "D_fit and D_dev must originate only from "
            "the Paper 1 training split."
        )

    calibration_sources = set(
        split.loc[
            split[
                "paper2_role"
            ]
            == ROLE_D_CAL,
            "paper1_split",
        ]
    )

    if calibration_sources != {
        "calibration",
    }:
        raise SplitContractError(
            "D_cal must originate only from the "
            "Paper 1 calibration split."
        )

    test_sources = set(
        split.loc[
            split[
                "paper2_role"
            ]
            == ROLE_D_TEST,
            "paper1_split",
        ]
    )

    if test_sources != {
        "test",
    }:
        raise SplitContractError(
            "D_test must originate only from the "
            "Paper 1 test split."
        )

    test_designations = set(
        split.loc[
            split[
                "paper2_role"
            ]
            == ROLE_D_TEST,
            "test_designation",
        ]
        .astype(str)
        .str.strip()
    )

    if test_designations != {
        "locked_reused_evaluation_cohort",
    }:
        raise SplitContractError(
            "D_test must retain its locked reused "
            "evaluation-cohort designation."
        )

    return split.sort_values(
        "volume_id"
    ).reset_index(
        drop=True
    )


def build_role_aware_index(
    cache_manifest_path: str | Path,
    split_path: str | Path,
    *,
    require_existing_paths: bool = True,
) -> pd.DataFrame:
    """Attach one frozen Paper 2 role to every cache row."""
    manifest_path = Path(
        cache_manifest_path
    ).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cache manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    required_manifest_columns = {
        "sample_id",
        "volume_id",
    }

    missing_columns = sorted(
        required_manifest_columns
        - set(
            manifest.columns
        )
    )

    if missing_columns:
        raise SplitContractError(
            "Cache manifest is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    path_column = _resolve_path_column(
        list(
            manifest.columns
        )
    )

    manifest = manifest.copy()

    manifest[
        "sample_id"
    ] = (
        manifest[
            "sample_id"
        ]
        .astype(str)
        .str.strip()
    )

    manifest[
        "volume_id"
    ] = (
        manifest[
            "volume_id"
        ]
        .astype(str)
        .str.strip()
    )

    if (
        manifest[
            "sample_id"
        ].duplicated().any()
    ):
        raise SplitContractError(
            "Cache manifest contains duplicate sample IDs."
        )

    split = load_locked_split(
        split_path
    )

    role_columns = split[
        [
            "volume_id",
            "paper1_split",
            "paper2_role",
            "assignment_seed",
            "assignment_algorithm",
            "locked",
            "test_designation",
        ]
    ]

    index = manifest.merge(
        role_columns,
        on="volume_id",
        how="left",
        validate="many_to_one",
        suffixes=(
            "",
            "_frozen",
        ),
    )

    if index[
        "paper2_role"
    ].isna().any():
        missing_volume_ids = sorted(
            index.loc[
                index[
                    "paper2_role"
                ].isna(),
                "volume_id",
            ].unique().tolist()
        )

        raise SplitContractError(
            "Cache rows have no frozen Paper 2 role: "
            + ", ".join(
                missing_volume_ids[
                    :10
                ]
            )
        )

    manifest_volume_ids = set(
        index[
            "volume_id"
        ]
    )

    split_volume_ids = set(
        split[
            "volume_id"
        ]
    )

    if manifest_volume_ids != split_volume_ids:
        raise SplitContractError(
            "Cache-manifest and frozen-split volume "
            "populations do not match."
        )

    index[
        "cache_path"
    ] = (
        index[
            path_column
        ]
        .astype(str)
        .map(
            lambda value:
                str(
                    Path(
                        value
                    ).resolve()
                )
        )
    )

    if require_existing_paths:
        missing_paths = [
            path
            for path in index[
                "cache_path"
            ]
            if not Path(
                path
            ).is_file()
        ]

        if missing_paths:
            raise SplitContractError(
                f"{len(missing_paths)} cache paths "
                "do not exist."
            )

    return index.sort_values(
        [
            "paper2_role",
            "volume_id",
            "sample_id",
        ]
    ).reset_index(
        drop=True
    )


def assert_role_access(
    *,
    paper2_role: str,
    purpose: str,
    barrier: FinalTestBarrier,
) -> None:
    """Enforce the one-purpose/one-role access contract."""
    normalized_role = str(
        paper2_role
    ).strip()

    normalized_purpose = str(
        purpose
    ).strip().lower()

    if normalized_role not in VALID_ROLES:
        raise SplitContractError(
            f"Unknown Paper 2 role: {normalized_role}"
        )

    if normalized_purpose not in VALID_PURPOSES:
        raise SplitContractError(
            f"Unknown access purpose: {normalized_purpose}"
        )

    required_role = PURPOSE_TO_ROLE[
        normalized_purpose
    ]

    if normalized_role != required_role:
        raise SplitAccessError(
            f"Purpose {normalized_purpose!r} permits only "
            f"{required_role}; received {normalized_role}."
        )

    if (
        normalized_role
        == ROLE_D_TEST
        and not barrier.is_open
    ):
        raise SplitAccessError(
            "D_test access denied because the final-test "
            "barrier is CLOSED."
        )


def select_authorized_rows(
    role_aware_index: pd.DataFrame,
    *,
    purpose: str,
    barrier: FinalTestBarrier | None = None,
) -> pd.DataFrame:
    """Select exactly the rows authorized for one purpose."""
    resolved_barrier = (
        barrier
        if barrier is not None
        else FinalTestBarrier()
    )

    normalized_purpose = str(
        purpose
    ).strip().lower()

    if normalized_purpose not in VALID_PURPOSES:
        raise SplitContractError(
            f"Unknown access purpose: {normalized_purpose}"
        )

    required_role = PURPOSE_TO_ROLE[
        normalized_purpose
    ]

    assert_role_access(
        paper2_role=required_role,
        purpose=normalized_purpose,
        barrier=resolved_barrier,
    )

    selected = role_aware_index.loc[
        role_aware_index[
            "paper2_role"
        ]
        == required_role
    ].copy()

    if selected.empty:
        raise SplitContractError(
            f"No rows are available for {required_role}."
        )

    if set(
        selected[
            "paper2_role"
        ]
    ) != {
        required_role,
    }:
        raise SplitContractError(
            "Authorized selection contains another role."
        )

    return selected.reset_index(
        drop=True
    )


class RoleAwareReliabilityDataset:
    """Lazy semantic cache dataset protected by split access."""

    def __init__(
        self,
        role_aware_index: pd.DataFrame,
        *,
        purpose: str,
        barrier: FinalTestBarrier | None = None,
        verify_transform: bool = False,
    ) -> None:
        self.purpose = str(
            purpose
        ).strip().lower()

        self.barrier = (
            barrier
            if barrier is not None
            else FinalTestBarrier()
        )

        self.verify_transform = bool(
            verify_transform
        )

        self.rows = select_authorized_rows(
            role_aware_index,
            purpose=self.purpose,
            barrier=self.barrier,
        )

        self.paper2_role = PURPOSE_TO_ROLE[
            self.purpose
        ]

    @classmethod
    def from_files(
        cls,
        *,
        cache_manifest_path: str | Path,
        split_path: str | Path,
        purpose: str,
        barrier: FinalTestBarrier | None = None,
        verify_transform: bool = False,
        require_existing_paths: bool = True,
    ) -> "RoleAwareReliabilityDataset":
        index = build_role_aware_index(
            cache_manifest_path,
            split_path,
            require_existing_paths=require_existing_paths,
        )

        return cls(
            index,
            purpose=purpose,
            barrier=barrier,
            verify_transform=verify_transform,
        )

    def __len__(self) -> int:
        return int(
            len(
                self.rows
            )
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        if not isinstance(
            index,
            int,
        ):
            raise TypeError(
                "Dataset index must be an integer."
            )

        if index < 0:
            index += len(
                self
            )

        if (
            index < 0
            or index >= len(
                self
            )
        ):
            raise IndexError(
                "Dataset index is out of range."
            )

        row = self.rows.iloc[
            index
        ]

        assert_role_access(
            paper2_role=str(
                row[
                    "paper2_role"
                ]
            ),
            purpose=self.purpose,
            barrier=self.barrier,
        )

        from paper2_uq_mri.cache import (
            load_reliability_cache_sample,
        )

        semantic_sample = (
            load_reliability_cache_sample(
                Path(
                    row[
                        "cache_path"
                    ]
                ),
                verify_transform=self.verify_transform,
            )
        )

        return {
            "sample_id":
                str(
                    row[
                        "sample_id"
                    ]
                ),

            "volume_id":
                str(
                    row[
                        "volume_id"
                    ]
                ),

            "paper2_role":
                str(
                    row[
                        "paper2_role"
                    ]
                ),

            "purpose":
                self.purpose,

            "cache_path":
                str(
                    row[
                        "cache_path"
                    ]
                ),

            "semantic_sample":
                semantic_sample,
        }

    @property
    def volume_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.rows[
                    "volume_id"
                ].unique().tolist()
            )
        )

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(
            self.rows[
                "sample_id"
            ].tolist()
        )
