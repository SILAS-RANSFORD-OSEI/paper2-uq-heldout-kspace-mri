"""Tests for role-aware cache access and leakage barriers."""

from pathlib import Path

import pandas as pd
import pytest

from paper2_uq_mri.split_access import (
    BARRIER_CLOSED,
    BARRIER_OPEN,
    PURPOSE_CALIBRATION,
    PURPOSE_FINAL_EVALUATION,
    PURPOSE_GRADIENT_FIT,
    PURPOSE_MODEL_SELECTION,
    FinalTestBarrier,
    SplitAccessError,
    SplitContractError,
    assert_role_access,
    build_role_aware_index,
    load_locked_split,
    select_authorized_rows,
)


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

SPLIT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "splits"
    / "paper2_split.csv"
)


def test_locked_split_contract() -> None:
    split = load_locked_split(
        SPLIT_PATH
    )

    assert len(split) == 281

    assert (
        split[
            "paper2_role"
        ]
        .value_counts()
        .to_dict()
        == {
            "D_fit": 181,
            "D_cal": 40,
            "D_test": 40,
            "D_dev": 20,
        }
    )


def test_expected_role_purpose_mapping() -> None:
    barrier = FinalTestBarrier(
        BARRIER_CLOSED
    )

    assert_role_access(
        paper2_role="D_fit",
        purpose=PURPOSE_GRADIENT_FIT,
        barrier=barrier,
    )

    assert_role_access(
        paper2_role="D_dev",
        purpose=PURPOSE_MODEL_SELECTION,
        barrier=barrier,
    )

    assert_role_access(
        paper2_role="D_cal",
        purpose=PURPOSE_CALIBRATION,
        barrier=barrier,
    )


@pytest.mark.parametrize(
    (
        "role",
        "purpose",
    ),
    [
        (
            "D_cal",
            PURPOSE_GRADIENT_FIT,
        ),
        (
            "D_test",
            PURPOSE_GRADIENT_FIT,
        ),
        (
            "D_fit",
            PURPOSE_MODEL_SELECTION,
        ),
        (
            "D_test",
            PURPOSE_MODEL_SELECTION,
        ),
        (
            "D_fit",
            PURPOSE_CALIBRATION,
        ),
        (
            "D_test",
            PURPOSE_CALIBRATION,
        ),
    ],
)
def test_forbidden_cross_role_access(
    role: str,
    purpose: str,
) -> None:
    with pytest.raises(
        SplitAccessError
    ):
        assert_role_access(
            paper2_role=role,
            purpose=purpose,
            barrier=FinalTestBarrier(
                BARRIER_CLOSED
            ),
        )


def test_D_test_is_blocked_while_barrier_closed() -> None:
    with pytest.raises(
        SplitAccessError,
        match="barrier is CLOSED",
    ):
        assert_role_access(
            paper2_role="D_test",
            purpose=PURPOSE_FINAL_EVALUATION,
            barrier=FinalTestBarrier(
                BARRIER_CLOSED
            ),
        )


def test_D_test_requires_explicit_open_barrier() -> None:
    assert_role_access(
        paper2_role="D_test",
        purpose=PURPOSE_FINAL_EVALUATION,
        barrier=FinalTestBarrier(
            BARRIER_OPEN
        ),
    )


def test_invalid_barrier_state_is_rejected() -> None:
    with pytest.raises(
        SplitContractError
    ):
        FinalTestBarrier(
            "UNKNOWN"
        )


def test_synthetic_role_index_filters_exactly(
    tmp_path: Path,
) -> None:
    split_rows = []

    role_specs = [
        (
            "D_fit",
            "train",
            181,
        ),
        (
            "D_dev",
            "train",
            20,
        ),
        (
            "D_cal",
            "calibration",
            40,
        ),
        (
            "D_test",
            "test",
            40,
        ),
    ]

    manifest_rows = []
    counter = 0

    for role, source, count in role_specs:
        for role_index in range(
            count
        ):
            counter += 1

            volume_id = (
                f"volume_{counter:03d}"
            )

            split_rows.append(
                {
                    "volume_id":
                        volume_id,

                    "paper1_split":
                        source,

                    "paper2_role":
                        role,

                    "assignment_seed":
                        20260720,

                    "assignment_algorithm":
                        (
                            "width_coil_ilp_"
                            "slice_balance_v1.0"
                        ),

                    "locked":
                        True,

                    "test_designation":
                        (
                            "locked_reused_"
                            "evaluation_cohort"
                            if role == "D_test"
                            else ""
                        ),
                }
            )

            manifest_rows.append(
                {
                    "sample_id":
                        f"{volume_id}_slice000",

                    "volume_id":
                        volume_id,

                    "_resolved_absolute_path":
                        str(
                            tmp_path
                            / f"{volume_id}.npz"
                        ),
                }
            )

    split_path = (
        tmp_path
        / "split.csv"
    )

    manifest_path = (
        tmp_path
        / "manifest.csv"
    )

    pd.DataFrame(
        split_rows
    ).to_csv(
        split_path,
        index=False,
    )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        manifest_path,
        index=False,
    )

    index = build_role_aware_index(
        manifest_path,
        split_path,
        require_existing_paths=False,
    )

    fit_rows = select_authorized_rows(
        index,
        purpose=PURPOSE_GRADIENT_FIT,
    )

    dev_rows = select_authorized_rows(
        index,
        purpose=PURPOSE_MODEL_SELECTION,
    )

    cal_rows = select_authorized_rows(
        index,
        purpose=PURPOSE_CALIBRATION,
    )

    assert set(
        fit_rows[
            "paper2_role"
        ]
    ) == {
        "D_fit",
    }

    assert set(
        dev_rows[
            "paper2_role"
        ]
    ) == {
        "D_dev",
    }

    assert set(
        cal_rows[
            "paper2_role"
        ]
    ) == {
        "D_cal",
    }

    assert len(fit_rows) == 181
    assert len(dev_rows) == 20
    assert len(cal_rows) == 40

    with pytest.raises(
        SplitAccessError
    ):
        select_authorized_rows(
            index,
            purpose=PURPOSE_FINAL_EVALUATION,
            barrier=FinalTestBarrier(
                BARRIER_CLOSED
            ),
        )
