"""Tests for the frozen Paper 2 volume split."""

from pathlib import Path
import hashlib
import json

import pandas as pd
import yaml


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

SPLIT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "splits"
    / "paper2_split.csv"
)

ALGORITHM_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "protocol"
    / "split_assignment_algorithm_v1.0.yaml"
)

BALANCE_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "audits"
    / "exp001d_split_balance.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_locked_split_counts() -> None:
    split = pd.read_csv(
        SPLIT_PATH
    )

    assert len(split) == 281
    assert split["volume_id"].nunique() == 281

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


def test_original_roles_are_preserved() -> None:
    split = pd.read_csv(
        SPLIT_PATH
    )

    assert set(
        split.loc[
            split[
                "paper2_role"
            ].isin(
                [
                    "D_fit",
                    "D_dev",
                ]
            ),
            "paper1_split",
        ]
    ) == {
        "train",
    }

    assert set(
        split.loc[
            split[
                "paper2_role"
            ]
            == "D_cal",
            "paper1_split",
        ]
    ) == {
        "calibration",
    }

    assert set(
        split.loc[
            split[
                "paper2_role"
            ]
            == "D_test",
            "paper1_split",
        ]
    ) == {
        "test",
    }


def test_split_is_locked_and_seeded() -> None:
    split = pd.read_csv(
        SPLIT_PATH
    )

    assert (
        split[
            "locked"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
        .all()
    )

    assert set(
        split[
            "assignment_seed"
        ]
    ) == {
        20260720,
    }

    assert set(
        split[
            "assignment_algorithm"
        ]
    ) == {
        "width_coil_ilp_slice_balance_v1.0",
    }


def test_test_designation_is_explicit() -> None:
    split = pd.read_csv(
        SPLIT_PATH
    )

    test_rows = split[
        split[
            "paper2_role"
        ]
        == "D_test"
    ]

    assert set(
        test_rows[
            "test_designation"
        ]
    ) == {
        "locked_reused_evaluation_cohort",
    }


def test_split_checksum_matches_frozen_algorithm() -> None:
    algorithm = yaml.safe_load(
        ALGORITHM_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        sha256_file(
            SPLIT_PATH
        )
        == algorithm[
            "output"
        ][
            "sha256"
        ]
    )


def test_split_balance_report_passed() -> None:
    report = json.loads(
        BALANCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "PASS"
    assert report["failed_checks"] == []
    assert all(
        report[
            "balance_checks"
        ].values()
    )

    governance = report[
        "governance"
    ]

    assert governance["split_created"] is True
    assert governance["volume_ids_assigned"] is True
    assert governance["split_locked"] is True
    assert governance["final_test_barrier"] == "CLOSED"
    assert governance["test_predictions_generated"] is False


def test_assignment_remained_outcome_blind() -> None:
    report = json.loads(
        BALANCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    blindness = report[
        "target_blindness"
    ]

    assert blindness["risk_targets_loaded"] is False
    assert blindness["cache_npz_arrays_opened"] is False
    assert blindness["predictions_loaded"] is False
    assert blindness["uncertainty_loaded"] is False
    assert blindness["performance_metrics_loaded"] is False


def test_patient_level_independence_is_not_claimed() -> None:
    report = json.loads(
        BALANCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    patient = report[
        "patient_identifier"
    ]

    assert patient["available"] is True
    assert patient["unique_per_volume"] is True

    assert (
        patient[
            "validated_cross_volume_identifier"
        ]
        is False
    )

    assert patient["used_in_assignment"] is False

    assert (
        patient["claim"]
        == "volume-level separation only"
    )


def test_p_values_are_descriptive_only() -> None:
    report = json.loads(
        BALANCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    for test in report[
        "descriptive_tests"
    ].values():
        if test.get(
            "available",
            False,
        ):
            assert (
                "not used"
                in test[
                    "interpretation"
                ]
            )
