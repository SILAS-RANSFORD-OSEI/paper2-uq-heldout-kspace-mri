"""Tests for the completed P2-Exp001C metadata audit."""

from pathlib import Path
import json

import yaml


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

AUDIT_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "audits"
    / "exp001c_acquisition_metadata_audit.json"
)

CANDIDATE_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "protocol"
    / "stratification_candidate_v0.1.yaml"
)

ALGORITHM_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "protocol"
    / "split_assignment_algorithm_v1.0.yaml"
)

SPLIT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "splits"
    / "paper2_split.csv"
)


def load_audit() -> dict:
    return json.loads(
        AUDIT_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_metadata_audit_passed() -> None:
    audit = load_audit()

    assert audit["status"] == "PASS"
    assert audit["population"]["volume_count"] == 201
    assert audit["population"]["planned_D_fit"] == 181
    assert audit["population"]["planned_D_dev"] == 20


def test_metadata_audit_was_target_blind() -> None:
    blindness = load_audit()[
        "target_blindness"
    ]

    assert blindness["status"] == "PASS"
    assert blindness["outcome_columns_loaded"] == []
    assert blindness["targets_loaded"] is False
    assert blindness["cache_npz_arrays_opened"] is False
    assert blindness["model_predictions_loaded"] is False
    assert blindness["performance_reports_loaded"] is False


def test_metadata_audit_preceded_assignment() -> None:
    governance = load_audit()[
        "governance"
    ]

    assert governance["candidate_only"] is True
    assert governance["algorithm_frozen"] is False
    assert governance["split_created"] is False
    assert governance["volume_ids_assigned"] is False
    assert governance["final_test_barrier"] == "CLOSED"


def test_candidate_was_superseded_by_frozen_algorithm() -> None:
    candidate = yaml.safe_load(
        CANDIDATE_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        candidate[
            "candidate"
        ][
            "status"
        ]
        == "SUPERSEDED_BY_FROZEN_V1.0"
    )

    assert (
        candidate[
            "candidate"
        ][
            "superseded_by"
        ]
        == (
            "reports/protocol/"
            "split_assignment_algorithm_v1.0.yaml"
        )
    )


def test_frozen_algorithm_and_split_now_exist() -> None:
    assert ALGORITHM_PATH.exists()
    assert SPLIT_PATH.exists()
