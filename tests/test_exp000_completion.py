"""Tests for closure of P2-Exp000."""

from pathlib import Path
import json

import yaml


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

COMPLETION_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "audits"
    / "exp000_completion.json"
)

PROTOCOL_STATUS_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "protocol"
    / "protocol_status.yaml"
)

EXP000_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "exp000_audit.yaml"
)


def test_exp000_completion_is_pass() -> None:
    completion = json.loads(
        COMPLETION_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert completion["status"] == "PASS"

    cache = completion["cache"]

    assert cache["manifest_rows"] == 4462
    assert cache["successful_files"] == 4462
    assert cache["failed_files"] == 0


def test_exp000_split_counts_are_unchanged() -> None:
    completion = json.loads(
        COMPLETION_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert completion["cache"]["split_rows"] == {
        "train": 3190,
        "calibration": 636,
        "test": 636,
    }


def test_exp000_final_test_barrier_remains_closed() -> None:
    completion = json.loads(
        COMPLETION_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        completion["final_test_barrier"]
        == "CLOSED"
    )

    assert (
        completion["paper2_models_trained"]
        is False
    )

    assert (
        completion["paper2_split_constructed"]
        is False
    )


def test_protocol_records_exp000_completion() -> None:
    status = yaml.safe_load(
        PROTOCOL_STATUS_PATH.read_text(
            encoding="utf-8"
        )
    )

    exp000 = status[
        "experiments"
    ][
        "P2-Exp000"
    ]

    assert exp000["status"] == "completed"
    assert exp000["result"] == "PASS"
    assert exp000["files_audited"] == 4462
    assert exp000["files_failed"] == 0
    assert exp000["final_test_opened"] is False


def test_exp000_config_is_completed() -> None:
    config = yaml.safe_load(
        EXP000_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        config[
            "experiment"
        ][
            "status"
        ]
        == "completed"
    )

    assert (
        config[
            "completion"
        ][
            "status"
        ]
        == "PASS"
    )
