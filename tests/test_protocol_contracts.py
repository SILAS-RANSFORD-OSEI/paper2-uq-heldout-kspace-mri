"""Tests for frozen Paper 2 research contracts."""

from pathlib import Path

from paper2_uq_mri.contracts import (
    load_yaml,
    validate_protocol_contracts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_contracts_have_no_errors() -> None:
    errors = validate_protocol_contracts(
        REPOSITORY_ROOT
    )

    assert errors == [], "\n".join(errors)


def test_protocol_is_frozen_before_implementation() -> None:
    status = load_yaml(
        REPOSITORY_ROOT
        / "reports/protocol/protocol_status.yaml"
    )

    protocol = status["protocol"]

    assert protocol["version"] == "1.1"
    assert protocol["status"] == (
        "frozen-before-implementation"
    )

    assert protocol["final_test_opened"] is False
    assert protocol["calibration_frozen"] is False
    assert protocol["paper2_models_trained"] is False


def test_final_test_barrier_has_required_prerequisites() -> None:
    governance = load_yaml(
        REPOSITORY_ROOT
        / "reports/protocol/data_governance_v1.0.yaml"
    )

    barrier = governance["governance"]["final_test_barrier"]

    required = {
        "all_method_configs_frozen",
        "all_random_seeds_frozen",
        "tau_hold_frozen",
        "all_metric_implementations_tested",
        "calibration_manifest_hashed",
        "git_worktree_clean",
    }

    assert barrier["opened"] is False
    assert set(barrier["prerequisites"]) == required


def test_primary_endpoint_hierarchy_is_locked() -> None:
    endpoints = load_yaml(
        REPOSITORY_ROOT
        / "reports/protocol/endpoint_registry_v1.0.yaml"
    )

    tasks = endpoints["tasks"]

    assert tasks["P"]["primary_metric"] == "mae"
    assert tasks["R"]["primary_metric"] == "auprc"
    assert tasks["E"]["primary_metric"] == "ause"

    assert tasks["R"]["higher_is_better"] is True
    assert tasks["E"]["higher_is_better"] is False
