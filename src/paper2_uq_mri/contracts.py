"""Validation of the Paper 2 protocol contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from paper2_uq_mri.notation import (
    collect_symbol_entries,
    load_notation_registry,
)


REQUIRED_METHOD_IDS = {
    "C0",
    "U1",
    "U2a",
    "U2b",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
}

REQUIRED_TASK_IDS = {
    "P",
    "R",
    "E",
}


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping: {path}"
        )

    return data


def validate_protocol_contracts(
    repository_root: Path | str,
) -> list[str]:
    """Return all protocol-contract validation errors."""
    root = Path(repository_root)

    notation = load_notation_registry(
        root
        / "reports/protocol/notation_registry_v1.1.yaml"
    )

    methods = load_yaml(
        root
        / "reports/protocol/method_registry_v1.0.yaml"
    )

    endpoints = load_yaml(
        root
        / "reports/protocol/endpoint_registry_v1.0.yaml"
    )

    governance = load_yaml(
        root
        / "reports/protocol/data_governance_v1.0.yaml"
    )

    status = load_yaml(
        root
        / "reports/protocol/protocol_status.yaml"
    )

    errors: list[str] = []

    notation_symbols = {
        entry["symbol"]
        for entry in collect_symbol_entries(notation)
    }

    method_dict = methods.get("methods", {})

    if set(method_dict) != REQUIRED_METHOD_IDS:
        errors.append(
            "Method IDs differ from the frozen set. "
            f"Found: {sorted(method_dict)}"
        )

    for method_id, method in method_dict.items():
        if method.get("method_id") != method_id:
            errors.append(
                f"Method key/ID mismatch for {method_id}."
            )

        suffix = method.get("symbol_suffix")

        if not isinstance(suffix, str) or not suffix:
            errors.append(
                f"Missing symbol suffix for {method_id}."
            )

    tasks = endpoints.get("tasks", {})

    if set(tasks) != REQUIRED_TASK_IDS:
        errors.append(
            "Endpoint task IDs differ from the frozen set. "
            f"Found: {sorted(tasks)}"
        )

    if tasks.get("R", {}).get("threshold_symbol") != "tau_hold":
        errors.append(
            "Task R threshold must be tau_hold."
        )

    if tasks.get("R", {}).get("target_symbol") != "u_hold_v":
        errors.append(
            "Task R target must be u_hold_v."
        )

    if tasks.get("E", {}).get("target_symbol") != "d_j_v":
        errors.append(
            "Task E target must be d_j_v."
        )

    required_notation_symbols = {
        "u_hold_v",
        "mu_j_v",
        "U_j_v",
        "tau_hold",
        "h_v",
        "d_j_v",
    }

    missing_symbols = sorted(
        required_notation_symbols - notation_symbols
    )

    if missing_symbols:
        errors.append(
            "Protocol contracts reference missing notation symbols: "
            + ", ".join(missing_symbols)
        )

    sets = (
        governance
        .get("governance", {})
        .get("paper2_sets", {})
    )

    expected_counts = {
        "D_fit": 181,
        "D_dev": 20,
        "D_cal": 40,
        "D_test": 40,
    }

    actual_counts = {
        key: value.get("planned_volumes")
        for key, value in sets.items()
    }

    if actual_counts != expected_counts:
        errors.append(
            "Planned Paper 2 split counts changed. "
            f"Found: {actual_counts}"
        )

    if sum(expected_counts.values()) != 281:
        errors.append(
            "Planned split does not sum to 281 volumes."
        )

    final_test_opened = (
        status
        .get("protocol", {})
        .get("final_test_opened")
    )

    if final_test_opened is not False:
        errors.append(
            "The Paper 2 final test barrier must remain closed."
        )

    return errors
