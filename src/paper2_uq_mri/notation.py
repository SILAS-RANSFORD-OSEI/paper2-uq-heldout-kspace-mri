"""Canonical notation-registry validation.

This module enforces the Paper 2 rule:

    one scientific concept -> one canonical symbol
    one canonical symbol -> one scientific meaning
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import yaml


REQUIRED_CANONICAL_SYMBOLS = {
    "y",
    "e_s_v",
    "d_j_v",
    "mu_j_v",
    "U_j_v",
    "tau_hold",
    "h_v",
    "u_risk_v",
    "u_hold_v",
}

PROHIBITED_CANONICAL_SYMBOLS = {
    "y_R",
    "e_j",
    "u_bar",
    "tau_R",
    "U_between",
    "q_hat",
}


def load_notation_registry(path: Path | str) -> dict[str, Any]:
    """Load and minimally validate the notation-registry YAML file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Notation registry does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "The notation registry must contain a YAML mapping."
        )

    if "registry" not in data:
        raise ValueError(
            "The notation registry requires a top-level 'registry' field."
        )

    return data


def iter_symbol_entries(
    node: Any,
    path: tuple[str, ...] = (),
) -> Iterator[dict[str, Any]]:
    """Recursively yield mappings containing canonical symbols."""
    if isinstance(node, dict):
        if (
            "symbol" in node
            and "meaning" in node
        ):
            yield {
                "path": ".".join(path),
                "symbol": str(node["symbol"]),
                "meaning": str(node["meaning"]),
                "code_name": (
                    str(node["code_name"])
                    if "code_name" in node
                    else ""
                ),
                "latex": (
                    str(node["latex"])
                    if "latex" in node
                    else ""
                ),
            }

        for key, value in node.items():
            yield from iter_symbol_entries(
                value,
                path + (str(key),),
            )

    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_symbol_entries(
                value,
                path + (str(index),),
            )


def collect_symbol_entries(
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return all canonical notation entries."""
    return list(iter_symbol_entries(registry))


def validate_notation_registry(
    registry: dict[str, Any],
) -> list[str]:
    """
    Return a list of validation errors.

    An empty list means the registry passes.
    """
    errors: list[str] = []

    registry_meta = registry.get("registry", {})

    required_metadata = {
        "title",
        "version",
        "status",
        "governing_rule",
    }

    missing_metadata = sorted(
        required_metadata - set(registry_meta)
    )

    if missing_metadata:
        errors.append(
            "Missing registry metadata: "
            + ", ".join(missing_metadata)
        )

    entries = collect_symbol_entries(registry)

    if len(entries) < 20:
        errors.append(
            f"Too few canonical symbol entries: {len(entries)}."
        )

    symbols_to_paths: dict[str, list[str]] = defaultdict(list)
    code_names_to_paths: dict[str, list[str]] = defaultdict(list)

    for entry in entries:
        symbol = entry["symbol"].strip()
        meaning = entry["meaning"].strip()
        code_name = entry["code_name"].strip()

        if not symbol:
            errors.append(
                f"Empty symbol at {entry['path']}."
            )

        if not meaning:
            errors.append(
                f"Empty meaning at {entry['path']}."
            )

        symbols_to_paths[symbol].append(entry["path"])

        if code_name:
            code_names_to_paths[code_name].append(
                entry["path"]
            )

    duplicate_symbols = {
        symbol: paths
        for symbol, paths in symbols_to_paths.items()
        if len(paths) > 1
    }

    for symbol, paths in duplicate_symbols.items():
        errors.append(
            f"Duplicate canonical symbol '{symbol}' at: "
            + ", ".join(paths)
        )

    duplicate_code_names = {
        code_name: paths
        for code_name, paths in code_names_to_paths.items()
        if len(paths) > 1
    }

    for code_name, paths in duplicate_code_names.items():
        errors.append(
            f"Duplicate code name '{code_name}' at: "
            + ", ".join(paths)
        )

    canonical_symbols = set(symbols_to_paths)

    missing_required = sorted(
        REQUIRED_CANONICAL_SYMBOLS
        - canonical_symbols
    )

    if missing_required:
        errors.append(
            "Missing required canonical symbols: "
            + ", ".join(missing_required)
        )

    prohibited_present = sorted(
        PROHIBITED_CANONICAL_SYMBOLS
        & canonical_symbols
    )

    if prohibited_present:
        errors.append(
            "Prohibited legacy symbols remain canonical: "
            + ", ".join(prohibited_present)
        )

    # --------------------------------------------------------
    # Semantic guards for the most collision-prone symbols
    # --------------------------------------------------------
    entry_by_symbol = {
        entry["symbol"]: entry
        for entry in entries
    }

    semantic_requirements = {
        "y": (
            "measured",
            "k-space",
        ),
        "e_s_v": (
            "residual",
            "energy",
        ),
        "d_j_v": (
            "absolute",
            "prediction",
            "deviation",
        ),
        "mu_j_v": (
            "mean",
            "prediction",
        ),
        "U_j_v": (
            "uncertainty",
            "score",
        ),
        "tau_hold": (
            "calibration",
            "threshold",
            "u_hold",
        ),
        "h_v": (
            "binary",
            "indicator",
        ),
    }

    for symbol, required_terms in semantic_requirements.items():
        if symbol not in entry_by_symbol:
            continue

        meaning = entry_by_symbol[symbol]["meaning"].lower()

        missing_terms = [
            term
            for term in required_terms
            if term.lower() not in meaning
        ]

        if missing_terms:
            errors.append(
                f"Symbol '{symbol}' has an incomplete meaning. "
                f"Missing terms: {missing_terms}. "
                f"Current meaning: {meaning}"
            )

    return errors


def notation_rows(
    registry: dict[str, Any],
) -> list[dict[str, str]]:
    """Create sorted rows suitable for a CSV notation table."""
    rows = collect_symbol_entries(registry)

    return sorted(
        rows,
        key=lambda row: (
            row["symbol"].lower(),
            row["path"],
        ),
    )
