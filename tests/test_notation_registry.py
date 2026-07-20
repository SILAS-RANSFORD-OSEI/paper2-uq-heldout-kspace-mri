"""Tests for the canonical Paper 2 notation registry."""

from pathlib import Path

from paper2_uq_mri.notation import (
    collect_symbol_entries,
    load_notation_registry,
    validate_notation_registry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "protocol"
    / "notation_registry_v1.1.yaml"
)


def test_notation_registry_has_no_validation_errors() -> None:
    registry = load_notation_registry(REGISTRY_PATH)
    errors = validate_notation_registry(registry)

    assert errors == [], "\n".join(errors)


def test_one_symbol_has_one_registry_entry() -> None:
    registry = load_notation_registry(REGISTRY_PATH)
    entries = collect_symbol_entries(registry)

    symbols = [
        entry["symbol"]
        for entry in entries
    ]

    assert len(symbols) == len(set(symbols))


def test_one_code_name_has_one_registry_entry() -> None:
    registry = load_notation_registry(REGISTRY_PATH)
    entries = collect_symbol_entries(registry)

    code_names = [
        entry["code_name"]
        for entry in entries
        if entry["code_name"]
    ]

    assert len(code_names) == len(set(code_names))


def test_collision_prone_symbols_are_reserved_correctly() -> None:
    registry = load_notation_registry(REGISTRY_PATH)
    entries = collect_symbol_entries(registry)

    by_symbol = {
        entry["symbol"]: entry
        for entry in entries
    }

    assert "measured multicoil k-space" in (
        by_symbol["y"]["meaning"].lower()
    )

    assert "residual energy" in (
        by_symbol["e_s_v"]["meaning"].lower()
    )

    assert "absolute prediction deviation" in (
        by_symbol["d_j_v"]["meaning"].lower()
    )

    assert "mean residual-risk prediction" in (
        by_symbol["mu_j_v"]["meaning"].lower()
    )

    assert "uncertainty score" in (
        by_symbol["U_j_v"]["meaning"].lower()
    )


def test_legacy_symbols_are_not_canonical() -> None:
    registry = load_notation_registry(REGISTRY_PATH)
    entries = collect_symbol_entries(registry)

    symbols = {
        entry["symbol"]
        for entry in entries
    }

    prohibited = {
        "y_R",
        "e_j",
        "u_bar",
        "tau_R",
        "U_between",
        "q_hat",
    }

    assert symbols.isdisjoint(prohibited)
