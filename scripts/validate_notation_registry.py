#!/usr/bin/env python3
"""Validate the canonical notation registry and export a CSV table."""

from __future__ import annotations

import csv
from pathlib import Path

from paper2_uq_mri.notation import (
    collect_symbol_entries,
    load_notation_registry,
    notation_rows,
    validate_notation_registry,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    registry_path = (
        repository_root
        / "reports"
        / "protocol"
        / "notation_registry_v1.1.yaml"
    )

    output_path = (
        repository_root
        / "reports"
        / "protocol"
        / "notation_table_v1.1.csv"
    )

    registry = load_notation_registry(registry_path)
    errors = validate_notation_registry(registry)

    if errors:
        print("=" * 72)
        print("NOTATION REGISTRY: FAIL")
        print("=" * 72)

        for index, error in enumerate(errors, start=1):
            print(f"{index}. {error}")

        raise SystemExit(1)

    rows = notation_rows(registry)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "symbol",
                "latex",
                "meaning",
                "code_name",
                "path",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("NOTATION REGISTRY VALIDATION")
    print("=" * 72)
    print(f"Registry:          {registry_path}")
    print(f"Canonical symbols:{len(collect_symbol_entries(registry)):>5}")
    print(f"Validation errors:{len(errors):>5}")
    print(f"Notation table:    {output_path}")
    print("=" * 72)
    print("NOTATION REGISTRY STATUS: PASS")


if __name__ == "__main__":
    main()
