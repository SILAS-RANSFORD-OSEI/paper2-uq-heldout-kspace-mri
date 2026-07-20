#!/usr/bin/env python3
"""Validate the canonical Paper 1 artifact registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from paper2_uq_mri.provenance import (
    load_artifact_registry,
    validate_artifact_files,
    validate_registry_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--drive-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry = load_artifact_registry(
        args.registry
    )

    schema_errors = (
        validate_registry_schema(
            registry
        )
    )

    checks = validate_artifact_files(
        registry=registry,
        source_root=args.source_root,
        drive_root=args.drive_root,
    )

    failed_checks = [
        check
        for check in checks
        if check["status"] != "PASS"
    ]

    status = (
        "PASS"
        if (
            not schema_errors
            and not failed_checks
        )
        else "FAIL"
    )

    report = {
        "experiment_id":
            "P2-Exp000",

        "stage":
            "artifact_registry_validation",

        "status":
            status,

        "schema_errors":
            schema_errors,

        "checks":
            checks,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status":
                    status,

                "artifact_count":
                    len(checks),

                "failed_count":
                    len(
                        failed_checks
                    ),

                "schema_error_count":
                    len(
                        schema_errors
                    ),

                "output":
                    str(args.output),
            },
            indent=2,
        )
    )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
