#!/usr/bin/env python3
"""Audit every NPZ archive in the Paper 1 reliability cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from paper2_uq_mri.cache import (
    load_reliability_cache_sample,
)


def normalize_split_name(value: object) -> str:
    """Normalize Paper 1 split names."""
    text = str(value).strip().lower()

    aliases = {
        "training": "train",
        "validation": "calibration",
        "val": "calibration",
        "cal": "calibration",
        "testing": "test",
    }

    return aliases.get(
        text,
        text,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = pd.read_csv(
        args.manifest
    )

    required_columns = {
        "sample_id",
        "volume_id",
        "split",
        "height",
        "width",
        "_resolved_absolute_path",
    }

    missing_columns = sorted(
        required_columns
        - set(
            manifest.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Resolved manifest is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    contract = yaml.safe_load(
        args.contract.read_text(
            encoding="utf-8"
        )
    )

    permitted_sizes = {
        tuple(
            int(value)
            for value in size
        )
        for size in contract[
            "spatial_template"
        ][
            "permitted_observed_sizes"
        ]
    }

    manifest[
        "_normalized_split"
    ] = manifest[
        "split"
    ].map(
        normalize_split_name
    )

    total_files = len(
        manifest
    )

    split_counts = Counter()
    matrix_size_counts = Counter()
    failure_type_counts = Counter()

    file_failures = []
    diagnostic_rows = []

    channel_minima = [
        float("inf")
        for _ in range(6)
    ]

    channel_maxima = [
        float("-inf")
        for _ in range(6)
    ]

    global_u_risk_minimum = float("inf")
    global_u_risk_maximum = float("-inf")
    global_z_risk_minimum = float("inf")
    global_z_risk_maximum = float("-inf")
    global_support_minimum = float("inf")
    global_support_maximum = float("-inf")

    maximum_transform_error = 0.0
    maximum_p99_transform_error = 0.0
    maximum_mean_transform_error = 0.0
    minimum_transform_correlation = 1.0

    successful_files = 0

    for position, row in manifest.iterrows():
        sample_id = str(
            row[
                "sample_id"
            ]
        )

        volume_id = str(
            row[
                "volume_id"
            ]
        )

        split_name = str(
            row[
                "_normalized_split"
            ]
        )

        expected_height = int(
            row[
                "height"
            ]
        )

        expected_width = int(
            row[
                "width"
            ]
        )

        expected_size = (
            expected_height,
            expected_width,
        )

        path = Path(
            str(
                row[
                    "_resolved_absolute_path"
                ]
            )
        )

        diagnostic = {
            "sample_id":
                sample_id,

            "volume_id":
                volume_id,

            "split":
                split_name,

            "height":
                expected_height,

            "width":
                expected_width,

            "file_size_bytes":
                (
                    path.stat().st_size
                    if path.is_file()
                    else None
                ),

            "maximum_absolute_error":
                None,

            "percentile_99_absolute_error":
                None,

            "mean_absolute_error":
                None,

            "pearson_correlation":
                None,

            "reconstructed_quantile":
                None,

            "status":
                "FAIL",
        }

        try:
            if expected_size not in permitted_sizes:
                raise ValueError(
                    "Manifest matrix size is not permitted by "
                    f"the frozen contract: {expected_size}."
                )

            sample = load_reliability_cache_sample(
                path,
                verify_transform=True,
            )

            if (
                sample.cache_input_6ch.shape
                != (
                    6,
                    expected_height,
                    expected_width,
                )
            ):
                raise ValueError(
                    "Six-channel input shape does not match "
                    "the manifest: "
                    f"{sample.cache_input_6ch.shape} vs "
                    f"{(6, expected_height, expected_width)}."
                )

            if (
                sample.predictor_input.shape
                != (
                    3,
                    expected_height,
                    expected_width,
                )
            ):
                raise ValueError(
                    "A4 predictor-input shape does not match "
                    "the frozen contract."
                )

            if (
                sample.u_risk.shape
                != expected_size
            ):
                raise ValueError(
                    "u_risk shape does not match the manifest."
                )

            if (
                sample.z_risk.shape
                != expected_size
            ):
                raise ValueError(
                    "z_risk shape does not match the manifest."
                )

            diagnostics = (
                sample.transform_diagnostics
            )

            if not diagnostics.passed:
                raise ValueError(
                    "Stored nonlinear target transformation "
                    "did not pass."
                )

            for channel_index in range(6):
                channel = sample.cache_input_6ch[
                    channel_index
                ]

                channel_minima[
                    channel_index
                ] = min(
                    channel_minima[
                        channel_index
                    ],
                    float(
                        channel.min()
                    ),
                )

                channel_maxima[
                    channel_index
                ] = max(
                    channel_maxima[
                        channel_index
                    ],
                    float(
                        channel.max()
                    ),
                )

            global_u_risk_minimum = min(
                global_u_risk_minimum,
                float(
                    sample.u_risk.min()
                ),
            )

            global_u_risk_maximum = max(
                global_u_risk_maximum,
                float(
                    sample.u_risk.max()
                ),
            )

            global_z_risk_minimum = min(
                global_z_risk_minimum,
                float(
                    sample.z_risk.min()
                ),
            )

            global_z_risk_maximum = max(
                global_z_risk_maximum,
                float(
                    sample.z_risk.max()
                ),
            )

            global_support_minimum = min(
                global_support_minimum,
                float(
                    sample.support_mask.min()
                ),
            )

            global_support_maximum = max(
                global_support_maximum,
                float(
                    sample.support_mask.max()
                ),
            )

            maximum_transform_error = max(
                maximum_transform_error,
                diagnostics.maximum_absolute_error,
            )

            maximum_p99_transform_error = max(
                maximum_p99_transform_error,
                diagnostics.percentile_99_absolute_error,
            )

            maximum_mean_transform_error = max(
                maximum_mean_transform_error,
                diagnostics.mean_absolute_error,
            )

            minimum_transform_correlation = min(
                minimum_transform_correlation,
                diagnostics.pearson_correlation,
            )

            split_counts[
                split_name
            ] += 1

            matrix_size_counts[
                expected_size
            ] += 1

            diagnostic.update(
                {
                    "maximum_absolute_error":
                        diagnostics.maximum_absolute_error,

                    "percentile_99_absolute_error":
                        diagnostics.percentile_99_absolute_error,

                    "mean_absolute_error":
                        diagnostics.mean_absolute_error,

                    "pearson_correlation":
                        diagnostics.pearson_correlation,

                    "reconstructed_quantile":
                        diagnostics.reconstructed_quantile,

                    "status":
                        "PASS",
                }
            )

            successful_files += 1

        except Exception as exc:
            failure_type = type(
                exc
            ).__name__

            failure_type_counts[
                failure_type
            ] += 1

            file_failures.append(
                {
                    "row_index":
                        int(
                            position
                        ),

                    "sample_id":
                        sample_id,

                    "volume_id":
                        volume_id,

                    "split":
                        split_name,

                    "height":
                        expected_height,

                    "width":
                        expected_width,

                    "path":
                        str(
                            path
                        ),

                    "failure_type":
                        failure_type,

                    "error":
                        repr(
                            exc
                        ),
                }
            )

        diagnostic_rows.append(
            diagnostic
        )

        completed = (
            position + 1
        )

        if (
            args.progress_every > 0
            and (
                completed
                % args.progress_every
                == 0
                or completed
                == total_files
            )
        ):
            print(
                f"AUDIT PROGRESS: "
                f"{completed:,}/{total_files:,} "
                f"files; "
                f"passed={successful_files:,}; "
                f"failed={len(file_failures):,}",
                flush=True,
            )

    args.diagnostics_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.diagnostics_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "sample_id",
            "volume_id",
            "split",
            "height",
            "width",
            "file_size_bytes",
            "maximum_absolute_error",
            "percentile_99_absolute_error",
            "mean_absolute_error",
            "pearson_correlation",
            "reconstructed_quantile",
            "status",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            diagnostic_rows
        )

    expected_split_counts = {
        "train": 3190,
        "calibration": 636,
        "test": 636,
    }

    observed_split_counts = {
        split_name:
            int(
                split_counts.get(
                    split_name,
                    0,
                )
            )
        for split_name
        in expected_split_counts
    }

    checks = {
        "expected_manifest_rows":
            total_files
            == 4462,

        "all_files_audited":
            (
                successful_files
                + len(
                    file_failures
                )
                == total_files
            ),

        "all_files_passed":
            successful_files
            == total_files,

        "no_file_failures":
            len(
                file_failures
            )
            == 0,

        "expected_split_counts":
            observed_split_counts
            == expected_split_counts,

        "all_matrix_sizes_permitted":
            all(
                size in permitted_sizes
                for size
                in matrix_size_counts
            ),

        "support_mask_within_unit_interval":
            (
                global_support_minimum
                >= -1.0e-5
                and global_support_maximum
                <= 1.0 + 1.0e-5
            ),

        "transform_maximum_error_within_loader_tolerance":
            maximum_transform_error
            <= 0.05,

        "transform_p99_error_within_loader_tolerance":
            maximum_p99_transform_error
            <= 0.02,

        "transform_correlation_within_loader_tolerance":
            minimum_transform_correlation
            >= 0.999,
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    status = (
        "PASS"
        if not failed_checks
        else "FAIL"
    )

    report = {
        "experiment_id":
            "P2-Exp000J",

        "stage":
            "full_reliability_cache_integrity_audit",

        "status":
            status,

        "runtime_device":
            "CPU",

        "manifest_path":
            str(
                args.manifest
            ),

        "contract_path":
            str(
                args.contract
            ),

        "counts": {
            "manifest_rows":
                int(
                    total_files
                ),

            "successful_files":
                int(
                    successful_files
                ),

            "failed_files":
                int(
                    len(
                        file_failures
                    )
                ),

            "split_rows":
                observed_split_counts,

            "matrix_sizes":
                {
                    f"{height}x{width}":
                        int(
                            count
                        )
                    for (
                        height,
                        width,
                    ), count
                    in sorted(
                        matrix_size_counts.items()
                    )
                },
        },

        "numerical_ranges": {
            "channel_minima":
                [
                    float(
                        value
                    )
                    for value
                    in channel_minima
                ],

            "channel_maxima":
                [
                    float(
                        value
                    )
                    for value
                    in channel_maxima
                ],

            "u_risk_minimum":
                float(
                    global_u_risk_minimum
                ),

            "u_risk_maximum":
                float(
                    global_u_risk_maximum
                ),

            "z_risk_minimum":
                float(
                    global_z_risk_minimum
                ),

            "z_risk_maximum":
                float(
                    global_z_risk_maximum
                ),

            "support_mask_minimum":
                float(
                    global_support_minimum
                ),

            "support_mask_maximum":
                float(
                    global_support_maximum
                ),
        },

        "target_transform": {
            "maximum_absolute_error":
                float(
                    maximum_transform_error
                ),

            "maximum_file_p99_absolute_error":
                float(
                    maximum_p99_transform_error
                ),

            "maximum_file_mean_absolute_error":
                float(
                    maximum_mean_transform_error
                ),

            "minimum_pearson_correlation":
                float(
                    minimum_transform_correlation
                ),
        },

        "failure_type_counts":
            dict(
                failure_type_counts
            ),

        "file_failures":
            file_failures,

        "checks":
            checks,

        "failed_checks":
            failed_checks,

        "diagnostics_csv":
            str(
                args.diagnostics_csv
            ),

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report.write_text(
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

                "manifest_rows":
                    total_files,

                "successful_files":
                    successful_files,

                "failed_files":
                    len(
                        file_failures
                    ),

                "failed_checks":
                    failed_checks,

                "report":
                    str(
                        args.report
                    ),

                "diagnostics_csv":
                    str(
                        args.diagnostics_csv
                    ),
            },
            indent=2,
        ),
        flush=True,
    )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
