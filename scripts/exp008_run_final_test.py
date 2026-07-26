# ============================================================
# CELL 46 — ACTUAL EXP008 PRIMARY FINAL-TEST IMPLEMENTATION
#
# Primary endpoints:
#   Task P: per-volume weighted MAE
#   Task R: per-volume weighted AUPRC at frozen tau_hold
#   Task E: per-volume normalized AUSE
#
# Methods:
#   Task P: C0, U1, U2a, U2b
#   Task R: U1, U2a, U2b
#   Task E: U1, U2a, U2b
#
# Inference:
#   - 40-volume unweighted mean
#   - 1,000 paired volume bootstrap replicates
#   - 95% percentile confidence intervals
#   - seed 20260725
#
# Restart-safe:
#   Completed volume records are validated and skipped.
#
# No training, model selection, threshold selection, or fitting.
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ------------------------------------------------------------
# 1. Frozen paths
# ------------------------------------------------------------
REPO = Path(
    "/content/paper2-uq-heldout-kspace-mri"
).resolve()

sys.path.insert(
    0,
    str(REPO / "src"),
)

from paper2_uq_mri.metrics.ause import (
    normalized_ause,
)


EXPECTED_COMMIT = (
    "58020824270afd67a125f0c04e993c591fdde791"
)

DRIVE_PROJECT = Path(
    "/content/drive/MyDrive/"
    "Paper2_UQ_Heldout_KSpace_MRI"
).resolve()

DTEST_ROOT = (
    DRIVE_PROJECT
    / "outputs"
    / "exp006_dtest_final"
    / "release_pending"
)

NEURAL_ROOT = (
    DTEST_ROOT
    / "neural_outputs"
)

UHOLD_ROOT = (
    DTEST_ROOT
    / "uhold_outputs"
)

NEURAL_CHUNK_ROOT = (
    NEURAL_ROOT
    / "chunks"
)

UHOLD_CHUNK_ROOT = (
    UHOLD_ROOT
    / "aligned_chunks"
)

CALIBRATION_RULES_PATH = (
    REPO
    / "results"
    / "exp004b_calibration_freeze"
    / "calibration_rules.json"
)

AUSE_CONTRACT_PATH = (
    REPO
    / "configs"
    / "exp006_ause_contract.json"
)

OUTPUT_ROOT = (
    DRIVE_PROJECT
    / "outputs"
    / "exp008_final_test_primary"
    / "commit58020824270a"
)

VOLUME_RESULT_ROOT = (
    OUTPUT_ROOT
    / "per_volume_records"
)

TASK_P_PATH = (
    OUTPUT_ROOT
    / "task_p_per_volume.csv"
)

TASK_R_PATH = (
    OUTPUT_ROOT
    / "task_r_per_volume.csv"
)

TASK_E_PATH = (
    OUTPUT_ROOT
    / "task_e_per_volume.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "primary_endpoint_summary.csv"
)

PAIRWISE_PATH = (
    OUTPUT_ROOT
    / "paired_method_differences.csv"
)

WIN_RATE_PATH = (
    OUTPUT_ROOT
    / "volume_level_win_rates.csv"
)

PROVENANCE_PATH = (
    OUTPUT_ROOT
    / "final_test_provenance.json"
)

COMPLETE_MARKER = (
    OUTPUT_ROOT
    / "COMPLETE"
)


# ------------------------------------------------------------
# 2. Frozen method definitions
# ------------------------------------------------------------
MEAN_FIELDS = {
    "C0":
        "c0_mean",

    "U1":
        "u1_mean",

    "U2a":
        "u2a_mean",

    "U2b":
        "u2b_mean",
}

UNCERTAINTY_VARIANCE_FIELDS = {
    "U1":
        "u1_variance",

    "U2a":
        "u2a_between_model_variance",

    "U2b":
        "u2b_total_predictive_variance",
}

CALIBRATION_METHOD_KEYS = {
    "C0":
        "C0",

    "U1":
        "MC",

    "U2a":
        "PE",

    "U2b":
        "DE",
}

TASK_P_METHODS = [
    "C0",
    "U1",
    "U2a",
    "U2b",
]

TASK_R_METHODS = [
    "U1",
    "U2a",
    "U2b",
]

TASK_E_METHODS = [
    "U1",
    "U2a",
    "U2b",
]

EXPECTED_VOLUMES = 40
EXPECTED_SLICES = 636
EXPECTED_BATCHES = 160
EXPECTED_PIXELS = 168_048_640

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260725
CONFIDENCE_LEVEL = 0.95


# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------
def git_output(*arguments):
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO,
        text=True,
    ).strip()


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(
                4 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def scalar(array):
    return np.asarray(array).item()


def atomic_json_write(
    destination,
    payload,
):
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        destination,
    )


def atomic_csv_write(
    destination,
    dataframe,
):
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    dataframe.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        destination,
    )


def weighted_mean(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    denominator = float(
        weights.sum(
            dtype=np.float64
        )
    )

    assert denominator > 0.0

    return float(
        np.sum(
            values * weights,
            dtype=np.float64,
        )
        / denominator
    )


def weighted_mae(
    target,
    prediction,
    weights,
):
    return weighted_mean(
        np.abs(
            np.asarray(target)
            - np.asarray(prediction)
        ),
        weights,
    )


def weighted_rmse(
    target,
    prediction,
    weights,
):
    mse = weighted_mean(
        (
            np.asarray(target)
            - np.asarray(prediction)
        )
        ** 2,
        weights,
    )

    return float(
        np.sqrt(mse)
    )


def weighted_bias(
    target,
    prediction,
    weights,
):
    return weighted_mean(
        np.asarray(prediction)
        - np.asarray(target),
        weights,
    )


def weighted_average_precision(
    labels,
    scores,
    weights,
):
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    positive_weight = float(
        weights[
            labels == 1
        ].sum(
            dtype=np.float64
        )
    )

    negative_weight = float(
        weights[
            labels == 0
        ].sum(
            dtype=np.float64
        )
    )

    if positive_weight <= 0.0:
        return 0.0

    if negative_weight <= 0.0:
        return 1.0

    return float(
        average_precision_score(
            labels,
            scores,
            sample_weight=weights,
        )
    )


def weighted_roc_auc(
    labels,
    scores,
    weights,
):
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )

    if np.unique(labels).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            labels,
            scores,
            sample_weight=np.asarray(
                weights,
                dtype=np.float64,
            ),
        )
    )


def safe_filename(
    value,
):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def percentile_interval(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

    return (
        float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
    )


# ------------------------------------------------------------
# 4. Frozen-state guards
# ------------------------------------------------------------
assert REPO.is_dir()

assert git_output(
    "rev-parse",
    "HEAD",
) == EXPECTED_COMMIT

assert git_output(
    "status",
    "--porcelain",
) == ""

assert (
    NEURAL_ROOT
    / "COMPLETE"
).is_file()

assert (
    UHOLD_ROOT
    / "COMPLETE"
).is_file()

assert CALIBRATION_RULES_PATH.is_file()
assert AUSE_CONTRACT_PATH.is_file()

neural_paths = sorted(
    NEURAL_CHUNK_ROOT.glob(
        "dtest_batch_*.npz"
    )
)

uhold_paths = sorted(
    UHOLD_CHUNK_ROOT.glob(
        "dtest_uhold_batch_*.npz"
    )
)

assert len(neural_paths) == EXPECTED_BATCHES
assert len(uhold_paths) == EXPECTED_BATCHES

calibration = json.loads(
    CALIBRATION_RULES_PATH.read_text(
        encoding="utf-8"
    )
)

ause_contract = json.loads(
    AUSE_CONTRACT_PATH.read_text(
        encoding="utf-8"
    )
)

assert calibration[
    "artifact_sha256"
] == (
    "f39b8274006328bd7a3b3dd74f91cb496957c25d705316ac89a9bcaebb4058ed"
)

TAU_HOLD = float(
    calibration[
        "task_R"
    ][
        "tau_hold"
    ]
)

assert TAU_HOLD == 1.7366089820861816

assert ause_contract[
    "endpoint_name"
] == "per-volume normalized AUSE"

assert ause_contract[
    "bootstrap"
][
    "replicates"
] == BOOTSTRAP_REPLICATES

assert ause_contract[
    "bootstrap"
][
    "seed"
] == BOOTSTRAP_SEED

assert ause_contract[
    "mean_calibration"
] == "none; raw method means are mandatory"


# ------------------------------------------------------------
# 5. Prevent duplicate successful execution
# ------------------------------------------------------------
if COMPLETE_MARKER.is_file():
    print(
        "Exp008 primary evaluation has already "
        "completed successfully."
    )

    print(
        f"Results: {OUTPUT_ROOT}"
    )

    raise SystemExit(
        "No rerun performed."
    )


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

VOLUME_RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# 6. Build batch-to-volume index using metadata only
# ------------------------------------------------------------
print("=" * 118)
print("P2-EXP008 — ACTUAL PRIMARY FINAL-TEST IMPLEMENTATION")
print("=" * 118)
print("Stage 1/4: indexing frozen D_test chunks...")

volume_to_batches = defaultdict(
    list
)

all_sample_ids = set()
all_volume_ids = set()

total_pixels_from_chunks = 0

for batch_position, (
    neural_path,
    uhold_path,
) in enumerate(
    zip(
        neural_paths,
        uhold_paths,
    )
):
    with np.load(
        neural_path,
        allow_pickle=False,
    ) as neural:

        batch_index = int(
            scalar(
                neural[
                    "batch_index"
                ]
            )
        )

        sample_ids = [
            str(value)
            for value in neural[
                "sample_ids"
            ].tolist()
        ]

        volume_ids = [
            str(value)
            for value in neural[
                "volume_ids"
            ].tolist()
        ]

        pixel_count = int(
            neural[
                "support_weight"
            ].size
        )

    with np.load(
        uhold_path,
        allow_pickle=False,
    ) as target_archive:

        target_batch_index = int(
            scalar(
                target_archive[
                    "batch_index"
                ]
            )
        )

        aligned_sample_ids = [
            str(value)
            for value in target_archive[
                "sample_ids"
            ].tolist()
        ]

        aligned_volume_ids = [
            str(value)
            for value in target_archive[
                "volume_ids"
            ].tolist()
        ]

        aligned_pixel_count = int(
            target_archive[
                "target_u_hold"
            ].size
        )

    assert batch_index == batch_position
    assert target_batch_index == batch_index

    assert sample_ids == aligned_sample_ids
    assert volume_ids == aligned_volume_ids

    assert pixel_count == aligned_pixel_count

    assert len(sample_ids) == len(volume_ids)

    for sample_id in sample_ids:
        assert sample_id not in all_sample_ids
        all_sample_ids.add(sample_id)

    for volume_id in sorted(
        set(volume_ids)
    ):
        volume_to_batches[
            volume_id
        ].append(
            batch_index
        )

        all_volume_ids.add(
            volume_id
        )

    total_pixels_from_chunks += pixel_count


assert len(all_sample_ids) == EXPECTED_SLICES
assert len(all_volume_ids) == EXPECTED_VOLUMES
assert total_pixels_from_chunks == EXPECTED_PIXELS

volume_ids_sorted = sorted(
    all_volume_ids
)

print(
    f"  volumes indexed:                     "
    f"{len(volume_ids_sorted)}"
)

print(
    f"  slices indexed:                      "
    f"{len(all_sample_ids)}"
)

print(
    f"  aligned pixels indexed:              "
    f"{total_pixels_from_chunks:,}"
)


# ------------------------------------------------------------
# 7. Evaluate one volume at a time
# ------------------------------------------------------------
print()
print(
    "Stage 2/4: calculating per-volume "
    "Task P, Task R and Task E endpoints..."
)

evaluation_start = datetime.now(
    timezone.utc
)

for volume_position, volume_id in enumerate(
    volume_ids_sorted,
    start=1,
):
    volume_record_path = (
        VOLUME_RESULT_ROOT
        / (
            f"volume_"
            f"{volume_position:03d}_"
            f"{safe_filename(volume_id)}.json"
        )
    )

    # Restart-safe validation.
    if volume_record_path.is_file():
        try:
            existing = json.loads(
                volume_record_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                existing.get("status") == "PASS"
                and existing.get("volume_id") == volume_id
                and set(
                    existing.get(
                        "task_p",
                        {},
                    ).keys()
                ) == set(TASK_P_METHODS)
                and set(
                    existing.get(
                        "task_r",
                        {},
                    ).keys()
                ) == set(TASK_R_METHODS)
                and set(
                    existing.get(
                        "task_e",
                        {},
                    ).keys()
                ) == set(TASK_E_METHODS)
            ):
                print(
                    f"  volume "
                    f"{volume_position:02d}/"
                    f"{EXPECTED_VOLUMES}: "
                    f"{volume_id} — reused"
                )

                continue

        except Exception:
            pass

    print(
        f"\n  volume "
        f"{volume_position:02d}/"
        f"{EXPECTED_VOLUMES}: "
        f"{volume_id}"
    )

    collected = {
        "target_u_hold":
            [],

        "support_weight":
            [],
    }

    for field in MEAN_FIELDS.values():
        collected[field] = []

    for field in (
        UNCERTAINTY_VARIANCE_FIELDS.values()
    ):
        collected[field] = []

    volume_sample_ids = set()

    for batch_index in volume_to_batches[
        volume_id
    ]:
        neural_path = neural_paths[
            batch_index
        ]

        uhold_path = uhold_paths[
            batch_index
        ]

        with np.load(
            neural_path,
            allow_pickle=False,
        ) as neural, np.load(
            uhold_path,
            allow_pickle=False,
        ) as target_archive:

            sample_ids = [
                str(value)
                for value in neural[
                    "sample_ids"
                ].tolist()
            ]

            volume_ids = [
                str(value)
                for value in neural[
                    "volume_ids"
                ].tolist()
            ]

            aligned_sample_ids = [
                str(value)
                for value in target_archive[
                    "sample_ids"
                ].tolist()
            ]

            aligned_volume_ids = [
                str(value)
                for value in target_archive[
                    "volume_ids"
                ].tolist()
            ]

            assert sample_ids == aligned_sample_ids
            assert volume_ids == aligned_volume_ids

            local_sample_indices = [
                local_index
                for local_index, local_volume_id
                in enumerate(volume_ids)
                if local_volume_id == volume_id
            ]

            assert local_sample_indices

            for local_index in local_sample_indices:
                volume_sample_ids.add(
                    sample_ids[
                        local_index
                    ]
                )

            pixel_sample_index = np.asarray(
                neural[
                    "pixel_sample_index"
                ],
                dtype=np.int64,
            )

            select = np.isin(
                pixel_sample_index,
                np.asarray(
                    local_sample_indices,
                    dtype=np.int64,
                ),
            )

            assert select.any()

            collected[
                "target_u_hold"
            ].append(
                np.asarray(
                    target_archive[
                        "target_u_hold"
                    ][select],
                    dtype=np.float32,
                )
            )

            collected[
                "support_weight"
            ].append(
                np.asarray(
                    neural[
                        "support_weight"
                    ][select],
                    dtype=np.float32,
                )
            )

            for field in MEAN_FIELDS.values():
                collected[field].append(
                    np.asarray(
                        neural[field][select],
                        dtype=np.float32,
                    )
                )

            for field in (
                UNCERTAINTY_VARIANCE_FIELDS.values()
            ):
                collected[field].append(
                    np.asarray(
                        neural[field][select],
                        dtype=np.float32,
                    )
                )

    arrays = {
        key:
            np.concatenate(
                pieces,
                axis=0,
            )

        for key, pieces
        in collected.items()
    }

    target = arrays[
        "target_u_hold"
    ]

    weights = arrays[
        "support_weight"
    ]

    assert target.ndim == 1
    assert weights.ndim == 1
    assert target.size == weights.size

    assert np.isfinite(
        target
    ).all()

    assert np.isfinite(
        weights
    ).all()

    positive_support = (
        weights > 0.0
    )

    assert positive_support.any()

    # Evaluation population is M_soft > 0.
    if not positive_support.all():
        arrays = {
            key:
                value[
                    positive_support
                ]

            for key, value
            in arrays.items()
        }

        target = arrays[
            "target_u_hold"
        ]

        weights = arrays[
            "support_weight"
        ]

    assert (
        weights > 0.0
    ).all()

    for key, value in arrays.items():
        assert value.size == target.size
        assert np.isfinite(value).all(), (
            f"Nonfinite values in {key}, "
            f"volume {volume_id}"
        )

    pixel_count = int(
        target.size
    )

    weight_sum = float(
        weights.sum(
            dtype=np.float64
        )
    )

    print(
        f"    supported pixels:                  "
        f"{pixel_count:,}"
    )

    print(
        f"    slices:                           "
        f"{len(volume_sample_ids)}"
    )


    # --------------------------------------------------------
    # Task P — weighted prediction quality
    # --------------------------------------------------------
    task_p = {}

    for method in TASK_P_METHODS:
        mean_field = MEAN_FIELDS[
            method
        ]

        prediction = arrays[
            mean_field
        ]

        raw_mae = weighted_mae(
            target,
            prediction,
            weights,
        )

        raw_rmse = weighted_rmse(
            target,
            prediction,
            weights,
        )

        raw_bias = weighted_bias(
            target,
            prediction,
            weights,
        )

        calibration_key = (
            CALIBRATION_METHOD_KEYS[
                method
            ]
        )

        affine_parameters = calibration[
            "task_P_affine_diagnostics"
        ][
            "parameters"
        ][
            calibration_key
        ]

        slope = float(
            affine_parameters[
                "slope"
            ]
        )

        intercept = float(
            affine_parameters[
                "intercept"
            ]
        )

        affine_prediction = (
            slope * prediction
            + intercept
        )

        task_p[
            method
        ] = {
            "raw_mae":
                raw_mae,

            "raw_rmse":
                raw_rmse,

            "raw_bias":
                raw_bias,

            "affine_mae_secondary":
                weighted_mae(
                    target,
                    affine_prediction,
                    weights,
                ),

            "affine_rmse_secondary":
                weighted_rmse(
                    target,
                    affine_prediction,
                    weights,
                ),

            "affine_slope":
                slope,

            "affine_intercept":
                intercept,
        }


    # --------------------------------------------------------
    # Task R — frozen high-risk threshold
    # --------------------------------------------------------
    high_risk = (
        target >= TAU_HOLD
    ).astype(
        np.int8
    )

    high_risk_prevalence = weighted_mean(
        high_risk,
        weights,
    )

    task_r = {}

    for method in TASK_R_METHODS:
        variance = arrays[
            UNCERTAINTY_VARIANCE_FIELDS[
                method
            ]
        ]

        assert (
            variance >= -1.0e-8
        ).all()

        uncertainty_score = np.sqrt(
            np.maximum(
                variance,
                0.0,
            )
        ).astype(
            np.float32,
            copy=False,
        )

        task_r[
            method
        ] = {
            "auprc":
                weighted_average_precision(
                    high_risk,
                    uncertainty_score,
                    weights,
                ),

            "auroc":
                weighted_roc_auc(
                    high_risk,
                    uncertainty_score,
                    weights,
                ),
        }


    # --------------------------------------------------------
    # Task E — frozen normalized AUSE
    # --------------------------------------------------------
    task_e = {}

    for method in TASK_E_METHODS:
        prediction = arrays[
            MEAN_FIELDS[
                method
            ]
        ]

        errors = np.abs(
            target
            - prediction
        ).astype(
            np.float32,
            copy=False,
        )

        uncertainty = np.maximum(
            arrays[
                UNCERTAINTY_VARIANCE_FIELDS[
                    method
                ]
            ],
            0.0,
        ).astype(
            np.float32,
            copy=False,
        )

        print(
            f"    Task E AUSE:                      "
            f"{method}...",
            end="",
            flush=True,
        )

        ause_result = normalized_ause(
            errors,
            uncertainty,
            weights=weights,
        )

        task_e[
            method
        ] = {
            "normalized_ause":
                float(
                    ause_result.normalized_ause
                ),

            "raw_ause":
                float(
                    ause_result.raw_ause
                ),

            "random_oracle_area":
                float(
                    ause_result.random_oracle_area
                ),

            "weighted_mean_error":
                float(
                    ause_result.weighted_mean_error
                ),

            "positive_weight_observations":
                int(
                    ause_result.positive_weight_observations
                ),

            "total_evaluation_weight":
                float(
                    ause_result.total_evaluation_weight
                ),
        }

        print(
            f" "
            f"{ause_result.normalized_ause:.6f}"
        )


    # --------------------------------------------------------
    # Persist completed volume record
    # --------------------------------------------------------
    record = {
        "schema_version":
            "paper2-exp008-primary-volume-v1.0",

        "status":
            "PASS",

        "volume_id":
            volume_id,

        "volume_position":
            volume_position,

        "sample_count":
            len(volume_sample_ids),

        "supported_pixels":
            pixel_count,

        "support_weight_sum":
            weight_sum,

        "tau_hold":
            TAU_HOLD,

        "high_risk_prevalence":
            high_risk_prevalence,

        "task_p":
            task_p,

        "task_r":
            task_r,

        "task_e":
            task_e,

        "parameter_fitting_performed":
            False,

        "threshold_selection_performed":
            False,

        "model_inference_performed":
            False,

        "completed_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    atomic_json_write(
        volume_record_path,
        record,
    )

    print(
        "    saved:                             "
        f"{volume_record_path.name}"
    )

    # Release large volume arrays.
    del arrays
    del target
    del weights
    del task_p
    del task_r
    del task_e


# ------------------------------------------------------------
# 8. Load all completed volume records
# ------------------------------------------------------------
print()
print(
    "Stage 3/4: aggregating 40 completed "
    "volume records..."
)

volume_records = []

for volume_position, volume_id in enumerate(
    volume_ids_sorted,
    start=1,
):
    record_path = (
        VOLUME_RESULT_ROOT
        / (
            f"volume_"
            f"{volume_position:03d}_"
            f"{safe_filename(volume_id)}.json"
        )
    )

    assert record_path.is_file()

    record = json.loads(
        record_path.read_text(
            encoding="utf-8"
        )
    )

    assert record[
        "status"
    ] == "PASS"

    assert record[
        "volume_id"
    ] == volume_id

    volume_records.append(
        record
    )


assert len(volume_records) == EXPECTED_VOLUMES

assert sum(
    int(record["supported_pixels"])
    for record in volume_records
) == EXPECTED_PIXELS


# ------------------------------------------------------------
# 9. Flatten per-volume tables
# ------------------------------------------------------------
task_p_rows = []
task_r_rows = []
task_e_rows = []

for record in volume_records:
    common = {
        "volume_id":
            record[
                "volume_id"
            ],

        "sample_count":
            record[
                "sample_count"
            ],

        "supported_pixels":
            record[
                "supported_pixels"
            ],

        "support_weight_sum":
            record[
                "support_weight_sum"
            ],
    }

    for method, values in record[
        "task_p"
    ].items():
        task_p_rows.append(
            {
                **common,
                "method":
                    method,
                **values,
            }
        )

    for method, values in record[
        "task_r"
    ].items():
        task_r_rows.append(
            {
                **common,

                "method":
                    method,

                "tau_hold":
                    record[
                        "tau_hold"
                    ],

                "high_risk_prevalence":
                    record[
                        "high_risk_prevalence"
                    ],

                **values,
            }
        )

    for method, values in record[
        "task_e"
    ].items():
        task_e_rows.append(
            {
                **common,
                "method":
                    method,
                **values,
            }
        )


task_p_dataframe = pd.DataFrame(
    task_p_rows
)

task_r_dataframe = pd.DataFrame(
    task_r_rows
)

task_e_dataframe = pd.DataFrame(
    task_e_rows
)

assert len(
    task_p_dataframe
) == (
    EXPECTED_VOLUMES
    * len(TASK_P_METHODS)
)

assert len(
    task_r_dataframe
) == (
    EXPECTED_VOLUMES
    * len(TASK_R_METHODS)
)

assert len(
    task_e_dataframe
) == (
    EXPECTED_VOLUMES
    * len(TASK_E_METHODS)
)

atomic_csv_write(
    TASK_P_PATH,
    task_p_dataframe,
)

atomic_csv_write(
    TASK_R_PATH,
    task_r_dataframe,
)

atomic_csv_write(
    TASK_E_PATH,
    task_e_dataframe,
)


# ------------------------------------------------------------
# 10. Paired volume bootstrap
# ------------------------------------------------------------
print(
    "Stage 4/4: running 1,000 paired "
    "volume bootstrap replicates..."
)

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_indices = rng.integers(
    low=0,
    high=EXPECTED_VOLUMES,
    size=(
        BOOTSTRAP_REPLICATES,
        EXPECTED_VOLUMES,
    ),
)

summary_rows = []
pairwise_rows = []
win_rate_rows = []


def summarize_endpoint(
    task,
    endpoint,
    dataframe,
    methods,
    direction,
):
    pivot = (
        dataframe.pivot(
            index="volume_id",
            columns="method",
            values=endpoint,
        )
        .reindex(
            index=volume_ids_sorted,
            columns=methods,
        )
    )

    values = pivot.to_numpy(
        dtype=np.float64
    )

    assert values.shape == (
        EXPECTED_VOLUMES,
        len(methods),
    )

    # Primary endpoints must be finite for all 40 volumes.
    assert np.isfinite(values).all(), (
        f"Nonfinite primary endpoint: "
        f"{task}/{endpoint}"
    )

    bootstrap_means = values[
        bootstrap_indices
    ].mean(
        axis=1
    )

    for method_index, method in enumerate(
        methods
    ):
        observed = float(
            values[
                :,
                method_index
            ].mean()
        )

        ci_low, ci_high = percentile_interval(
            bootstrap_means[
                :,
                method_index
            ]
        )

        summary_rows.append(
            {
                "task":
                    task,

                "endpoint":
                    endpoint,

                "method":
                    method,

                "n_volumes":
                    EXPECTED_VOLUMES,

                "estimate":
                    observed,

                "ci_low":
                    ci_low,

                "ci_high":
                    ci_high,

                "confidence_level":
                    CONFIDENCE_LEVEL,

                "bootstrap_replicates":
                    BOOTSTRAP_REPLICATES,

                "bootstrap_seed":
                    BOOTSTRAP_SEED,

                "direction":
                    direction,
            }
        )

    for first_index, second_index in combinations(
        range(len(methods)),
        2,
    ):
        first_method = methods[
            first_index
        ]

        second_method = methods[
            second_index
        ]

        observed_difference = float(
            (
                values[
                    :,
                    first_index
                ]
                - values[
                    :,
                    second_index
                ]
            ).mean()
        )

        bootstrap_difference = (
            bootstrap_means[
                :,
                first_index
            ]
            - bootstrap_means[
                :,
                second_index
            ]
        )

        ci_low, ci_high = percentile_interval(
            bootstrap_difference
        )

        if direction == "lower_is_better":
            probability_first_better = float(
                np.mean(
                    bootstrap_difference
                    < 0.0
                )
            )

            volume_win_rate = float(
                np.mean(
                    values[
                        :,
                        first_index
                    ]
                    < values[
                        :,
                        second_index
                    ]
                )
            )

        else:
            probability_first_better = float(
                np.mean(
                    bootstrap_difference
                    > 0.0
                )
            )

            volume_win_rate = float(
                np.mean(
                    values[
                        :,
                        first_index
                    ]
                    > values[
                        :,
                        second_index
                    ]
                )
            )

        tie_rate = float(
            np.mean(
                values[
                    :,
                    first_index
                ]
                == values[
                    :,
                    second_index
                ]
            )
        )

        pairwise_rows.append(
            {
                "task":
                    task,

                "endpoint":
                    endpoint,

                "first_method":
                    first_method,

                "second_method":
                    second_method,

                "difference_definition":
                    "first_minus_second",

                "observed_mean_difference":
                    observed_difference,

                "ci_low":
                    ci_low,

                "ci_high":
                    ci_high,

                "direction":
                    direction,

                "bootstrap_probability_first_better":
                    probability_first_better,

                "bootstrap_replicates":
                    BOOTSTRAP_REPLICATES,

                "bootstrap_seed":
                    BOOTSTRAP_SEED,
            }
        )

        win_rate_rows.append(
            {
                "task":
                    task,

                "endpoint":
                    endpoint,

                "first_method":
                    first_method,

                "second_method":
                    second_method,

                "direction":
                    direction,

                "first_method_volume_win_rate":
                    volume_win_rate,

                "volume_tie_rate":
                    tie_rate,

                "n_volumes":
                    EXPECTED_VOLUMES,
            }
        )


summarize_endpoint(
    task="P",
    endpoint="raw_mae",
    dataframe=task_p_dataframe,
    methods=TASK_P_METHODS,
    direction="lower_is_better",
)

summarize_endpoint(
    task="R",
    endpoint="auprc",
    dataframe=task_r_dataframe,
    methods=TASK_R_METHODS,
    direction="higher_is_better",
)

summarize_endpoint(
    task="E",
    endpoint="normalized_ause",
    dataframe=task_e_dataframe,
    methods=TASK_E_METHODS,
    direction="lower_is_better",
)


summary_dataframe = pd.DataFrame(
    summary_rows
)

pairwise_dataframe = pd.DataFrame(
    pairwise_rows
)

win_rate_dataframe = pd.DataFrame(
    win_rate_rows
)

atomic_csv_write(
    SUMMARY_PATH,
    summary_dataframe,
)

atomic_csv_write(
    PAIRWISE_PATH,
    pairwise_dataframe,
)

atomic_csv_write(
    WIN_RATE_PATH,
    win_rate_dataframe,
)


# ------------------------------------------------------------
# 11. Save provenance
# ------------------------------------------------------------
evaluation_end = datetime.now(
    timezone.utc
)

provenance = {
    "schema_version":
        "paper2-exp008-primary-final-test-v1.0",

    "status":
        "PASS",

    "experiment_id":
        "P2-Exp008",

    "repository_commit":
        EXPECTED_COMMIT,

    "locked_evaluation_cohort":
        "D_test",

    "D_test_description":
        "locked reused evaluation cohort",

    "D_test_volumes":
        EXPECTED_VOLUMES,

    "D_test_slices":
        EXPECTED_SLICES,

    "D_test_supported_pixels":
        EXPECTED_PIXELS,

    "task_P_primary_endpoint":
        "unweighted mean of 40 per-volume weighted MAE values",

    "task_R_primary_endpoint":
        "unweighted mean of 40 per-volume weighted AUPRC values",

    "task_R_tau_hold":
        TAU_HOLD,

    "task_E_primary_endpoint":
        "unweighted mean of 40 per-volume normalized AUSE values",

    "task_E_uncertainty_fields":
        UNCERTAINTY_VARIANCE_FIELDS,

    "task_E_mean_fields":
        {
            method:
                MEAN_FIELDS[method]
            for method in TASK_E_METHODS
        },

    "bootstrap": {
        "scheme":
            "paired volume bootstrap",

        "replicates":
            BOOTSTRAP_REPLICATES,

        "confidence_level":
            CONFIDENCE_LEVEL,

        "seed":
            BOOTSTRAP_SEED,
    },

    "calibration_artifact_sha256":
        calibration[
            "artifact_sha256"
        ],

    "calibration_rules_file_sha256":
        sha256_file(
            CALIBRATION_RULES_PATH
        ),

    "ause_contract_file_sha256":
        sha256_file(
            AUSE_CONTRACT_PATH
        ),

    "threshold_selection_performed":
        False,

    "parameter_fitting_performed":
        False,

    "model_training_performed":
        False,

    "model_selection_performed":
        False,

    "new_model_inference_performed":
        False,

    "volume_records":
        EXPECTED_VOLUMES,

    "evaluation_start_utc":
        evaluation_start.isoformat(),

    "evaluation_end_utc":
        evaluation_end.isoformat(),

    "output_files": {
        "task_p_per_volume":
            str(TASK_P_PATH),

        "task_r_per_volume":
            str(TASK_R_PATH),

        "task_e_per_volume":
            str(TASK_E_PATH),

        "primary_summary":
            str(SUMMARY_PATH),

        "pairwise_differences":
            str(PAIRWISE_PATH),

        "volume_win_rates":
            str(WIN_RATE_PATH),
    },
}

atomic_json_write(
    PROVENANCE_PATH,
    provenance,
)


# COMPLETE written only after all calculations and writes pass.
COMPLETE_MARKER.write_text(
    json.dumps(
        {
            "status":
                "PASS",

            "experiment_id":
                "P2-Exp008",

            "D_test_volumes":
                EXPECTED_VOLUMES,

            "bootstrap_replicates":
                BOOTSTRAP_REPLICATES,

            "completed_utc":
                evaluation_end.isoformat(),
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# 12. Final report
# ------------------------------------------------------------
print()
print("=" * 118)
print("P2-EXP008 — PRIMARY FINAL-TEST RESULTS")
print("=" * 118)

for task in (
    "P",
    "R",
    "E",
):
    task_rows = summary_dataframe[
        summary_dataframe[
            "task"
        ] == task
    ]

    endpoint = task_rows[
        "endpoint"
    ].iloc[0]

    direction = task_rows[
        "direction"
    ].iloc[0]

    print()
    print(
        f"TASK {task}: {endpoint} "
        f"({direction})"
    )

    print("-" * 118)

    for row in task_rows.itertuples(
        index=False
    ):
        print(
            f"{row.method:<5s}  "
            f"{row.estimate:.6f}  "
            f"[{row.ci_low:.6f}, "
            f"{row.ci_high:.6f}]"
        )


print()
print("OUTPUT FILES")
print("-" * 118)

print(f"Task P per volume:                     {TASK_P_PATH}")
print(f"Task R per volume:                     {TASK_R_PATH}")
print(f"Task E per volume:                     {TASK_E_PATH}")
print(f"Primary summary:                       {SUMMARY_PATH}")
print(f"Paired differences:                    {PAIRWISE_PATH}")
print(f"Volume win rates:                      {WIN_RATE_PATH}")
print(f"Provenance:                            {PROVENANCE_PATH}")
print(f"COMPLETE marker:                       {COMPLETE_MARKER}")

print()
print("GOVERNANCE")
print("-" * 118)

print("Threshold selection performed:          NO")
print("Parameter fitting performed:            NO")
print("Model training performed:               NO")
print("Model selection performed:              NO")
print("New model inference performed:          NO")
print("Paired bootstrap replicates:             1,000")
print("Bootstrap seed:                         20260725")

print("=" * 118)
print("\nCELL 46 STATUS: PASS")
