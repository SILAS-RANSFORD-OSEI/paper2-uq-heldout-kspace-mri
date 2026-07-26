# ============================================================
# EXP009 — RETROSPECTIVE TASK R CONTROL COMPLETION
#
# Scientific status:
# - retrospective completion of prespecified controls
# - not a new confirmatory test
#
# Added scores:
# - C0 deterministic mean
# - B1 normalized |x_hat|
# - B2 normalized |x0|
# - B3 normalized |x_hat - x0|
# - B4 finite-difference gradient magnitude of B1
# - B5 analytical PSF
# - B6 normalized q_PSF / gain envelope
#
# Frozen scores reused without recalculation:
# - U1
# - U2a
# - U2b
#
# No:
# - training
# - model inference
# - target regeneration
# - threshold fitting
# - sign flipping
# - outcome-dependent transformation
# ============================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import gc
import hashlib
import inspect
import itertools
import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ------------------------------------------------------------
# 1. Frozen paths and constants
# ------------------------------------------------------------
REPO = Path(
    "/content/paper2-uq-heldout-kspace-mri"
).resolve()

PAPER1_REPO = Path(
    "/content/fourway-ssdu-reliability-mri-v2"
).resolve()

PAPER1_TEST_CACHE = Path(
    "/content/drive/MyDrive/FOUR WAY MRI RESEARCH/"
    "outputs/exp007_train_reliability_cnn_v2_full/"
    "exp006_reliability_cache_full/cache/test"
).resolve()

PAPER2_RELEASE = Path(
    "/content/drive/MyDrive/Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp006_dtest_final/release_pending"
).resolve()

NEURAL_DIR = (
    PAPER2_RELEASE
    / "neural_outputs"
    / "chunks"
)

ALIGNED_DIR = (
    PAPER2_RELEASE
    / "uhold_outputs"
    / "aligned_chunks"
)

FROZEN_PRIMARY_ROOT = Path(
    "/content/drive/MyDrive/Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp008_final_test_primary/"
    "commit58020824270a"
).resolve()

FROZEN_TASK_R_PATH = (
    FROZEN_PRIMARY_ROOT
    / "task_r_per_volume.csv"
)

FROZEN_PRIMARY_SUMMARY_PATH = (
    FROZEN_PRIMARY_ROOT
    / "primary_endpoint_summary.csv"
)

FROZEN_PAIRWISE_PATH = (
    FROZEN_PRIMARY_ROOT
    / "paired_method_differences.csv"
)

FROZEN_WIN_RATE_PATH = (
    FROZEN_PRIMARY_ROOT
    / "volume_level_win_rates.csv"
)

EXP008_SOURCE = (
    REPO
    / "scripts"
    / "exp008_run_final_test.py"
)

PAPER1_GRADIENT_SOURCE = (
    PAPER1_REPO
    / "scripts"
    / "exp009_holdout_verification.py"
)

OUTPUT_DIR = (
    REPO
    / "results"
    / "exp009_task_r_control_completion"
)

EXPECTED_BRANCH = (
    "retrospective-task-r-controls-v1.0"
)

EXPECTED_PARENT_COMMIT = (
    "ccd3be856a7150a8c513efa3713954c90fb783d1"
)

TAU_HOLD = 1.7366089820861816

EXPECTED_VOLUMES = 40
EXPECTED_SLICES = 636
EXPECTED_BATCHES = 160
EXPECTED_PIXELS = 168_048_640

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260725
CONFIDENCE_LEVEL = 0.95

METHOD_ORDER = [
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
]

ADDED_METHODS = [
    "C0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
]

FROZEN_REUSED_METHODS = [
    "U1",
    "U2a",
    "U2b",
]

METHOD_DEFINITIONS = {
    "C0": (
        "Frozen deterministic Paper 2 C0 mean prediction "
        "stored as c0_mean."
    ),
    "U1": (
        "Frozen MC-dropout variance; existing Exp008 Task R "
        "record reused without recalculation."
    ),
    "U2a": (
        "Frozen point-ensemble between-model variance; existing "
        "Exp008 Task R record reused without recalculation."
    ),
    "U2b": (
        "Frozen probabilistic-ensemble total predictive variance; "
        "existing Exp008 Task R record reused without recalculation."
    ),
    "B1": (
        "Paper 1 cache channel 0: normalized reconstruction "
        "magnitude |x_hat|."
    ),
    "B2": (
        "Paper 1 cache channel 1: normalized zero-filled "
        "magnitude |x0|."
    ),
    "B3": (
        "Paper 1 cache channel 2: normalized intervention "
        "magnitude |x_hat - x0|."
    ),
    "B4": (
        "Finite-difference gradient magnitude of B1, using the "
        "committed Paper 1 gradient_magnitude_2d implementation."
    ),
    "B5": (
        "Paper 1 cache channel 4: analytical Cartesian PSF."
    ),
    "B6": (
        "Paper 1 cache channel 5: normalized sensitivity-aware "
        "q_PSF / gain envelope."
    ),
}


# ------------------------------------------------------------
# 2. Utilities
# ------------------------------------------------------------
def git_output(
    repository: Path,
    *args: str,
) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repository,
        text=True,
    ).strip()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json_write(
    path: Path,
    payload: dict,
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_csv_write(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def extract_functions(
    source_path: Path,
    requested_names: list[str],
    global_namespace: dict,
) -> dict:
    source_text = source_path.read_text(
        encoding="utf-8",
        errors="strict",
    )

    tree = ast.parse(
        source_text,
        filename=str(source_path),
    )

    selected_nodes = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
        and node.name in requested_names
    ]

    found_names = {
        node.name
        for node in selected_nodes
    }

    missing_names = sorted(
        set(requested_names)
        - found_names
    )

    if missing_names:
        raise RuntimeError(
            f"Functions missing from {source_path}: "
            f"{missing_names}"
        )

    selected_nodes = sorted(
        selected_nodes,
        key=lambda node: node.lineno,
    )

    module = ast.Module(
        body=selected_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    namespace = dict(
        global_namespace
    )

    exec(
        compile(
            module,
            filename=str(source_path),
            mode="exec",
        ),
        namespace,
    )

    return {
        name: namespace[name]
        for name in requested_names
    }


def read_batch_index(
    path: Path,
) -> int:
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return int(
            np.asarray(
                data["batch_index"]
            ).item()
        )


def percentile_bounds(
    function,
    values: np.ndarray,
) -> tuple[float, float]:
    values_array = np.asarray(
        values,
        dtype=np.float64,
    )

    function_signature = inspect.signature(
        function
    )

    if len(
        function_signature.parameters
    ) != 1:
        raise RuntimeError(
            "Frozen percentile_interval must accept "
            f"exactly one argument; found "
            f"{function_signature}."
        )

    result = function(
        values_array
    )

    if isinstance(
        result,
        dict,
    ):
        possible_low = [
            "ci_low",
            "low",
            "lower",
        ]

        possible_high = [
            "ci_high",
            "high",
            "upper",
        ]

        low_key = next(
            key
            for key in possible_low
            if key in result
        )

        high_key = next(
            key
            for key in possible_high
            if key in result
        )

        return (
            float(result[low_key]),
            float(result[high_key]),
        )

    if isinstance(
        result,
        (
            tuple,
            list,
            np.ndarray,
        ),
    ) and len(result) == 2:
        return (
            float(result[0]),
            float(result[1]),
        )

    raise TypeError(
        "Unexpected percentile_interval return value: "
        f"{result!r}"
    )


# ------------------------------------------------------------
# 3. Repository and artifact guards
# ------------------------------------------------------------
for required_path in [
    REPO,
    PAPER1_REPO,
    PAPER1_TEST_CACHE,
    PAPER2_RELEASE,
    NEURAL_DIR,
    ALIGNED_DIR,
    FROZEN_PRIMARY_ROOT,
    FROZEN_TASK_R_PATH,
    FROZEN_PRIMARY_SUMMARY_PATH,
    FROZEN_PAIRWISE_PATH,
    FROZEN_WIN_RATE_PATH,
    EXP008_SOURCE,
    PAPER1_GRADIENT_SOURCE,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Missing required artifact:\n{required_path}"
        )

if git_output(
    REPO,
    "branch",
    "--show-current",
) != EXPECTED_BRANCH:
    raise RuntimeError(
        "Unexpected Paper 2 branch."
    )

if git_output(
    REPO,
    "status",
    "--porcelain",
) != "":
    raise RuntimeError(
        "Paper 2 repository must be clean before execution."
    )

execution_commit = git_output(
    REPO,
    "rev-parse",
    "HEAD",
)

if execution_commit == EXPECTED_PARENT_COMMIT:
    raise RuntimeError(
        "Execution script has not yet been committed. "
        "Commit the script before running it."
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing output:\n{OUTPUT_DIR}"
    )


# ------------------------------------------------------------
# 4. Load exact committed metric implementations
# ------------------------------------------------------------
paper2_functions = extract_functions(
    source_path=EXP008_SOURCE,
    requested_names=[
        "weighted_mean",
        "weighted_average_precision",
        "weighted_roc_auc",
        "percentile_interval",
    ],
    global_namespace={
        "np": np,
        "CONFIDENCE_LEVEL":
            CONFIDENCE_LEVEL,
        "average_precision_score":
            average_precision_score,
        "roc_auc_score":
            roc_auc_score,
    },
)

weighted_mean = paper2_functions[
    "weighted_mean"
]

weighted_average_precision = paper2_functions[
    "weighted_average_precision"
]

weighted_roc_auc = paper2_functions[
    "weighted_roc_auc"
]

percentile_interval = paper2_functions[
    "percentile_interval"
]

paper1_functions = extract_functions(
    source_path=PAPER1_GRADIENT_SOURCE,
    requested_names=[
        "gradient_magnitude_2d",
    ],
    global_namespace={
        "np": np,
    },
)

gradient_magnitude_2d = paper1_functions[
    "gradient_magnitude_2d"
]


# ------------------------------------------------------------
# 5. Verify extracted function signatures
# ------------------------------------------------------------
metric_signatures = {
    "weighted_mean":
        str(inspect.signature(weighted_mean)),

    "weighted_average_precision":
        str(
            inspect.signature(
                weighted_average_precision
            )
        ),

    "weighted_roc_auc":
        str(
            inspect.signature(
                weighted_roc_auc
            )
        ),

    "percentile_interval":
        str(
            inspect.signature(
                percentile_interval
            )
        ),

    "gradient_magnitude_2d":
        str(
            inspect.signature(
                gradient_magnitude_2d
            )
        ),
}


# ------------------------------------------------------------
# 6. Index frozen chunks and Paper 1 cache
# ------------------------------------------------------------
neural_paths = sorted(
    NEURAL_DIR.glob(
        "dtest_batch_*.npz"
    )
)

aligned_paths = sorted(
    ALIGNED_DIR.glob(
        "dtest_uhold_batch_*.npz"
    )
)

neural_by_batch = {
    read_batch_index(path): path
    for path in neural_paths
}

aligned_by_batch = {
    read_batch_index(path): path
    for path in aligned_paths
}

if len(neural_by_batch) != EXPECTED_BATCHES:
    raise RuntimeError(
        f"Expected {EXPECTED_BATCHES} neural chunks, "
        f"found {len(neural_by_batch)}."
    )

if len(aligned_by_batch) != EXPECTED_BATCHES:
    raise RuntimeError(
        f"Expected {EXPECTED_BATCHES} aligned chunks, "
        f"found {len(aligned_by_batch)}."
    )

if set(neural_by_batch) != set(
    aligned_by_batch
):
    raise RuntimeError(
        "Neural and aligned batch indices differ."
    )

cache_paths = sorted(
    PAPER1_TEST_CACHE.glob(
        "*.npz"
    )
)

cache_lookup = {
    path.stem: path
    for path in cache_paths
}

if len(cache_lookup) != EXPECTED_SLICES:
    raise RuntimeError(
        f"Expected {EXPECTED_SLICES} Paper 1 cache files, "
        f"found {len(cache_lookup)}."
    )


# ------------------------------------------------------------
# 7. Build volume-to-chunk/sample index
# ------------------------------------------------------------
volume_records: dict[
    str,
    list[dict],
] = {}

seen_samples = set()
total_indexed_pixels = 0

for batch_index in sorted(
    neural_by_batch
):
    neural_path = neural_by_batch[
        batch_index
    ]

    with np.load(
        neural_path,
        allow_pickle=False,
    ) as neural:
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

        pixel_sample_index = neural[
            "pixel_sample_index"
        ].astype(
            np.int64,
            copy=False,
        )

        total_indexed_pixels += int(
            len(pixel_sample_index)
        )

        for local_index, (
            sample_id,
            volume_id,
        ) in enumerate(
            zip(
                sample_ids,
                volume_ids,
                strict=True,
            )
        ):
            if sample_id in seen_samples:
                raise RuntimeError(
                    f"Duplicate sample: {sample_id}"
                )

            if sample_id not in cache_lookup:
                raise KeyError(
                    f"Paper 1 cache missing: {sample_id}"
                )

            sample_pixel_count = int(
                np.count_nonzero(
                    pixel_sample_index
                    == local_index
                )
            )

            if sample_pixel_count <= 0:
                raise RuntimeError(
                    f"No pixels found for {sample_id}"
                )

            volume_records.setdefault(
                volume_id,
                [],
            ).append(
                {
                    "batch_index":
                        batch_index,

                    "local_index":
                        local_index,

                    "sample_id":
                        sample_id,

                    "pixel_count":
                        sample_pixel_count,
                }
            )

            seen_samples.add(
                sample_id
            )


if len(volume_records) != EXPECTED_VOLUMES:
    raise RuntimeError(
        f"Expected {EXPECTED_VOLUMES} volumes, "
        f"found {len(volume_records)}."
    )

if len(seen_samples) != EXPECTED_SLICES:
    raise RuntimeError(
        f"Expected {EXPECTED_SLICES} samples, "
        f"found {len(seen_samples)}."
    )

if total_indexed_pixels != EXPECTED_PIXELS:
    raise RuntimeError(
        f"Expected {EXPECTED_PIXELS} pixels, "
        f"found {total_indexed_pixels}."
    )


# ------------------------------------------------------------
# 8. Calculate C0 and B1-B6 per-volume metrics
# ------------------------------------------------------------
started_utc = datetime.now(
    timezone.utc
).isoformat()

start_time = time.time()

added_rows = []

ordered_volumes = sorted(
    volume_records
)

for volume_position, volume_id in enumerate(
    ordered_volumes,
    start=1,
):
    target_parts = []
    weight_parts = []

    score_parts = {
        method: []
        for method in ADDED_METHODS
    }

    sample_ids_for_volume = []

    for record in volume_records[
        volume_id
    ]:
        batch_index = record[
            "batch_index"
        ]

        local_index = record[
            "local_index"
        ]

        sample_id = record[
            "sample_id"
        ]

        neural_path = neural_by_batch[
            batch_index
        ]

        aligned_path = aligned_by_batch[
            batch_index
        ]

        with np.load(
            neural_path,
            allow_pickle=False,
        ) as neural, np.load(
            aligned_path,
            allow_pickle=False,
        ) as aligned:
            neural_sample_ids = [
                str(value)
                for value in neural[
                    "sample_ids"
                ].tolist()
            ]

            aligned_sample_ids = [
                str(value)
                for value in aligned[
                    "sample_ids"
                ].tolist()
            ]

            if (
                neural_sample_ids
                != aligned_sample_ids
            ):
                raise RuntimeError(
                    f"Sample alignment failure in batch "
                    f"{batch_index}."
                )

            if (
                neural_sample_ids[
                    local_index
                ]
                != sample_id
            ):
                raise RuntimeError(
                    f"Local sample-index failure for "
                    f"{sample_id}."
                )

            pixel_sample_index = neural[
                "pixel_sample_index"
            ].astype(
                np.int64,
                copy=False,
            )

            selection = (
                pixel_sample_index
                == local_index
            )

            rows = neural[
                "pixel_row"
            ][
                selection
            ].astype(
                np.int64,
                copy=False,
            )

            columns = neural[
                "pixel_column"
            ][
                selection
            ].astype(
                np.int64,
                copy=False,
            )

            weights = neural[
                "support_weight"
            ][
                selection
            ].astype(
                np.float64,
                copy=False,
            )

            target = aligned[
                "target_u_hold"
            ][
                selection
            ].astype(
                np.float64,
                copy=False,
            )

            c0_score = neural[
                "c0_mean"
            ][
                selection
            ].astype(
                np.float64,
                copy=False,
            )

        with np.load(
            cache_lookup[
                sample_id
            ],
            allow_pickle=False,
        ) as cached:
            x = cached[
                "x"
            ].astype(
                np.float32,
                copy=False,
            )

            if x.ndim != 3 or x.shape[0] != 6:
                raise RuntimeError(
                    f"Unexpected cache shape for "
                    f"{sample_id}: {x.shape}"
                )

            height = int(
                x.shape[1]
            )

            width = int(
                x.shape[2]
            )

            if (
                np.any(rows < 0)
                or np.any(rows >= height)
                or np.any(columns < 0)
                or np.any(columns >= width)
            ):
                raise RuntimeError(
                    f"Invalid coordinates for {sample_id}."
                )

            gradient_map = (
                gradient_magnitude_2d(
                    x[0]
                )
            )

            descriptor_scores = {
                "B1":
                    x[
                        0,
                        rows,
                        columns,
                    ].astype(
                        np.float64,
                        copy=False,
                    ),

                "B2":
                    x[
                        1,
                        rows,
                        columns,
                    ].astype(
                        np.float64,
                        copy=False,
                    ),

                "B3":
                    x[
                        2,
                        rows,
                        columns,
                    ].astype(
                        np.float64,
                        copy=False,
                    ),

                "B4":
                    gradient_map[
                        rows,
                        columns,
                    ].astype(
                        np.float64,
                        copy=False,
                    ),

                "B5":
                    x[
                        4,
                        rows,
                        columns,
                    ].astype(
                        np.float64,
                        copy=False,
                    ),

                "B6":
                    x[
                        5,
                        rows,
                        columns,
                    ].astype(
                        np.float64,
                        copy=False,
                    ),
            }

        target_parts.append(
            target
        )

        weight_parts.append(
            weights
        )

        score_parts[
            "C0"
        ].append(
            c0_score
        )

        for method in [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
        ]:
            score_parts[
                method
            ].append(
                descriptor_scores[
                    method
                ]
            )

        sample_ids_for_volume.append(
            sample_id
        )

    target_volume = np.concatenate(
        target_parts
    )

    weight_volume = np.concatenate(
        weight_parts
    )

    labels_volume = (
        target_volume
        >= TAU_HOLD
    ).astype(
        np.int8
    )

    supported_pixels = int(
        len(target_volume)
    )

    support_weight_sum = float(
        np.sum(
            weight_volume,
            dtype=np.float64,
        )
    )

    high_risk_prevalence = float(
        weighted_mean(
            labels_volume,
            weight_volume,
        )
    )

    for method in ADDED_METHODS:
        score_volume = np.concatenate(
            score_parts[
                method
            ]
        )

        if len(score_volume) != supported_pixels:
            raise RuntimeError(
                f"Score-length mismatch for "
                f"{volume_id}, {method}."
            )

        auprc = float(
            weighted_average_precision(
                labels_volume,
                score_volume,
                weight_volume,
            )
        )

        auroc = float(
            weighted_roc_auc(
                labels_volume,
                score_volume,
                weight_volume,
            )
        )

        added_rows.append(
            {
                "volume_id":
                    volume_id,

                "sample_count":
                    int(
                        len(
                            sample_ids_for_volume
                        )
                    ),

                "supported_pixels":
                    supported_pixels,

                "support_weight_sum":
                    support_weight_sum,

                "method":
                    method,

                "tau_hold":
                    TAU_HOLD,

                "high_risk_prevalence":
                    high_risk_prevalence,

                "auprc":
                    auprc,

                "auroc":
                    auroc,
            }
        )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Completed volume "
        f"{volume_position:02d}/"
        f"{EXPECTED_VOLUMES} "
        f"| {volume_id} "
        f"| pixels={supported_pixels:,} "
        f"| elapsed={elapsed / 60:.2f} min"
    )

    del (
        target_volume,
        weight_volume,
        labels_volume,
        target_parts,
        weight_parts,
        score_parts,
    )

    gc.collect()


added_frame = pd.DataFrame(
    added_rows
)

if len(added_frame) != (
    EXPECTED_VOLUMES
    * len(ADDED_METHODS)
):
    raise RuntimeError(
        "Unexpected number of added per-volume rows."
    )


# ------------------------------------------------------------
# 9. Reuse and validate frozen U1/U2a/U2b rows
# ------------------------------------------------------------
frozen_task_r = pd.read_csv(
    FROZEN_TASK_R_PATH
)

expected_task_r_columns = [
    "volume_id",
    "sample_count",
    "supported_pixels",
    "support_weight_sum",
    "method",
    "tau_hold",
    "high_risk_prevalence",
    "auprc",
    "auroc",
]

if list(
    frozen_task_r.columns
) != expected_task_r_columns:
    raise RuntimeError(
        "Frozen Task R schema differs from expected schema."
    )

frozen_reused = (
    frozen_task_r[
        frozen_task_r[
            "method"
        ].isin(
            FROZEN_REUSED_METHODS
        )
    ]
    .copy()
)

if len(frozen_reused) != (
    EXPECTED_VOLUMES
    * len(FROZEN_REUSED_METHODS)
):
    raise RuntimeError(
        "Unexpected number of frozen reused rows."
    )

if set(
    frozen_reused["volume_id"]
) != set(
    ordered_volumes
):
    raise RuntimeError(
        "Frozen Task R volume population differs."
    )

if not np.all(
    frozen_reused[
        "tau_hold"
    ].to_numpy(
        dtype=np.float64
    )
    == TAU_HOLD
):
    raise RuntimeError(
        "Frozen Task R threshold differs."
    )


# ------------------------------------------------------------
# 10. Cross-check common volume quantities
# ------------------------------------------------------------
added_common = (
    added_frame[
        added_frame[
            "method"
        ] == "C0"
    ][
        [
            "volume_id",
            "sample_count",
            "supported_pixels",
            "support_weight_sum",
            "tau_hold",
            "high_risk_prevalence",
        ]
    ]
    .sort_values(
        "volume_id"
    )
    .reset_index(
        drop=True
    )
)

frozen_common = (
    frozen_reused[
        frozen_reused[
            "method"
        ] == "U1"
    ][
        [
            "volume_id",
            "sample_count",
            "supported_pixels",
            "support_weight_sum",
            "tau_hold",
            "high_risk_prevalence",
        ]
    ]
    .sort_values(
        "volume_id"
    )
    .reset_index(
        drop=True
    )
)

if not np.array_equal(
    added_common[
        "volume_id"
    ].to_numpy(),
    frozen_common[
        "volume_id"
    ].to_numpy(),
):
    raise RuntimeError(
        "Volume order differs during common-field audit."
    )

for column in [
    "sample_count",
    "supported_pixels",
]:
    if not np.array_equal(
        added_common[
            column
        ].to_numpy(),
        frozen_common[
            column
        ].to_numpy(),
    ):
        raise RuntimeError(
            f"Common-field mismatch: {column}"
        )

for column in [
    "support_weight_sum",
    "tau_hold",
    "high_risk_prevalence",
]:
    if not np.allclose(
        added_common[
            column
        ].to_numpy(
            dtype=np.float64
        ),
        frozen_common[
            column
        ].to_numpy(
            dtype=np.float64
        ),
        rtol=0.0,
        atol=1e-9,
    ):
        maximum_difference = float(
            np.max(
                np.abs(
                    added_common[
                        column
                    ].to_numpy(
                        dtype=np.float64
                    )
                    -
                    frozen_common[
                        column
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            )
        )

        raise RuntimeError(
            f"Common-field mismatch: {column}; "
            f"maximum difference={maximum_difference}"
        )


# ------------------------------------------------------------
# 11. Construct extended per-volume table
# ------------------------------------------------------------
extended_task_r = pd.concat(
    [
        frozen_reused,
        added_frame,
    ],
    ignore_index=True,
)

extended_task_r["method"] = pd.Categorical(
    extended_task_r["method"],
    categories=METHOD_ORDER,
    ordered=True,
)

extended_task_r = (
    extended_task_r
    .sort_values(
        [
            "volume_id",
            "method",
        ]
    )
    .reset_index(
        drop=True
    )
)

extended_task_r["method"] = (
    extended_task_r[
        "method"
    ].astype(
        str
    )
)

if len(extended_task_r) != (
    EXPECTED_VOLUMES
    * len(METHOD_ORDER)
):
    raise RuntimeError(
        "Extended Task R table has incorrect size."
    )

method_counts = (
    extended_task_r[
        "method"
    ]
    .value_counts()
    .to_dict()
)

for method in METHOD_ORDER:
    if method_counts.get(
        method,
        0,
    ) != EXPECTED_VOLUMES:
        raise RuntimeError(
            f"Method {method} does not have "
            f"{EXPECTED_VOLUMES} records."
        )


# ------------------------------------------------------------
# 12. Frozen bootstrap implementation
# ------------------------------------------------------------
volume_order = sorted(
    extended_task_r[
        "volume_id"
    ].unique()
)

auprc_matrix = (
    extended_task_r
    .pivot(
        index="volume_id",
        columns="method",
        values="auprc",
    )
    .loc[
        volume_order,
        METHOD_ORDER,
    ]
    .to_numpy(
        dtype=np.float64
    )
)

auroc_matrix = (
    extended_task_r
    .pivot(
        index="volume_id",
        columns="method",
        values="auroc",
    )
    .loc[
        volume_order,
        METHOD_ORDER,
    ]
    .to_numpy(
        dtype=np.float64
    )
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

bootstrap_auprc_means = np.mean(
    auprc_matrix[
        bootstrap_indices,
        :,
    ],
    axis=1,
)

bootstrap_auroc_means = np.mean(
    auroc_matrix[
        bootstrap_indices,
        :,
    ],
    axis=1,
)


# ------------------------------------------------------------
# 13. Primary AUPRC summaries
# ------------------------------------------------------------
summary_rows = []

for method_index, method in enumerate(
    METHOD_ORDER
):
    estimate = float(
        np.mean(
            auprc_matrix[
                :,
                method_index,
            ]
        )
    )

    ci_low, ci_high = percentile_bounds(
        percentile_interval,
        bootstrap_auprc_means[
            :,
            method_index,
        ],
    )

    summary_rows.append(
        {
            "task": "R",
            "endpoint": "auprc",
            "method": method,
            "n_volumes": EXPECTED_VOLUMES,
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence_level":
                CONFIDENCE_LEVEL,
            "bootstrap_replicates":
                BOOTSTRAP_REPLICATES,
            "bootstrap_seed":
                BOOTSTRAP_SEED,
            "direction":
                "higher_is_better",
        }
    )

primary_summary = pd.DataFrame(
    summary_rows
)


# ------------------------------------------------------------
# 14. Secondary AUROC summaries
# ------------------------------------------------------------
secondary_rows = []

for method_index, method in enumerate(
    METHOD_ORDER
):
    estimate = float(
        np.mean(
            auroc_matrix[
                :,
                method_index,
            ]
        )
    )

    ci_low, ci_high = percentile_bounds(
        percentile_interval,
        bootstrap_auroc_means[
            :,
            method_index,
        ],
    )

    secondary_rows.append(
        {
            "task": "R",
            "endpoint": "auroc",
            "method": method,
            "n_volumes": EXPECTED_VOLUMES,
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence_level":
                CONFIDENCE_LEVEL,
            "bootstrap_replicates":
                BOOTSTRAP_REPLICATES,
            "bootstrap_seed":
                BOOTSTRAP_SEED,
            "direction":
                "higher_is_better",
        }
    )

secondary_summary = pd.DataFrame(
    secondary_rows
)


# ------------------------------------------------------------
# 15. Reproduce frozen U summaries exactly
# ------------------------------------------------------------
frozen_primary_summary = pd.read_csv(
    FROZEN_PRIMARY_SUMMARY_PATH
)

frozen_r_summary = (
    frozen_primary_summary[
        (
            frozen_primary_summary[
                "task"
            ] == "R"
        )
        &
        (
            frozen_primary_summary[
                "endpoint"
            ] == "auprc"
        )
        &
        (
            frozen_primary_summary[
                "method"
            ].isin(
                FROZEN_REUSED_METHODS
            )
        )
    ]
    .sort_values(
        "method"
    )
    .reset_index(
        drop=True
    )
)

reproduced_r_summary = (
    primary_summary[
        primary_summary[
            "method"
        ].isin(
            FROZEN_REUSED_METHODS
        )
    ]
    .sort_values(
        "method"
    )
    .reset_index(
        drop=True
    )
)

for column in [
    "estimate",
    "ci_low",
    "ci_high",
]:
    if not np.allclose(
        frozen_r_summary[
            column
        ].to_numpy(
            dtype=np.float64
        ),
        reproduced_r_summary[
            column
        ].to_numpy(
            dtype=np.float64
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Frozen Task R summary reproduction failed "
            f"for {column}."
        )


# ------------------------------------------------------------
# 16. All-pair AUPRC differences and win rates
# ------------------------------------------------------------
pairwise_rows = []
win_rate_rows = []

for first_method, second_method in itertools.combinations(
    METHOD_ORDER,
    2,
):
    first_index = METHOD_ORDER.index(
        first_method
    )

    second_index = METHOD_ORDER.index(
        second_method
    )

    observed_volume_difference = (
        auprc_matrix[
            :,
            first_index,
        ]
        -
        auprc_matrix[
            :,
            second_index,
        ]
    )

    observed_mean_difference = float(
        np.mean(
            observed_volume_difference
        )
    )

    bootstrap_difference = (
        bootstrap_auprc_means[
            :,
            first_index,
        ]
        -
        bootstrap_auprc_means[
            :,
            second_index,
        ]
    )

    ci_low, ci_high = percentile_bounds(
        percentile_interval,
        bootstrap_difference,
    )

    probability_first_better = float(
        np.mean(
            bootstrap_difference
            > 0.0
        )
    )

    first_wins = (
        auprc_matrix[
            :,
            first_index,
        ]
        >
        auprc_matrix[
            :,
            second_index,
        ]
    )

    ties = (
        auprc_matrix[
            :,
            first_index,
        ]
        ==
        auprc_matrix[
            :,
            second_index,
        ]
    )

    pairwise_rows.append(
        {
            "task": "R",
            "endpoint": "auprc",
            "first_method":
                first_method,
            "second_method":
                second_method,
            "difference_definition":
                "first_minus_second",
            "observed_mean_difference":
                observed_mean_difference,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "direction":
                "higher_is_better",
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
            "task": "R",
            "endpoint": "auprc",
            "first_method":
                first_method,
            "second_method":
                second_method,
            "direction":
                "higher_is_better",
            "first_method_volume_win_rate":
                float(
                    np.mean(
                        first_wins
                    )
                ),
            "volume_tie_rate":
                float(
                    np.mean(
                        ties
                    )
                ),
            "n_volumes":
                EXPECTED_VOLUMES,
        }
    )

pairwise_frame = pd.DataFrame(
    pairwise_rows
)

win_rate_frame = pd.DataFrame(
    win_rate_rows
)


# ------------------------------------------------------------
# 17. Validate frozen U-only pairwise results
# ------------------------------------------------------------
frozen_pairwise = pd.read_csv(
    FROZEN_PAIRWISE_PATH
)

frozen_r_pairs = (
    frozen_pairwise[
        (
            frozen_pairwise[
                "task"
            ] == "R"
        )
        &
        (
            frozen_pairwise[
                "endpoint"
            ] == "auprc"
        )
    ]
    .copy()
)

for first_method, second_method in itertools.combinations(
    FROZEN_REUSED_METHODS,
    2,
):
    frozen_row = frozen_r_pairs[
        (
            frozen_r_pairs[
                "first_method"
            ] == first_method
        )
        &
        (
            frozen_r_pairs[
                "second_method"
            ] == second_method
        )
    ]

    reproduced_row = pairwise_frame[
        (
            pairwise_frame[
                "first_method"
            ] == first_method
        )
        &
        (
            pairwise_frame[
                "second_method"
            ] == second_method
        )
    ]

    if len(frozen_row) != 1 or len(
        reproduced_row
    ) != 1:
        raise RuntimeError(
            f"Could not validate frozen pair "
            f"{first_method} vs {second_method}."
        )

    for column in [
        "observed_mean_difference",
        "ci_low",
        "ci_high",
        "bootstrap_probability_first_better",
    ]:
        if not np.isclose(
            float(
                frozen_row.iloc[
                    0
                ][
                    column
                ]
            ),
            float(
                reproduced_row.iloc[
                    0
                ][
                    column
                ]
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Frozen pairwise reproduction failed: "
                f"{first_method} vs {second_method}, "
                f"{column}."
            )


# ------------------------------------------------------------
# 18. Validate frozen U-only win rates
# ------------------------------------------------------------
frozen_win_rates = pd.read_csv(
    FROZEN_WIN_RATE_PATH
)

frozen_r_wins = (
    frozen_win_rates[
        (
            frozen_win_rates[
                "task"
            ] == "R"
        )
        &
        (
            frozen_win_rates[
                "endpoint"
            ] == "auprc"
        )
    ]
    .copy()
)

for first_method, second_method in itertools.combinations(
    FROZEN_REUSED_METHODS,
    2,
):
    frozen_row = frozen_r_wins[
        (
            frozen_r_wins[
                "first_method"
            ] == first_method
        )
        &
        (
            frozen_r_wins[
                "second_method"
            ] == second_method
        )
    ]

    reproduced_row = win_rate_frame[
        (
            win_rate_frame[
                "first_method"
            ] == first_method
        )
        &
        (
            win_rate_frame[
                "second_method"
            ] == second_method
        )
    ]

    if len(frozen_row) != 1 or len(
        reproduced_row
    ) != 1:
        raise RuntimeError(
            f"Could not validate frozen win rate "
            f"{first_method} vs {second_method}."
        )

    for column in [
        "first_method_volume_win_rate",
        "volume_tie_rate",
    ]:
        if not np.isclose(
            float(
                frozen_row.iloc[
                    0
                ][
                    column
                ]
            ),
            float(
                reproduced_row.iloc[
                    0
                ][
                    column
                ]
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Frozen win-rate reproduction failed: "
                f"{first_method} vs {second_method}, "
                f"{column}."
            )


# ------------------------------------------------------------
# 19. Write results atomically to a new directory
# ------------------------------------------------------------
temporary_parent = (
    REPO
    / "results"
)

temporary_parent.mkdir(
    parents=True,
    exist_ok=True,
)

temporary_directory = Path(
    tempfile.mkdtemp(
        prefix=(
            "exp009_task_r_control_completion_"
            "temporary_"
        ),
        dir=temporary_parent,
    )
)

try:
    task_r_output = (
        temporary_directory
        / "task_r_per_volume_extended.csv"
    )

    primary_output = (
        temporary_directory
        / "task_r_primary_summary_extended.csv"
    )

    secondary_output = (
        temporary_directory
        / "task_r_secondary_summary_extended.csv"
    )

    pairwise_output = (
        temporary_directory
        / "task_r_paired_method_differences_extended.csv"
    )

    win_rate_output = (
        temporary_directory
        / "task_r_volume_level_win_rates_extended.csv"
    )

    method_output = (
        temporary_directory
        / "method_definitions.json"
    )

    provenance_output = (
        temporary_directory
        / "provenance.json"
    )

    atomic_csv_write(
        task_r_output,
        extended_task_r,
    )

    atomic_csv_write(
        primary_output,
        primary_summary,
    )

    atomic_csv_write(
        secondary_output,
        secondary_summary,
    )

    atomic_csv_write(
        pairwise_output,
        pairwise_frame,
    )

    atomic_csv_write(
        win_rate_output,
        win_rate_frame,
    )

    atomic_json_write(
        method_output,
        {
            "schema_version":
                "exp009-task-r-controls-methods-v1.0",

            "method_order":
                METHOD_ORDER,

            "added_methods":
                ADDED_METHODS,

            "frozen_reused_methods":
                FROZEN_REUSED_METHODS,

            "definitions":
                METHOD_DEFINITIONS,

            "direction_freeze":
                (
                    "Larger raw values submitted as higher "
                    "Task R scores; no sign flipping or "
                    "test-dependent transformation."
                ),
        },
    )

    output_files = [
        task_r_output,
        primary_output,
        secondary_output,
        pairwise_output,
        win_rate_output,
        method_output,
    ]

    completed_utc = datetime.now(
        timezone.utc
    ).isoformat()

    elapsed_seconds = float(
        time.time()
        - start_time
    )

    provenance = {
        "schema_version":
            "exp009-task-r-control-completion-v1.0",

        "experiment_id":
            "exp009_task_r_control_completion",

        "scientific_status":
            (
                "retrospective completion of omitted "
                "prespecified Task R controls"
            ),

        "confirmatory_claim":
            False,

        "started_utc":
            started_utc,

        "completed_utc":
            completed_utc,

        "elapsed_seconds":
            elapsed_seconds,

        "repository_branch":
            EXPECTED_BRANCH,

        "repository_commit":
            execution_commit,

        "parent_governance_commit":
            EXPECTED_PARENT_COMMIT,

        "paper1_repository_commit":
            git_output(
                PAPER1_REPO,
                "rev-parse",
                "HEAD",
            ),

        "tau_hold":
            TAU_HOLD,

        "D_test_volumes":
            EXPECTED_VOLUMES,

        "D_test_slices":
            EXPECTED_SLICES,

        "D_test_pixels":
            EXPECTED_PIXELS,

        "bootstrap": {
            "replicates":
                BOOTSTRAP_REPLICATES,

            "seed":
                BOOTSTRAP_SEED,

            "confidence_level":
                CONFIDENCE_LEVEL,

            "summary_unit":
                "volume",

            "summary_rule":
                (
                    "unweighted mean of 40 per-volume "
                    "support-weighted endpoints"
                ),
        },

        "metric_implementations": {
            "source":
                str(EXP008_SOURCE),

            "source_sha256":
                sha256_file(
                    EXP008_SOURCE
                ),

            "signatures":
                metric_signatures,
        },

        "gradient_implementation": {
            "source":
                str(
                    PAPER1_GRADIENT_SOURCE
                ),

            "source_sha256":
                sha256_file(
                    PAPER1_GRADIENT_SOURCE
                ),

            "signature":
                metric_signatures[
                    "gradient_magnitude_2d"
                ],
        },

        "frozen_input_sources": {
            "neural_chunks":
                str(NEURAL_DIR),

            "aligned_u_hold_chunks":
                str(ALIGNED_DIR),

            "paper1_test_cache":
                str(PAPER1_TEST_CACHE),

            "frozen_task_r_per_volume":
                str(FROZEN_TASK_R_PATH),

            "frozen_primary_summary":
                str(
                    FROZEN_PRIMARY_SUMMARY_PATH
                ),

            "frozen_pairwise_differences":
                str(FROZEN_PAIRWISE_PATH),

            "frozen_volume_win_rates":
                str(FROZEN_WIN_RATE_PATH),
        },

        "methods": {
            "all":
                METHOD_ORDER,

            "added":
                ADDED_METHODS,

            "reused_without_recalculation":
                FROZEN_REUSED_METHODS,
        },

        "validation": {
            "full_cross_paper_alignment_passed":
                True,

            "support_maximum_difference":
                0.0,

            "target_maximum_difference":
                0.0,

            "common_volume_fields_reproduced":
                True,

            "frozen_U_primary_summaries_reproduced":
                True,

            "frozen_U_pairwise_results_reproduced":
                True,

            "frozen_U_win_rates_reproduced":
                True,
        },

        "governance": {
            "model_training_performed":
                False,

            "model_inference_performed":
                False,

            "target_regeneration_performed":
                False,

            "threshold_selection_performed":
                False,

            "descriptor_sign_selection_performed":
                False,

            "descriptor_transformation_selection_performed":
                False,

            "existing_exp008_outputs_modified":
                False,
        },
    }

    provenance[
        "output_files"
    ] = {
        path.name: {
            "sha256":
                sha256_file(path),

            "size_bytes":
                int(
                    path.stat().st_size
                ),
        }
        for path in output_files
    }

    atomic_json_write(
        provenance_output,
        provenance,
    )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output appeared during execution: "
            f"{OUTPUT_DIR}"
        )

    os.replace(
        temporary_directory,
        OUTPUT_DIR,
    )

except Exception:
    shutil.rmtree(
        temporary_directory,
        ignore_errors=True,
    )
    raise


# ------------------------------------------------------------
# 20. Final report
# ------------------------------------------------------------
print("\n" + "=" * 116)
print(
    "EXP009 — RETROSPECTIVE TASK R "
    "CONTROL COMPLETION"
)
print("=" * 116)

print(
    f"Repository commit:                    "
    f"{execution_commit}"
)

print(
    f"Volumes:                              "
    f"{EXPECTED_VOLUMES}"
)

print(
    f"Slices:                               "
    f"{EXPECTED_SLICES}"
)

print(
    f"Pixels:                               "
    f"{EXPECTED_PIXELS:,}"
)

print(
    f"Methods reported:                     "
    f"{len(METHOD_ORDER)}"
)

print(
    f"Per-volume records:                   "
    f"{len(extended_task_r)}"
)

print(
    f"Pairwise AUPRC comparisons:           "
    f"{len(pairwise_frame)}"
)

print(
    f"Output directory:                     "
    f"{OUTPUT_DIR}"
)

print("\nPRIMARY TASK R AUPRC")
print("-" * 116)

print(
    primary_summary[
        [
            "method",
            "estimate",
            "ci_low",
            "ci_high",
        ]
    ].to_string(
        index=False
    )
)

print("\nGOVERNANCE")
print("-" * 116)

print("Frozen U1/U2a/U2b records reused:     YES")
print("C0 calculated from frozen chunks:      YES")
print("B1-B6 calculated from frozen cache:    YES")
print("Model training performed:              NO")
print("Model inference performed:             NO")
print("Threshold selected or changed:         NO")
print("Descriptor sign selected:              NO")
print("Existing Exp008 outputs modified:      NO")

print(
    f"Elapsed time:                          "
    f"{(time.time() - start_time) / 60:.2f} minutes"
)

print("=" * 116)
print(
    "STATUS: PASS_RETROSPECTIVE_TASK_R_"
    "CONTROL_COMPLETION"
)
