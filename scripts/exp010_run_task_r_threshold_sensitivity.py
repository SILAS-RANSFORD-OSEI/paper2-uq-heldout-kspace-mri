# ============================================================
# EXP010 — TASK R THRESHOLD SENSITIVITY
#
# Retrospective sensitivity analysis at the frozen D_cal
# thresholds q85 and q95.
#
# This script does not perform:
# - reconstruction
# - model training
# - model inference
# - target regeneration
# - threshold fitting
# - score sign selection
# - score transformation selection
#
# All ten methods are evaluated under the same:
# - D_test cohort
# - target_u_hold
# - support_weight
# - metric implementation
# - volume-level summary
# - paired bootstrap indices
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
# 1. Frozen paths
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

EXP009_RESULT_DIR = (
    REPO
    / "results"
    / "exp009_task_r_control_completion"
)

EXP009_PRIMARY_PATH = (
    EXP009_RESULT_DIR
    / "task_r_primary_summary_extended.csv"
)

OUTPUT_DIR = (
    REPO
    / "results"
    / "exp010_task_r_threshold_sensitivity"
)


# ------------------------------------------------------------
# 2. Frozen repository and scientific constants
# ------------------------------------------------------------
EXPECTED_BRANCH = (
    "retrospective-task-r-controls-v1.0"
)

EXPECTED_PARENT_COMMIT = (
    "7cc55768141c9a538d5a9647d17cde7b3dc84182"
)

EXPECTED_PAPER1_COMMIT = (
    "da563ead8fb653539e1eeca29248b31f0121ca12"
)

THRESHOLDS = {
    "q85": 1.4107755422592163,
    "q95": 2.0520856380462646,
}

Q90_REFERENCE = 1.7366089820861816

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

NEURAL_SCORE_FIELDS = {
    "C0": "c0_mean",
    "U1": "u1_variance",
    "U2a": "u2a_between_model_variance",
    "U2b": "u2b_total_predictive_variance",
}

DESCRIPTOR_CHANNELS = {
    "B1": 0,
    "B2": 1,
    "B3": 2,
    "B5": 4,
    "B6": 5,
}


# ------------------------------------------------------------
# 3. General helpers
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


def extract_functions(
    source_path: Path,
    requested_names: list[str],
    global_namespace: dict,
) -> dict:
    tree = ast.parse(
        source_path.read_text(
            encoding="utf-8"
        ),
        filename=str(source_path),
    )

    selected = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
        and node.name in requested_names
    ]

    found = {
        node.name
        for node in selected
    }

    missing = sorted(
        set(requested_names)
        - found
    )

    if missing:
        raise RuntimeError(
            f"Missing functions in {source_path}: "
            f"{missing}"
        )

    selected = sorted(
        selected,
        key=lambda node: node.lineno,
    )

    module = ast.Module(
        body=selected,
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


# ------------------------------------------------------------
# 4. Repository and artifact guards
# ------------------------------------------------------------
required_paths = [
    REPO,
    PAPER1_REPO,
    PAPER1_TEST_CACHE,
    PAPER2_RELEASE,
    NEURAL_DIR,
    ALIGNED_DIR,
    EXP008_SOURCE,
    PAPER1_GRADIENT_SOURCE,
    EXP009_PRIMARY_PATH,
]

for required_path in required_paths:
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
        "Paper 2 repository must be clean."
    )

execution_commit = git_output(
    REPO,
    "rev-parse",
    "HEAD",
)

if execution_commit == EXPECTED_PARENT_COMMIT:
    raise RuntimeError(
        "The sensitivity script must be committed "
        "before execution."
    )

paper1_commit = git_output(
    PAPER1_REPO,
    "rev-parse",
    "HEAD",
)

if paper1_commit != EXPECTED_PAPER1_COMMIT:
    raise RuntimeError(
        f"Unexpected Paper 1 commit: {paper1_commit}"
    )

if git_output(
    PAPER1_REPO,
    "status",
    "--porcelain",
) != "":
    raise RuntimeError(
        "Paper 1 repository must be clean."
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Refusing to overwrite:\n{OUTPUT_DIR}"
    )

if not (
    THRESHOLDS["q85"]
    < Q90_REFERENCE
    < THRESHOLDS["q95"]
):
    raise RuntimeError(
        "Frozen threshold ordering is invalid."
    )


# ------------------------------------------------------------
# 5. Load exact committed metric implementations
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
# 6. Index neural, aligned, and cache artifacts
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
        f"Expected {EXPECTED_SLICES} cache files, "
        f"found {len(cache_lookup)}."
    )


# ------------------------------------------------------------
# 7. Build volume/sample index
# ------------------------------------------------------------
volume_records: dict[
    str,
    list[dict],
] = {}

seen_samples = set()
indexed_pixels = 0

for batch_index in sorted(
    neural_by_batch
):
    with np.load(
        neural_by_batch[
            batch_index
        ],
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

        indexed_pixels += int(
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
                    f"Missing Paper 1 cache: {sample_id}"
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
        f"Expected {EXPECTED_SLICES} slices, "
        f"found {len(seen_samples)}."
    )

if indexed_pixels != EXPECTED_PIXELS:
    raise RuntimeError(
        f"Expected {EXPECTED_PIXELS} pixels, "
        f"found {indexed_pixels}."
    )


# ------------------------------------------------------------
# 8. Calculate q85 and q95 per-volume metrics
# ------------------------------------------------------------
started_utc = datetime.now(
    timezone.utc
).isoformat()

start_time = time.time()

rows = []

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
        for method in METHOD_ORDER
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

        with np.load(
            neural_by_batch[
                batch_index
            ],
            allow_pickle=False,
        ) as neural, np.load(
            aligned_by_batch[
                batch_index
            ],
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

            if neural_sample_ids != aligned_sample_ids:
                raise RuntimeError(
                    f"Chunk alignment failure: {batch_index}"
                )

            if (
                neural_sample_ids[
                    local_index
                ]
                != sample_id
            ):
                raise RuntimeError(
                    f"Local index failure: {sample_id}"
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

            pixel_rows = neural[
                "pixel_row"
            ][
                selection
            ].astype(
                np.int64,
                copy=False,
            )

            pixel_columns = neural[
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

            neural_scores = {
                method:
                    neural[
                        field_name
                    ][
                        selection
                    ].astype(
                        np.float64,
                        copy=False,
                    )
                for method, field_name
                in NEURAL_SCORE_FIELDS.items()
            }

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
                np.any(pixel_rows < 0)
                or np.any(pixel_rows >= height)
                or np.any(pixel_columns < 0)
                or np.any(pixel_columns >= width)
            ):
                raise RuntimeError(
                    f"Invalid coordinates: {sample_id}"
                )

            gradient_map = (
                gradient_magnitude_2d(
                    x[0]
                )
            )

            descriptor_scores = {
                method:
                    x[
                        channel_index,
                        pixel_rows,
                        pixel_columns,
                    ].astype(
                        np.float64,
                        copy=False,
                    )
                for method, channel_index
                in DESCRIPTOR_CHANNELS.items()
            }

            descriptor_scores[
                "B4"
            ] = gradient_map[
                pixel_rows,
                pixel_columns,
            ].astype(
                np.float64,
                copy=False,
            )

        target_parts.append(
            target
        )

        weight_parts.append(
            weights
        )

        for method in NEURAL_SCORE_FIELDS:
            score_parts[
                method
            ].append(
                neural_scores[
                    method
                ]
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

    scores_volume = {
        method:
            np.concatenate(
                score_parts[
                    method
                ]
            )
        for method in METHOD_ORDER
    }

    supported_pixels = int(
        len(target_volume)
    )

    support_weight_sum = float(
        np.sum(
            weight_volume,
            dtype=np.float64,
        )
    )

    for threshold_name, threshold_value in (
        THRESHOLDS.items()
    ):
        labels = (
            target_volume
            >= threshold_value
        ).astype(
            np.int8
        )

        prevalence = float(
            weighted_mean(
                labels,
                weight_volume,
            )
        )

        for method in METHOD_ORDER:
            scores = scores_volume[
                method
            ]

            auprc = float(
                weighted_average_precision(
                    labels,
                    scores,
                    weight_volume,
                )
            )

            auroc = float(
                weighted_roc_auc(
                    labels,
                    scores,
                    weight_volume,
                )
            )

            rows.append(
                {
                    "threshold_name":
                        threshold_name,
                    "threshold_value":
                        threshold_value,
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
                    "high_risk_prevalence":
                        prevalence,
                    "auprc":
                        auprc,
                    "auroc":
                        auroc,
                }
            )

    elapsed_minutes = (
        time.time()
        - start_time
    ) / 60.0

    print(
        f"Completed volume "
        f"{volume_position:02d}/"
        f"{EXPECTED_VOLUMES} "
        f"| {volume_id} "
        f"| pixels={supported_pixels:,} "
        f"| elapsed={elapsed_minutes:.2f} min"
    )

    del (
        target_parts,
        weight_parts,
        score_parts,
        target_volume,
        weight_volume,
        scores_volume,
    )

    gc.collect()


per_volume = pd.DataFrame(
    rows
)

expected_rows = (
    EXPECTED_VOLUMES
    * len(METHOD_ORDER)
    * len(THRESHOLDS)
)

if len(per_volume) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} rows, "
        f"found {len(per_volume)}."
    )

if per_volume[
    [
        "support_weight_sum",
        "high_risk_prevalence",
        "auprc",
        "auroc",
    ]
].isna().any().any():
    raise RuntimeError(
        "Sensitivity results contain missing metrics."
    )

if not np.isfinite(
    per_volume[
        [
            "support_weight_sum",
            "high_risk_prevalence",
            "auprc",
            "auroc",
        ]
    ].to_numpy(
        dtype=np.float64
    )
).all():
    raise RuntimeError(
        "Sensitivity results contain nonfinite metrics."
    )


# ------------------------------------------------------------
# 9. Common paired volume bootstrap
# ------------------------------------------------------------
volume_order = sorted(
    per_volume[
        "volume_id"
    ].unique()
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

primary_rows = []
secondary_rows = []
pairwise_rows = []
win_rate_rows = []

for threshold_name, threshold_value in (
    THRESHOLDS.items()
):
    threshold_frame = per_volume[
        per_volume[
            "threshold_name"
        ] == threshold_name
    ]

    auprc_matrix = (
        threshold_frame
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
        threshold_frame
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

    bootstrap_auprc = np.mean(
        auprc_matrix[
            bootstrap_indices,
            :,
        ],
        axis=1,
    )

    bootstrap_auroc = np.mean(
        auroc_matrix[
            bootstrap_indices,
            :,
        ],
        axis=1,
    )

    for method_index, method in enumerate(
        METHOD_ORDER
    ):
        estimate_auprc = float(
            np.mean(
                auprc_matrix[
                    :,
                    method_index,
                ]
            )
        )

        auprc_ci = percentile_interval(
            bootstrap_auprc[
                :,
                method_index,
            ]
        )

        primary_rows.append(
            {
                "threshold_name":
                    threshold_name,
                "threshold_value":
                    threshold_value,
                "task":
                    "R",
                "endpoint":
                    "auprc",
                "method":
                    method,
                "n_volumes":
                    EXPECTED_VOLUMES,
                "estimate":
                    estimate_auprc,
                "ci_low":
                    float(auprc_ci[0]),
                "ci_high":
                    float(auprc_ci[1]),
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

        estimate_auroc = float(
            np.mean(
                auroc_matrix[
                    :,
                    method_index,
                ]
            )
        )

        auroc_ci = percentile_interval(
            bootstrap_auroc[
                :,
                method_index,
            ]
        )

        secondary_rows.append(
            {
                "threshold_name":
                    threshold_name,
                "threshold_value":
                    threshold_value,
                "task":
                    "R",
                "endpoint":
                    "auroc",
                "method":
                    method,
                "n_volumes":
                    EXPECTED_VOLUMES,
                "estimate":
                    estimate_auroc,
                "ci_low":
                    float(auroc_ci[0]),
                "ci_high":
                    float(auroc_ci[1]),
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

    for first_method, second_method in (
        itertools.combinations(
            METHOD_ORDER,
            2,
        )
    ):
        first_index = METHOD_ORDER.index(
            first_method
        )

        second_index = METHOD_ORDER.index(
            second_method
        )

        observed_difference = (
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

        bootstrap_difference = (
            bootstrap_auprc[
                :,
                first_index,
            ]
            -
            bootstrap_auprc[
                :,
                second_index,
            ]
        )

        difference_ci = percentile_interval(
            bootstrap_difference
        )

        pairwise_rows.append(
            {
                "threshold_name":
                    threshold_name,
                "threshold_value":
                    threshold_value,
                "task":
                    "R",
                "endpoint":
                    "auprc",
                "first_method":
                    first_method,
                "second_method":
                    second_method,
                "difference_definition":
                    "first_minus_second",
                "observed_mean_difference":
                    float(
                        np.mean(
                            observed_difference
                        )
                    ),
                "ci_low":
                    float(
                        difference_ci[0]
                    ),
                "ci_high":
                    float(
                        difference_ci[1]
                    ),
                "direction":
                    "higher_is_better",
                "bootstrap_probability_first_better":
                    float(
                        np.mean(
                            bootstrap_difference
                            > 0.0
                        )
                    ),
                "bootstrap_replicates":
                    BOOTSTRAP_REPLICATES,
                "bootstrap_seed":
                    BOOTSTRAP_SEED,
            }
        )

        first_values = auprc_matrix[
            :,
            first_index,
        ]

        second_values = auprc_matrix[
            :,
            second_index,
        ]

        win_rate_rows.append(
            {
                "threshold_name":
                    threshold_name,
                "threshold_value":
                    threshold_value,
                "task":
                    "R",
                "endpoint":
                    "auprc",
                "first_method":
                    first_method,
                "second_method":
                    second_method,
                "direction":
                    "higher_is_better",
                "first_method_volume_win_rate":
                    float(
                        np.mean(
                            first_values
                            > second_values
                        )
                    ),
                "volume_tie_rate":
                    float(
                        np.mean(
                            first_values
                            == second_values
                        )
                    ),
                "n_volumes":
                    EXPECTED_VOLUMES,
            }
        )


primary_summary = pd.DataFrame(
    primary_rows
)

secondary_summary = pd.DataFrame(
    secondary_rows
)

pairwise = pd.DataFrame(
    pairwise_rows
)

win_rates = pd.DataFrame(
    win_rate_rows
)


# ------------------------------------------------------------
# 10. Structural validation
# ------------------------------------------------------------
if len(primary_summary) != 20:
    raise RuntimeError(
        "Expected 20 primary summary rows."
    )

if len(secondary_summary) != 20:
    raise RuntimeError(
        "Expected 20 secondary summary rows."
    )

if len(pairwise) != 90:
    raise RuntimeError(
        "Expected 90 paired comparisons."
    )

if len(win_rates) != 90:
    raise RuntimeError(
        "Expected 90 win-rate records."
    )

for threshold_name in THRESHOLDS:
    threshold_methods = set(
        primary_summary[
            primary_summary[
                "threshold_name"
            ] == threshold_name
        ][
            "method"
        ]
    )

    if threshold_methods != set(
        METHOD_ORDER
    ):
        raise RuntimeError(
            f"Incomplete method set at {threshold_name}."
        )


# ------------------------------------------------------------
# 11. Write outputs atomically
# ------------------------------------------------------------
OUTPUT_DIR.parent.mkdir(
    parents=True,
    exist_ok=True,
)

temporary_directory = Path(
    tempfile.mkdtemp(
        prefix=(
            "exp010_task_r_threshold_sensitivity_"
            "temporary_"
        ),
        dir=OUTPUT_DIR.parent,
    )
)

try:
    output_paths = {
        "task_r_sensitivity_per_volume.csv":
            temporary_directory
            / "task_r_sensitivity_per_volume.csv",

        "task_r_sensitivity_primary_summary.csv":
            temporary_directory
            / "task_r_sensitivity_primary_summary.csv",

        "task_r_sensitivity_secondary_summary.csv":
            temporary_directory
            / "task_r_sensitivity_secondary_summary.csv",

        "task_r_sensitivity_paired_differences.csv":
            temporary_directory
            / "task_r_sensitivity_paired_differences.csv",

        "task_r_sensitivity_win_rates.csv":
            temporary_directory
            / "task_r_sensitivity_win_rates.csv",

        "thresholds.json":
            temporary_directory
            / "thresholds.json",

        "provenance.json":
            temporary_directory
            / "provenance.json",
    }

    atomic_csv_write(
        output_paths[
            "task_r_sensitivity_per_volume.csv"
        ],
        per_volume,
    )

    atomic_csv_write(
        output_paths[
            "task_r_sensitivity_primary_summary.csv"
        ],
        primary_summary,
    )

    atomic_csv_write(
        output_paths[
            "task_r_sensitivity_secondary_summary.csv"
        ],
        secondary_summary,
    )

    atomic_csv_write(
        output_paths[
            "task_r_sensitivity_paired_differences.csv"
        ],
        pairwise,
    )

    atomic_csv_write(
        output_paths[
            "task_r_sensitivity_win_rates.csv"
        ],
        win_rates,
    )

    atomic_json_write(
        output_paths[
            "thresholds.json"
        ],
        {
            "schema_version":
                "exp010-task-r-thresholds-v1.0",
            "source":
                "frozen D_cal quantiles",
            "thresholds":
                THRESHOLDS,
            "q90_reference":
                Q90_REFERENCE,
            "selection_on_D_test":
                False,
        },
    )

    elapsed_seconds = float(
        time.time()
        - start_time
    )

    provenance = {
        "schema_version":
            "exp010-task-r-threshold-sensitivity-v1.0",
        "experiment_id":
            "exp010_task_r_threshold_sensitivity",
        "scientific_status":
            "retrospective frozen-threshold sensitivity analysis",
        "confirmatory_claim":
            False,
        "started_utc":
            started_utc,
        "completed_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "elapsed_seconds":
            elapsed_seconds,
        "repository_branch":
            EXPECTED_BRANCH,
        "repository_commit":
            execution_commit,
        "parent_result_commit":
            EXPECTED_PARENT_COMMIT,
        "paper1_repository_commit":
            paper1_commit,
        "thresholds":
            THRESHOLDS,
        "q90_reference":
            Q90_REFERENCE,
        "D_test_volumes":
            EXPECTED_VOLUMES,
        "D_test_slices":
            EXPECTED_SLICES,
        "D_test_pixels":
            EXPECTED_PIXELS,
        "methods":
            METHOD_ORDER,
        "bootstrap": {
            "replicates":
                BOOTSTRAP_REPLICATES,
            "seed":
                BOOTSTRAP_SEED,
            "confidence_level":
                CONFIDENCE_LEVEL,
            "summary_unit":
                "volume",
            "paired_indices_shared_across_methods":
                True,
            "paired_indices_shared_across_thresholds":
                True,
        },
        "metric_implementations": {
            "paper2_source":
                str(EXP008_SOURCE),
            "paper2_source_sha256":
                sha256_file(
                    EXP008_SOURCE
                ),
            "paper1_gradient_source":
                str(
                    PAPER1_GRADIENT_SOURCE
                ),
            "paper1_gradient_source_sha256":
                sha256_file(
                    PAPER1_GRADIENT_SOURCE
                ),
            "signatures":
                metric_signatures,
        },
        "governance": {
            "model_training_performed":
                False,
            "model_inference_performed":
                False,
            "reconstruction_performed":
                False,
            "target_regeneration_performed":
                False,
            "threshold_selection_performed":
                False,
            "score_sign_selection_performed":
                False,
            "score_transformation_selection_performed":
                False,
            "exp009_q90_results_modified":
                False,
        },
    }

    files_before_provenance = [
        path
        for name, path in output_paths.items()
        if name != "provenance.json"
    ]

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
        for path in files_before_provenance
    }

    atomic_json_write(
        output_paths[
            "provenance.json"
        ],
        provenance,
    )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output appeared during execution:\n"
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
# 12. Final report
# ------------------------------------------------------------
print("\n" + "=" * 120)
print("EXP010 — TASK R THRESHOLD SENSITIVITY")
print("=" * 120)

print(
    f"Repository commit:              "
    f"{execution_commit}"
)

print(
    f"Volumes:                        "
    f"{EXPECTED_VOLUMES}"
)

print(
    f"Slices:                         "
    f"{EXPECTED_SLICES}"
)

print(
    f"Methods:                        "
    f"{len(METHOD_ORDER)}"
)

print(
    f"Thresholds:                     "
    f"{list(THRESHOLDS)}"
)

print(
    f"Per-volume records:             "
    f"{len(per_volume)}"
)

print(
    f"Paired comparisons:             "
    f"{len(pairwise)}"
)

print(
    f"Output directory:               "
    f"{OUTPUT_DIR}"
)

for threshold_name in THRESHOLDS:
    print(
        f"\nPRIMARY AUPRC — "
        f"{threshold_name.upper()}"
    )

    print("-" * 120)

    ranking = (
        primary_summary[
            primary_summary[
                "threshold_name"
            ] == threshold_name
        ][
            [
                "method",
                "estimate",
                "ci_low",
                "ci_high",
            ]
        ]
        .sort_values(
            "estimate",
            ascending=False,
        )
    )

    print(
        ranking.to_string(
            index=False
        )
    )

print("\nGOVERNANCE")
print("-" * 120)

print("Model training performed:       NO")
print("Model inference performed:      NO")
print("Reconstruction performed:       NO")
print("Threshold selected on D_test:   NO")
print("Score sign selected:            NO")
print("Exp009 q90 results modified:    NO")

print(
    f"Elapsed time:                    "
    f"{(time.time() - start_time) / 60:.2f} minutes"
)

print("=" * 120)
print(
    "STATUS: PASS_EXP010_TASK_R_THRESHOLD_SENSITIVITY"
)
