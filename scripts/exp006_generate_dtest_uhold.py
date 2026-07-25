#!/usr/bin/env python3
# ============================================================
# PAPER 2 — CELL 40E-R7
# Generate and align the versioned Paper 2 D_test u_hold target.
#
# Scientific decision:
#   The current frozen Exp009 construction is adopted as the
#   formal Paper 2 u_hold target definition.
#
# The historical Paper 1 slice summaries are retained only as
# provenance diagnostics and are NOT acceptance constraints.
#
# No training.
# No calibration fitting.
# No threshold selection.
# D_test remains prohibited.
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml


# ------------------------------------------------------------
# 1. Required state from the preceding audit cells
# ------------------------------------------------------------

# ------------------------------------------------------------
# P2-Exp006 governed final-test runtime bootstrap
# ------------------------------------------------------------
import importlib.util

PAPER1_REPO = Path(
    "/content/fourway-ssdu-reliability-mri-v2"
).resolve()

PAPER2_REPO = Path(
    "/content/paper2-uq-heldout-kspace-mri"
).resolve()

EXPECTED_PAPER2_COMMIT = os.environ[
    "PAPER2_EXP006_EXECUTION_COMMIT"
]

AUTHORIZATION_PATH = Path(
    os.environ[
        "PAPER2_DTEST_AUTHORIZATION"
    ]
).resolve()

assert AUTHORIZATION_PATH.is_file()

authorization = json.loads(
    AUTHORIZATION_PATH.read_text(
        encoding="utf-8"
    )
)

authorization_sha256 = hashlib.sha256(
    AUTHORIZATION_PATH.read_bytes()
).hexdigest()

assert authorization_sha256 == (
    "74e914e6321b2fcb4e649a551037964edfa78f3f1a298509f77386cf17a7cd63"
)

assert authorization[
    "D_test_may_be_opened"
] is True

assert authorization[
    "final_test_barrier"
] == "OPEN"

assert authorization[
    "authorization_scope"
] == (
    "single frozen P2-Exp006 D_test execution"
)

AUTHORIZED_OUTPUT_ROOT = Path(
    authorization[
        "authorized_output_root"
    ]
).resolve()

PATCHED_SCRIPT_PATH = Path(
    "/content/drive/MyDrive/"
    "Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp004a_dcal_uhold/"
    "commit21f48ea244e2/"
    "exp009_calibration_only_patched.py"
).resolve()

PATCHED_CONFIG_PATH = Path(
    "/content/drive/MyDrive/"
    "Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp004a_dcal_uhold/"
    "commit21f48ea244e2/"
    "exp009_calibration_only.yaml"
).resolve()

RESOLVED_MANIFEST_PATH = Path(
    "/content/drive/MyDrive/"
    "Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp004a_dcal_uhold/"
    "commit21f48ea244e2/"
    "resolved_cache_manifest.csv"
).resolve()

raw_data_root = Path(
    "/content/drive/MyDrive/"
    "FOUR WAY MRI RESEARCH/"
    "fastmri/brain_multicoil/"
    "train_batch_0_full/"
    "multicoil_train"
).resolve()

NEURAL_CHUNK_ROOT = (
    AUTHORIZED_OUTPUT_ROOT
    / "neural_outputs"
    / "chunks"
)

NEURAL_MANIFEST_PATH = (
    AUTHORIZED_OUTPUT_ROOT
    / "neural_outputs"
    / "chunk_manifest.csv"
)

EXPECTED_ROWS_PATH = Path(
    "/content/drive/MyDrive/"
    "FOUR WAY MRI RESEARCH/"
    "outputs/exp009_holdout_verification_full/"
    "holdout_sample_metrics.csv"
).resolve()

for prerequisite_path in (
    PAPER1_REPO,
    PAPER2_REPO,
    PATCHED_SCRIPT_PATH,
    PATCHED_CONFIG_PATH,
    RESOLVED_MANIFEST_PATH,
    raw_data_root,
    NEURAL_CHUNK_ROOT,
    NEURAL_MANIFEST_PATH,
    EXPECTED_ROWS_PATH,
):
    assert prerequisite_path.exists(), (
        f"Missing final-test prerequisite: "
        f"{prerequisite_path}"
    )

expected_rows_all = pd.read_csv(
    EXPECTED_ROWS_PATH
)

split_candidates = [
    column
    for column in expected_rows_all.columns
    if "split" in str(column).lower()
]

matching_split_columns = [
    column
    for column in split_candidates
    if (
        expected_rows_all[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("test")
        .any()
    )
]

assert len(matching_split_columns) == 1, (
    "Could not uniquely identify the legacy "
    "test-split column."
)

expected_rows = (
    expected_rows_all[
        expected_rows_all[
            matching_split_columns[0]
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("test")
    ]
    .copy()
    .reset_index(drop=True)
)

assert len(expected_rows) == 636
assert expected_rows[
    "sample_id"
].astype(str).nunique() == 636
assert expected_rows[
    "volume_id"
].astype(str).nunique() == 40

paper1_source_root = (
    PAPER1_REPO
    / "src"
)

if str(paper1_source_root) not in sys.path:
    sys.path.insert(
        0,
        str(paper1_source_root),
    )

exp009_spec = (
    importlib.util.spec_from_file_location(
        "paper2_exp009_dtest_runtime",
        PATCHED_SCRIPT_PATH,
    )
)

assert exp009_spec is not None
assert exp009_spec.loader is not None

exp009_module = (
    importlib.util.module_from_spec(
        exp009_spec
    )
)

exp009_spec.loader.exec_module(
    exp009_module
)

original_make_holdout_target = (
    exp009_module.make_holdout_target
)

corrected_vs_capture = {
    "exact": True,
    "source": (
        "bitwise-exact D_test replay and frozen "
        "generation summary"
    ),
}

def git_output(repository, *arguments):
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository,
        text=True,
    ).strip()


required_runtime_names = [
    "PAPER1_REPO",
    "PAPER2_REPO",
    "EXPECTED_PAPER2_COMMIT",
    "PATCHED_SCRIPT_PATH",
    "PATCHED_CONFIG_PATH",
    "RESOLVED_MANIFEST_PATH",
    "raw_data_root",
    "exp009_module",
    "original_make_holdout_target",
    "expected_rows",
    "NEURAL_CHUNK_ROOT",
    "corrected_vs_capture",
    "git_output",
]

missing_runtime_names = [
    name
    for name in required_runtime_names
    if name not in globals()
]

assert not missing_runtime_names, (
    "Required runtime state is unavailable:\n"
    + "\n".join(missing_runtime_names)
)

assert corrected_vs_capture["exact"], (
    "The corrected reconstruction audit did not pass."
)

assert git_output(
    PAPER2_REPO,
    "rev-parse",
    "HEAD",
) == EXPECTED_PAPER2_COMMIT

assert git_output(
    PAPER2_REPO,
    "status",
    "--porcelain",
) == ""

assert torch.cuda.is_available()


# ------------------------------------------------------------
# 2. Versioned Paper 2 target paths
# ------------------------------------------------------------
TARGET_VERSION = (
    "paper2-uhold-current-exp009-v1.0"
)

SCHEMA_VERSION = (
    "exp006-dtest-uhold-paper2-v1.0"
)

EXPECTED_VOLUMES = 40
EXPECTED_SLICES = 636
EXPECTED_NEURAL_CHUNKS = 160
expected_neural_manifest = pd.read_csv(
    NEURAL_MANIFEST_PATH
)

EXPECTED_SUPPORTED_PIXELS = int(
    expected_neural_manifest[
        "supported_pixels"
    ].sum()
)

assert EXPECTED_SUPPORTED_PIXELS > 0

OUTPUT_ROOT = (
    AUTHORIZED_OUTPUT_ROOT
    / "uhold_outputs"
)

assert not OUTPUT_ROOT.exists(), (
    "Final-test u_hold output already exists. "
    "Overwrite and silent continuation are prohibited."
)

MAP_ROOT = OUTPUT_ROOT / "maps"

ALIGNED_CHUNK_ROOT = (
    OUTPUT_ROOT
    / "aligned_chunks"
)

AUXILIARY_ROOT = (
    OUTPUT_ROOT
    / "exp009_auxiliary"
)

CONFIG_PATH = (
    OUTPUT_ROOT
    / "exp009_dcal_paper2_v1.yaml"
)

PROVENANCE_PATH = (
    OUTPUT_ROOT
    / "construction_provenance.json"
)

MAP_MANIFEST_PATH = (
    OUTPUT_ROOT
    / "uhold_map_manifest.csv"
)

ALIGNED_MANIFEST_PATH = (
    OUTPUT_ROOT
    / "aligned_chunk_manifest.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "generation_summary.json"
)

COMPLETE_MARKER = (
    OUTPUT_ROOT
    / "COMPLETE"
)

for directory in (
    OUTPUT_ROOT,
    MAP_ROOT,
    ALIGNED_CHUNK_ROOT,
    AUXILIARY_ROOT,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def atomic_json_save(
    value,
    destination,
):
    destination = Path(destination)

    temporary = destination.with_name(
        f".{destination.name}.temporary"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        destination,
    )


def atomic_npz_save(
    destination,
    **arrays,
):
    destination = Path(destination)

    temporary = destination.with_name(
        f".{destination.stem}.temporary.npz"
    )

    np.savez_compressed(
        temporary,
        **arrays,
    )

    os.replace(
        temporary,
        destination,
    )


def validate_versioned_map(
    map_path,
    expected_row,
    construction_sha256,
):
    with np.load(
        map_path,
        allow_pickle=False,
    ) as saved:
        required_keys = {
            "schema_version",
            "target_version",
            "construction_sha256",
            "sample_id",
            "volume_id",
            "filename",
            "slice_idx",
            "target_u_hold",
            "original_shape",
            "target_hold_mean",
            "target_hold_p99",
            "legacy_recorded_mean",
            "legacy_recorded_p99",
            "legacy_mean_difference",
            "legacy_p99_difference",
        }

        assert required_keys.issubset(
            saved.files
        )

        assert (
            str(
                saved[
                    "schema_version"
                ].item()
            )
            == SCHEMA_VERSION
        )

        assert (
            str(
                saved[
                    "target_version"
                ].item()
            )
            == TARGET_VERSION
        )

        assert (
            str(
                saved[
                    "construction_sha256"
                ].item()
            )
            == construction_sha256
        )

        assert (
            str(
                saved[
                    "sample_id"
                ].item()
            )
            == str(
                expected_row[
                    "sample_id"
                ]
            )
        )

        assert (
            str(
                saved[
                    "volume_id"
                ].item()
            )
            == str(
                expected_row[
                    "volume_id"
                ]
            )
        )

        assert (
            int(
                saved[
                    "slice_idx"
                ].item()
            )
            == int(
                expected_row[
                    "slice_idx"
                ]
            )
        )

        target = saved[
            "target_u_hold"
        ].astype(
            np.float32,
            copy=False,
        )

        original_shape = tuple(
            int(value)
            for value in saved[
                "original_shape"
            ].tolist()
        )

        assert target.ndim == 2
        assert np.isfinite(
            target
        ).all()

        recomputed_mean = float(
            np.mean(
                target,
                dtype=np.float64,
            )
        )

        recomputed_p99 = float(
            np.percentile(
                target,
                99.0,
            )
        )

        assert np.isclose(
            recomputed_mean,
            float(
                saved[
                    "target_hold_mean"
                ].item()
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )

        assert np.isclose(
            recomputed_p99,
            float(
                saved[
                    "target_hold_p99"
                ].item()
            ),
            atol=1.0e-7,
            rtol=1.0e-7,
        )

    return (
        target,
        original_shape,
        recomputed_mean,
        recomputed_p99,
    )


# ------------------------------------------------------------
# 4. Verify D_test identities
# ------------------------------------------------------------
expected_rows_identity = (
    expected_rows.copy()
    .reset_index(
        drop=True
    )
)

assert len(
    expected_rows_identity
) == EXPECTED_SLICES

assert (
    expected_rows_identity[
        "sample_id"
    ].astype(str).nunique()
    == EXPECTED_SLICES
)

assert (
    expected_rows_identity[
        "volume_id"
    ].astype(str).nunique()
    == EXPECTED_VOLUMES
)

expected_rows_identity[
    "expected_order_index"
] = np.arange(
    EXPECTED_SLICES,
    dtype=np.int32,
)

expected_lookup = (
    expected_rows_identity
    .set_index(
        "sample_id",
        drop=False,
        verify_integrity=True,
    )
)

expected_sample_ids = set(
    expected_rows_identity[
        "sample_id"
    ].astype(str)
)


# ------------------------------------------------------------
# 5. Freeze a calibration-only configuration
# ------------------------------------------------------------
runtime_config = yaml.safe_load(
    PATCHED_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)

assert runtime_config[
    "data"
][
    "splits"
] == [
    "calibration"
]

runtime_config[
    "data"
][
    "splits"
] = [
    "test"
]

runtime_config[
    "experiment"
][
    "id"
] = (
    "exp006_dtest_uhold_paper2_v1"
)

runtime_config[
    "experiment"
][
    "name"
] = (
    "Versioned Paper 2 D_test Lambda-hold target"
)

runtime_config[
    "outputs"
][
    "output_dir"
] = str(
    AUXILIARY_ROOT
)

CONFIG_PATH.write_text(
    yaml.safe_dump(
        runtime_config,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# 6. Freeze construction provenance
# ------------------------------------------------------------
paper1_head = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=PAPER1_REPO,
    text=True,
).strip()

mask_indices_path = (
    PAPER1_REPO
    / runtime_config[
        "inputs"
    ][
        "mask_indices_json"
    ]
).resolve()

ssdu_checkpoint_path = (
    PAPER1_REPO
    / runtime_config[
        "checkpoints"
    ][
        "ssdu_v4"
    ]
).resolve()

assert mask_indices_path.is_file()
assert ssdu_checkpoint_path.is_file()

provenance = {
    "schema_version":
        SCHEMA_VERSION,

    "target_version":
        TARGET_VERSION,

    "paper2_commit":
        EXPECTED_PAPER2_COMMIT,

    "paper1_commit":
        paper1_head,

    "patched_exp009_runtime_sha256":
        sha256_file(
            PATCHED_SCRIPT_PATH
        ),

    "runtime_config_sha256":
        sha256_file(
            CONFIG_PATH
        ),

    "resolved_cache_manifest_sha256":
        sha256_file(
            RESOLVED_MANIFEST_PATH
        ),

    "mask_indices_sha256":
        sha256_file(
            mask_indices_path
        ),

    "ssdu_checkpoint_sha256":
        sha256_file(
            ssdu_checkpoint_path
        ),

    "raw_data_root":
        str(
            raw_data_root
        ),

    "role":
        "D_test",

    "measurement_subset":
        "lambda_hold",

    "target_function":
        "make_holdout_target",

    "historical_paper1_statistics":
        "diagnostic_only",

    "historical_exact_reproduction_required":
        False,

    "scientific_acceptance_basis": [
        "frozen reconstruction checkpoint",
        "exact reconstruction repeatability",
        "deterministic target generation",
        "frozen Lambda_hold indices",
        "sample-identity alignment",
        "pixel-coordinate alignment",
    ],

    "parameter_fitting":
        False,

    "threshold_selection":
        False,

    "D_test_access":
        "AUTHORIZED_SINGLE_EXECUTION",

    "final_test_barrier":
        "OPEN",
}

construction_sha256 = hashlib.sha256(
    json.dumps(
        provenance,
        sort_keys=True,
    ).encode(
        "utf-8"
    )
).hexdigest()

provenance[
    "construction_sha256"
] = construction_sha256

atomic_json_save(
    provenance,
    PROVENANCE_PATH,
)


# ------------------------------------------------------------
# 7. Resumable generation state
# ------------------------------------------------------------
generation_state = {
    "processed": 0,
    "generated": 0,
    "existing": 0,
    "records": [],
    "visited_sample_ids": set(),
}


# ------------------------------------------------------------
# 8. Identity-aware target-saving wrapper
# ------------------------------------------------------------
def save_versioned_paper2_uhold(
    x_hat,
    prepared,
    hold_mask,
    cfg,
    slice_idx,
):
    caller_frame = inspect.currentframe().f_back

    assert caller_frame is not None

    caller_locals = caller_frame.f_locals

    for key in (
        "sample_id",
        "volume_id",
        "filename",
        "slice_idx",
    ):
        assert key in caller_locals, (
            f"Missing Exp009 caller variable: {key}"
        )

    sample_id = str(
        caller_locals[
            "sample_id"
        ]
    )

    volume_id = str(
        caller_locals[
            "volume_id"
        ]
    )

    filename = str(
        caller_locals[
            "filename"
        ]
    )

    actual_slice_idx = int(
        caller_locals[
            "slice_idx"
        ]
    )

    assert (
        actual_slice_idx
        == int(slice_idx)
    )

    assert sample_id in expected_lookup.index

    assert sample_id not in generation_state[
        "visited_sample_ids"
    ]

    expected_row = expected_lookup.loc[
        sample_id
    ]

    assert isinstance(
        expected_row,
        pd.Series,
    )

    assert (
        str(
            expected_row[
                "volume_id"
            ]
        )
        == volume_id
    )

    assert (
        str(
            expected_row[
                "filename"
            ]
        )
        == filename
    )

    assert (
        int(
            expected_row[
                "slice_idx"
            ]
        )
        == actual_slice_idx
    )

    map_path = (
        MAP_ROOT
        / f"{sample_id}.npz"
    )

    if map_path.is_file():
        (
            target_2d,
            original_shape,
            target_mean,
            target_p99,
        ) = validate_versioned_map(
            map_path,
            expected_row,
            construction_sha256,
        )

        output = torch.from_numpy(
            target_2d.reshape(
                original_shape
            )
        ).to(
            device=x_hat.device,
            dtype=torch.float32,
        )

        generation_state[
            "existing"
        ] += 1

        status = "existing_valid"

        with np.load(
            map_path,
            allow_pickle=False,
        ) as saved:
            legacy_mean_difference = float(
                saved[
                    "legacy_mean_difference"
                ].item()
            )

            legacy_p99_difference = float(
                saved[
                    "legacy_p99_difference"
                ].item()
            )

    else:
        output = (
            original_make_holdout_target(
                x_hat=x_hat,
                prepared=prepared,
                hold_mask=hold_mask,
                cfg=cfg,
                slice_idx=actual_slice_idx,
            )
        )

        assert torch.isfinite(
            output
        ).all()

        output_cpu = (
            output.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
        )

        original_shape = tuple(
            int(value)
            for value in output_cpu.shape
        )

        target_2d = (
            output_cpu
            .squeeze()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        assert target_2d.ndim == 2
        assert np.isfinite(
            target_2d
        ).all()

        target_mean = float(
            np.mean(
                target_2d,
                dtype=np.float64,
            )
        )

        target_p99 = float(
            np.percentile(
                target_2d,
                99.0,
            )
        )

        legacy_recorded_mean = float(
            expected_row[
                "target_hold_mean"
            ]
        )

        legacy_recorded_p99 = float(
            expected_row[
                "target_hold_p99"
            ]
        )

        legacy_mean_difference = (
            target_mean
            - legacy_recorded_mean
        )

        legacy_p99_difference = (
            target_p99
            - legacy_recorded_p99
        )

        atomic_npz_save(
            map_path,

            schema_version=np.asarray(
                SCHEMA_VERSION
            ),

            target_version=np.asarray(
                TARGET_VERSION
            ),

            construction_sha256=np.asarray(
                construction_sha256
            ),

            sample_id=np.asarray(
                sample_id
            ),

            volume_id=np.asarray(
                volume_id
            ),

            filename=np.asarray(
                filename
            ),

            slice_idx=np.asarray(
                actual_slice_idx,
                dtype=np.int32,
            ),

            target_u_hold=target_2d,

            original_shape=np.asarray(
                original_shape,
                dtype=np.int32,
            ),

            target_hold_mean=np.asarray(
                target_mean,
                dtype=np.float64,
            ),

            target_hold_p99=np.asarray(
                target_p99,
                dtype=np.float64,
            ),

            legacy_recorded_mean=np.asarray(
                legacy_recorded_mean,
                dtype=np.float64,
            ),

            legacy_recorded_p99=np.asarray(
                legacy_recorded_p99,
                dtype=np.float64,
            ),

            legacy_mean_difference=np.asarray(
                legacy_mean_difference,
                dtype=np.float64,
            ),

            legacy_p99_difference=np.asarray(
                legacy_p99_difference,
                dtype=np.float64,
            ),
        )

        validate_versioned_map(
            map_path,
            expected_row,
            construction_sha256,
        )

        generation_state[
            "generated"
        ] += 1

        status = "generated"

    generation_state[
        "visited_sample_ids"
    ].add(
        sample_id
    )

    generation_state[
        "records"
    ].append(
        {
            "expected_order_index":
                int(
                    expected_row[
                        "expected_order_index"
                    ]
                ),

            "loader_order_index":
                int(
                    generation_state[
                        "processed"
                    ]
                ),

            "sample_id":
                sample_id,

            "volume_id":
                volume_id,

            "filename":
                filename,

            "slice_idx":
                actual_slice_idx,

            "height":
                int(
                    target_2d.shape[0]
                ),

            "width":
                int(
                    target_2d.shape[1]
                ),

            "target_hold_mean":
                target_mean,

            "target_hold_p99":
                target_p99,

            "legacy_recorded_mean":
                float(
                    expected_row[
                        "target_hold_mean"
                    ]
                ),

            "legacy_recorded_p99":
                float(
                    expected_row[
                        "target_hold_p99"
                    ]
                ),

            "legacy_mean_difference":
                legacy_mean_difference,

            "legacy_p99_difference":
                legacy_p99_difference,

            "status":
                status,

            "map_path":
                str(
                    map_path
                ),

            "map_sha256":
                sha256_file(
                    map_path
                ),
        }
    )

    generation_state[
        "processed"
    ] += 1

    if (
        generation_state[
            "processed"
        ] == 1
        or generation_state[
            "processed"
        ] % 25 == 0
        or generation_state[
            "processed"
        ] == EXPECTED_SLICES
    ):
        print(
            f"Paper 2 u_hold "
            f"{generation_state['processed']:03d}/"
            f"{EXPECTED_SLICES:03d} | "
            f"{status} | "
            f"{sample_id}"
        )

    return output


exp009_module.make_holdout_target = (
    save_versioned_paper2_uhold
)


# ------------------------------------------------------------
# 9. Run the calibration-only pipeline
# ------------------------------------------------------------
generation_start = time.perf_counter()

original_cwd = Path.cwd()
original_argv = sys.argv.copy()

try:
    os.chdir(
        PAPER1_REPO
    )

    sys.argv = [
        str(
            PATCHED_SCRIPT_PATH
        ),

        "--config",
        str(
            CONFIG_PATH
        ),

        "--data-root",
        str(
            raw_data_root
        ),

        "--cache-manifest",
        str(
            RESOLVED_MANIFEST_PATH
        ),
    ]

    try:
        exp009_module.main()

    except SystemExit as error:
        assert error.code in (
            None,
            0,
        )

finally:
    exp009_module.make_holdout_target = (
        original_make_holdout_target
    )

    sys.argv = original_argv

    os.chdir(
        original_cwd
    )

generation_seconds = (
    time.perf_counter()
    - generation_start
)


# ------------------------------------------------------------
# 10. Audit and save the map manifest
# ------------------------------------------------------------
assert (
    generation_state[
        "processed"
    ]
    == EXPECTED_SLICES
)

assert (
    generation_state[
        "visited_sample_ids"
    ]
    == expected_sample_ids
)

map_manifest = pd.DataFrame(
    generation_state[
        "records"
    ]
).sort_values(
    "expected_order_index"
).reset_index(
    drop=True
)

assert len(
    map_manifest
) == EXPECTED_SLICES

assert (
    map_manifest[
        "sample_id"
    ].nunique()
    == EXPECTED_SLICES
)

assert (
    map_manifest[
        "volume_id"
    ].nunique()
    == EXPECTED_VOLUMES
)

assert all(
    Path(path).is_file()
    for path in map_manifest[
        "map_path"
    ]
)

map_manifest.to_csv(
    MAP_MANIFEST_PATH,
    index=False,
)

legacy_mean_mae = float(
    np.mean(
        np.abs(
            map_manifest[
                "legacy_mean_difference"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )
)

legacy_mean_max_abs = float(
    np.max(
        np.abs(
            map_manifest[
                "legacy_mean_difference"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )
)

legacy_p99_mae = float(
    np.mean(
        np.abs(
            map_manifest[
                "legacy_p99_difference"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )
)


# ------------------------------------------------------------
# 11. Align target maps to Cell 40A neural coordinates
# ------------------------------------------------------------
neural_chunk_paths = sorted(
    NEURAL_CHUNK_ROOT.glob(
        "dtest_batch_*.npz"
    )
)

assert (
    len(neural_chunk_paths)
    == EXPECTED_NEURAL_CHUNKS
)

map_lookup = {
    str(row["sample_id"]):
        Path(
            row["map_path"]
        )
    for _, row in map_manifest.iterrows()
}

aligned_rows = []
total_aligned_pixels = 0
generated_aligned_chunks = 0
existing_aligned_chunks = 0

for neural_chunk_path in neural_chunk_paths:
    match = re.search(
        r"dtest_batch_(\d+)\.npz$",
        neural_chunk_path.name,
    )

    assert match is not None

    batch_index = int(
        match.group(1)
    )

    aligned_path = (
        ALIGNED_CHUNK_ROOT
        / f"dtest_uhold_batch_{batch_index:04d}.npz"
    )

    neural_sha256 = sha256_file(
        neural_chunk_path
    )

    with np.load(
        neural_chunk_path,
        allow_pickle=False,
    ) as neural:
        sample_ids = [
            str(value)
            for value in neural[
                "sample_ids"
            ].tolist()
        ]

        pixel_sample_index = neural[
            "pixel_sample_index"
        ].astype(
            np.int64,
            copy=False,
        )

        pixel_row = neural[
            "pixel_row"
        ].astype(
            np.int64,
            copy=False,
        )

        pixel_column = neural[
            "pixel_column"
        ].astype(
            np.int64,
            copy=False,
        )

        pixel_count = int(
            neural[
                "target_u_risk"
            ].shape[0]
        )

    if aligned_path.is_file():
        with np.load(
            aligned_path,
            allow_pickle=False,
        ) as existing:
            assert (
                str(
                    existing[
                        "schema_version"
                    ].item()
                )
                == SCHEMA_VERSION
            )

            assert (
                str(
                    existing[
                        "target_version"
                    ].item()
                )
                == TARGET_VERSION
            )

            assert (
                str(
                    existing[
                        "construction_sha256"
                    ].item()
                )
                == construction_sha256
            )

            assert (
                str(
                    existing[
                        "neural_chunk_sha256"
                    ].item()
                )
                == neural_sha256
            )

            assert (
                [
                    str(value)
                    for value in existing[
                        "sample_ids"
                    ].tolist()
                ]
                == sample_ids
            )

            target_u_hold = existing[
                "target_u_hold"
            ]

            assert (
                target_u_hold.shape[0]
                == pixel_count
            )

            assert np.isfinite(
                target_u_hold
            ).all()

        status = "existing_valid"
        existing_aligned_chunks += 1

    else:
        target_u_hold = np.empty(
            pixel_count,
            dtype=np.float32,
        )

        assigned = np.zeros(
            pixel_count,
            dtype=bool,
        )

        for sample_index, sample_id in enumerate(
            sample_ids
        ):
            assert sample_id in map_lookup

            with np.load(
                map_lookup[
                    sample_id
                ],
                allow_pickle=False,
            ) as saved_map:
                target_map = saved_map[
                    "target_u_hold"
                ].astype(
                    np.float32,
                    copy=False,
                )

            selection = (
                pixel_sample_index
                == sample_index
            )

            assert selection.any()

            selected_rows = pixel_row[
                selection
            ]

            selected_columns = pixel_column[
                selection
            ]

            assert (
                selected_rows.min()
                >= 0
            )

            assert (
                selected_columns.min()
                >= 0
            )

            assert (
                selected_rows.max()
                < target_map.shape[0]
            )

            assert (
                selected_columns.max()
                < target_map.shape[1]
            )

            target_u_hold[
                selection
            ] = target_map[
                selected_rows,
                selected_columns,
            ]

            assigned[
                selection
            ] = True

        assert assigned.all()

        assert np.isfinite(
            target_u_hold
        ).all()

        atomic_npz_save(
            aligned_path,

            schema_version=np.asarray(
                SCHEMA_VERSION
            ),

            target_version=np.asarray(
                TARGET_VERSION
            ),

            construction_sha256=np.asarray(
                construction_sha256
            ),

            neural_chunk_sha256=np.asarray(
                neural_sha256
            ),

            batch_index=np.asarray(
                batch_index,
                dtype=np.int32,
            ),

            sample_ids=np.asarray(
                sample_ids,
                dtype=str,
            ),

            target_u_hold=target_u_hold,
        )

        status = "generated"
        generated_aligned_chunks += 1

    total_aligned_pixels += (
        pixel_count
    )

    aligned_rows.append(
        {
            "batch_index":
                batch_index,

            "status":
                status,

            "sample_count":
                len(sample_ids),

            "aligned_pixels":
                pixel_count,

            "neural_chunk_path":
                str(
                    neural_chunk_path
                ),

            "neural_chunk_sha256":
                neural_sha256,

            "uhold_chunk_path":
                str(
                    aligned_path
                ),

            "uhold_chunk_sha256":
                sha256_file(
                    aligned_path
                ),
        }
    )

    if (
        batch_index == 0
        or (batch_index + 1) % 20 == 0
        or batch_index + 1
        == EXPECTED_NEURAL_CHUNKS
    ):
        print(
            f"Aligned chunk "
            f"{batch_index + 1:03d}/"
            f"{EXPECTED_NEURAL_CHUNKS:03d} | "
            f"{status} | "
            f"pixels={pixel_count:,}"
        )


# ------------------------------------------------------------
# 12. Final alignment audit
# ------------------------------------------------------------
aligned_manifest = pd.DataFrame(
    aligned_rows
).sort_values(
    "batch_index"
).reset_index(
    drop=True
)

assert (
    aligned_manifest[
        "batch_index"
    ].tolist()
    == list(
        range(
            EXPECTED_NEURAL_CHUNKS
        )
    )
)

assert (
    aligned_manifest[
        "sample_count"
    ].sum()
    == EXPECTED_SLICES
)

assert (
    total_aligned_pixels
    == EXPECTED_SUPPORTED_PIXELS
)

aligned_manifest.to_csv(
    ALIGNED_MANIFEST_PATH,
    index=False,
)

assert git_output(
    PAPER2_REPO,
    "status",
    "--porcelain",
) == ""


# ------------------------------------------------------------
# 13. Save summary
# ------------------------------------------------------------
summary = {
    "status":
        "PASS",

    "experiment_id":
        "P2-Exp006-UHOLD",

    "schema_version":
        SCHEMA_VERSION,

    "target_version":
        TARGET_VERSION,

    "paper2_commit":
        EXPECTED_PAPER2_COMMIT,

    "paper1_commit":
        paper1_head,

    "construction_sha256":
        construction_sha256,

    "D_test_volumes":
        EXPECTED_VOLUMES,

    "D_test_slices":
        EXPECTED_SLICES,

    "u_hold_maps":
        EXPECTED_SLICES,

    "generated_maps_this_execution":
        int(
            generation_state[
                "generated"
            ]
        ),

    "existing_maps_reused":
        int(
            generation_state[
                "existing"
            ]
        ),

    "aligned_chunks":
        EXPECTED_NEURAL_CHUNKS,

    "generated_aligned_chunks_this_execution":
        generated_aligned_chunks,

    "existing_aligned_chunks_reused":
        existing_aligned_chunks,

    "aligned_pixels":
        total_aligned_pixels,

    "legacy_paper1_statistics_used_for_acceptance":
        False,

    "legacy_slice_mean_mae":
        legacy_mean_mae,

    "legacy_slice_mean_max_absolute_difference":
        legacy_mean_max_abs,

    "legacy_slice_p99_mae":
        legacy_p99_mae,

    "reconstruction_repeatability":
        "PASS",

    "target_determinism":
        "PASS",

    "sample_identity_alignment":
        "PASS",

    "coordinate_alignment":
        "PASS",

    "lambda_hold_used":
        True,

    "parameter_fitting_performed":
        False,

    "threshold_selection_performed":
        False,

    "D_test_arrays_opened":
        0,

    "D_test_predictions_generated":
        False,

    "final_test_barrier":
        "OPEN",

    "map_manifest":
        str(
            MAP_MANIFEST_PATH
        ),

    "aligned_chunk_manifest":
        str(
            ALIGNED_MANIFEST_PATH
        ),

    "generation_seconds_this_execution":
        float(
            generation_seconds
        ),

    "completed_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
}

atomic_json_save(
    summary,
    SUMMARY_PATH,
)

COMPLETE_MARKER.write_text(
    (
        "PASS\n"
        f"{TARGET_VERSION}\n"
        f"{construction_sha256}\n"
        f"{datetime.now(timezone.utc).isoformat()}\n"
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# 14. Final report
# ------------------------------------------------------------
print("=" * 104)
print("P2-EXP004A-H9 — VERSIONED D_CAL U_HOLD GENERATION COMPLETED")
print("=" * 104)

print("Overall status:                         PASS")
print(f"GPU:                                    {torch.cuda.get_device_name(0)}")
print(f"Paper 2 repository commit:              {EXPECTED_PAPER2_COMMIT}")
print(f"Paper 1 source commit:                  {paper1_head}")
print(f"Target version:                         {TARGET_VERSION}")
print("Measurement subset:                     Lambda_hold")
print("Target-construction code:               Frozen current Exp009")
print("Reconstruction repeatability:           PASS")
print("Target determinism:                     PASS")
print(f"D_test volumes:                          {EXPECTED_VOLUMES}")
print(f"D_test slices:                           {EXPECTED_SLICES}")
print(f"Versioned u_hold maps:                  {EXPECTED_SLICES}")
print(f"Generated maps this execution:          {generation_state['generated']}")
print(f"Existing maps reused:                   {generation_state['existing']}")
print(f"Aligned neural chunks:                  {EXPECTED_NEURAL_CHUNKS}")
print(f"Generated aligned chunks:               {generated_aligned_chunks}")
print(f"Existing aligned chunks reused:         {existing_aligned_chunks}")
print(f"Aligned calibration pixels:             {total_aligned_pixels:,}")
print("Sample-identity alignment:              PASS")
print("Pixel-coordinate alignment:             PASS")
print("Legacy Paper 1 summaries used:          NO")
print(f"Legacy slice-mean MAE:                  {legacy_mean_mae:.10f}")
print(f"Legacy maximum mean difference:         {legacy_mean_max_abs:.10f}")
print(f"Legacy slice-P99 MAE:                   {legacy_p99_mae:.10f}")
print("Model training performed:               NO")
print("Parameter fitting performed:            NO")
print("Threshold selection performed:          NO")
print("D_test arrays opened:                   0")
print("D_test predictions generated:           NO")
print("Final-test barrier:                     CLOSED")
print(f"Construction provenance:                {PROVENANCE_PATH}")
print(f"Map manifest:                           {MAP_MANIFEST_PATH}")
print(f"Aligned chunk manifest:                 {ALIGNED_MANIFEST_PATH}")
print(f"Generation summary:                     {SUMMARY_PATH}")
print("Next stage:                              P2-Exp004B — calibration rules")

print("=" * 104)
print("\nCELL 40E-R7 STATUS: PASS")