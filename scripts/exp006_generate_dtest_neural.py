#!/usr/bin/env python3
# ============================================================
# PAPER 2 — CELL 40A
# P2-Exp006: Generate frozen neural outputs on D_test
#
# Models:
#   C0  — deterministic control mean
#   U1  — MC-dropout mean and population variance, T=20
#   U2a — point-ensemble mean and between-model variance
#   U2b — probabilistic ensemble mean plus within-, between-,
#         and total predictive variance
#
# Outputs:
#   - compressed, resumable NPZ chunks
#   - supported-pixel coordinates and weights
#   - u_risk_v calibration target
#   - chunk manifest and generation summary
#
# No parameter fitting.
# No threshold selection.
# No gradients.
# D_test remains prohibited.
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# 1. Frozen paths and identifiers
# ------------------------------------------------------------
REPO = Path(
    "/content/paper2-uq-heldout-kspace-mri"
).resolve()

SOURCE_ROOT = REPO / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )

LOCAL_MANIFEST = Path(
    "/content/paper2_exp000_cache_manifest_local.csv"
).resolve()

SPLIT_PATH = (
    REPO
    / "data"
    / "splits"
    / "paper2_split.csv"
)

EXPECTED_COMMIT = (
    os.environ["PAPER2_EXP006_EXECUTION_COMMIT"]
)

C0_CHECKPOINT = Path(
    "/content/drive/MyDrive/"
    "Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp003a_c0/"
    "seed20260720_commit9091c53af159_fp32/"
    "best_model.pt"
).resolve()

U1_CHECKPOINT = Path(
    "/content/drive/MyDrive/"
    "Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp003b_u1/"
    "seed20260720_p010_mc20_commitf7d1205b8909_fp32/"
    "best_model.pt"
).resolve()

U2A_CHECKPOINT = Path(
    "/content/drive/MyDrive/"
    "Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp003c_u2a/"
    "ensemble3_commit0dff1d39fc25_fp32/"
    "ensemble_best.pt"
).resolve()

U2B_CHECKPOINT = Path(
    "/content/drive/MyDrive/"
    "Paper2_UQ_Heldout_KSpace_MRI/"
    "outputs/exp003d_u2b/"
    "ensemble3_commit985ef999e2ad_fp32/"
    "ensemble_best.pt"
).resolve()

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

OUTPUT_ROOT = (
    AUTHORIZED_OUTPUT_ROOT
    / "neural_outputs"
)

assert not OUTPUT_ROOT.exists(), (
    "Final-test neural output already exists. "
    "Overwrite and silent continuation are prohibited."
)

CHUNK_ROOT = (
    OUTPUT_ROOT
    / "chunks"
)

RUN_CONFIG_PATH = (
    OUTPUT_ROOT
    / "generation_config.json"
)

MANIFEST_PATH = (
    OUTPUT_ROOT
    / "chunk_manifest.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "generation_summary.json"
)

COMPLETE_MARKER = (
    OUTPUT_ROOT
    / "COMPLETE"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNK_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MEMBER_SEEDS = (
    20260720,
    20260721,
    20260722,
)

BASE_SEED = 20260730
MC_PASSES = 20
BATCH_SIZE = 4
EXPECTED_D_TEST_SLICES = 636
EXPECTED_D_TEST_VOLUMES = 40
SCHEMA_VERSION = "exp006-dtest-neural-v1.0"


# ------------------------------------------------------------
# 2. General helpers
# ------------------------------------------------------------
def git_output(*arguments):
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPO,
        text=True,
    ).strip()


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


def load_checkpoint(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint is not a dictionary."
        )

    for key in (
        "model_state_dict",
        "ensemble_state_dict",
        "state_dict",
    ):
        candidate = checkpoint.get(key)

        if isinstance(candidate, dict):
            return candidate, key

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint, "root_state_dict"

    raise KeyError(
        "No supported state-dictionary key found."
    )


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


def normalize_mean_output(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in (
            "mean",
            "prediction",
            "predicted_mean",
            "output",
        ):
            value = output.get(key)

            if torch.is_tensor(value):
                return value

    if (
        isinstance(output, (tuple, list))
        and len(output) >= 1
        and torch.is_tensor(output[0])
    ):
        return output[0]

    raise TypeError(
        "Unsupported point-prediction output."
    )


def normalize_mean_variance_output(output):
    if isinstance(output, dict):
        mean = None
        variance = None

        for key in (
            "mean",
            "predictive_mean",
            "prediction",
        ):
            if torch.is_tensor(
                output.get(key)
            ):
                mean = output[key]
                break

        for key in (
            "variance",
            "predictive_variance",
            "population_variance",
        ):
            if torch.is_tensor(
                output.get(key)
            ):
                variance = output[key]
                break

        if mean is not None and variance is not None:
            return mean, variance

    if (
        isinstance(output, (tuple, list))
        and len(output) >= 2
        and torch.is_tensor(output[0])
        and torch.is_tensor(output[1])
    ):
        return output[0], output[1]

    raise TypeError(
        "Unsupported mean/variance output."
    )


def volume_id_from_sample_id(sample_id):
    return re.sub(
        r"_slice\d+$",
        "",
        str(sample_id),
    )


# ------------------------------------------------------------
# 3. Verify frozen repository and checkpoints
# ------------------------------------------------------------
assert torch.cuda.is_available(), (
    "CUDA is unavailable."
)

assert git_output(
    "rev-parse",
    "HEAD",
) == EXPECTED_COMMIT

assert git_output(
    "status",
    "--porcelain",
) == ""

assert LOCAL_MANIFEST.is_file()
assert SPLIT_PATH.is_file()

for checkpoint_path in (
    C0_CHECKPOINT,
    U1_CHECKPOINT,
    U2A_CHECKPOINT,
    U2B_CHECKPOINT,
):
    assert checkpoint_path.is_file(), (
        f"Missing checkpoint: {checkpoint_path}"
    )

checkpoint_hashes = {
    "C0": sha256_file(
        C0_CHECKPOINT
    ),
    "U1": sha256_file(
        U1_CHECKPOINT
    ),
    "U2a": sha256_file(
        U2A_CHECKPOINT
    ),
    "U2b": sha256_file(
        U2B_CHECKPOINT
    ),
}


# ------------------------------------------------------------
# 4. Frozen generation configuration
# ------------------------------------------------------------
generation_config = {
    "schema_version":
        SCHEMA_VERSION,

    "experiment_id":
        "P2-Exp006",

    "repository_commit":
        EXPECTED_COMMIT,

    "role":
        "D_test",

    "purpose":
        "single frozen final-test neural-output generation",

    "input":
        "three-channel C_v",

    "target":
        "u_risk_v",

    "support":
        "M_soft",

    "batch_size":
        BATCH_SIZE,

    "MC_dropout_passes":
        MC_PASSES,

    "MC_dropout_variance":
        "population variance",

    "U2a_between_model_variance":
        "population variance",

    "U2b_between_model_variance":
        "population variance",

    "member_seeds":
        list(MEMBER_SEEDS),

    "numerical_precision":
        "FP32",

    "automatic_mixed_precision":
        False,

    "tf32_enabled":
        False,

    "checkpoint_paths": {
        "C0": str(C0_CHECKPOINT),
        "U1": str(U1_CHECKPOINT),
        "U2a": str(U2A_CHECKPOINT),
        "U2b": str(U2B_CHECKPOINT),
    },

    "checkpoint_sha256":
        checkpoint_hashes,

    "parameter_fitting":
        False,

    "threshold_selection":
        False,

    "D_test_access":
        "AUTHORIZED_SINGLE_EXECUTION",

    "final_test_barrier":
        "OPEN",
}

configuration_sha256 = hashlib.sha256(
    json.dumps(
        generation_config,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

generation_config[
    "configuration_sha256"
] = configuration_sha256

if RUN_CONFIG_PATH.is_file():
    existing_config = json.loads(
        RUN_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        existing_config[
            "configuration_sha256"
        ]
        == configuration_sha256
    ), (
        "Existing D_test output directory uses "
        "a different frozen configuration."
    )

else:
    atomic_json_save(
        generation_config,
        RUN_CONFIG_PATH,
    )


# ------------------------------------------------------------
# 5. Deterministic FP32 runtime
# ------------------------------------------------------------
os.environ["PYTHONHASHSEED"] = str(
    BASE_SEED
)

random.seed(BASE_SEED)
np.random.seed(BASE_SEED)

torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed_all(BASE_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.use_deterministic_algorithms(
    True,
    warn_only=True,
)

device = torch.device("cuda:0")


# ------------------------------------------------------------
# 6. Reload frozen project modules
# ------------------------------------------------------------
import paper2_uq_mri.batching as batching_module
import paper2_uq_mri.split_access as split_access_module
import paper2_uq_mri.uncertainty_models as model_module

batching_module = importlib.reload(
    batching_module
)

split_access_module = importlib.reload(
    split_access_module
)

model_module = importlib.reload(
    model_module
)


BARRIER_OPEN = getattr(
    split_access_module,
    "BARRIER_OPEN",
)

FinalTestBarrier = getattr(
    split_access_module,
    "FinalTestBarrier",
)

final_evaluation_constant_name = (
    "PURPOSE_FINAL_EVALUATION"
)

final_evaluation_purpose = getattr(
    split_access_module,
    final_evaluation_constant_name,
)

open_barrier = FinalTestBarrier(
    BARRIER_OPEN
)


# ------------------------------------------------------------
# 7. Construct frozen models
# ------------------------------------------------------------
C0_MODEL = (
    model_module.DeterministicA4Model(
        base_channels=8,
    )
).to(device)

U1_MODEL = (
    model_module.MCDropoutA4Model(
        base_channels=8,
        dropout_probability=0.10,
    )
).to(device)

U2A_MODEL = (
    model_module.PointPredictorEnsemble(
        member_seeds=MEMBER_SEEDS,
        base_channels=8,
    )
).to(device)

U2B_MODEL = (
    model_module.ProbabilisticDeepEnsemble(
        member_seeds=MEMBER_SEEDS,
        base_channels=8,
        variance_floor=1.0e-6,
    )
).to(device)

models = {
    "C0": C0_MODEL,
    "U1": U1_MODEL,
    "U2a": U2A_MODEL,
    "U2b": U2B_MODEL,
}

checkpoint_paths = {
    "C0": C0_CHECKPOINT,
    "U1": U1_CHECKPOINT,
    "U2a": U2A_CHECKPOINT,
    "U2b": U2B_CHECKPOINT,
}

checkpoint_state_keys = {}

for model_code, model in models.items():
    checkpoint = load_checkpoint(
        checkpoint_paths[
            model_code
        ]
    )

    state_dict, state_key = (
        extract_state_dict(
            checkpoint
        )
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    checkpoint_state_keys[
        model_code
    ] = state_key

    model.eval()

    assert all(
        torch.isfinite(
            parameter
        ).all()
        for parameter in model.parameters()
    )


# ------------------------------------------------------------
# 8. Build authorized role-locked D_test loader
# ------------------------------------------------------------
cal_loader = (
    batching_module.build_shape_safe_loader(
        cache_manifest_path=str(
            LOCAL_MANIFEST
        ),
        split_path=str(
            SPLIT_PATH
        ),
        purpose=final_evaluation_purpose,
        barrier=open_barrier,
        batch_size=BATCH_SIZE,
        seed=BASE_SEED,
        shuffle=False,
        drop_last=False,
        max_batches_per_shape=None,
        verify_transform=False,
        num_workers=0,
    )
)

assert (
    len(cal_loader.dataset)
    == EXPECTED_D_TEST_SLICES
)

expected_batches = len(
    cal_loader
)


# ------------------------------------------------------------
# 9. Chunk validation helper
# ------------------------------------------------------------
required_chunk_keys = {
    "schema_version",
    "configuration_sha256",
    "batch_index",
    "batch_seed",
    "sample_ids",
    "volume_ids",
    "pixel_sample_index",
    "pixel_row",
    "pixel_column",
    "support_weight",
    "target_u_risk",
    "c0_mean",
    "u1_mean",
    "u1_variance",
    "u2a_mean",
    "u2a_between_model_variance",
    "u2b_mean",
    "u2b_within_model_variance",
    "u2b_between_model_variance",
    "u2b_total_predictive_variance",
}


def inspect_existing_chunk(
    chunk_path,
    expected_batch_index,
):
    with np.load(
        chunk_path,
        allow_pickle=False,
    ) as chunk:
        assert required_chunk_keys.issubset(
            chunk.files
        )

        assert (
            str(
                chunk[
                    "schema_version"
                ].item()
            )
            == SCHEMA_VERSION
        )

        assert (
            str(
                chunk[
                    "configuration_sha256"
                ].item()
            )
            == configuration_sha256
        )

        assert (
            int(
                chunk[
                    "batch_index"
                ].item()
            )
            == expected_batch_index
        )

        supported_pixels = int(
            chunk[
                "target_u_risk"
            ].shape[0]
        )

        sample_ids = [
            str(value)
            for value in chunk[
                "sample_ids"
            ].tolist()
        ]

        for key in (
            "support_weight",
            "target_u_risk",
            "c0_mean",
            "u1_mean",
            "u1_variance",
            "u2a_mean",
            "u2a_between_model_variance",
            "u2b_mean",
            "u2b_within_model_variance",
            "u2b_between_model_variance",
            "u2b_total_predictive_variance",
        ):
            assert (
                chunk[key].shape[0]
                == supported_pixels
            )

            assert np.isfinite(
                chunk[key]
            ).all()

        assert (
            chunk[
                "support_weight"
            ] > 0
        ).all()

        assert (
            chunk[
                "u1_variance"
            ] >= 0
        ).all()

        assert (
            chunk[
                "u2a_between_model_variance"
            ] >= 0
        ).all()

        assert (
            chunk[
                "u2b_within_model_variance"
            ] > 0
        ).all()

        assert (
            chunk[
                "u2b_between_model_variance"
            ] >= 0
        ).all()

        assert (
            chunk[
                "u2b_total_predictive_variance"
            ] > 0
        ).all()

    return sample_ids, supported_pixels


# ------------------------------------------------------------
# 10. Generate or resume D_test chunks
# ------------------------------------------------------------
manifest_rows = []
all_sample_ids = []
generated_batches = 0
skipped_batches = 0
total_supported_pixels = 0

generation_start = time.perf_counter()

for batch_index, batch in enumerate(
    cal_loader
):
    chunk_path = (
        CHUNK_ROOT
        / f"dtest_batch_{batch_index:04d}.npz"
    )

    if chunk_path.is_file():
        (
            existing_sample_ids,
            supported_pixels,
        ) = inspect_existing_chunk(
            chunk_path,
            batch_index,
        )

        all_sample_ids.extend(
            existing_sample_ids
        )

        total_supported_pixels += (
            supported_pixels
        )

        skipped_batches += 1

        manifest_rows.append(
            {
                "batch_index":
                    batch_index,

                "status":
                    "existing_valid",

                "sample_count":
                    len(
                        existing_sample_ids
                    ),

                "supported_pixels":
                    supported_pixels,

                "chunk_path":
                    str(chunk_path),

                "chunk_sha256":
                    sha256_file(
                        chunk_path
                    ),
            }
        )

        print(
            f"Batch {batch_index + 1:03d}/"
            f"{expected_batches:03d}: "
            f"existing valid chunk skipped"
        )

        continue

    assert (
        batch["paper2_role"]
        == "D_test"
    )

    assert (
        batch["purpose"]
        == final_evaluation_purpose
    )

    sample_ids = [
        str(value)
        for value in batch[
            "sample_id"
        ]
    ]

    volume_ids = [
        volume_id_from_sample_id(
            sample_id
        )
        for sample_id in sample_ids
    ]

    C_v = batch["C_v"].to(
        device=device,
        dtype=torch.float32,
    )

    target = batch[
        "u_risk_v"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    support = batch[
        "M_soft"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    assert C_v.ndim == 4
    assert C_v.shape[1] == 3

    assert target.ndim == 4
    assert target.shape[1] == 1

    assert support.ndim == 4
    assert support.shape[1] == 1

    assert C_v.shape[0] == len(
        sample_ids
    )

    assert C_v.shape[-2:] == (
        target.shape[-2:]
    )

    assert C_v.shape[-2:] == (
        support.shape[-2:]
    )

    assert torch.isfinite(C_v).all()
    assert torch.isfinite(target).all()
    assert torch.isfinite(support).all()

    batch_seed = (
        BASE_SEED
        + batch_index
    )

    random.seed(batch_seed)
    np.random.seed(batch_seed)

    torch.manual_seed(batch_seed)
    torch.cuda.manual_seed_all(
        batch_seed
    )

    with torch.inference_mode():
        c0_mean = normalize_mean_output(
            C0_MODEL(C_v)
        )

        (
            u1_mean,
            u1_variance,
        ) = normalize_mean_variance_output(
            U1_MODEL.mc_predict(
                C_v,
                passes=MC_PASSES,
            )
        )

        u2a_statistics = (
            U2A_MODEL.predictive_statistics(
                C_v
            )
        )

        u2a_mean = u2a_statistics[
            "mean"
        ]

        u2a_between_variance = (
            u2a_statistics[
                "between_model_variance"
            ]
        )

        u2b_statistics = (
            U2B_MODEL.predictive_statistics(
                C_v
            )
        )

        u2b_mean = u2b_statistics[
            "mean"
        ]

        u2b_within_variance = (
            u2b_statistics[
                "within_model_variance"
            ]
        )

        u2b_between_variance = (
            u2b_statistics[
                "between_model_variance"
            ]
        )

        u2b_total_variance = (
            u2b_statistics[
                "total_predictive_variance"
            ]
        )

    output_tensors = {
        "c0_mean":
            c0_mean,

        "u1_mean":
            u1_mean,

        "u1_variance":
            u1_variance,

        "u2a_mean":
            u2a_mean,

        "u2a_between_model_variance":
            u2a_between_variance,

        "u2b_mean":
            u2b_mean,

        "u2b_within_model_variance":
            u2b_within_variance,

        "u2b_between_model_variance":
            u2b_between_variance,

        "u2b_total_predictive_variance":
            u2b_total_variance,
    }

    for name, tensor in output_tensors.items():
        assert tensor.shape == target.shape, (
            f"{name} shape mismatch."
        )

        assert torch.isfinite(
            tensor
        ).all(), (
            f"{name} contains non-finite values."
        )

    assert torch.all(
        u1_variance >= 0
    )

    assert torch.all(
        u2a_between_variance >= 0
    )

    assert torch.all(
        u2b_within_variance > 0
    )

    assert torch.all(
        u2b_between_variance >= 0
    )

    assert torch.all(
        u2b_total_variance > 0
    )

    assert torch.equal(
        u2b_total_variance,
        (
            u2b_within_variance
            + u2b_between_variance
        ),
    )

    # --------------------------------------------------------
    # Flatten only supported calibration pixels
    # --------------------------------------------------------
    pixel_sample_index_parts = []
    pixel_row_parts = []
    pixel_column_parts = []

    support_parts = []
    target_parts = []

    output_parts = {
        name: []
        for name in output_tensors
    }

    for sample_index in range(
        C_v.shape[0]
    ):
        valid = (
            support[
                sample_index,
                0,
            ] > 0
        )

        coordinates = torch.nonzero(
            valid,
            as_tuple=False,
        )

        supported_count = int(
            coordinates.shape[0]
        )

        assert supported_count > 0

        pixel_sample_index_parts.append(
            np.full(
                supported_count,
                sample_index,
                dtype=np.uint16,
            )
        )

        pixel_row_parts.append(
            coordinates[
                :,
                0,
            ].cpu().numpy().astype(
                np.int32,
                copy=False,
            )
        )

        pixel_column_parts.append(
            coordinates[
                :,
                1,
            ].cpu().numpy().astype(
                np.int32,
                copy=False,
            )
        )

        support_parts.append(
            support[
                sample_index,
                0,
            ][valid].cpu().numpy().astype(
                np.float32,
                copy=False,
            )
        )

        target_parts.append(
            target[
                sample_index,
                0,
            ][valid].cpu().numpy().astype(
                np.float32,
                copy=False,
            )
        )

        for name, tensor in output_tensors.items():
            output_parts[
                name
            ].append(
                tensor[
                    sample_index,
                    0,
                ][valid].cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )

    pixel_sample_index = np.concatenate(
        pixel_sample_index_parts
    )

    pixel_row = np.concatenate(
        pixel_row_parts
    )

    pixel_column = np.concatenate(
        pixel_column_parts
    )

    support_weight = np.concatenate(
        support_parts
    )

    target_u_risk = np.concatenate(
        target_parts
    )

    flattened_outputs = {
        name: np.concatenate(
            parts
        )
        for name, parts in output_parts.items()
    }

    supported_pixels = int(
        target_u_risk.shape[0]
    )

    assert supported_pixels > 0

    assert (
        pixel_sample_index.shape[0]
        == supported_pixels
    )

    assert (
        pixel_row.shape[0]
        == supported_pixels
    )

    assert (
        pixel_column.shape[0]
        == supported_pixels
    )

    assert (
        support_weight.shape[0]
        == supported_pixels
    )

    for array in flattened_outputs.values():
        assert (
            array.shape[0]
            == supported_pixels
        )

    atomic_npz_save(
        chunk_path,

        schema_version=np.asarray(
            SCHEMA_VERSION
        ),

        configuration_sha256=np.asarray(
            configuration_sha256
        ),

        batch_index=np.asarray(
            batch_index,
            dtype=np.int32,
        ),

        batch_seed=np.asarray(
            batch_seed,
            dtype=np.int64,
        ),

        sample_ids=np.asarray(
            sample_ids,
            dtype=str,
        ),

        volume_ids=np.asarray(
            volume_ids,
            dtype=str,
        ),

        pixel_sample_index=(
            pixel_sample_index
        ),

        pixel_row=pixel_row,

        pixel_column=pixel_column,

        support_weight=(
            support_weight
        ),

        target_u_risk=(
            target_u_risk
        ),

        **flattened_outputs,
    )

    (
        verified_sample_ids,
        verified_supported_pixels,
    ) = inspect_existing_chunk(
        chunk_path,
        batch_index,
    )

    assert (
        verified_sample_ids
        == sample_ids
    )

    assert (
        verified_supported_pixels
        == supported_pixels
    )

    generated_batches += 1

    all_sample_ids.extend(
        sample_ids
    )

    total_supported_pixels += (
        supported_pixels
    )

    manifest_rows.append(
        {
            "batch_index":
                batch_index,

            "status":
                "generated",

            "sample_count":
                len(sample_ids),

            "supported_pixels":
                supported_pixels,

            "chunk_path":
                str(chunk_path),

            "chunk_sha256":
                sha256_file(
                    chunk_path
                ),
        }
    )

    print(
        f"Batch {batch_index + 1:03d}/"
        f"{expected_batches:03d}: "
        f"generated | "
        f"slices={len(sample_ids)} | "
        f"supported pixels="
        f"{supported_pixels:,}"
    )


# ------------------------------------------------------------
# 11. Final completeness and governance audit
# ------------------------------------------------------------
assert len(
    manifest_rows
) == expected_batches

assert len(
    all_sample_ids
) == EXPECTED_D_TEST_SLICES

assert len(
    set(all_sample_ids)
) == EXPECTED_D_TEST_SLICES

all_volume_ids = {
    volume_id_from_sample_id(
        sample_id
    )
    for sample_id in all_sample_ids
}

assert len(
    all_volume_ids
) == EXPECTED_D_TEST_VOLUMES

manifest = pd.DataFrame(
    manifest_rows
).sort_values(
    "batch_index"
)

assert manifest[
    "batch_index"
].tolist() == list(
    range(
        expected_batches
    )
)

assert (
    manifest[
        "sample_count"
    ].sum()
    == EXPECTED_D_TEST_SLICES
)

assert (
    manifest[
        "supported_pixels"
    ].sum()
    == total_supported_pixels
)

manifest.to_csv(
    MANIFEST_PATH,
    index=False,
)

generation_seconds = (
    time.perf_counter()
    - generation_start
)

summary = {
    "status":
        "PASS",

    "experiment_id":
        "P2-Exp006",

    "schema_version":
        SCHEMA_VERSION,

    "configuration_sha256":
        configuration_sha256,

    "repository_commit":
        EXPECTED_COMMIT,

    "final_evaluation_purpose_constant":
        final_evaluation_constant_name,

    "final_evaluation_purpose_value":
        str(final_evaluation_purpose),

    "D_test_volumes":
        EXPECTED_D_TEST_VOLUMES,

    "D_test_slices":
        EXPECTED_D_TEST_SLICES,

    "D_test_batches":
        expected_batches,

    "supported_pixels":
        int(
            total_supported_pixels
        ),

    "generated_batches_this_execution":
        int(
            generated_batches
        ),

    "existing_batches_skipped":
        int(
            skipped_batches
        ),

    "MC_dropout_passes":
        MC_PASSES,

    "checkpoint_state_keys":
        checkpoint_state_keys,

    "checkpoint_sha256":
        checkpoint_hashes,

    "chunk_manifest":
        str(MANIFEST_PATH),

    "chunk_directory":
        str(CHUNK_ROOT),

    "generation_seconds_this_execution":
        float(
            generation_seconds
        ),

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
        f"PASS\n"
        f"{configuration_sha256}\n"
        f"{datetime.now(timezone.utc).isoformat()}\n"
    ),
    encoding="utf-8",
)

assert git_output(
    "status",
    "--porcelain",
) == ""


# ------------------------------------------------------------
# 12. Final report
# ------------------------------------------------------------
print("=" * 98)
print("P2-EXP004A — D_CAL NEURAL OUTPUT GENERATION COMPLETED")
print("=" * 98)

print("Overall status:                         PASS")
print(f"GPU:                                    {torch.cuda.get_device_name(0)}")
print(f"Repository commit:                      {EXPECTED_COMMIT}")
print(f"D_test purpose constant:                 {final_evaluation_constant_name}")
print(f"D_test volumes:                          {EXPECTED_D_TEST_VOLUMES}")
print(f"D_test slices:                           {EXPECTED_D_TEST_SLICES}")
print(f"D_test batches:                          {expected_batches}")
print(f"Supported calibration pixels:           {total_supported_pixels:,}")
print(f"Generated batches this execution:       {generated_batches}")
print(f"Existing valid batches skipped:         {skipped_batches}")
print(f"MC-dropout passes:                      {MC_PASSES}")
print("C0 deterministic means:                 SAVED")
print("U1 MC means and variances:              SAVED")
print("U2a means and between-model variances:  SAVED")
print("U2b means and variance components:      SAVED")
print("Parameter fitting performed:            NO")
print("Threshold selection performed:          NO")
print("D_test arrays opened:                   0")
print("D_test predictions generated:           NO")
print("Final-test barrier:                     CLOSED")
print("Repository modified:                    NO")
print(f"Chunk manifest:                         {MANIFEST_PATH}")
print(f"Generation summary:                     {SUMMARY_PATH}")
print("Next stage:                              P2-Exp004B — calibration rules")

print("=" * 98)
print("\nCELL 40A STATUS: PASS")