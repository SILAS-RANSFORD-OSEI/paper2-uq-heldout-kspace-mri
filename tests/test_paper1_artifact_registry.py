"""Tests for the canonical reusable Paper 1 registry."""

from pathlib import Path

from paper2_uq_mri.provenance import (
    iter_artifact_records,
    load_artifact_registry,
    validate_registry_schema,
)


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

REGISTRY_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "protocol"
    / "paper1_reusable_artifacts_v1.2.json"
)


def test_registry_schema_passes() -> None:
    registry = load_artifact_registry(
        REGISTRY_PATH
    )

    errors = validate_registry_schema(
        registry
    )

    assert errors == [], "\n".join(
        errors
    )


def test_registry_contains_no_absolute_paths() -> None:
    registry = load_artifact_registry(
        REGISTRY_PATH
    )

    for (
        _,
        _,
        record,
    ) in iter_artifact_records(
        registry
    ):
        path = Path(
            record["relative_path"]
        )

        assert not path.is_absolute()
        assert ".." not in path.parts


def test_registry_uses_canonical_source_commit() -> None:
    registry = load_artifact_registry(
        REGISTRY_PATH
    )

    assert (
        registry[
            "paper1_source_repository"
        ]["commit"]
        ==
        "da563ead8fb653539e1eeca29248b31f0121ca12"
    )


def test_registry_uses_canonical_checkpoints() -> None:
    registry = load_artifact_registry(
        REGISTRY_PATH
    )

    artifacts = registry[
        "drive_artifacts"
    ]

    assert (
        artifacts[
            "a4_checkpoint"
        ]["relative_path"]
        ==
        "outputs/exp008_reliability_ablation_full/"
        "A4_image_only/best_model.pt"
    )

    assert (
        artifacts[
            "ssdu_checkpoint"
        ]["relative_path"]
        ==
        "outputs/exp004_train_ssdu_v4_full/"
        "best_model.pt"
    )


def test_required_artifacts_are_registered() -> None:
    registry = load_artifact_registry(
        REGISTRY_PATH
    )

    required_source = {
        "dataset_manifest",
        "paper1_split_manifest",
        "fourway_mask_config",
        "mask_generation_script",
        "mask_module",
        "mask_test",
        "a4_training_config",
        "holdout_verification_config",
    }

    required_drive = {
        "a4_checkpoint",
        "a4_summary",
        "a4_final_split_metrics",
        "ssdu_checkpoint",
        "reliability_cache_manifest",
        "reliability_cache_summary",
    }

    assert required_source.issubset(
        registry[
            "source_artifacts"
        ]
    )

    assert required_drive.issubset(
        registry[
            "drive_artifacts"
        ]
    )
