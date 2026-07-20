"""Paper 1 artifact-registry and provenance utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(
    path: Path | str,
    block_size: int = 1024 * 1024,
) -> str:
    """Return a file SHA-256 checksum."""
    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_artifact_registry(
    path: Path | str,
) -> dict[str, Any]:
    """Load a Paper 1 reusable-artifact registry."""
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = json.load(file)

    if not isinstance(registry, dict):
        raise ValueError(
            "Artifact registry must contain a JSON object."
        )

    return registry


def iter_artifact_records(
    registry: dict[str, Any],
):
    """Yield source and Drive artifact records."""
    for group_name in [
        "source_artifacts",
        "drive_artifacts",
    ]:
        group = registry.get(
            group_name,
            {}
        )

        for artifact_key, record in group.items():
            if record is None:
                continue

            yield (
                group_name,
                artifact_key,
                record,
            )


def validate_registry_schema(
    registry: dict[str, Any],
) -> list[str]:
    """Return all path-safe registry schema errors."""
    errors: list[str] = []

    metadata = registry.get(
        "registry",
        {}
    )

    if metadata.get("status") != "PASS":
        errors.append(
            "Registry status must be PASS."
        )

    source = registry.get(
        "paper1_source_repository",
        {}
    )

    commit = str(
        source.get("commit", "")
    )

    if len(commit) != 40:
        errors.append(
            "Paper 1 source commit must be a "
            "40-character Git hash."
        )

    for (
        group_name,
        artifact_key,
        record,
    ) in iter_artifact_records(registry):

        relative_path = Path(
            record.get(
                "relative_path",
                ""
            )
        )

        if not str(relative_path):
            errors.append(
                f"{group_name}.{artifact_key} "
                "has no relative path."
            )

        if relative_path.is_absolute():
            errors.append(
                f"{group_name}.{artifact_key} "
                "contains an absolute path."
            )

        if ".." in relative_path.parts:
            errors.append(
                f"{group_name}.{artifact_key} "
                "escapes its declared root."
            )

        checksum = str(
            record.get("sha256", "")
        )

        if len(checksum) != 64:
            errors.append(
                f"{group_name}.{artifact_key} "
                "has an invalid SHA-256 value."
            )

        size_bytes = record.get(
            "size_bytes"
        )

        if (
            not isinstance(
                size_bytes,
                int,
            )
            or size_bytes <= 0
        ):
            errors.append(
                f"{group_name}.{artifact_key} "
                "has an invalid file size."
            )

    return errors


def validate_artifact_files(
    registry: dict[str, Any],
    source_root: Path | str,
    drive_root: Path | str,
) -> list[dict[str, Any]]:
    """Validate artifact existence, size, and hash."""
    source_root = Path(
        source_root
    ).resolve()

    drive_root = Path(
        drive_root
    ).resolve()

    roots = {
        "source_artifacts":
            source_root,

        "drive_artifacts":
            drive_root,
    }

    checks = []

    for (
        group_name,
        artifact_key,
        record,
    ) in iter_artifact_records(registry):

        root = roots[group_name]

        path = (
            root
            / record["relative_path"]
        ).resolve()

        try:
            path.relative_to(root)
            within_root = True

        except ValueError:
            within_root = False

        exists = (
            within_root
            and path.is_file()
        )

        observed_size = (
            path.stat().st_size
            if exists
            else None
        )

        observed_hash = (
            sha256_file(path)
            if exists
            else None
        )

        size_matches = (
            observed_size
            ==
            record["size_bytes"]
        )

        hash_matches = (
            observed_hash
            ==
            record["sha256"]
        )

        passed = (
            within_root
            and exists
            and size_matches
            and hash_matches
        )

        checks.append(
            {
                "group":
                    group_name,

                "artifact_key":
                    artifact_key,

                "relative_path":
                    record[
                        "relative_path"
                    ],

                "within_root":
                    within_root,

                "exists":
                    exists,

                "expected_size_bytes":
                    record[
                        "size_bytes"
                    ],

                "observed_size_bytes":
                    observed_size,

                "size_matches":
                    size_matches,

                "expected_sha256":
                    record[
                        "sha256"
                    ],

                "observed_sha256":
                    observed_hash,

                "hash_matches":
                    hash_matches,

                "status":
                    (
                        "PASS"
                        if passed
                        else "FAIL"
                    ),
            }
        )

    return checks
