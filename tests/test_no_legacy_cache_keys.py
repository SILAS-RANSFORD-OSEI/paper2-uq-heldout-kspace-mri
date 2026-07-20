"""Prevent legacy NPZ names from leaking into analysis modules."""

from pathlib import Path
import re


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

SOURCE_ROOT = (
    REPOSITORY_ROOT
    / "src"
    / "paper2_uq_mri"
)

ALLOWED_BOUNDARY_MODULES = {
    "cache.py",
}

LEGACY_INDEX_PATTERN = re.compile(
    r"""\[\s*["'](?:x|y|y_raw)["']\s*\]"""
)


def test_legacy_cache_indexing_is_confined_to_loader() -> None:
    violations = []

    for path in SOURCE_ROOT.rglob(
        "*.py"
    ):
        if (
            path.name
            in ALLOWED_BOUNDARY_MODULES
        ):
            continue

        text = path.read_text(
            encoding="utf-8"
        )

        matches = list(
            LEGACY_INDEX_PATTERN.finditer(
                text
            )
        )

        if matches:
            violations.append(
                str(
                    path.relative_to(
                        REPOSITORY_ROOT
                    )
                )
            )

    assert violations == [], (
        "Legacy cache keys escaped the boundary loader: "
        + ", ".join(
            violations
        )
    )
