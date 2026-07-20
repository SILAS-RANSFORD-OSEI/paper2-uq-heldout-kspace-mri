"""Focused checks for the full-cache audit entry point."""

from pathlib import Path


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "exp000_audit_full_cache.py"
)


def test_full_cache_audit_script_exists() -> None:
    assert SCRIPT_PATH.is_file()


def test_full_cache_audit_uses_semantic_loader() -> None:
    text = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        "load_reliability_cache_sample"
        in text
    )

    assert (
        "verify_transform=True"
        in text
    )


def test_full_cache_audit_does_not_use_pickle() -> None:
    text = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "allow_pickle=True" not in text
