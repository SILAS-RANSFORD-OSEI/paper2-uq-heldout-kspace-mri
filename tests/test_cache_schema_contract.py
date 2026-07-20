"""Tests for the frozen reliability-cache semantic contract."""

from pathlib import Path

from paper2_uq_mri.cache_contract import (
    load_cache_contract,
    validate_cache_contract,
)


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "protocol"
    / "cache_schema_contract_v1.0.yaml"
)


def test_cache_contract_has_no_errors() -> None:
    contract = load_cache_contract(
        CONTRACT_PATH
    )

    errors = validate_cache_contract(
        contract
    )

    assert errors == [], "\n".join(
        errors
    )


def test_matrix_size_is_template_based() -> None:
    contract = load_cache_contract(
        CONTRACT_PATH
    )

    template = contract[
        "spatial_template"
    ]

    assert (
        template[
            "fixed_matrix_size_required"
        ]
        is False
    )

    assert [
        640,
        320,
    ] in template[
        "permitted_observed_sizes"
    ]

    assert [
        768,
        396,
    ] in template[
        "permitted_observed_sizes"
    ]


def test_a4_input_is_exactly_three_channels() -> None:
    contract = load_cache_contract(
        CONTRACT_PATH
    )

    a4 = contract[
        "a4_predictor_input"
    ]

    assert a4[
        "channel_indices"
    ] == [
        0,
        1,
        2,
    ]

    assert a4[
        "prohibited_channels"
    ] == [
        3,
        4,
        5,
    ]
