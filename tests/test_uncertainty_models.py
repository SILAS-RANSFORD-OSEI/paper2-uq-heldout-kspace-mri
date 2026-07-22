"""Architecture-contract tests for C0, U1, U2a, and U2b."""

import pytest
import torch
import torch.nn as nn

from paper2_uq_mri.uncertainty_models import (
    ArchitectureContractError,
    DeterministicA4Model,
    GaussianA4Member,
    MCDropoutA4Model,
    PointPredictorEnsemble,
    ProbabilisticDeepEnsemble,
    gaussian_nll,
    masked_mae,
)


MEMBER_SEEDS = [
    20260720,
    20260721,
    20260722,
]


def first_convolution(
    model: nn.Module,
) -> nn.Conv2d:
    return next(
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    )


def test_all_members_use_native_three_channel_input() -> None:
    C0 = DeterministicA4Model()
    U1 = MCDropoutA4Model()
    U2a = PointPredictorEnsemble(
        MEMBER_SEEDS
    )
    U2b = ProbabilisticDeepEnsemble(
        MEMBER_SEEDS
    )

    assert first_convolution(C0).in_channels == 3
    assert first_convolution(U1).in_channels == 3

    assert all(
        first_convolution(member).in_channels == 3
        for member in U2a.members
    )

    assert all(
        first_convolution(member).in_channels == 3
        for member in U2b.members
    )


def test_six_channel_input_is_rejected() -> None:
    model = DeterministicA4Model()

    with pytest.raises(
        ArchitectureContractError,
        match="exactly three",
    ):
        model(
            torch.zeros(
                1,
                6,
                16,
                16,
            )
        )


def test_C0_output_contract() -> None:
    model = DeterministicA4Model()
    model.eval()

    C_v = torch.randn(
        2,
        3,
        17,
        19,
    )

    with torch.no_grad():
        output_1 = model(C_v)
        output_2 = model(C_v)

    assert output_1.shape == (
        2,
        1,
        17,
        19,
    )

    assert torch.equal(
        output_1,
        output_2,
    )


def test_U1_deterministic_and_MC_contracts() -> None:
    model = MCDropoutA4Model(
        dropout_probability=0.10
    )

    model.eval()

    C_v = torch.randn(
        1,
        3,
        17,
        19,
    )

    with torch.no_grad():
        deterministic_1 = model(C_v)
        deterministic_2 = model(C_v)

    assert torch.equal(
        deterministic_1,
        deterministic_2,
    )

    statistics = model.mc_predict(
        C_v,
        passes=5,
    )

    assert statistics["samples"].shape == (
        5,
        1,
        1,
        17,
        19,
    )

    assert statistics["mean"].shape == (
        1,
        1,
        17,
        19,
    )

    assert statistics["variance"].shape == (
        1,
        1,
        17,
        19,
    )

    assert torch.all(
        statistics["variance"] >= 0
    )

    assert torch.any(
        statistics["variance"] > 0
    )


def test_U2a_three_member_contract() -> None:
    model = PointPredictorEnsemble(
        MEMBER_SEEDS
    )

    model.eval()

    C_v = torch.randn(
        1,
        3,
        17,
        19,
    )

    with torch.no_grad():
        statistics = (
            model.predictive_statistics(C_v)
        )

    assert len(model.members) == 3

    assert statistics[
        "member_predictions"
    ].shape == (
        3,
        1,
        1,
        17,
        19,
    )

    assert statistics["mean"].shape == (
        1,
        1,
        17,
        19,
    )

    assert statistics[
        "between_model_variance"
    ].shape == (
        1,
        1,
        17,
        19,
    )

    assert torch.any(
        statistics[
            "between_model_variance"
        ] > 0
    )


def test_U2b_variance_decomposition() -> None:
    model = ProbabilisticDeepEnsemble(
        MEMBER_SEEDS
    )

    model.eval()

    C_v = torch.randn(
        1,
        3,
        17,
        19,
    )

    with torch.no_grad():
        statistics = (
            model.predictive_statistics(C_v)
        )

    assert len(model.members) == 3

    assert statistics[
        "member_means"
    ].shape == (
        3,
        1,
        1,
        17,
        19,
    )

    assert statistics[
        "member_variances"
    ].shape == (
        3,
        1,
        1,
        17,
        19,
    )

    assert torch.all(
        statistics[
            "member_variances"
        ] > 0
    )

    assert torch.allclose(
        statistics[
            "total_predictive_variance"
        ],
        (
            statistics[
                "within_model_variance"
            ]
            + statistics[
                "between_model_variance"
            ]
        ),
    )


def test_independent_seed_initialization() -> None:
    ensemble = PointPredictorEnsemble(
        MEMBER_SEEDS
    )

    first_weights = [
        first_convolution(member)
        .weight.detach()
        for member in ensemble.members
    ]

    assert not torch.equal(
        first_weights[0],
        first_weights[1],
    )

    assert not torch.equal(
        first_weights[1],
        first_weights[2],
    )


def test_masked_mae_contract() -> None:
    prediction = torch.tensor(
        [[[[1.0, 3.0]]]]
    )

    target = torch.tensor(
        [[[[0.0, 1.0]]]]
    )

    mask = torch.tensor(
        [[[[1.0, 0.0]]]]
    )

    loss = masked_mae(
        prediction,
        target,
        mask,
    )

    assert torch.isclose(
        loss,
        torch.tensor(1.0),
    )


def test_gaussian_nll_is_finite() -> None:
    mean = torch.zeros(
        1,
        1,
        4,
        4,
    )

    variance = torch.ones(
        1,
        1,
        4,
        4,
    )

    target = torch.zeros_like(mean)

    loss = gaussian_nll(
        mean,
        variance,
        target,
    )

    assert torch.isfinite(loss)
