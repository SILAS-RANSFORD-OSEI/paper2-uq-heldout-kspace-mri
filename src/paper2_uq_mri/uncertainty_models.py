"""Frozen neural-model contracts for Paper 2.

Scientific input:
    C_v with shape (B, 3, H, W).

Model codes:
    C0  deterministic point predictor
    U1  MC-dropout point predictor
    U2a point-predictor ensemble
    U2b probabilistic deep ensemble

The trainable Paper 2 models use native three-channel inputs
and are initialized from scratch. The Paper 1 A4 checkpoint
remains a compatibility reference and is not used to initialize
these models.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourway_mri.reliability_model import (
    ReliabilityUNetSmall,
)


MODEL_CODE_C0 = "C0"
MODEL_CODE_U1 = "U1"
MODEL_CODE_U2A = "U2a"
MODEL_CODE_U2B = "U2b"

PAPER2_INPUT_CHANNELS = 3
PAPER2_BASE_CHANNELS = 8

DEFAULT_DROPOUT_PROBABILITY = 0.10
DEFAULT_MC_PASSES = 20
DEFAULT_ENSEMBLE_SIZE = 3
DEFAULT_VARIANCE_FLOOR = 1.0e-6
DEFAULT_LOG_VARIANCE_MINIMUM = -10.0
DEFAULT_LOG_VARIANCE_MAXIMUM = 10.0


class ArchitectureContractError(ValueError):
    """Raised when a Paper 2 model contract is violated."""


def validate_C_v(
    C_v: torch.Tensor,
) -> None:
    """Validate the shared three-channel scientific input."""
    if C_v.ndim != 4:
        raise ArchitectureContractError(
            "C_v must have shape (B, 3, H, W)."
        )

    if C_v.shape[1] != PAPER2_INPUT_CHANNELS:
        raise ArchitectureContractError(
            "C_v must contain exactly three channels."
        )

    if not torch.is_floating_point(C_v):
        raise ArchitectureContractError(
            "C_v must be a floating-point tensor."
        )

    if not torch.isfinite(C_v).all():
        raise ArchitectureContractError(
            "C_v contains non-finite values."
        )


def _seeded_model(
    constructor,
    seed: int,
) -> nn.Module:
    """Construct one independently initialized model."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return constructor()


class DeterministicA4Model(nn.Module):
    """C0: native three-channel deterministic point predictor."""

    model_code = MODEL_CODE_C0

    def __init__(
        self,
        base_channels: int = PAPER2_BASE_CHANNELS,
    ) -> None:
        super().__init__()

        self.base_channels = int(base_channels)

        self.backbone = ReliabilityUNetSmall(
            in_channels=PAPER2_INPUT_CHANNELS,
            out_channels=1,
            base_channels=self.base_channels,
        )

    def forward(
        self,
        C_v: torch.Tensor,
    ) -> torch.Tensor:
        validate_C_v(C_v)

        prediction = self.backbone(C_v)

        expected_shape = (
            C_v.shape[0],
            1,
            C_v.shape[2],
            C_v.shape[3],
        )

        if tuple(prediction.shape) != expected_shape:
            raise ArchitectureContractError(
                "C0 output must have shape (B, 1, H, W)."
            )

        return prediction


class MCDropoutA4Model(
    ReliabilityUNetSmall
):
    """U1: A4 topology with spatial dropout.

    Dropout is applied after every encoder, bottleneck, and
    decoder block. During evaluation it is enabled only when
    MC sampling is explicitly activated.
    """

    model_code = MODEL_CODE_U1

    def __init__(
        self,
        base_channels: int = PAPER2_BASE_CHANNELS,
        dropout_probability: float = (
            DEFAULT_DROPOUT_PROBABILITY
        ),
    ) -> None:
        if not 0.0 < dropout_probability < 1.0:
            raise ArchitectureContractError(
                "dropout_probability must lie in (0, 1)."
            )

        super().__init__(
            in_channels=PAPER2_INPUT_CHANNELS,
            out_channels=1,
            base_channels=int(base_channels),
        )

        self.base_channels = int(base_channels)

        self.dropout_probability = float(
            dropout_probability
        )

        self.mc_sampling_enabled = False

    def set_mc_sampling(
        self,
        enabled: bool,
    ) -> None:
        self.mc_sampling_enabled = bool(enabled)

    def _drop(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return F.dropout2d(
            value,
            p=self.dropout_probability,
            training=(
                self.training
                or self.mc_sampling_enabled
            ),
        )

    def forward(
        self,
        C_v: torch.Tensor,
    ) -> torch.Tensor:
        validate_C_v(C_v)

        e1 = self._drop(
            self.enc1(C_v)
        )

        e2 = self._drop(
            self.enc2(
                self.pool(e1)
            )
        )

        e3 = self._drop(
            self.enc3(
                self.pool(e2)
            )
        )

        bottleneck = self._drop(
            self.bottleneck(
                self.pool(e3)
            )
        )

        d3 = self._upsample_to(
            bottleneck,
            e3,
        )

        d3 = self._drop(
            self.dec3(
                torch.cat(
                    [
                        d3,
                        e3,
                    ],
                    dim=1,
                )
            )
        )

        d2 = self._upsample_to(
            d3,
            e2,
        )

        d2 = self._drop(
            self.dec2(
                torch.cat(
                    [
                        d2,
                        e2,
                    ],
                    dim=1,
                )
            )
        )

        d1 = self._upsample_to(
            d2,
            e1,
        )

        d1 = self._drop(
            self.dec1(
                torch.cat(
                    [
                        d1,
                        e1,
                    ],
                    dim=1,
                )
            )
        )

        prediction = self.out_conv(d1)

        expected_shape = (
            C_v.shape[0],
            1,
            C_v.shape[2],
            C_v.shape[3],
        )

        if tuple(prediction.shape) != expected_shape:
            raise ArchitectureContractError(
                "U1 output must have shape (B, 1, H, W)."
            )

        return prediction

    @torch.no_grad()
    def mc_predict(
        self,
        C_v: torch.Tensor,
        passes: int = DEFAULT_MC_PASSES,
    ) -> dict[str, torch.Tensor]:
        """Return MC samples, mean, and population variance."""
        validate_C_v(C_v)

        if passes < 2:
            raise ArchitectureContractError(
                "MC prediction requires at least two passes."
            )

        previous_training_state = self.training
        previous_mc_state = self.mc_sampling_enabled

        try:
            self.eval()
            self.set_mc_sampling(True)

            samples = torch.stack(
                [
                    self(C_v)
                    for _ in range(int(passes))
                ],
                dim=0,
            )
        finally:
            self.train(previous_training_state)
            self.set_mc_sampling(
                previous_mc_state
            )

        return {
            "samples":
                samples,

            "mean":
                samples.mean(dim=0),

            "variance":
                samples.var(
                    dim=0,
                    unbiased=False,
                ),
        }


class PointPredictorEnsemble(nn.Module):
    """U2a: independently initialized point-predictor ensemble."""

    model_code = MODEL_CODE_U2A

    def __init__(
        self,
        member_seeds: Sequence[int],
        base_channels: int = PAPER2_BASE_CHANNELS,
    ) -> None:
        super().__init__()

        if len(member_seeds) < 2:
            raise ArchitectureContractError(
                "U2a requires at least two members."
            )

        if len(set(member_seeds)) != len(member_seeds):
            raise ArchitectureContractError(
                "U2a member seeds must be unique."
            )

        self.member_seeds = tuple(
            int(seed)
            for seed in member_seeds
        )

        self.base_channels = int(base_channels)

        self.members = nn.ModuleList(
            [
                _seeded_model(
                    lambda: DeterministicA4Model(
                        base_channels=self.base_channels
                    ),
                    seed,
                )
                for seed in self.member_seeds
            ]
        )

    def forward_members(
        self,
        C_v: torch.Tensor,
    ) -> torch.Tensor:
        validate_C_v(C_v)

        return torch.stack(
            [
                member(C_v)
                for member in self.members
            ],
            dim=0,
        )

    def forward(
        self,
        C_v: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_members(
            C_v
        ).mean(dim=0)

    def predictive_statistics(
        self,
        C_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        member_predictions = (
            self.forward_members(C_v)
        )

        return {
            "member_predictions":
                member_predictions,

            "mean":
                member_predictions.mean(
                    dim=0
                ),

            "between_model_variance":
                member_predictions.var(
                    dim=0,
                    unbiased=False,
                ),
        }


class GaussianA4Member(nn.Module):
    """One U2b member predicting mean and variance."""

    def __init__(
        self,
        base_channels: int = PAPER2_BASE_CHANNELS,
        variance_floor: float = DEFAULT_VARIANCE_FLOOR,
        log_variance_minimum: float = (
            DEFAULT_LOG_VARIANCE_MINIMUM
        ),
        log_variance_maximum: float = (
            DEFAULT_LOG_VARIANCE_MAXIMUM
        ),
    ) -> None:
        super().__init__()

        if variance_floor <= 0.0:
            raise ArchitectureContractError(
                "variance_floor must be positive."
            )

        if (
            log_variance_minimum
            >= log_variance_maximum
        ):
            raise ArchitectureContractError(
                "Invalid log-variance bounds."
            )

        self.base_channels = int(base_channels)

        self.variance_floor = float(
            variance_floor
        )

        self.log_variance_minimum = float(
            log_variance_minimum
        )

        self.log_variance_maximum = float(
            log_variance_maximum
        )

        self.backbone = ReliabilityUNetSmall(
            in_channels=PAPER2_INPUT_CHANNELS,
            out_channels=2,
            base_channels=self.base_channels,
        )

    def forward(
        self,
        C_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        validate_C_v(C_v)

        raw_output = self.backbone(C_v)

        mean = raw_output[:, 0:1]

        raw_log_variance = raw_output[:, 1:2]

        bounded_log_variance = torch.clamp(
            raw_log_variance,
            min=self.log_variance_minimum,
            max=self.log_variance_maximum,
        )

        variance = (
            torch.exp(
                bounded_log_variance
            )
            + self.variance_floor
        )

        return {
            "mean":
                mean,

            "variance":
                variance,

            "bounded_log_variance":
                bounded_log_variance,
        }


class ProbabilisticDeepEnsemble(nn.Module):
    """U2b: probabilistic ensemble with variance decomposition."""

    model_code = MODEL_CODE_U2B

    def __init__(
        self,
        member_seeds: Sequence[int],
        base_channels: int = PAPER2_BASE_CHANNELS,
        variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    ) -> None:
        super().__init__()

        if len(member_seeds) < 2:
            raise ArchitectureContractError(
                "U2b requires at least two members."
            )

        if len(set(member_seeds)) != len(member_seeds):
            raise ArchitectureContractError(
                "U2b member seeds must be unique."
            )

        self.member_seeds = tuple(
            int(seed)
            for seed in member_seeds
        )

        self.base_channels = int(base_channels)

        self.variance_floor = float(
            variance_floor
        )

        self.members = nn.ModuleList(
            [
                _seeded_model(
                    lambda: GaussianA4Member(
                        base_channels=self.base_channels,
                        variance_floor=(
                            self.variance_floor
                        ),
                    ),
                    seed,
                )
                for seed in self.member_seeds
            ]
        )

    def predictive_statistics(
        self,
        C_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        validate_C_v(C_v)

        outputs = [
            member(C_v)
            for member in self.members
        ]

        member_means = torch.stack(
            [
                output["mean"]
                for output in outputs
            ],
            dim=0,
        )

        member_variances = torch.stack(
            [
                output["variance"]
                for output in outputs
            ],
            dim=0,
        )

        predictive_mean = (
            member_means.mean(dim=0)
        )

        within_model_variance = (
            member_variances.mean(dim=0)
        )

        between_model_variance = (
            member_means.var(
                dim=0,
                unbiased=False,
            )
        )

        total_predictive_variance = (
            within_model_variance
            + between_model_variance
        )

        return {
            "member_means":
                member_means,

            "member_variances":
                member_variances,

            "mean":
                predictive_mean,

            "within_model_variance":
                within_model_variance,

            "between_model_variance":
                between_model_variance,

            "total_predictive_variance":
                total_predictive_variance,
        }

    def forward(
        self,
        C_v: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictive_statistics(
            C_v
        )["mean"]


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MAE for C0, U1, and U2a."""
    if prediction.shape != target.shape:
        raise ArchitectureContractError(
            "Prediction and target shapes must match."
        )

    absolute_error = torch.abs(
        prediction - target
    )

    if mask is None:
        return absolute_error.mean()

    if mask.shape != target.shape:
        raise ArchitectureContractError(
            "Mask and target shapes must match."
        )

    denominator = mask.sum().clamp_min(1.0)

    return (
        absolute_error * mask
    ).sum() / denominator


def gaussian_nll(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian NLL for each U2b member."""
    if not (
        mean.shape
        == variance.shape
        == target.shape
    ):
        raise ArchitectureContractError(
            "Mean, variance, and target shapes must match."
        )

    if torch.any(variance <= 0):
        raise ArchitectureContractError(
            "Variance must be strictly positive."
        )

    elementwise_loss = 0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(variance)
        + torch.square(target - mean)
        / variance
    )

    if mask is None:
        return elementwise_loss.mean()

    if mask.shape != target.shape:
        raise ArchitectureContractError(
            "Mask and target shapes must match."
        )

    denominator = mask.sum().clamp_min(1.0)

    return (
        elementwise_loss * mask
    ).sum() / denominator
