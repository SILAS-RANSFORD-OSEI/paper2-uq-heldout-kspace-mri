"""Compatibility interface for the frozen Paper 1 A4 model.

Paper 2 exposes only the three-channel predictor input C_v.

The Paper 1 A4 ablation retained channels 0-2 and zeroed
channels 3-5 before passing the input to its six-channel
ReliabilityUNetSmall. The adapter reproduces that operation
internally and prevents auxiliary channels from entering the
Paper 2 scientific interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class A4CompatibilityError(ValueError):
    """Raised when the frozen A4 interface is violated."""


class A4ThreeChannelAdapter(nn.Module):
    """Run a frozen six-channel A4 model from three-channel C_v."""

    def __init__(
        self,
        frozen_model: nn.Module,
    ) -> None:
        super().__init__()

        first_convolution = next(
            (
                module
                for module in frozen_model.modules()
                if isinstance(module, nn.Conv2d)
            ),
            None,
        )

        if first_convolution is None:
            raise A4CompatibilityError(
                "Frozen model contains no Conv2d layer."
            )

        if first_convolution.in_channels != 6:
            raise A4CompatibilityError(
                "Frozen A4 model must expect six channels."
            )

        self.frozen_model = frozen_model

    def expand_C_v(
        self,
        C_v: torch.Tensor,
    ) -> torch.Tensor:
        """Append three exact-zero channels to C_v."""
        if C_v.ndim != 4:
            raise A4CompatibilityError(
                "C_v must have shape (B, 3, H, W)."
            )

        if C_v.shape[1] != 3:
            raise A4CompatibilityError(
                "C_v must contain exactly three channels."
            )

        zero_channels = torch.zeros(
            (
                C_v.shape[0],
                3,
                C_v.shape[2],
                C_v.shape[3],
            ),
            dtype=C_v.dtype,
            device=C_v.device,
        )

        expanded = torch.cat(
            [
                C_v,
                zero_channels,
            ],
            dim=1,
        )

        if expanded.shape[1] != 6:
            raise A4CompatibilityError(
                "Expanded input must contain six channels."
            )

        return expanded

    def forward(
        self,
        C_v: torch.Tensor,
    ) -> torch.Tensor:
        expanded = self.expand_C_v(C_v)
        prediction = self.frozen_model(expanded)

        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)

        if (
            prediction.ndim != 4
            or prediction.shape[1] != 1
        ):
            raise A4CompatibilityError(
                "A4 output must have shape (B, 1, H, W)."
            )

        return prediction
