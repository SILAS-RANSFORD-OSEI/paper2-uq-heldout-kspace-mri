"""Tests for the frozen A4 compatibility adapter."""

import pytest
import torch
import torch.nn as nn

from paper2_uq_mri.a4_compatibility import (
    A4CompatibilityError,
    A4ThreeChannelAdapter,
)


class DummyA4Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.convolution = nn.Conv2d(
            6,
            1,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self.convolution(value)


def test_adapter_appends_exact_zero_channels() -> None:
    adapter = A4ThreeChannelAdapter(
        DummyA4Model()
    )

    C_v = torch.randn(
        2,
        3,
        8,
        7,
    )

    expanded = adapter.expand_C_v(C_v)

    assert expanded.shape == (
        2,
        6,
        8,
        7,
    )

    assert torch.equal(
        expanded[:, :3],
        C_v,
    )

    assert (
        torch.count_nonzero(
            expanded[:, 3:]
        ).item()
        == 0
    )


def test_adapter_matches_direct_zero_padding() -> None:
    model = DummyA4Model()
    adapter = A4ThreeChannelAdapter(model)

    C_v = torch.randn(
        2,
        3,
        8,
        7,
    )

    expanded = torch.cat(
        [
            C_v,
            torch.zeros_like(C_v),
        ],
        dim=1,
    )

    assert torch.equal(
        adapter(C_v),
        model(expanded),
    )


def test_adapter_rejects_six_channel_interface() -> None:
    adapter = A4ThreeChannelAdapter(
        DummyA4Model()
    )

    with pytest.raises(
        A4CompatibilityError,
        match="exactly three",
    ):
        adapter(
            torch.zeros(
                1,
                6,
                8,
                7,
            )
        )
