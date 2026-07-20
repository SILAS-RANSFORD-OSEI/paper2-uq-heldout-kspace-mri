"""Validation of the frozen reliability-cache contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_LEGACY_KEYS = {
    "x",
    "y",
    "y_raw",
}

REQUIRED_CHANNEL_CODE_NAMES = {
    "reconstruction_magnitude_normalized",
    "zero_filled_magnitude_normalized",
    "intervention_magnitude_normalized",
    "support_mask",
    "analytical_psf",
    "psf_gain_descriptor",
}


def load_cache_contract(
    path: Path | str,
) -> dict[str, Any]:
    """Load the machine-readable cache contract."""
    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        contract = yaml.safe_load(
            file
        )

    if not isinstance(
        contract,
        dict,
    ):
        raise ValueError(
            "Cache contract must be a YAML mapping."
        )

    return contract


def validate_cache_contract(
    contract: dict[str, Any],
) -> list[str]:
    """Return all cache-contract validation errors."""
    errors: list[str] = []

    metadata = contract.get(
        "contract",
        {}
    )

    if metadata.get(
        "version"
    ) != "1.0":
        errors.append(
            "Cache contract version must be 1.0."
        )

    archive = contract.get(
        "archive",
        {}
    )

    legacy_keys = set(
        archive.get(
            "required_legacy_keys",
            [],
        )
    )

    if legacy_keys != REQUIRED_LEGACY_KEYS:
        errors.append(
            "Legacy key set changed. "
            f"Found: {sorted(legacy_keys)}"
        )

    if archive.get(
        "allow_pickle"
    ) is not False:
        errors.append(
            "allow_pickle must remain false."
        )

    legacy_input = contract.get(
        "legacy_input_tensor",
        {}
    )

    channels = legacy_input.get(
        "channels",
        []
    )

    indices = [
        channel.get(
            "index"
        )
        for channel in channels
    ]

    if indices != list(
        range(6)
    ):
        errors.append(
            "Cache channel indices must be 0 through 5."
        )

    channel_code_names = {
        channel.get(
            "semantic_code_name"
        )
        for channel in channels
    }

    if (
        channel_code_names
        != REQUIRED_CHANNEL_CODE_NAMES
    ):
        errors.append(
            "The six-channel semantic mapping changed."
        )

    a4 = contract.get(
        "a4_predictor_input",
        {}
    )

    if a4.get(
        "manuscript_symbol"
    ) != "C_v":
        errors.append(
            "A4 predictor input must use C_v."
        )

    if a4.get(
        "channel_indices"
    ) != [
        0,
        1,
        2,
    ]:
        errors.append(
            "A4 must use only channels 0, 1, and 2."
        )

    risk_target = contract.get(
        "risk_learning_target",
        {}
    )

    if risk_target.get(
        "semantic_code_name"
    ) != "u_risk":
        errors.append(
            "Legacy y must map to u_risk."
        )

    if risk_target.get(
        "manuscript_symbol"
    ) != "u_risk_v":
        errors.append(
            "Risk-learning target symbol must be u_risk_v."
        )

    raw_target = contract.get(
        "pre_log_risk_quantity",
        {}
    )

    if raw_target.get(
        "semantic_code_name"
    ) != "z_risk":
        errors.append(
            "Legacy y_raw must map to z_risk."
        )

    transformation = risk_target.get(
        "transformation",
        {}
    )

    if float(
        transformation.get(
            "alpha_log",
            -1,
        )
    ) != 10.0:
        errors.append(
            "alpha_log must remain 10."
        )

    if float(
        transformation.get(
            "quantile",
            -1,
        )
    ) != 0.99:
        errors.append(
            "Target quantile must remain 0.99."
        )

    return errors
