"""Tests for the frozen Paper 2 split policy."""

from pathlib import Path
import json

import yaml


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

POLICY_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "protocol"
    / "split_policy_v1.0.yaml"
)

FEASIBILITY_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "audits"
    / "exp001_split_feasibility.json"
)

CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "exp001_split.yaml"
)

SPLIT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "splits"
    / "paper2_split.csv"
)


def load_policy() -> dict:
    return yaml.safe_load(
        POLICY_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_split_counts_total_281() -> None:
    policy = load_policy()

    counts = {
        role:
            record[
                "volume_count"
            ]
        for role, record
        in policy[
            "paper2_roles"
        ].items()
    }

    assert counts == {
        "D_fit": 181,
        "D_dev": 20,
        "D_cal": 40,
        "D_test": 40,
    }

    assert sum(
        counts.values()
    ) == 281


def test_original_paper1_roles_are_preserved() -> None:
    roles = load_policy()[
        "paper2_roles"
    ]

    assert (
        roles[
            "D_fit"
        ][
            "source_population"
        ]
        == "Paper 1 train"
    )

    assert (
        roles[
            "D_dev"
        ][
            "source_population"
        ]
        == "Paper 1 train"
    )

    assert (
        roles[
            "D_cal"
        ][
            "source_population"
        ]
        == "Paper 1 calibration"
    )

    assert (
        roles[
            "D_test"
        ][
            "source_population"
        ]
        == "Paper 1 test"
    )


def test_test_cohort_is_not_claimed_as_fresh() -> None:
    test_role = load_policy()[
        "paper2_roles"
    ][
        "D_test"
    ]

    assert (
        test_role[
            "designation"
        ]
        == "locked_reused_evaluation_cohort"
    )

    assert (
        test_role[
            "previously_reported_in_paper1"
        ]
        is True
    )

    assert (
        test_role[
            "informed_paper2_design"
        ]
        is True
    )

    assert (
        test_role[
            "fresh_pristine_test_claim_permitted"
        ]
        is False
    )


def test_split_is_created_but_test_barrier_remains_closed() -> None:
    policy = load_policy()

    assert (
        policy[
            "test_barrier"
        ][
            "status"
        ]
        == "CLOSED"
    )

    assert (
        policy[
            "governance"
        ][
            "split_created"
        ]
        is True
    )

    assert (
        policy[
            "governance"
        ][
            "volume_ids_assigned"
        ]
        is True
    )

    assert (
        policy[
            "governance"
        ][
            "paper2_test_predictions_generated"
        ]
        is False
    )


def test_fit_dev_assignment_is_frozen_and_outcome_blind() -> None:
    allocation = load_policy()[
        "fit_dev_allocation"
    ]

    assert (
        allocation[
            "assignment_seed"
        ]
        == 20260720
    )

    assert (
        allocation[
            "assignment_method_status"
        ]
        == "frozen"
    )

    assert (
        allocation[
            "assignment_algorithm"
        ]
        == "width_coil_ilp_slice_balance_v1.0"
    )

    assert (
        allocation[
            "split_file"
        ]
        == "data/splits/paper2_split.csv"
    )

    prohibited = set(
        allocation[
            "prohibited_assignment_inputs"
        ]
    )

    assert {
        "u_risk",
        "u_hold",
        "model predictions",
        "uncertainty scores",
        "performance metrics",
    }.issubset(
        prohibited
    )


def test_selected_strategy_matches_feasibility_audit() -> None:
    policy = load_policy()

    feasibility = json.loads(
        FEASIBILITY_PATH.read_text(
            encoding="utf-8"
        )
    )

    strategy_id = policy[
        "policy"
    ][
        "selected_strategy"
    ]

    strategies = {
        item[
            "strategy_id"
        ]:
            item
        for item
        in feasibility[
            "candidate_strategies"
        ]
    }

    assert strategy_id in strategies

    strategy = strategies[
        strategy_id
    ]

    assert strategy[
        "reuses_existing_cache"
    ] is True

    assert strategy[
        "requires_ssdu_retraining"
    ] is False

    assert strategy[
        "test_gradient_exposure"
    ] is False

    assert strategy[
        "test_model_selection_exposure"
    ] is False


def test_exp001_config_records_completed_split() -> None:
    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        config[
            "experiment"
        ][
            "status"
        ]
        == "completed"
    )

    assert (
        config[
            "experiment"
        ][
            "result"
        ]
        == "PASS"
    )

    assert (
        config[
            "fit_dev_assignment"
        ][
            "status"
        ]
        == "frozen"
    )

    assert (
        config[
            "fit_dev_assignment"
        ][
            "split_file"
        ]
        == "data/splits/paper2_split.csv"
    )

    assert (
        config[
            "test_barrier"
        ][
            "status"
        ]
        == "CLOSED"
    )

    assert (
        config[
            "test_barrier"
        ][
            "test_predictions_generated"
        ]
        is False
    )

    assert SPLIT_PATH.exists()
