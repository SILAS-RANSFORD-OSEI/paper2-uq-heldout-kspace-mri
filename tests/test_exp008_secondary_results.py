from pathlib import Path
import json

import pandas as pd


REPO = Path(__file__).resolve().parents[1]

ROOT = (
    REPO
    / "results"
    / "exp008_final_test_secondary"
)


def _estimate(
    dataframe,
    **conditions,
):
    selected = dataframe.copy()

    for column, value in conditions.items():
        selected = selected[
            selected[column] == value
        ]

    assert len(selected) == 1

    return float(
        selected.iloc[0]["estimate"]
    )


def test_secondary_freeze_manifest():
    path = (
        ROOT
        / "exp008_secondary_results_freeze.json"
    )

    assert path.is_file()

    manifest = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert manifest["status"] == "FROZEN"

    assert (
        manifest[
            "descriptor_baseline_decision"
        ]["status"]
        == "NOT_PERFORMED"
    )

    assert (
        manifest[
            "metrics_recalculated_during_archive"
        ]
        is False
    )


def test_threshold_sensitivity_results():
    summary = pd.read_csv(
        ROOT
        / "task_r_threshold_sensitivity"
        / "task_r_threshold_sensitivity_summary.csv"
    )

    assert len(summary) == 18

    expected = {
        ("q85", "auprc", "U1"):
            0.856402,

        ("q90", "auprc", "U1"):
            0.714307,

        ("q95", "auprc", "U1"):
            0.481641,

        ("q90", "auroc", "U1"):
            0.840965,
    }

    for key, target in expected.items():
        threshold_name, endpoint, method = key

        observed = _estimate(
            summary,
            threshold_name=threshold_name,
            endpoint=endpoint,
            method=method,
        )

        assert abs(
            observed - target
        ) < 1.0e-6


def test_u1_is_best_across_frozen_thresholds():
    summary = pd.read_csv(
        ROOT
        / "task_r_threshold_sensitivity"
        / "task_r_threshold_sensitivity_summary.csv"
    )

    for threshold_name in (
        "q85",
        "q90",
        "q95",
    ):
        for endpoint in (
            "auprc",
            "auroc",
        ):
            rows = summary[
                (
                    summary["threshold_name"]
                    == threshold_name
                )
                & (
                    summary["endpoint"]
                    == endpoint
                )
            ]

            best_method = (
                rows
                .sort_values(
                    "estimate",
                    ascending=False,
                )
                .iloc[0]["method"]
            )

            assert best_method == "U1"


def test_u2b_decomposition_results():
    summary = pd.read_csv(
        ROOT
        / "u2b_decomposition"
        / "u2b_decomposition_summary.csv"
    )

    within_ause = _estimate(
        summary,
        task="E",
        endpoint="normalized_ause",
        component="within",
    )

    between_ause = _estimate(
        summary,
        task="E",
        endpoint="normalized_ause",
        component="between",
    )

    total_ause = _estimate(
        summary,
        task="E",
        endpoint="normalized_ause",
        component="total",
    )

    assert abs(
        within_ause - 0.296325
    ) < 1.0e-6

    assert abs(
        between_ause - 0.471323
    ) < 1.0e-6

    assert abs(
        total_ause - 0.296751
    ) < 1.0e-6

    assert within_ause < total_ause
    assert total_ause < between_ause


def test_u2b_variance_composition():
    path = (
        ROOT
        / "u2b_decomposition"
        / "u2b_variance_composition_summary.json"
    )

    composition = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    within = float(
        composition[
            "global_within_fraction"
        ]
    )

    between = float(
        composition[
            "global_between_fraction"
        ]
    )

    assert abs(
        within - 0.982341
    ) < 1.0e-6

    assert abs(
        between - 0.017659
    ) < 1.0e-6

    assert abs(
        within + between - 1.0
    ) < 1.0e-6
