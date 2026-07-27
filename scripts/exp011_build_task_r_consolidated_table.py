# ============================================================
# EXP011 — CONSOLIDATED TASK R AUPRC TABLE
#
# Builds a manuscript-ready table from frozen Exp009 q90 and
# Exp010 q85/q95 summary files.
#
# No metrics are recalculated.
# No scientific arrays are opened.
# ============================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


REPO = Path(
    "/content/paper2-uq-heldout-kspace-mri"
).resolve()

EXP009_DIR = (
    REPO
    / "results"
    / "exp009_task_r_control_completion"
)

EXP010_DIR = (
    REPO
    / "results"
    / "exp010_task_r_threshold_sensitivity"
)

Q90_PATH = (
    EXP009_DIR
    / "task_r_primary_summary_extended.csv"
)

SENSITIVITY_PATH = (
    EXP010_DIR
    / "task_r_sensitivity_primary_summary.csv"
)

OUTPUT_DIR = (
    REPO
    / "results"
    / "exp011_task_r_consolidated_table"
)

EXPECTED_BRANCH = (
    "retrospective-task-r-controls-v1.0"
)

EXPECTED_PARENT_COMMIT = (
    "bd8a02b3352ebb11bc3f424175e34042e61dfc57"
)

THRESHOLDS = {
    "q85": 1.4107755422592163,
    "q90": 1.7366089820861816,
    "q95": 2.0520856380462646,
}

METHOD_ORDER = [
    "C0",
    "U1",
    "U2a",
    "U2b",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
]

METHOD_LABELS = {
    "C0": "Direct residual-risk predictor",
    "U1": "MC-dropout uncertainty",
    "U2a": "Point-ensemble between-model variance",
    "U2b": "Probabilistic-ensemble total predictive variance",
    "B1": "Reconstructed-image magnitude",
    "B2": "Zero-filled image magnitude",
    "B3": "Reconstruction–zero-filled discrepancy",
    "B4": "Reconstructed-image gradient magnitude",
    "B5": "Analytical PSF descriptor",
    "B6": "qPSF/gain descriptor",
}

METHOD_GROUPS = {
    "C0": "Direct predictor",
    "U1": "Model uncertainty",
    "U2a": "Model uncertainty",
    "U2b": "Model uncertainty",
    "B1": "Deterministic descriptor",
    "B2": "Deterministic descriptor",
    "B3": "Deterministic descriptor",
    "B4": "Deterministic descriptor",
    "B5": "Deterministic descriptor",
    "B6": "Deterministic descriptor",
}


def git_output(
    *args: str,
) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
    ).strip()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def format_result(
    estimate: float,
    ci_low: float,
    ci_high: float,
) -> str:
    return (
        f"{estimate:.3f} "
        f"[{ci_low:.3f}, {ci_high:.3f}]"
    )


# ------------------------------------------------------------
# 1. Guards
# ------------------------------------------------------------
required_paths = [
    Q90_PATH,
    SENSITIVITY_PATH,
]

for path in required_paths:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen result file:\n{path}"
        )

if git_output(
    "branch",
    "--show-current",
) != EXPECTED_BRANCH:
    raise RuntimeError(
        "Unexpected repository branch."
    )

execution_commit = git_output(
    "rev-parse",
    "HEAD",
)

if execution_commit == EXPECTED_PARENT_COMMIT:
    raise RuntimeError(
        "Table-generation script must be committed "
        "before execution."
    )

if git_output(
    "status",
    "--porcelain",
) != "":
    raise RuntimeError(
        "Repository must be clean before execution."
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Refusing to overwrite:\n{OUTPUT_DIR}"
    )


# ------------------------------------------------------------
# 2. Load frozen summary records
# ------------------------------------------------------------
q90 = pd.read_csv(
    Q90_PATH
)

sensitivity = pd.read_csv(
    SENSITIVITY_PATH
)


# ------------------------------------------------------------
# 3. Validate frozen records
# ------------------------------------------------------------
required_summary_columns = {
    "task",
    "endpoint",
    "method",
    "n_volumes",
    "estimate",
    "ci_low",
    "ci_high",
    "confidence_level",
    "bootstrap_replicates",
    "bootstrap_seed",
    "direction",
}

if not required_summary_columns.issubset(
    q90.columns
):
    raise RuntimeError(
        "Unexpected Exp009 summary schema."
    )

if not (
    required_summary_columns
    | {
        "threshold_name",
        "threshold_value",
    }
).issubset(
    sensitivity.columns
):
    raise RuntimeError(
        "Unexpected Exp010 summary schema."
    )

q90 = q90[
    (
        q90["task"] == "R"
    )
    &
    (
        q90["endpoint"] == "auprc"
    )
].copy()

sensitivity = sensitivity[
    (
        sensitivity["task"] == "R"
    )
    &
    (
        sensitivity["endpoint"] == "auprc"
    )
].copy()

if len(q90) != 10:
    raise RuntimeError(
        f"Expected 10 q90 rows, found {len(q90)}."
    )

if len(sensitivity) != 20:
    raise RuntimeError(
        "Expected 20 q85/q95 rows."
    )

if set(q90["method"]) != set(METHOD_ORDER):
    raise RuntimeError(
        "Incomplete q90 method set."
    )

for threshold_name in [
    "q85",
    "q95",
]:
    frame = sensitivity[
        sensitivity[
            "threshold_name"
        ] == threshold_name
    ]

    if len(frame) != 10:
        raise RuntimeError(
            f"Expected 10 rows for {threshold_name}."
        )

    if set(frame["method"]) != set(
        METHOD_ORDER
    ):
        raise RuntimeError(
            f"Incomplete method set for {threshold_name}."
        )

for frame in [
    q90,
    sensitivity,
]:
    numeric = frame[
        [
            "estimate",
            "ci_low",
            "ci_high",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(numeric).all():
        raise RuntimeError(
            "Nonfinite summary value detected."
        )

    if np.any(
        frame["ci_low"].to_numpy()
        >
        frame["estimate"].to_numpy()
    ):
        raise RuntimeError(
            "Estimate below confidence interval."
        )

    if np.any(
        frame["estimate"].to_numpy()
        >
        frame["ci_high"].to_numpy()
    ):
        raise RuntimeError(
            "Estimate above confidence interval."
        )


# ------------------------------------------------------------
# 4. Harmonize q85, q90, and q95
# ------------------------------------------------------------
q90["threshold_name"] = "q90"
q90["threshold_value"] = THRESHOLDS[
    "q90"
]

combined_long = pd.concat(
    [
        sensitivity[
            [
                "threshold_name",
                "threshold_value",
                "method",
                "n_volumes",
                "estimate",
                "ci_low",
                "ci_high",
                "confidence_level",
                "bootstrap_replicates",
                "bootstrap_seed",
                "direction",
            ]
        ],
        q90[
            [
                "threshold_name",
                "threshold_value",
                "method",
                "n_volumes",
                "estimate",
                "ci_low",
                "ci_high",
                "confidence_level",
                "bootstrap_replicates",
                "bootstrap_seed",
                "direction",
            ]
        ],
    ],
    ignore_index=True,
)

threshold_order = {
    "q85": 0,
    "q90": 1,
    "q95": 2,
}

method_order = {
    method: index
    for index, method in enumerate(
        METHOD_ORDER
    )
}

combined_long[
    "_threshold_order"
] = combined_long[
    "threshold_name"
].map(
    threshold_order
)

combined_long[
    "_method_order"
] = combined_long[
    "method"
].map(
    method_order
)

combined_long = (
    combined_long
    .sort_values(
        [
            "_threshold_order",
            "_method_order",
        ]
    )
    .drop(
        columns=[
            "_threshold_order",
            "_method_order",
        ]
    )
    .reset_index(
        drop=True
    )
)

if len(combined_long) != 30:
    raise RuntimeError(
        "Expected 30 consolidated records."
    )


# ------------------------------------------------------------
# 5. Validate threshold-specific rankings
# ------------------------------------------------------------
for threshold_name in [
    "q85",
    "q90",
    "q95",
]:
    ranking = (
        combined_long[
            combined_long[
                "threshold_name"
            ] == threshold_name
        ]
        .sort_values(
            "estimate",
            ascending=False,
        )
    )

    if ranking.iloc[0]["method"] != "C0":
        raise RuntimeError(
            f"C0 not ranked first at {threshold_name}."
        )

    if ranking.iloc[1]["method"] != "U1":
        raise RuntimeError(
            f"U1 not ranked second at {threshold_name}."
        )

    descriptor_ranking = ranking[
        ranking[
            "method"
        ].str.startswith("B")
    ]

    if (
        descriptor_ranking.iloc[0]["method"]
        != "B2"
    ):
        raise RuntimeError(
            f"B2 not strongest descriptor at "
            f"{threshold_name}."
        )


# ------------------------------------------------------------
# 6. Create manuscript-wide table
# ------------------------------------------------------------
wide_rows = []

for method in METHOD_ORDER:
    row = {
        "Method code": method,
        "Method": METHOD_LABELS[method],
        "Method group": METHOD_GROUPS[method],
    }

    for threshold_name in [
        "q85",
        "q90",
        "q95",
    ]:
        record = combined_long[
            (
                combined_long[
                    "method"
                ] == method
            )
            &
            (
                combined_long[
                    "threshold_name"
                ] == threshold_name
            )
        ]

        if len(record) != 1:
            raise RuntimeError(
                f"Missing unique result for "
                f"{method}, {threshold_name}."
            )

        record = record.iloc[0]

        row[
            f"{threshold_name} estimate"
        ] = float(
            record["estimate"]
        )

        row[
            f"{threshold_name} CI low"
        ] = float(
            record["ci_low"]
        )

        row[
            f"{threshold_name} CI high"
        ] = float(
            record["ci_high"]
        )

        row[
            f"{threshold_name} AUPRC [95% CI]"
        ] = format_result(
            float(record["estimate"]),
            float(record["ci_low"]),
            float(record["ci_high"]),
        )

    wide_rows.append(row)

wide = pd.DataFrame(
    wide_rows
)


# ------------------------------------------------------------
# 7. Markdown manuscript table
# ------------------------------------------------------------
markdown_lines = [
    "| Method | Category | "
    "$q_{85}$ AUPRC [95% CI] | "
    "$q_{90}$ AUPRC [95% CI] | "
    "$q_{95}$ AUPRC [95% CI] |",
    "|---|---|---:|---:|---:|",
]

for _, row in wide.iterrows():
    method_text = (
        f"{row['Method code']}: "
        f"{row['Method']}"
    )

    q85_text = row[
        "q85 AUPRC [95% CI]"
    ]

    q90_text = row[
        "q90 AUPRC [95% CI]"
    ]

    q95_text = row[
        "q95 AUPRC [95% CI]"
    ]

    if row["Method code"] == "C0":
        q85_text = f"**{q85_text}**"
        q90_text = f"**{q90_text}**"
        q95_text = f"**{q95_text}**"

    markdown_lines.append(
        f"| {method_text} "
        f"| {row['Method group']} "
        f"| {q85_text} "
        f"| {q90_text} "
        f"| {q95_text} |"
    )

markdown_lines.extend(
    [
        "",
        "**Table note.** Values are mean "
        "volume-level support-weighted AUPRC "
        "with 95% paired volume-bootstrap "
        "confidence intervals across 40 test "
        "volumes. Thresholds were fixed from "
        "the calibration split. Higher values "
        "indicate stronger localization of "
        "elevated measurement-derived "
        "residual-consistency risk. Absolute "
        "AUPRC values should not be interpreted "
        "as directly comparable across "
        "thresholds because high-risk prevalence "
        "changes with the threshold.",
        "",
    ]
)

markdown_text = "\n".join(
    markdown_lines
)


# ------------------------------------------------------------
# 8. LaTeX manuscript table
# ------------------------------------------------------------
latex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Task R localization performance "
    r"across frozen residual-risk thresholds.}",
    r"\label{tab:task_r_threshold_sensitivity}",
    r"\begin{tabular}{llccc}",
    r"\hline",
    r"Method & Category & "
    r"$q_{85}$ AUPRC [95\% CI] & "
    r"$q_{90}$ AUPRC [95\% CI] & "
    r"$q_{95}$ AUPRC [95\% CI] \\",
    r"\hline",
]

for _, row in wide.iterrows():
    method_text = (
        f"{row['Method code']}: "
        f"{row['Method']}"
    ).replace(
        "–",
        "--",
    )

    group_text = str(
        row["Method group"]
    )

    q85_text = row[
        "q85 AUPRC [95% CI]"
    ]

    q90_text = row[
        "q90 AUPRC [95% CI]"
    ]

    q95_text = row[
        "q95 AUPRC [95% CI]"
    ]

    if row["Method code"] == "C0":
        q85_text = (
            r"\textbf{"
            + q85_text
            + "}"
        )

        q90_text = (
            r"\textbf{"
            + q90_text
            + "}"
        )

        q95_text = (
            r"\textbf{"
            + q95_text
            + "}"
        )

    latex_lines.append(
        f"{method_text} & "
        f"{group_text} & "
        f"{q85_text} & "
        f"{q90_text} & "
        f"{q95_text} \\\\"
    )

latex_lines.extend(
    [
        r"\hline",
        r"\end{tabular}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize",
        r"\textit{Note:} Values are mean "
        r"volume-level support-weighted AUPRC "
        r"with 95\% paired volume-bootstrap "
        r"confidence intervals across 40 test "
        r"volumes. Thresholds were fixed from "
        r"the calibration split. Higher values "
        r"indicate stronger localization of "
        r"elevated measurement-derived "
        r"residual-consistency risk. Absolute "
        r"AUPRC values should not be interpreted "
        r"as directly comparable across "
        r"thresholds because high-risk prevalence "
        r"changes with the threshold.",
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ]
)

latex_text = "\n".join(
    latex_lines
)


# ------------------------------------------------------------
# 9. Write outputs atomically
# ------------------------------------------------------------
OUTPUT_DIR.parent.mkdir(
    parents=True,
    exist_ok=True,
)

temporary_directory = Path(
    tempfile.mkdtemp(
        prefix="exp011_task_r_table_tmp_",
        dir=OUTPUT_DIR.parent,
    )
)

try:
    long_path = (
        temporary_directory
        / "task_r_auprc_consolidated_long.csv"
    )

    wide_path = (
        temporary_directory
        / "task_r_auprc_consolidated_manuscript.csv"
    )

    markdown_path = (
        temporary_directory
        / "task_r_auprc_consolidated_table.md"
    )

    latex_path = (
        temporary_directory
        / "task_r_auprc_consolidated_table.tex"
    )

    provenance_path = (
        temporary_directory
        / "provenance.json"
    )

    combined_long.to_csv(
        long_path,
        index=False,
    )

    wide.to_csv(
        wide_path,
        index=False,
    )

    markdown_path.write_text(
        markdown_text,
        encoding="utf-8",
    )

    latex_path.write_text(
        latex_text,
        encoding="utf-8",
    )

    provenance = {
        "schema_version":
            "exp011-task-r-consolidated-table-v1.0",
        "experiment_id":
            "exp011_task_r_consolidated_table",
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "repository_commit":
            execution_commit,
        "parent_result_commit":
            EXPECTED_PARENT_COMMIT,
        "source_files": {
            str(
                Q90_PATH.relative_to(REPO)
            ): {
                "sha256":
                    sha256_file(Q90_PATH),
            },
            str(
                SENSITIVITY_PATH.relative_to(REPO)
            ): {
                "sha256":
                    sha256_file(
                        SENSITIVITY_PATH
                    ),
            },
        },
        "thresholds":
            THRESHOLDS,
        "methods":
            METHOD_ORDER,
        "records":
            int(len(combined_long)),
        "governance": {
            "metrics_recalculated":
                False,
            "scientific_arrays_opened":
                False,
            "thresholds_changed":
                False,
            "model_training_performed":
                False,
            "model_inference_performed":
                False,
        },
    }

    output_files = [
        long_path,
        wide_path,
        markdown_path,
        latex_path,
    ]

    provenance[
        "output_files"
    ] = {
        path.name: {
            "sha256":
                sha256_file(path),
            "size_bytes":
                int(path.stat().st_size),
        }
        for path in output_files
    }

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary_directory,
        OUTPUT_DIR,
    )

except Exception:
    shutil.rmtree(
        temporary_directory,
        ignore_errors=True,
    )
    raise


# ------------------------------------------------------------
# 10. Final report
# ------------------------------------------------------------
print("=" * 116)
print("EXP011 — CONSOLIDATED TASK R AUPRC TABLE")
print("=" * 116)

print(f"Repository commit:          {execution_commit}")
print(f"Source q90 rows:            {len(q90)}")
print(f"Source q85/q95 rows:        {len(sensitivity)}")
print(f"Consolidated records:       {len(combined_long)}")
print(f"Methods:                    {len(METHOD_ORDER)}")
print(f"Thresholds:                 q85, q90, q95")
print(f"Output directory:           {OUTPUT_DIR}")

print("\nMANUSCRIPT TABLE")
print("-" * 116)
print(markdown_text)

print("GOVERNANCE")
print("-" * 116)
print("Metrics recalculated:       NO")
print("Scientific arrays opened:   NO")
print("Thresholds changed:         NO")
print("Model training performed:   NO")
print("Model inference performed:  NO")

print("=" * 116)
print(
    "STATUS: PASS_EXP011_CONSOLIDATED_TASK_R_TABLE"
)
