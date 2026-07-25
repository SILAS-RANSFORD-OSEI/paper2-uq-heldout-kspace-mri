from pathlib import Path
import ast
import json


REPO = Path(__file__).resolve().parents[1]

NEURAL = (
    REPO
    / "scripts"
    / "exp006_generate_dtest_neural.py"
)

UHOLD = (
    REPO
    / "scripts"
    / "exp006_generate_dtest_uhold.py"
)

CONTRACT = (
    REPO
    / "configs"
    / "exp006_final_execution_contract.json"
)


def source(path):
    return path.read_text(
        encoding="utf-8"
    )


def test_final_execution_files_exist():
    assert NEURAL.is_file()
    assert UHOLD.is_file()
    assert CONTRACT.is_file()


def test_final_scripts_parse():
    ast.parse(
        source(NEURAL),
        filename=str(NEURAL),
    )

    ast.parse(
        source(UHOLD),
        filename=str(UHOLD),
    )


def test_neural_uses_open_final_evaluation_barrier():
    text = source(NEURAL)

    assert "PURPOSE_FINAL_EVALUATION" in text
    assert "BARRIER_OPEN" in text
    assert "open_barrier" in text
    assert "PURPOSE_CALIBRATION" not in text
    assert "BARRIER_CLOSED" not in text
    assert "dtest_batch_" in text
    assert "dcal_batch_" not in text


def test_uhold_uses_test_split_and_dtest_chunks():
    text = source(UHOLD)

    assert '"test"' in text
    assert "dtest_batch_" in text
    assert "dcal_batch_" not in text
    assert "AUTHORIZED_SINGLE_EXECUTION" in text
    assert "EXPECTED_SUPPORTED_PIXELS = 168_146_944" not in text


def test_both_scripts_require_authorization():
    for path in (
        NEURAL,
        UHOLD,
    ):
        text = source(path)

        assert "PAPER2_DTEST_AUTHORIZATION" in text
        assert "D_test_may_be_opened" in text
        assert "final_test_barrier" in text
        assert "single frozen P2-Exp006 D_test execution" in text


def test_execution_contract_is_single_frozen_run():
    contract = json.loads(
        CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    assert contract[
        "status"
    ] == "FROZEN_PRE_DTEST_PRECOMMIT"

    assert contract[
        "authorization"
    ][
        "barrier"
    ] == "OPEN"

    assert contract[
        "execution_policy"
    ][
        "single_frozen_execution"
    ] is True

    assert contract[
        "execution_policy"
    ][
        "overwrite_existing_outputs"
    ] is False

    assert contract[
        "execution_policy"
    ][
        "continue_after_partial_failure"
    ] is False

    assert contract[
        "D_test_access_during_build"
    ] is False
