"""SWE-bench-Live: image naming, entry script, spec-free pytest grading."""

import json

import pytest

from src.config import reset_config
from src.benchmark.swebench_live import (
    SWEBenchLiveRunner,
    _entry_script,
    _test_ids,
    grade_pytest_log,
    live_image_ref,
)
from src.benchmark.swebench_runner import ExperimentConfig


@pytest.fixture(autouse=True)
def _reset_config():
    reset_config()
    yield
    reset_config()


def test_live_image_ref_matches_official_convention():
    assert live_image_ref("conan-io__conan-15377") == "starryzhang/sweb.eval.x86_64.conan-io_1776_conan-15377"
    assert live_image_ref("A__B-1", namespace="ns") == "ns/sweb.eval.x86_64.a_1776_b-1"


def test_test_ids_parsing():
    assert _test_ids(["a::b", "c::d"]) == ["a::b", "c::d"]
    assert _test_ids(json.dumps(["x::y"])) == ["x::y"]
    assert _test_ids(None) == []
    assert _test_ids("") == []


def test_entry_script_applies_patches_and_runs_pytest():
    from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS

    script = _entry_script("'pkg/test_x.py::test_a'")
    assert "cd /testbed" in script
    assert "git apply --verbose /self_evolver_eval/patch.diff" in script
    assert "/self_evolver_eval/test.patch" in script
    assert "pytest -rA" in script
    assert APPLY_PATCH_PASS in script and APPLY_PATCH_FAIL in script


def _instance(f2p, p2p):
    return {"instance_id": "org__repo-1", "repo": "org/repo", "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p}


def test_grade_pytest_log_resolved():
    log = (
        "PASSED pkg/test_x.py::test_a\n"
        "PASSED pkg/test_x.py::test_b\n"
        "2 passed in 0.10s\n"
    )
    outcome = grade_pytest_log(
        _instance(["pkg/test_x.py::test_a"], ["pkg/test_x.py::test_b"]), log, applied=True
    )
    assert outcome.resolved is True
    assert outcome.f2p_passed == 1 and outcome.f2p_total == 1
    assert outcome.p2p_passed == 1 and outcome.p2p_total == 1


def test_grade_pytest_log_f2p_failure_is_unresolved():
    log = (
        "FAILED pkg/test_x.py::test_a - AssertionError\n"
        "PASSED pkg/test_x.py::test_b\n"
        "1 failed, 1 passed in 0.10s\n"
    )
    outcome = grade_pytest_log(
        _instance(["pkg/test_x.py::test_a"], ["pkg/test_x.py::test_b"]), log, applied=True
    )
    assert outcome.resolved is False
    assert outcome.f2p_passed == 0


def test_grade_pytest_log_regression_blocks_resolve():
    log = (
        "PASSED pkg/test_x.py::test_a\n"
        "FAILED pkg/test_x.py::test_b - regression\n"
        "1 failed, 1 passed in 0.10s\n"
    )
    outcome = grade_pytest_log(
        _instance(["pkg/test_x.py::test_a"], ["pkg/test_x.py::test_b"]), log, applied=True
    )
    assert outcome.f2p_passed == 1
    assert outcome.p2p_no_regression is False
    assert outcome.resolved is False


def test_grade_not_applied_is_unresolved():
    outcome = grade_pytest_log(_instance(["t::a"], []), "APPLY_PATCH_FAIL", applied=False)
    assert outcome.resolved is False


def test_live_runner_dataset_split_maps_variant(tmp_path):
    runner = SWEBenchLiveRunner(
        dataset="lite",
        output_dir=tmp_path / "run",
        workspace_dir=tmp_path / "ws",
        experiment=ExperimentConfig(stage="eval"),
    )
    assert runner.name == "swebench_live"
    assert runner._dataset_split("test") == ("swebench_live", "lite")
    other = SWEBenchLiveRunner(
        dataset="unknown-variant",
        output_dir=tmp_path / "run2",
        workspace_dir=tmp_path / "ws2",
        experiment=ExperimentConfig(stage="eval"),
    )
    assert other._dataset_split("test") == ("swebench_live", "full")
