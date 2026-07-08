"""Execution-utility math, skill-write gate, evolution utility, config loading."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from src.environment import models
from src.environment.models import Issue, PatchInfo
from src.reward.reward_model import (
    DEFAULT_REWARD_WEIGHTS,
    RewardModel,
    default_config_path,
)


@dataclass
class FakeVerification:
    patch_applied: bool = True
    canonical_patch_content: str = ""
    test_result: Any = None
    eval_outcome: Any = None


@dataclass
class FakeRecord:
    verification_result: Any = None


@dataclass
class FakeExecution:
    success: bool = False
    total_tokens: int = 0
    iterations_used: int = 1
    status: Any = field(default_factory=lambda: SimpleNamespace(value="success"))
    final_patch: Any = None
    iteration_records: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _execution(tokens: int = 0, status: str = "success") -> FakeExecution:
    """Clean-process execution: patch applied, non-empty canonical diff."""
    return FakeExecution(
        total_tokens=tokens,
        status=SimpleNamespace(value=status),
        final_patch=PatchInfo(content="--- a/x.py\n+++ b/x.py\n", modified_files=["x.py"]),
        iteration_records=[FakeRecord(verification_result=FakeVerification())],
    )


def test_full_resolve_scores_one():
    outcome = {"f2p_passed": 3, "f2p_total": 3, "p2p_passed": 10, "p2p_total": 10, "resolved": True}
    reward = RewardModel().score(_execution(tokens=0), eval_outcome=outcome)
    assert reward.total == 1.0
    assert reward.components == {
        "resolved": 1.0,
        "f2p_fraction": 1.0,
        "p2p_no_regression": 1.0,
        "cost_efficiency": 1.0,
        "process": 1.0,
    }


def test_gate_reachable_without_full_resolve():
    """A focused-subset pass (partial progress) + clean process clears the gate.

    Official resolution over the full suite is False, but all targeted F2P
    tests pass with no observed P2P regression, so the in-loop resolved
    predicate holds and the utility exceeds the 0.55 skill-write gate.
    """
    model = RewardModel()
    outcome = {"f2p_passed": 1, "f2p_total": 1, "p2p_passed": 0, "p2p_total": 0, "resolved": False}
    reward = model.score(_execution(tokens=30000), eval_outcome=outcome)
    assert reward.components["resolved"] == 1.0
    assert reward.total == pytest.approx(0.5 + 0.2 + 0.1 + 0.05 + 0.1)
    assert reward.total >= model.skill_write_gate


def test_partial_f2p_progress_stays_below_gate():
    model = RewardModel()
    outcome = {"f2p_passed": 1, "f2p_total": 2, "p2p_passed": 3, "p2p_total": 3, "resolved": False}
    reward = model.score(_execution(tokens=0), eval_outcome=outcome)
    assert reward.total == pytest.approx(0.2 * 0.5 + 0.1 + 0.1 + 0.1)
    assert reward.total < model.skill_write_gate


def test_p2p_regression_blocks_resolved_and_p2p_credit():
    outcome = {"f2p_passed": 2, "f2p_total": 2, "p2p_passed": 2, "p2p_total": 3, "resolved": False}
    reward = RewardModel().score(_execution(tokens=0), eval_outcome=outcome)
    assert reward.components["resolved"] == 0.0
    assert reward.components["p2p_no_regression"] == 0.0
    assert reward.total == pytest.approx(0.2 + 0.1 + 0.1)


def test_no_eval_outcome_gives_no_test_credit():
    reward = RewardModel().score(_execution(tokens=0))
    assert reward.components["resolved"] == 0.0
    assert reward.components["f2p_fraction"] == 0.0
    assert reward.components["p2p_no_regression"] == 0.0
    assert reward.total == pytest.approx(0.1 + 0.1)  # cost + process only


def test_cost_efficiency_hits_zero_at_budget():
    outcome = {"f2p_passed": 0, "f2p_total": 2, "p2p_passed": 0, "p2p_total": 0, "resolved": False}
    reward = RewardModel().score(_execution(tokens=60000), eval_outcome=outcome)
    assert reward.components["cost_efficiency"] == 0.0


def test_process_requires_applied_patch_and_budget():
    outcome = {"f2p_passed": 0, "f2p_total": 1, "p2p_passed": 0, "p2p_total": 0, "resolved": False}
    no_patch = FakeExecution(iteration_records=[FakeRecord(FakeVerification(patch_applied=False))])
    assert RewardModel().score(no_patch, eval_outcome=outcome).components["process"] == 0.0
    over_budget = _execution(status="max_iterations_reached")
    assert RewardModel().score(over_budget, eval_outcome=outcome).components["process"] == 0.0


def test_view_derived_from_real_per_test_statuses():
    """Without an attached EvalOutcome, per-test statuses + F2P/P2P lists decide."""
    test_result = models.TestResult(
        passed=False,
        test_cases=[
            models.TestCase(name="tests/test_a.py::test_one", status=models.TestStatus.PASSED),
            models.TestCase(name="tests/test_b.py::test_keep", status=models.TestStatus.PASSED),
            # second F2P test missing from the run -> counts as not passed
        ],
    )
    execution = FakeExecution(
        iteration_records=[FakeRecord(FakeVerification(test_result=test_result))],
        final_patch=PatchInfo(content="diff", modified_files=["a.py"]),
    )
    issue = Issue(
        id="repo__1",
        description="bug",
        metadata={
            "fail_to_pass": '["tests/test_a.py::test_one", "tests/test_a.py::test_two"]',
            "pass_to_pass": '["tests/test_b.py::test_keep", "tests/test_b.py::test_not_run"]',
        },
    )
    reward = RewardModel().score(execution, issue=issue)
    assert reward.components["f2p_fraction"] == 0.5
    assert reward.components["p2p_no_regression"] == 1.0  # only run P2P tests count
    assert reward.components["resolved"] == 0.0


def test_explicit_eval_outcome_wins_over_derived():
    test_result = models.TestResult(
        passed=True,
        test_cases=[models.TestCase("t::a", models.TestStatus.PASSED)],
    )
    execution = FakeExecution(
        iteration_records=[FakeRecord(FakeVerification(test_result=test_result))],
    )
    issue = Issue(id="i", description="d", metadata={"fail_to_pass": '["t::a"]'})
    outcome = {"f2p_passed": 0, "f2p_total": 1, "p2p_passed": 0, "p2p_total": 0, "resolved": False}
    reward = RewardModel().score(execution, eval_outcome=outcome, issue=issue)
    assert reward.components["f2p_fraction"] == 0.0


def test_evolution_utility_is_advantage_over_ema_baseline():
    model = RewardModel(baseline_alpha=0.3)
    assert model.evolution_utility(0.5) == 0.0  # first observation seeds baseline
    assert model.evolution_utility(0.9) == pytest.approx(0.4)
    # baseline is now 0.7*0.5 + 0.3*0.9 = 0.62
    assert model.evolution_utility(0.9) == pytest.approx(0.28)


def test_score_exposes_evolution_utility():
    model = RewardModel()
    outcome = {"f2p_passed": 1, "f2p_total": 1, "p2p_passed": 1, "p2p_total": 1, "resolved": True}
    first = model.score(_execution(tokens=0), eval_outcome=outcome)
    assert first.evolution_utility == 0.0
    second = model.score(FakeExecution(), eval_outcome=None)
    assert second.evolution_utility == pytest.approx(second.total - first.total)


def test_repo_config_is_loaded_by_default_and_matches_code_defaults():
    assert default_config_path().exists()
    model = RewardModel.from_config_file()
    assert model.weights == DEFAULT_REWARD_WEIGHTS
    assert model.cost_token_budget == 60000
    assert model.skill_write_gate == 0.55
    assert model.baseline_alpha == 0.3


def test_config_file_overrides_and_parser_features(tmp_path):
    path = tmp_path / "reward.yaml"
    path.write_text(
        "# comment line\n"
        "weights:\n"
        "  resolved: 0.6  # inline comment\n"
        "  f2p_fraction: 0.1\n"
        'cost_token_budget: "40000"\n'
        "skill_write_gate: 0.7\n",
        encoding="utf-8",
    )
    model = RewardModel.from_config_file(path)
    assert model.weights["resolved"] == 0.6
    assert model.weights["f2p_fraction"] == 0.1
    assert model.weights["process"] == 0.1  # unspecified keys keep defaults
    assert model.cost_token_budget == 40000
    assert model.skill_write_gate == 0.7


def test_config_errors_are_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        RewardModel.from_config_file(tmp_path / "missing.yaml")

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("weights:\n  tests_pass: 0.35\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tests_pass"):
        RewardModel.from_config_file(unknown)

    listy = tmp_path / "listy.yaml"
    listy.write_text("weights:\n  - resolved\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        RewardModel.from_config_file(listy)

    deep = tmp_path / "deep.yaml"
    deep.write_text("weights:\n  nested:\n    resolved: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nesting"):
        RewardModel.from_config_file(deep)
