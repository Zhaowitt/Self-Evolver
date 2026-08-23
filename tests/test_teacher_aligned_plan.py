"""Acceptance tests for the teacher-aligned next-stage design."""

from types import SimpleNamespace

import pytest

from src.environment.f2p_subset import F2PSubsetSelector, test_scope as scope_key
from src.memory.hard_case_buffer import HardCaseRecord
from src.reflection.clustering import cluster_records, qualifying_clusters
from src.reflection.failure_diagnosis import diagnose_failed_trajectory
from src.reward.reward_model import RewardModel
from src.skills.skill_schema import StructuredSkill, lint_skill_markdown, render_skill_markdown


def test_f2p_subset_is_round_stable_and_scope_diverse():
    tests = ["a.py::A::x", "a.py::A::y", "b.py::B::z", "c.py::C::w"]
    selector = F2PSubsetSelector(subset_size=2)
    first = selector.select(tests, "instance", "round-1")
    assert first == selector.select(tests, "instance", "round-1")
    assert len(first) == 2
    assert len({scope_key(item) for item in first}) == 2


def test_subset_evidence_never_becomes_task_outcome_and_cost_is_negative():
    execution = SimpleNamespace(
        total_tokens=60000,
        iteration_records=[],
        final_patch=None,
        status=SimpleNamespace(value="failed"),
    )
    subset = {
        "f2p_passed": 1,
        "f2p_total": 1,
        "p2p_passed": 0,
        "p2p_total": 0,
        "resolved": True,
        "evaluation_scope": "subset",
    }
    reward = RewardModel(cluster_baselines={}).score(execution, eval_outcome=subset)
    assert reward.components["official_resolved"] == 0.0
    assert reward.components["official_f2p_fraction"] == 0.0
    assert reward.total == -0.05


def test_cluster_contribution_is_in_overall_and_uses_frozen_baseline():
    execution = SimpleNamespace(
        total_tokens=0,
        iteration_records=[],
        final_patch=None,
        status=SimpleNamespace(value="failed"),
    )
    full = {
        "f2p_passed": 1,
        "f2p_total": 2,
        "p2p_passed": 1,
        "p2p_total": 1,
        "resolved": False,
        "evaluation_scope": "full",
    }
    model = RewardModel(
        cluster_baselines={"cluster-a": {"ema": 0.2, "count": 5}},
        contribution_weight=0.1,
    )
    reward = model.score(
        execution,
        eval_outcome=full,
        cluster_id="cluster-a",
        selected_skill_ids=["skill_b", "skill_a"],
    )
    assert reward.task_reward == 0.35
    assert reward.skill_contribution == pytest.approx(0.15)
    assert reward.total == pytest.approx(0.365)
    assert reward.skill_set_key == ["skill_a", "skill_b"]


def test_max_iterations_is_termination_evidence_not_mechanism():
    verification = SimpleNamespace(
        status=SimpleNamespace(value="patch_failed"),
        patch_applied=False,
        summary="git apply malformed hunk",
        error_message="",
        subset_eval_outcome=None,
        full_eval_outcome=None,
        eval_outcome=None,
    )
    record = SimpleNamespace(
        error=None,
        judge_decision=None,
        verification_result=verification,
    )
    diagnosis = diagnose_failed_trajectory([record], "max_iterations")
    assert diagnosis.mechanism == "patch_application_failure"
    assert diagnosis.evidence["termination_reason"] == "max_iterations"


def _diagnosed(issue_id: str, mechanism: str, summary: str) -> HardCaseRecord:
    return HardCaseRecord(
        issue_id=issue_id,
        failure_diagnosis={
            "mechanism": mechanism,
            "summary": summary,
            "error_signatures": [summary],
            "evidence": {
                "f2p_progress": [{"passed": 1, "total": 2}],
                "retry_pattern": {"statuses": ["tests_failed"]},
            },
            "confidence": 0.9,
        },
    )


def test_two_layer_cluster_never_crosses_mechanism_and_uses_evidence_gates():
    records = [
        _diagnosed("a", "incomplete_fix", "parser returns wrong nested shape"),
        _diagnosed("b", "incomplete_fix", "parser returns wrong nested shape"),
        _diagnosed("c", "incomplete_fix", "parser returns wrong nested shape"),
        _diagnosed("d", "wrong_root_cause", "parser returns wrong nested shape"),
    ]
    clusters = cluster_records(records)
    assert all(len({item.failure_diagnosis["mechanism"] for item in cluster.members}) == 1 for cluster in clusters)
    qualified = qualifying_clusters(clusters)
    assert len(qualified) == 1
    assert qualified[0].mechanism == "incomplete_fix"
    assert qualified[0].unique_issue_count == 3


def test_generated_skill_requires_seed_skill_sections():
    content = render_skill_markdown(
        StructuredSkill(
            title="Verify Parser Shape",
            description="Repair parser shape mismatches shown by focused F2P evidence.",
            purpose="Prevent partial fixes that only handle one nested branch.",
            when_to_use=["F2P shows a nested return-shape mismatch."],
            how_to_apply=[
                "Trace the failing assertion to the branch that constructs the nested result.",
                "Patch that branch and rerun the focused assertion before the full suite.",
            ],
            constraints=["Do not flatten unrelated parser outputs."],
            action_type="INSPECT + PATCH + VERIFY",
            applicable_capability_label="incomplete_fix",
        )
    )
    assert lint_skill_markdown(content) == []
    assert "## Purpose" in content and "## Constraints" in content
