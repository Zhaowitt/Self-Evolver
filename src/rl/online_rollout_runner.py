"""Shared online rollout execution for Controller reward evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from src.controller.schema import ControllerSignal
from src.critic.judge import CriticJudge
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.orchestrator.orchestrator import ExecutionOrchestrator, ExecutionResult
from src.reward.reward_model import RewardModel, RewardResult
from src.rl.rollout_writer import RolloutWriter, build_rollout_record
from src.skills.skill_evolver import SkillEvolver


@dataclass
class OnlineRolloutResult:
    """Result of one controller-guided repair rollout."""

    execution_result: ExecutionResult
    evaluation: Any
    reward: RewardResult
    rollout_record: Optional[dict[str, Any]] = None
    skill_evolution: Optional[dict[str, Any]] = None


def run_online_rollout(
    issue: Issue,
    env: ProjectEnvironment,
    controller_signal: Optional[ControllerSignal],
    max_iterations: Optional[int] = None,
    reward_model: Optional[RewardModel] = None,
    rollout_writer: Optional[RolloutWriter] = None,
    skill_evolver: Optional[SkillEvolver] = None,
    judge: Optional[CriticJudge] = None,
    llm_client: Optional[LLMClient] = None,
) -> OnlineRolloutResult:
    """Run the repair loop, score it, and optionally append a rollout record."""
    judge = judge or CriticJudge()
    reward_model = reward_model or RewardModel()

    result = ExecutionOrchestrator(
        env=env,
        llm_client=llm_client,
        max_iterations=max_iterations,
        controller_signal=controller_signal,
    ).run(issue)

    evaluation = judge.evaluate(result)
    reward = reward_model.score(result, controller_signal=controller_signal)

    skill_evolution = None
    if skill_evolver and controller_signal:
        skill_evolution = skill_evolver.update_from_rollout(controller_signal, reward)

    rollout_record = build_rollout_record(
        issue,
        controller_signal,
        result,
        evaluation=evaluation,
        reward=reward,
        skill_evolution=skill_evolution,
    )
    if rollout_writer:
        rollout_writer.append(rollout_record)

    return OnlineRolloutResult(
        execution_result=result,
        evaluation=evaluation,
        reward=reward,
        rollout_record=rollout_record,
        skill_evolution=skill_evolution,
    )


def build_targeted_test_cmd(issue: Issue) -> Optional[str]:
    """Build a focused pytest command from SWE-bench FAIL_TO_PASS metadata."""
    raw = issue.metadata.get("fail_to_pass")
    if not raw:
        return None
    try:
        tests: list[str] = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not tests:
        return None
    test_args = " ".join(f'"{test}"' for test in tests)
    return f"python3 -m pytest {test_args} -x --tb=short -q"
