"""Offline mock-controller rollout helper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.controller.controller_client import ControllerClient
from src.critic.judge import CriticJudge
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.orchestrator.orchestrator import ExecutionOrchestrator
from src.reward.reward_model import RewardModel
from src.rl.rollout_writer import RolloutWriter, build_rollout_record
from src.skills.skill_selector import SkillSelector


def run_mock_rollout(
    issue: Issue,
    repo_path: Path,
    output_path: Path,
    test_cmd: Optional[str] = None,
    stage: str = "train",
    max_iterations: Optional[int] = None,
) -> dict:
    """Run one local repair attempt with a mock controller and write a rollout."""
    skill = SkillSelector().select()
    signal = ControllerClient(mode="mock").generate(issue, stage=stage, skill=skill)
    env = ProjectEnvironment(repo_path, test_cmd=test_cmd)
    result = ExecutionOrchestrator(
        env,
        max_iterations=max_iterations,
        controller_signal=signal,
    ).run(issue)
    evaluation = CriticJudge().evaluate(result)
    reward = RewardModel().score(result, controller_signal=signal)
    record = build_rollout_record(issue, signal, result, evaluation=evaluation, reward=reward)
    RolloutWriter(output_path).append(record)
    return record
