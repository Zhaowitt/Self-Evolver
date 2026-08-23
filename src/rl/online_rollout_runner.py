"""Shared online rollout execution for Controller reward evaluation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from src.config import get_config
from src.controller.schema import ControllerSignal
from src.critic.judge import CriticJudge
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.orchestrator.orchestrator import ExecutionOrchestrator, ExecutionResult
from src.reward.reward_model import RewardModel, RewardResult
from src.rl.rollout_writer import RolloutWriter, build_rollout_record
from src.skills.skill_evolver import SkillEvolver

logger = logging.getLogger(__name__)


@dataclass
class OnlineRolloutResult:
    """Result of one controller-guided repair rollout."""

    execution_result: ExecutionResult
    evaluation: Any
    reward: RewardResult
    eval_outcome: Any = None
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
    test_backend: Any = None,
    eval_timeout: Optional[int] = None,
    stage: Optional[str] = None,
    seed: Optional[int] = None,
    experiment: Optional[str] = None,
    verify_in_loop: bool = True,
) -> OnlineRolloutResult:
    """Run the repair loop, grade the final patch, score it, and log the rollout.

    When ``test_backend`` (a ``src.environment.test_backend`` backend) is
    provided and the episode produced a non-empty patch, the patch is graded
    with official SWE-bench eval semantics and the resulting ``EvalOutcome``
    drives the test components of the utility.

    With ``verify_in_loop`` each train patch first runs its frozen F2P subset.
    Only subset-passing patches (plus the final usable patch) run full official
    F2P+P2P.  Final reward reads the full outcome only.
    """
    judge = judge or CriticJudge()
    reward_model = reward_model or RewardModel.from_config_file()

    result = ExecutionOrchestrator(
        env=env,
        llm_client=llm_client,
        max_iterations=max_iterations,
        controller_signal=controller_signal,
        test_backend=test_backend if verify_in_loop else None,
        stage=stage,
    ).run(issue)

    # Reuse only a final full in-loop outcome.  A subset-only final attempt is
    # not silently promoted, nor is an earlier patch re-graded as the final one.
    eval_outcome = _final_eval_outcome(result) if verify_in_loop else None
    if eval_outcome is None and test_backend is not None and not verify_in_loop:
        patch = model_patch_from_execution(result)
        if patch.strip():
            evaluator = getattr(test_backend, "run_full_eval", test_backend.run_swebench_eval)
            eval_outcome = evaluator(
                swebench_instance_from_issue(issue), patch,
                timeout=eval_timeout or get_config().docker.timeout,
            )
        else:
            logger.info("Empty final patch for %s; skipping container eval", issue.id)

    evaluation = judge.evaluate(result)
    reward = reward_model.score(
        result,
        eval_outcome=eval_outcome,
        issue=issue,
        selected_skill_ids=(controller_signal.selected_skill_ids if controller_signal else []),
    )

    skill_evolution = None
    if skill_evolver and controller_signal:
        official_success = None
        if eval_outcome is not None:
            official_success = bool(
                eval_outcome.get("resolved", False)
                if isinstance(eval_outcome, dict)
                else getattr(eval_outcome, "resolved", False)
            )
        skill_evolution = skill_evolver.update_from_rollout(
            controller_signal,
            reward,
            success=official_success,
            round_id=os.getenv("SELF_EVOLVER_ROUND_ID") or None,
        )

    rollout_record = build_rollout_record(
        issue,
        controller_signal,
        result,
        evaluation=evaluation,
        reward=reward,
        skill_evolution=skill_evolution,
        eval_outcome=eval_outcome,
        stage=stage,
        seed=seed,
        experiment=experiment,
    )
    if rollout_writer:
        rollout_writer.append(rollout_record)

    return OnlineRolloutResult(
        execution_result=result,
        evaluation=evaluation,
        reward=reward,
        eval_outcome=eval_outcome,
        rollout_record=rollout_record,
        skill_evolution=skill_evolution,
    )


def _final_eval_outcome(result: ExecutionResult) -> Any:
    """Full outcome of the final attempted candidate, never a subset/older patch."""
    records = list(getattr(result, "iteration_records", []) or [])
    if not records:
        return None
    verification = getattr(records[-1], "verification_result", None)
    if not verification or getattr(verification, "evaluation_scope", "full") != "full":
        return None
    return getattr(verification, "eval_outcome", None)


def model_patch_from_execution(result: ExecutionResult) -> str:
    """Best canonical (git-generated) patch produced by an execution."""
    final_patch = getattr(result, "final_patch", None)
    if final_patch and getattr(final_patch, "content", "").strip():
        return final_patch.content
    for record in reversed(getattr(result, "iteration_records", []) or []):
        verification = getattr(record, "verification_result", None)
        if verification and getattr(verification, "canonical_patch_content", "").strip():
            return verification.canonical_patch_content
    return ""


def swebench_instance_from_issue(issue: Issue) -> dict[str, Any]:
    """Rebuild the official SWE-bench instance dict a test backend needs."""
    instance: dict[str, Any] = {
        "instance_id": issue.id,
        "repo": issue.repo_name,
        "base_commit": issue.base_commit,
        "problem_statement": issue.description,
        "test_patch": issue.test_patch,
        "version": issue.metadata.get("version"),
        "FAIL_TO_PASS": issue.metadata.get("fail_to_pass"),
        "PASS_TO_PASS": issue.metadata.get("pass_to_pass"),
    }
    setup_commit = issue.metadata.get("environment_setup_commit")
    if setup_commit:
        instance["environment_setup_commit"] = setup_commit
    return instance


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
