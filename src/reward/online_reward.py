"""EasyR1 online reward function for Controller policy training.

``compute_score`` runs one full repair episode per policy sample, grades the
final patch with official SWE-bench eval semantics inside the per-instance
container (``ContainerTestBackend``; engine from ``SELF_EVOLVER_TEST_BACKEND``,
default ``apptainer``), and returns the execution utility (Proposal 2.6).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.controller.parser import parse_controller_response
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.reward.reward_model import DEFAULT_REWARD_WEIGHTS, RewardModel
from src.rl.online_rollout_runner import build_targeted_test_cmd, run_online_rollout
from src.rl.rollout_writer import RolloutWriter
from src.skills.skill_evolver import SkillEvolutionConfig, SkillEvolver

logger = logging.getLogger(__name__)


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs: Any,
) -> List[Dict[str, float]]:
    """
    Score EasyR1 policy responses by running the full repair environment.

    Each input must include a policy-generated Controller JSON string in
    ``response`` and a serialized SWE-bench instance in ``ground_truth``.
    """
    from src.environment.test_backend import resolve_backend

    reward_model = RewardModel.from_config_file(_optional_path(
        kwargs.get("reward_config") or os.getenv("SELF_EVOLVER_REWARD_CONFIG")
    ))
    rollout_writer = _rollout_writer(
        kwargs.get("rollout_jsonl") or os.getenv("SELF_EVOLVER_ROLLOUT_JSONL")
    )
    skill_evolver = None
    if _env_flag(kwargs.get("enable_skill_evolution"), "SELF_EVOLVER_ENABLE_SKILL_EVOLUTION"):
        skill_evolver = SkillEvolver(
            config=SkillEvolutionConfig(
                skill_write_utility_threshold=reward_model.skill_write_gate,
            )
        )
    test_backend = resolve_backend(
        str(kwargs.get("test_backend") or os.getenv("SELF_EVOLVER_TEST_BACKEND", "apptainer")),
        None,
    )

    scores: List[Dict[str, float]] = []
    for reward_input in reward_inputs:
        scores.append(
            _score_one(
                reward_input,
                reward_model=reward_model,
                rollout_writer=rollout_writer,
                skill_evolver=skill_evolver,
                test_backend=test_backend,
                default_workspace=kwargs.get("workspace_root"),
                max_iterations=kwargs.get("max_iterations"),
            )
        )
    return scores


def _score_one(
    reward_input: Dict[str, Any],
    reward_model: RewardModel,
    rollout_writer: RolloutWriter,
    skill_evolver: Optional[SkillEvolver],
    test_backend: Any,
    default_workspace: Any = None,
    max_iterations: Any = None,
) -> Dict[str, float]:
    extra_info = _coerce_dict(reward_input.get("extra_info") or reward_input.get("metadata"))
    ground_truth = _coerce_dict(
        reward_input.get("ground_truth")
        or reward_input.get("answer")
        or extra_info.get("ground_truth")
    )
    stage = str(extra_info.get("stage") or ground_truth.get("stage") or "train")
    response = _extract_response(reward_input)

    signal = parse_controller_response(response, mode=stage, source="policy")
    if signal.parse_error:
        logger.warning("Invalid Controller response: %s", signal.parse_error)
        return _failure_score(parse_valid=0.0, worker_executed=0.0)

    try:
        issue = issue_from_payload(ground_truth, extra_info)
        env = prepare_environment(issue, ground_truth, extra_info, default_workspace)
        rollout = run_online_rollout(
            issue,
            env,
            controller_signal=signal,
            max_iterations=_int_or_none(
                max_iterations
                or extra_info.get("max_iterations")
                or os.getenv("SELF_EVOLVER_MAX_ITERATIONS")
            ),
            reward_model=reward_model,
            rollout_writer=rollout_writer,
            skill_evolver=skill_evolver,
            test_backend=test_backend,
            stage=stage,
            seed=_int_or_none(extra_info.get("seed")),
            experiment=str(extra_info.get("experiment") or "rl_online"),
        )
    except Exception as exc:
        logger.exception("Online reward execution failed: %s", exc)
        return _failure_score(parse_valid=1.0, worker_executed=0.0)

    payload = {
        "overall": float(rollout.reward.total),
        "parse_valid": 1.0,
        "worker_executed": 1.0,
    }
    payload.update({
        key: float(value)
        for key, value in rollout.reward.components.items()
    })
    return payload


def issue_from_payload(
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
) -> Issue:
    """Reconstruct an Issue from EasyR1 ground-truth metadata."""
    extra_info = extra_info or {}
    issue_id = str(
        ground_truth.get("instance_id")
        or ground_truth.get("id")
        or extra_info.get("instance_id")
        or ""
    )
    description = str(
        ground_truth.get("problem_statement")
        or ground_truth.get("description")
        or extra_info.get("problem_statement")
        or ""
    )
    if not issue_id or not description:
        raise ValueError("ground_truth must include instance_id and problem_statement")

    return Issue(
        id=issue_id,
        description=description,
        repo_name=ground_truth.get("repo_name") or ground_truth.get("repo") or extra_info.get("repo_name"),
        base_commit=ground_truth.get("base_commit") or extra_info.get("base_commit"),
        hints=ground_truth.get("hints_text") or ground_truth.get("hints") or extra_info.get("hints"),
        test_patch=ground_truth.get("test_patch") or extra_info.get("test_patch"),
        metadata={
            "version": ground_truth.get("version"),
            "environment_setup_commit": (
                ground_truth.get("environment_setup_commit")
                or extra_info.get("environment_setup_commit")
            ),
            "fail_to_pass": (
                ground_truth.get("FAIL_TO_PASS")
                or ground_truth.get("fail_to_pass")
                or extra_info.get("fail_to_pass")
            ),
            "pass_to_pass": (
                ground_truth.get("PASS_TO_PASS")
                or ground_truth.get("pass_to_pass")
                or extra_info.get("pass_to_pass")
            ),
        },
    )


def prepare_environment(
    issue: Issue,
    ground_truth: Dict[str, Any],
    extra_info: Dict[str, Any],
    default_workspace: Any = None,
) -> ProjectEnvironment:
    """Prepare a cached repository checkout for online reward execution."""
    repo_path_value = extra_info.get("repo_path") or ground_truth.get("repo_path")
    workspace_root = (
        default_workspace
        or extra_info.get("workspace_root")
        or os.getenv("SELF_EVOLVER_REWARD_WORKSPACE")
        or (get_config().environment.workspace_dir / "online_reward")
    )
    if repo_path_value:
        repo_dir = Path(str(repo_path_value)).expanduser().resolve()
    else:
        repo_dir = Path(workspace_root).expanduser().resolve() / issue.id.replace("/", "_")

    repo_dir.mkdir(parents=True, exist_ok=True)
    test_cmd = extra_info.get("test_cmd") or build_targeted_test_cmd(issue)
    timeout = int(extra_info.get("timeout") or get_config().agent.timeout_seconds)

    env = ProjectEnvironment(repo_dir, test_cmd=test_cmd, timeout=timeout)
    if not (repo_dir / ".git").exists():
        if not issue.repo_name:
            raise ValueError("repo_name is required when repo_path is not an existing git repo")
        repo_url = f"https://github.com/{issue.repo_name}.git"
        if not env.clone_repo(repo_url):
            raise RuntimeError(f"failed to clone repository: {repo_url}")
    elif issue.base_commit and not env.reset_to_commit(issue.base_commit):
        # Cached checkouts are reused across GRPO samples of the same instance;
        # a hard reset prevents state leaking from the previous episode.
        raise RuntimeError(f"failed to reset cached repo {repo_dir} to {issue.base_commit}")
    return env


def _extract_response(reward_input: Dict[str, Any]) -> str:
    response = (
        reward_input.get("response")
        or reward_input.get("predict_str")
        or reward_input.get("completion")
        or ""
    )
    return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(loaded) if isinstance(loaded, dict) else {}
    return {}


def _failure_score(parse_valid: float, worker_executed: float) -> Dict[str, float]:
    payload = {
        "overall": 0.0,
        "parse_valid": float(parse_valid),
        "worker_executed": float(worker_executed),
    }
    payload.update({name: 0.0 for name in DEFAULT_REWARD_WEIGHTS})
    return payload


def _rollout_writer(path_value: Any) -> RolloutWriter:
    path = Path(str(path_value or "benchmark_results/online_rollouts.jsonl"))
    return RolloutWriter(path)


def _optional_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    return Path(str(value))


def _int_or_none(value: Any) -> Optional[int]:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_flag(kwargs_value: Any, env_var: str, default: str = "0") -> bool:
    """Explicit kwargs win over the environment (False must not fall through)."""
    if kwargs_value is not None:
        return _as_bool(kwargs_value)
    return _as_bool(os.getenv(env_var, default))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
