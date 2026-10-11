"""EasyR1 online reward: one repair episode per policy sample, graded by official SWE-bench eval."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.config import get_config
from src.controller.injection import ground_controller_skills
from src.controller.parser import parse_controller_response
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.reward.reward_model import RewardModel
from src.rl.online_rollout_runner import (
    build_targeted_test_cmd, run_online_rollout, swebench_instance_from_issue,
)
from src.rl.rollout_writer import RolloutWriter
from src.skills.file_lock import file_lock
from src.skills.skill_evolver import SkillEvolutionConfig, SkillEvolver

logger = logging.getLogger(__name__)


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs: Any,
) -> List[Dict[str, float]]:
    """Score Controller JSON ``response``s against serialized SWE-bench ``ground_truth``s."""
    from src.environment.test_backend import resolve_backend

    reward_model = RewardModel.from_config_file(_optional_path(
        kwargs.get("reward_config") or os.getenv("SELF_EVOLVER_REWARD_CONFIG")
    ))
    rollout_writer = _rollout_writer(
        kwargs.get("rollout_jsonl") or os.getenv("SELF_EVOLVER_ROLLOUT_JSONL")
    )
    skill_evolver = None
    if _env_flag(kwargs.get("enable_skill_evolution"), "SELF_EVOLVER_ENABLE_SKILL_EVOLUTION"):
        # configs/skill_evolution.yaml is the source of truth for credit knobs.
        skill_evolver = SkillEvolver(
            config=replace(
                SkillEvolutionConfig.load(),
                skill_write_utility_threshold=reward_model.skill_write_gate,
            )
        )
    test_backend = resolve_backend(
        str(kwargs.get("test_backend") or os.getenv("SELF_EVOLVER_TEST_BACKEND", "apptainer")),
        None,
    )

    default_workspace = kwargs.get("workspace_root")
    parallel_samples = _env_flag(
        kwargs.get("parallel_samples"), "SELF_EVOLVER_PARALLEL_SAMPLES", default="1"
    )
    slots = _checkout_slots(reward_inputs) if parallel_samples else {}
    groups = _group_reward_inputs(reward_inputs, default_workspace, slots=slots)
    max_workers = _parallel_workers(kwargs.get("parallel_workers"))

    def run_group(
        group: List[Tuple[int, Dict[str, Any]]],
    ) -> List[Tuple[int, Dict[str, float]]]:
        # Samples sharing a checkout stay sequential: each episode resets and
        # mutates it before official evaluation.
        try:
            return [
                (
                    index,
                    _score_one(
                        reward_input,
                        reward_model=reward_model,
                        rollout_writer=rollout_writer,
                        skill_evolver=skill_evolver,
                        test_backend=test_backend,
                        default_workspace=default_workspace,
                        max_iterations=kwargs.get("max_iterations"),
                        checkout_slot=slots.get(index, 0),
                    ),
                )
                for index, reward_input in group
            ]
        finally:
            if _env_flag(
                kwargs.get("cleanup_reward_workspaces"),
                "SELF_EVOLVER_CLEANUP_REWARD_WORKSPACES",
            ):
                _cleanup_group_checkout(
                    group, default_workspace, slot=slots.get(group[0][0], 0) if group else 0
                )

    def score_group(group):
        extra, truth = _reward_payload(group[0][1])
        instance_id = truth.get("instance_id") or truth.get("id") or extra.get("instance_id")
        if not instance_id or extra.get("repo_path") or truth.get("repo_path"):
            return run_group(group)
        checkout = checkout_dir_for(_workspace_root(default_workspace, extra), str(instance_id),
                                    slots.get(group[0][0], 0))
        # Covers reset, patch execution and cleanup, including other processes.
        with file_lock(checkout.with_name(checkout.name + ".episode")):
            return run_group(group)

    scores: List[Dict[str, float]] = [{} for _ in reward_inputs]
    worker_count = min(max_workers, len(groups))
    if worker_count <= 1:
        completed_groups = (score_group(group) for group in groups)
        for completed in completed_groups:
            for index, score in completed:
                scores[index] = score
        return _finish_scores(reward_inputs, scores)

    logger.info(
        "Scoring %d online rollouts in %d instance-safe parallel groups",
        len(reward_inputs),
        worker_count,
    )
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="online-reward",
    ) as executor:
        futures = [executor.submit(score_group, group) for group in groups]
        for future in as_completed(futures):
            for index, score in future.result():
                scores[index] = score
    return _finish_scores(reward_inputs, scores)


class ValidationInfrastructureError(RuntimeError):
    """An environment/API failure is not a valid policy evaluation score."""


def _finish_scores(reward_inputs, scores):
    invalid = []
    for reward_input, score in zip(reward_inputs, scores):
        extra_info, truth = _reward_payload(reward_input)
        stage = str(extra_info.get("stage") or truth.get("stage") or "train")
        if stage != "train" and score.get("infra_error"):
            invalid.append(str(truth.get("instance_id") or extra_info.get("instance_id") or "unknown"))
    if invalid:
        raise ValidationInfrastructureError(
            f"Validation infrastructure failure in {len(invalid)} episode(s), "
            f"tasks={sorted(set(invalid))[:12]}; refusing to return zero/imputed validation scores. "
            "Fix environments/provider access and retry validation; existing checkpoints remain valid."
        )
    # Train-time neutralization is unchanged. Never impute validation rewards.
    return _neutralize_infra_errors(reward_inputs, scores)


def _neutralize_infra_errors(
    reward_inputs: List[Dict[str, Any]],
    scores: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Infra errors get their instance group's valid mean: a fixed 0.0 would out-rank honest (negative) failures."""
    by_instance: Dict[str, List[int]] = {}
    for index, reward_input in enumerate(reward_inputs):
        _, ground_truth = _reward_payload(reward_input)
        by_instance.setdefault(str(ground_truth.get("instance_id") or index), []).append(index)
    for indices in by_instance.values():
        valid = [scores[i]["overall"] for i in indices if not scores[i].get("infra_error")]
        if not valid:
            continue
        mean = sum(valid) / len(valid)
        for i in indices:
            if scores[i].get("infra_error"):
                scores[i]["overall"] = mean
    return scores


def _checkout_slots(reward_inputs: List[Dict[str, Any]]) -> Dict[int, int]:
    """Per-sample checkout slot: k-th occurrence of an instance in the batch."""
    seen: Dict[str, int] = {}
    slots: Dict[int, int] = {}
    for index, reward_input in enumerate(reward_inputs):
        extra_info, ground_truth = _reward_payload(reward_input)
        if extra_info.get("repo_path") or ground_truth.get("repo_path"):
            continue  # caller-owned checkout: shared, stays sequential
        instance_id = str(
            ground_truth.get("instance_id") or ground_truth.get("id") or extra_info.get("instance_id") or ""
        )
        if not instance_id:
            continue
        slots[index] = seen.get(instance_id, 0)
        seen[instance_id] = slots[index] + 1
    return slots


def checkout_dir_for(workspace: Path, instance_id: str, slot: int = 0) -> Path:
    """Slot 0 keeps the historical cache path; other samples get siblings."""
    name = str(instance_id).replace("/", "_")
    return Path(workspace) / (name if slot <= 0 else f"{name}__s{slot}")


def _group_reward_inputs(
    reward_inputs: List[Dict[str, Any]],
    default_workspace: Any = None,
    slots: Optional[Dict[int, int]] = None,
) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """Group samples that may mutate the same cached checkout."""
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for index, reward_input in enumerate(reward_inputs):
        extra_info = _coerce_dict(
            reward_input.get("extra_info") or reward_input.get("metadata")
        )
        ground_truth = _coerce_dict(
            reward_input.get("ground_truth")
            or reward_input.get("answer")
            or extra_info.get("ground_truth")
        )
        repo_path = extra_info.get("repo_path") or ground_truth.get("repo_path")
        instance_id = (
            ground_truth.get("instance_id")
            or ground_truth.get("id")
            or extra_info.get("instance_id")
        )
        if repo_path:
            key = f"repo:{Path(str(repo_path)).expanduser().resolve()}"
        elif instance_id:
            checkout = checkout_dir_for(
                _workspace_root(default_workspace, extra_info),
                str(instance_id),
                (slots or {}).get(index, 0),
            )
            key = f"repo:{checkout}"
        else:
            # Malformed samples do not reach a checkout, so they need not block
            # otherwise independent reward work.
            key = f"input:{index}"
        groups.setdefault(key, []).append((index, reward_input))
    return list(groups.values())


def _workspace_root(default_workspace: Any, extra_info: Dict[str, Any]) -> Path:
    return Path(
        default_workspace
        or extra_info.get("workspace_root")
        or os.getenv("SELF_EVOLVER_REWARD_WORKSPACE")
        or (get_config().environment.workspace_dir / "online_reward")
    ).expanduser().resolve()


def _parallel_workers(kwargs_value: Any) -> int:
    value = (
        kwargs_value
        if kwargs_value is not None
        else os.getenv("SELF_EVOLVER_REWARD_WORKERS", "2")
    )
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        logger.warning("Invalid online reward worker count %r; using 2", value)
        return 2


def _cleanup_group_checkout(
    group: List[Tuple[int, Dict[str, Any]]],
    default_workspace: Any = None,
    slot: int = 0,
) -> None:
    """Remove an auto-created checkout; caller-owned ``repo_path`` inputs are never removed."""
    if not group:
        return
    reward_input = group[0][1]
    extra_info, ground_truth = _reward_payload(reward_input)
    if extra_info.get("repo_path") or ground_truth.get("repo_path"):
        return
    instance_id = (
        ground_truth.get("instance_id")
        or ground_truth.get("id")
        or extra_info.get("instance_id")
    )
    if not instance_id:
        return
    workspace = _workspace_root(default_workspace, extra_info)
    checkout = checkout_dir_for(workspace, str(instance_id), slot).resolve()
    if checkout.parent != workspace or not checkout.is_dir():
        return
    base = checkout_dir_for(workspace, str(instance_id), 0).resolve()
    try:
        # Never delete the slot-0 base while another sample is cloned from it.
        with file_lock(base.with_name(base.name + ".clone")):
            shutil.rmtree(checkout)
        logger.info("Removed completed online-reward checkout %s", checkout)
    except OSError as exc:
        logger.warning("Could not remove completed checkout %s: %s", checkout, exc)


def _reward_payload(reward_input: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(extra_info, ground_truth); EasyR1 drops extra_info, so the copy embedded in ground_truth fills in."""
    extra_info = _coerce_dict(reward_input.get("extra_info") or reward_input.get("metadata"))
    ground_truth = _coerce_dict(
        reward_input.get("ground_truth")
        or reward_input.get("answer")
        or extra_info.get("ground_truth")
    )
    embedded = _coerce_dict(ground_truth.get("extra_info"))
    return {**embedded, **extra_info}, ground_truth


def _score_one(
    reward_input: Dict[str, Any],
    reward_model: RewardModel,
    rollout_writer: RolloutWriter,
    skill_evolver: Optional[SkillEvolver],
    test_backend: Any,
    default_workspace: Any = None,
    max_iterations: Any = None,
    checkout_slot: int = 0,
) -> Dict[str, float]:
    extra_info, ground_truth = _reward_payload(reward_input)
    stage = str(extra_info.get("stage") or ground_truth.get("stage") or "train")
    response = _extract_response(reward_input)

    signal = parse_controller_response(response, mode=stage, source="policy")
    if signal.parse_error:
        logger.warning("Invalid Controller response: %s", signal.parse_error)
        # Strictly below every valid outcome (valid totals are >= -0.15), so a
        # malformed sample can never out-rank an honest failed attempt in GRPO.
        return _failure_score(
            parse_valid=0.0,
            worker_executed=0.0,
            overall=reward_model.format_failure_reward,
        )

    try:
        allowed_skill_ids = ground_truth.get("allowed_skill_ids")
        if allowed_skill_ids is None:
            allowed_skill_ids = extra_info.get("allowed_skill_ids")
        signal = ground_controller_skills(
            signal,
            allowed_skill_ids=(
                allowed_skill_ids if isinstance(allowed_skill_ids, list) else None
            ),
        )
        issue = issue_from_payload(ground_truth, extra_info)
        if stage != "train":
            # Check the image *before* spending worker API calls or modifying a
            # checkout. A CPU source builder holds the same image lock until a
            # candidate has passed its controls and been atomically published.
            ensure_image = getattr(test_backend, "ensure_image", None)
            if callable(ensure_image):
                ensure_image(swebench_instance_from_issue(issue))
        env = prepare_environment(
            issue, ground_truth, extra_info, default_workspace, slot=checkout_slot
        )
        rollout = run_online_rollout(
            issue,
            env,
            controller_signal=signal,
            # Controller/payload output cannot alter the system-owned cap.
            max_iterations=3,
            reward_model=reward_model,
            rollout_writer=rollout_writer,
            # Validation/final evaluation must be frozen even when the process
            # has train-time skill credit enabled globally.
            skill_evolver=skill_evolver if stage == "train" else None,
            test_backend=test_backend,
            verify_in_loop=True,
            stage=stage,
            seed=_int_or_none(extra_info.get("seed")),
            experiment=str(extra_info.get("experiment") or "rl_online"),
        )
    except Exception as exc:
        # Infrastructure failure: not the policy's fault. Training uses group
        # mean neutralization; evaluation refuses to return an invalid metric.
        logger.exception("Online reward execution failed: %s", exc)
        return _failure_score(parse_valid=1.0, worker_executed=0.0, infra_error=1.0, signal=signal)

    payload = {
        "overall": float(rollout.reward.total),
        "parse_valid": 1.0,
        "worker_executed": 1.0,
        "infra_error": 0.0,
        **_action_metrics(signal),
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
            "train_fail_to_pass_subset": (
                ground_truth.get("TRAIN_FAIL_TO_PASS_SUBSET")
                or ground_truth.get("train_fail_to_pass_subset")
                or extra_info.get("TRAIN_FAIL_TO_PASS_SUBSET")
                or extra_info.get("train_fail_to_pass_subset")
            ),
            "f2p_subset_policy": (
                ground_truth.get("F2P_SUBSET_POLICY")
                or extra_info.get("F2P_SUBSET_POLICY")
            ),
            "round_id": ground_truth.get("round_id") or extra_info.get("round_id"),
            "observed_failure_label": (
                ground_truth.get("observed_failure_label")
                or extra_info.get("observed_failure_label")
            ),
            "stable_cluster_id": (
                ground_truth.get("stable_cluster_id")
                or extra_info.get("stable_cluster_id")
            ),
            "sampling_source": (
                ground_truth.get("sampling_source") or extra_info.get("sampling_source")
            ),
        },
    )


def prepare_environment(
    issue: Issue,
    ground_truth: Dict[str, Any],
    extra_info: Dict[str, Any],
    default_workspace: Any = None,
    slot: int = 0,
) -> ProjectEnvironment:
    """``slot`` > 0 gives a sample its own checkout cloned from the shared mirror."""
    repo_path_value = extra_info.get("repo_path") or ground_truth.get("repo_path")
    workspace_root = _workspace_root(default_workspace, extra_info)
    if repo_path_value:
        repo_dir = Path(str(repo_path_value)).expanduser().resolve()
    else:
        repo_dir = checkout_dir_for(workspace_root, issue.id, slot)

    test_cmd = extra_info.get("test_cmd") or build_targeted_test_cmd(issue)
    if not test_cmd:
        # Without FAIL_TO_PASS the Verifier would run the whole repo suite with
        # bare ``pytest``: slow and not the SWE-bench reward signal.
        raise ValueError(
            f"{issue.id}: no FAIL_TO_PASS metadata and no explicit extra_info['test_cmd']; "
            "refusing to run the full repository test suite. Fix the dataset's "
            "FAIL_TO_PASS / PASS_TO_PASS or pass extra_info['test_cmd']."
        )
    timeout = int(extra_info.get("timeout") or get_config().agent.timeout_seconds)

    if repo_path_value:
        env = ProjectEnvironment(repo_dir, test_cmd=test_cmd, timeout=timeout)
        if issue.base_commit and not env.reset_to_commit(issue.base_commit):
            raise RuntimeError(f"failed to reset caller-owned repo {repo_dir} to {issue.base_commit}")
        return env
    if not issue.repo_name:
        raise ValueError("repo_name is required for a managed checkout")
    if repo_dir.resolve().parent != workspace_root.resolve():
        raise ValueError("Managed checkout must be directly inside its workspace")
    with file_lock(repo_dir.with_name(repo_dir.name + ".clone")):
        mirror = _repo_mirror(workspace_root, issue.repo_name)
        ensure_repository_commit(mirror, issue.base_commit, "origin")
        # Clone each slot from the immutable bare source, never an actively
        # modified sibling. Failed checkouts are isolated, not unlocked in place.
        for attempt in range(2):
            try:
                if not (repo_dir / ".git").exists():
                    if repo_dir.exists():
                        raise RuntimeError(f"Incomplete managed checkout: {repo_dir}")
                    _publish_checkout(mirror, repo_dir, issue.base_commit)
                ensure_repository_commit(repo_dir, issue.base_commit, str(mirror))
                env = ProjectEnvironment(repo_dir, test_cmd=test_cmd, timeout=timeout)
                if not issue.base_commit or env.reset_to_commit(issue.base_commit):
                    return env
                raise RuntimeError(f"failed to reset cached repo {repo_dir} to {issue.base_commit}")
            except (RuntimeError, subprocess.CalledProcessError):
                if attempt or not repo_dir.exists():
                    raise
                quarantine = workspace_root / "_failed_checkouts"
                quarantine.mkdir(exist_ok=True)
                saved = quarantine / f"{repo_dir.name}-{uuid.uuid4().hex}"
                repo_dir.rename(saved)
                logger.warning("Rebuilding failed managed checkout; original preserved at %s", saved)
    raise RuntimeError("Managed checkout recovery exhausted")


def _git(*args: str) -> None:
    try:
        subprocess.run(["git", *args], check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Git {' '.join(args)} failed: {(exc.stderr or '').strip()[-2000:]}") from exc


def ensure_repository_commit(repo: Path, commit: Optional[str], source: str) -> None:
    """Fetch the exact required object, including commits outside current refs."""
    if not commit:
        return
    if not all(c in "0123456789abcdefABCDEF" for c in commit) or len(commit) != 40:
        raise ValueError(f"Expected a full Git commit SHA: {commit!r}")
    def present():
        return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
                              capture_output=True, timeout=30).returncode == 0
    if not present():
        with file_lock(repo):
            if not present():
                _git("-C", str(repo), "fetch", "--quiet", source, commit)
            if not present():
                raise RuntimeError(f"Required commit {commit} unavailable in {repo}")


def _publish_checkout(mirror: Path, checkout: Path, commit: Optional[str]) -> None:
    candidate = checkout.with_name(f".{checkout.name}.clone-{uuid.uuid4().hex}")
    try:
        with file_lock(mirror):
            # Node-local checkouts cannot hard-link objects from shared scratch.
            copy_args = ["--no-hardlinks"] if mirror.stat().st_dev != checkout.parent.stat().st_dev else []
            _git("clone", "--quiet", "--no-checkout", "--local", *copy_args,
                 str(mirror.resolve()), str(candidate))
            ensure_repository_commit(candidate, commit, str(mirror))
        if commit:
            _git("-C", str(candidate), "reset", "--hard", commit)
        candidate.rename(checkout)
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)


def _repo_mirror(workspace: Path, repo_name: str, refresh: bool = False) -> Path:
    """Bare per-repository mirror under ``<workspace>/_mirrors``."""
    mirror = Path(workspace) / "_mirrors" / (repo_name.replace("/", "__") + ".git")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(mirror):
        # --bare, not --mirror: a GitHub mirror also downloads every
        # refs/pull/* head; SWE-bench base commits live on branch history.
        if not (mirror / "HEAD").exists():
            shutil.rmtree(mirror, ignore_errors=True)
            _git("clone", "--quiet", "--bare", f"https://github.com/{repo_name}.git", str(mirror))
        elif refresh:
            _git(
                "-C", str(mirror), "fetch", "--quiet", "origin",
                "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*",
            )
    return mirror


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


# Every sample must report the same keys: EasyR1 averages each metric over the
# samples that carry it, so missing keys on failures bias the logged means.
_COMPONENT_KEYS = (
    "official_resolved",
    "official_f2p_fraction",
    "token_cost",
    "resolved",
    "f2p_fraction",
    "p2p_no_regression",
    "cost_efficiency",
    "process",
)


def _action_metrics(signal: Any = None) -> Dict[str, float]:
    """Controller action statistics (means over the batch expose collapse)."""
    selected = list(getattr(signal, "selected_skill_ids", None) or [])
    return {
        "num_selected_skills": float(len(selected)),
        "empty_selection": 1.0 if signal is not None and not selected else 0.0,
        "strategy_present": 1.0 if getattr(signal, "strategy", "") else 0.0,
    }


def _failure_score(
    parse_valid: float,
    worker_executed: float,
    overall: float = 0.0,
    infra_error: float = 0.0,
    signal: Any = None,
) -> Dict[str, float]:
    payload = {
        "overall": float(overall),
        "parse_valid": float(parse_valid),
        "worker_executed": float(worker_executed),
        "infra_error": float(infra_error),
        **_action_metrics(signal),
    }
    payload.update({name: 0.0 for name in _COMPONENT_KEYS})
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
