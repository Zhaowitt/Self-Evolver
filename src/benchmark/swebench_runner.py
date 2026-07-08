"""SWE-bench runner: generate predictions and grade them with official semantics.

Generation routes every instance through the shared repair rollout. In-loop
verification uses a real per-instance container image (``ContainerTestBackend``,
apptainer or docker) so the reward sees the official FAIL_TO_PASS / PASS_TO_PASS
per-test outcomes rather than an unreliable host run.

Train stage (``--stage train``) draws instances from an evolving ``TaskPool``
and, when skill evolution is on, runs the ``Reflector`` every N rollouts to
turn recurring failures into skill and task-distribution updates. Eval stage
(``--stage eval``) is frozen: the skill bank is snapshotted into the run
directory and read-only, and no skill / task / memory evolution runs.

Each configuration of the flags below is one *experiment* (see ``ExperimentConfig``).
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmark import datasets
from src.benchmark.base import BenchmarkRunner
from src.config import get_config
from src.controller.controller_client import ControllerClient
from src.controller.schema import ControllerSignal, SkillSignal
from src.critic.judge import CriticJudge
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.environment.test_backend import default_eval_timeout, resolve_backend
from src.memory.memory_retriever import MemoryRetriever
from src.reflection.reflector import Reflector
from src.reward.reward_model import RewardModel
from src.rl.online_rollout_runner import (
    build_targeted_test_cmd,
    model_patch_from_execution,
    run_online_rollout,
    swebench_instance_from_issue,
)
from src.rl.rollout_writer import RolloutWriter, build_rollout_record
from src.skills.skill_bank import SkillBank
from src.skills.skill_evolver import SkillEvolutionConfig, SkillEvolver
from src.skills.skill_selector import SkillSelector
from src.skills.skill_store import SkillStore
from src.tasks.task_pool import TaskPool
from src.tasks.variants import base_instance_id
from src.tasks.verification import verify_task

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """One benchmark configuration (baselines and the full method are experiments)."""

    agent_mode: str = "mas"        # single | mas
    skills: str = "static"          # off | static | evolve
    memory: str = "on"              # on | off
    task_evolution: str = "off"     # on | off  (train stage only)
    controller_mode: str = "off"    # off | llm
    stage: str = "eval"             # train | eval
    seed: int = 0
    test_backend: str = "auto"      # auto | docker | apptainer | host
    hints: bool = False             # surface human hints_text to the worker
    validate_skills: int = 0        # replay M held-out instances before a skill write
    label: str = ""                 # descriptive experiment name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    """Outcome of one repair episode used to update the task pool and metrics."""

    instance_id: str
    patch: str
    resolved: bool
    utility: float
    execution_result: Any = None
    eval_outcome: Any = None


class _ImageBaseAdapter:
    """Grade an instance in its base image.

    Focused task variants (``<id>::focus-<n>``) run in their parent's official
    image; this adapter rewrites the grading instance id to the base id so the
    right image is used while the subset FAIL_TO_PASS is still evaluated.
    """

    def __init__(self, backend: Any):
        self.backend = backend

    def run_swebench_eval(self, instance: dict, model_patch: str, timeout: Optional[int] = None):
        base = base_instance_id(instance)
        if base and base != instance.get("instance_id"):
            instance = {**instance, "instance_id": base}
        return self.backend.run_swebench_eval(instance, model_patch, timeout=timeout)


class SWEBenchRunner(BenchmarkRunner):
    """Generate and grade SWE-bench predictions under one experiment configuration."""

    def __init__(
        self,
        dataset: str = "lite",
        output_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
        model_name: str = "self-evolver",
        run_id: str = "self-evolver",
        experiment: Optional[ExperimentConfig] = None,
        reward_config: Optional[Path] = None,
        train_ids_path: Optional[Path] = None,
    ):
        super().__init__(name="swebench", output_dir=output_dir)
        self.run_dir = self.output_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.workspace_dir = workspace_dir or Path(tempfile.mkdtemp(prefix="swebench_"))
        self.model_name = model_name
        self.run_id = run_id
        self.experiment = experiment or ExperimentConfig()
        self._train_ids_path = Path(train_ids_path) if train_ids_path else None
        self.reward_model = RewardModel.from_config_file(reward_config)
        self.judge = CriticJudge()

        # Isolate this run: hard cases and workspace-relative state live under the
        # run directory instead of the shared repo workspace.
        get_config().environment.workspace_dir = self.run_dir
        self._hard_cases_path = self.run_dir / "hard_cases.jsonl"

        self._is_eval = self.experiment.stage == "eval"
        self._skill_evolve = (
            self.experiment.stage == "train" and self.experiment.skills == "evolve"
        )
        self._skills_dir = self._frozen_skills_dir() if self._is_eval else self._repo_skills_dir()
        self._bank = SkillBank(self._skills_dir)
        self._selector = (
            SkillSelector(self._bank) if self.experiment.skills != "off" else None
        )
        self._evolver = (
            SkillEvolver(store=SkillStore(self._skills_dir)) if self._skill_evolve else None
        )
        self._controller_client = (
            ControllerClient(mode=self.experiment.controller_mode)
            if self.experiment.controller_mode != "off"
            else None
        )
        self.rollout_writer = RolloutWriter(self.run_dir / "rollouts.jsonl")

        self._backend_attempted = False
        self._backend: Any = None
        self._host_warned = False
        self._eval_cache: Optional[Dict[str, dict]] = None
        self._train_rows_by_id: Dict[str, dict] = {}

        self._write_manifest()

    # ------------------------------------------------------------ skill bank

    @staticmethod
    def _repo_skills_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "skills"

    def _frozen_skills_dir(self) -> Path:
        """Snapshot the live skill bank into the run directory for read-only eval."""
        source = self._repo_skills_dir()
        snapshot = self.run_dir / "skills_snapshot"
        if snapshot.exists():
            shutil.rmtree(snapshot)
        if source.exists():
            shutil.copytree(source, snapshot)
        else:
            snapshot.mkdir(parents=True, exist_ok=True)
        logger.info("Eval stage: froze skill bank snapshot at %s", snapshot)
        return snapshot

    # ------------------------------------------------------------- datasets

    def _dataset_split(self, split: str) -> tuple[str, str]:
        """Map the CLI (dataset, split) to a (registry key, registry split)."""
        return self.dataset, split

    def load_rows(
        self,
        split: str,
        limit: Optional[int] = None,
        exclude_ids: Optional[set] = None,
    ) -> List[dict]:
        key, registry_split = self._dataset_split(split)
        return list(datasets.iter_rows(key, registry_split, limit=limit, exclude_ids=exclude_ids))

    def load_instances(self, split: str = "test") -> List[Issue]:
        return [datasets.row_to_issue(row, use_hints=self.experiment.hints) for row in self.load_rows(split)]

    def setup_instance_environment(self, issue: Issue) -> ProjectEnvironment:
        """Clone (or reset) the repo and check out the base commit."""
        repo_dir = self.workspace_dir / issue.id.replace("/", "_")
        repo_url = f"https://github.com/{issue.repo_name}.git"
        if not repo_dir.exists() or not any(repo_dir.iterdir()):
            self.logger.info("Cloning %s into %s", issue.repo_name, repo_dir)
            repo_dir.mkdir(parents=True, exist_ok=True)
            env = ProjectEnvironment(repo_dir)
            if not env.clone_repo(repo_url):
                raise RuntimeError(f"Failed to clone repository: {repo_url}")
        else:
            env = ProjectEnvironment(repo_dir)
            env.revert_changes()  # drop a prior instance's edits before re-checkout
        if issue.base_commit and not env.checkout_commit(issue.base_commit):
            raise RuntimeError(f"Failed to checkout commit: {issue.base_commit}")
        return env

    # ------------------------------------------------------------- backends

    def _container_backend(self) -> Any:
        """Resolve the container test backend once (None for the host backend)."""
        if self._backend_attempted:
            return self._backend
        self._backend_attempted = True
        if self.experiment.test_backend == "host":
            return None
        self._backend = resolve_backend(self.experiment.test_backend)
        return self._backend

    def _grading_backend(self, env: Optional[ProjectEnvironment]) -> Any:
        """In-loop grading backend for a rollout (None => host approximation)."""
        if self.experiment.test_backend == "host":
            if not self._host_warned:
                self.logger.warning(
                    "Host test backend: benchmark grading is approximate and lacks "
                    "official PASS_TO_PASS regression checks."
                )
                self._host_warned = True
            return None
        backend = self._container_backend()
        return _ImageBaseAdapter(backend) if backend else None

    def _eval_backend_for(self, issue: Issue) -> Any:
        """Grading backend for the evaluate phase (host builds an env per instance)."""
        if self.experiment.test_backend == "host":
            env = self.setup_instance_environment(issue)
            env.setup_issue(issue)
            return resolve_backend("host", env=env)
        backend = self._container_backend()
        return _ImageBaseAdapter(backend) if backend else None

    # ---------------------------------------------------------- controller

    def _build_controller_signal(
        self,
        issue: Issue,
        forced_skills: Optional[List[SkillSignal]] = None,
    ) -> Optional[ControllerSignal]:
        if forced_skills is not None:
            return ControllerSignal(
                mode=self.experiment.stage,
                skills=list(forced_skills),
                selected_skill_ids=[s.id for s in forced_skills if s.id],
                source="skill-validation",
            ).enforce_mode()

        selected: List[SkillSignal] = []
        if self._selector is not None:
            hard_cases = self._retrieve_hard_cases(issue)
            query = " ".join(str(c.get("failure_type", "")) for c in hard_cases) or issue.description
            selected = [
                skill.to_skill_signal()
                for skill in self._selector.select_many(memory_query=query, limit=2)
            ]

        if self._controller_client is not None:
            hard_cases = self._retrieve_hard_cases(issue)
            return self._controller_client.generate(
                issue, stage=self.experiment.stage, skills=selected, hard_cases=hard_cases
            )
        if selected:
            return ControllerSignal(
                mode=self.experiment.stage,
                skills=selected,
                selected_skill_ids=[s.id for s in selected if s.id],
                source="skill-selection",
            ).enforce_mode()
        return None

    def _retrieve_hard_cases(self, issue: Issue) -> List[dict]:
        if self.experiment.memory == "off" or not self._hard_cases_path.exists():
            return []
        retriever = MemoryRetriever(self._hard_cases_path)
        return [record.to_dict() for record in retriever.retrieve(repo_name=issue.repo_name, limit=3)]

    @staticmethod
    def _max_iterations(signal: Optional[ControllerSignal]) -> int:
        if signal and signal.budget:
            return signal.budget
        return get_config().agent.max_iterations

    # -------------------------------------------------------------- episode

    def _run_episode(
        self,
        issue: Issue,
        forced_skills: Optional[List[SkillSignal]] = None,
    ) -> EpisodeResult:
        env = self.setup_instance_environment(issue)
        test_cmd = build_targeted_test_cmd(issue)
        if test_cmd:
            env.test_cmd = test_cmd
        backend = self._grading_backend(env)
        signal = self._build_controller_signal(issue, forced_skills=forced_skills)

        if self.experiment.agent_mode == "single":
            return self._single_episode(issue, env, backend, signal)

        rollout = run_online_rollout(
            issue,
            env=env,
            controller_signal=signal,
            max_iterations=self._max_iterations(signal),
            reward_model=self.reward_model,
            rollout_writer=self.rollout_writer,
            skill_evolver=self._evolver if self._skill_evolve else None,
            judge=self.judge,
            test_backend=backend,
            eval_timeout=default_eval_timeout(),
            stage=self.experiment.stage,
            seed=self.experiment.seed,
            experiment=self.experiment.label or None,
        )
        eval_outcome = rollout.eval_outcome
        resolved = eval_outcome.resolved if eval_outcome else rollout.execution_result.success
        self._cache_eval(issue.id, eval_outcome)
        return EpisodeResult(
            instance_id=issue.id,
            patch=model_patch_from_execution(rollout.execution_result),
            resolved=resolved,
            utility=rollout.reward.total,
            execution_result=rollout.execution_result,
            eval_outcome=eval_outcome,
        )

    def _single_episode(
        self,
        issue: Issue,
        env: ProjectEnvironment,
        backend: Any,
        signal: Optional[ControllerSignal],
    ) -> EpisodeResult:
        try:
            from src.workers.single_agent import run_single_agent
        except ImportError as exc:
            raise RuntimeError(
                "agent-mode 'single' requires src/workers/single_agent.py "
                f"(run_single_agent(issue, env) -> ExecutionResult); not available: {exc}"
            ) from exc

        exec_result = run_single_agent(issue, env)
        patch = model_patch_from_execution(exec_result)
        eval_outcome = None
        if backend is not None and patch.strip():
            eval_outcome = backend.run_swebench_eval(
                swebench_instance_from_issue(issue), patch, timeout=default_eval_timeout()
            )
        evaluation = self.judge.evaluate(exec_result)
        reward = self.reward_model.score(exec_result, eval_outcome=eval_outcome, issue=issue)
        record = build_rollout_record(
            issue,
            signal,
            exec_result,
            evaluation=evaluation,
            reward=reward,
            eval_outcome=eval_outcome,
            stage=self.experiment.stage,
            seed=self.experiment.seed,
        )
        self.rollout_writer.append(record)
        self._cache_eval(issue.id, eval_outcome)
        resolved = eval_outcome.resolved if eval_outcome else exec_result.success
        return EpisodeResult(
            instance_id=issue.id,
            patch=patch,
            resolved=resolved,
            utility=reward.total,
            execution_result=exec_result,
            eval_outcome=eval_outcome,
        )

    # ------------------------------------------------------------ reflection

    def _run_reflection(self, pool: Optional[TaskPool]) -> None:
        if self._skill_evolve:
            validator = self._validate_skills if self.experiment.validate_skills > 0 else None
            result = Reflector(
                skill_bank=self._bank,
                skill_evolver=self._evolver,
                buffer_path=self._hard_cases_path,
            ).reflect(stage="train", validator=validator)
            signals = result.task_signals
        else:
            signals = self._task_signals_from_memory()
        if pool is not None and signals:
            pool.apply_reflection(signals)

    def _task_signals_from_memory(self) -> Dict[str, Any]:
        """Instance boosts from hard-case clusters (task evolution without skill writes)."""
        from src.reflection.clustering import cluster_records, qualifying_clusters

        records = MemoryRetriever(self._hard_cases_path).retrieve(stage="train", limit=200)
        clusters = qualifying_clusters(cluster_records(records))
        boosts: Dict[str, int] = defaultdict(int)
        for cluster in clusters:
            for instance_id in cluster.instance_ids:
                boosts[instance_id] += 1
        return {"instance_boosts": dict(boosts), "family_boosts": {}}

    def _validate_skills(self, proposals, clusters):
        """Replay held-out cluster instances with the candidates forced in; gate on mean utility."""
        forced = [
            SkillSignal(
                id=proposal.skill_id,
                title=proposal.title,
                summary=proposal.summary,
                target_failure_type=proposal.target_failure_type,
            )
            for proposal in proposals
        ]
        picks: List[str] = []
        seen: set = set()
        for cluster in clusters:
            for instance_id in cluster.instance_ids:
                if instance_id in seen:
                    continue
                seen.add(instance_id)
                picks.append(instance_id)
                if len(picks) >= self.experiment.validate_skills:
                    break
            if len(picks) >= self.experiment.validate_skills:
                break

        utilities: List[float] = []
        for instance_id in picks:
            row = self._train_rows_by_id.get(base_instance_id({"instance_id": instance_id}))
            if row is None:
                continue
            issue = datasets.row_to_issue(row, use_hints=self.experiment.hints)
            utilities.append(self._run_episode(issue, forced_skills=forced).utility)
        mean_utility = sum(utilities) / len(utilities) if utilities else None
        return proposals, mean_utility

    # ------------------------------------------------------------- generate

    def generate_predictions(
        self,
        num_instances: Optional[int] = None,
        split: str = "test",
        predictions_path: Optional[Path] = None,
        resume: bool = True,
    ) -> Path:
        predictions_path = predictions_path or self.run_dir / "predictions.json"
        if self.experiment.stage == "train" and self.experiment.task_evolution == "on":
            self._generate_train(num_instances, split, predictions_path)
        else:
            self._generate_direct(num_instances, split, predictions_path, resume)
        self._write_final_summary_from_cache(predictions_path)
        return predictions_path

    def _generate_train(
        self,
        num_rollouts: Optional[int],
        split: str,
        predictions_path: Path,
    ) -> None:
        rows = self.load_rows(split)
        self._train_rows_by_id = {row["instance_id"]: row for row in rows}
        backend = self._container_backend()
        verifier = (lambda inst: verify_task(inst, backend)) if backend is not None else None
        if verifier is None:
            self.logger.warning("No container backend: focused task variants will not be verified.")
        pool = TaskPool.from_instances(rows, self.run_dir / "task_pool.json", verifier=verifier)
        rng = random.Random(self.experiment.seed)
        reflect_every = max(1, SkillEvolutionConfig.load().reflect_every_n_rollouts)
        budget = num_rollouts or len(pool)
        predictions = self._load_predictions(predictions_path)

        self.logger.info("Train stage: %d rollouts over a pool of %d instances", budget, len(pool))
        for step in range(budget):
            sampled = pool.sample(1, rng)
            if not sampled:
                break
            instance = sampled[0]
            issue = datasets.row_to_issue(instance, use_hints=self.experiment.hints)
            try:
                episode = self._run_episode(issue)
            except Exception as exc:  # keep the loop alive; log and continue
                self.logger.error("Rollout failed for %s: %s", issue.id, exc)
                continue
            pool.record_outcome(instance["instance_id"], episode.resolved, episode.utility)
            predictions[issue.id] = self._prediction_dict(issue.id, episode.patch)
            self._save_predictions(predictions_path, predictions)
            if self.experiment.task_evolution == "on" and (step + 1) % reflect_every == 0:
                self._run_reflection(pool)
            pool.save()

        self._run_reflection(pool)  # end-of-iteration reflection
        pool.save()

    def _generate_direct(
        self,
        num_instances: Optional[int],
        split: str,
        predictions_path: Path,
        resume: bool,
    ) -> None:
        exclude = self._eval_contamination_guard()
        rows = self.load_rows(split, limit=num_instances, exclude_ids=exclude)
        predictions = self._load_predictions(predictions_path) if resume else {}
        for row in rows:
            issue = datasets.row_to_issue(row, use_hints=self.experiment.hints)
            if resume and predictions.get(issue.id, {}).get("model_patch", "").strip():
                continue
            try:
                episode = self._run_episode(issue)
                patch = episode.patch
            except Exception as exc:
                self.logger.error("Prediction failed for %s: %s", issue.id, exc)
                patch = ""
            predictions[issue.id] = self._prediction_dict(issue.id, patch)
            self._save_predictions(predictions_path, predictions)

    def _eval_contamination_guard(self) -> set:
        """Refuse eval on ids listed in the training-id file (contamination control)."""
        if not self._train_ids_path:
            return set()
        train_ids = read_id_file(self._train_ids_path)
        self.logger.info("Excluding %d training ids from the eval set", len(train_ids))
        return train_ids

    # ------------------------------------------------------------- evaluate

    def evaluate_predictions(
        self,
        predictions_path: Path,
        split: str = "test",
        run_id: Optional[str] = None,
        max_workers: int = 2,
        cleanup_images: bool = True,
        official_harness: Optional[bool] = None,
    ) -> dict:
        """Grade a predictions file and summarize resolved / unresolved instances.

        The official swebench Docker harness is used when a docker engine is
        available (canonical leaderboard numbers); otherwise each prediction is
        graded per-instance with the same swebench grading code via apptainer or
        the host backend.
        """
        predictions = self._load_predictions(predictions_path)
        if official_harness is None:
            official_harness = (
                self.experiment.test_backend in ("auto", "docker")
                and shutil.which("docker") is not None
            )
        if official_harness:
            summary = self._evaluate_official_docker(
                predictions, predictions_path, split, run_id or self.run_id,
                max_workers, cleanup_images,
            )
        else:
            summary = self._evaluate_with_backend(predictions, split)
        (self.run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _evaluate_with_backend(self, predictions: Dict[str, dict], split: str) -> dict:
        key, registry_split = self._dataset_split(split)
        id_to_row = {row["instance_id"]: row for row in datasets.iter_rows(key, registry_split)}
        cache = self._load_eval_cache()
        resolved: List[str] = []
        unresolved: List[str] = []
        totals = {"f2p_passed": 0, "f2p_total": 0, "p2p_passed": 0, "p2p_total": 0}

        for instance_id, prediction in predictions.items():
            patch = prediction.get("model_patch", "").strip()
            row = id_to_row.get(instance_id)
            if not patch or row is None:
                unresolved.append(instance_id)
                continue
            outcome = cache.get(instance_id)
            if outcome is None:
                issue = datasets.row_to_issue(row, use_hints=self.experiment.hints)
                backend = self._eval_backend_for(issue)
                if backend is None:
                    unresolved.append(instance_id)
                    continue
                outcome = _eval_outcome_dict(
                    backend.run_swebench_eval(
                        swebench_instance_from_issue(issue), patch, timeout=default_eval_timeout()
                    )
                )
                cache[instance_id] = outcome
            (resolved if outcome["resolved"] else unresolved).append(instance_id)
            for field_name in totals:
                totals[field_name] += outcome[field_name]
        self._save_eval_cache(cache)

        total = len(resolved) + len(unresolved)
        return {
            "backend": self.experiment.test_backend,
            "resolved": resolved,
            "unresolved": unresolved,
            "resolved_count": len(resolved),
            "unresolved_count": len(unresolved),
            "total_evaluated": total,
            "resolve_rate": len(resolved) / max(1, total),
            **totals,
        }

    def _evaluate_official_docker(
        self,
        predictions: Dict[str, dict],
        predictions_path: Path,
        split: str,
        run_id: str,
        max_workers: int,
        cleanup_images: bool,
    ) -> dict:
        try:
            from swebench.harness.run_evaluation import main as swebench_eval
        except ImportError as exc:
            raise ImportError("swebench package required. Install with: pip install swebench") from exc

        key, registry_split = self._dataset_split(split)
        dataset_name = datasets.get_spec(key).hf_name
        repo_groups: Dict[str, List[str]] = defaultdict(list)
        for instance_id, prediction in predictions.items():
            if prediction.get("model_patch", "").strip():
                repo_groups[instance_id.rsplit("__", 1)[0]].append(instance_id)

        for repo, instance_ids in sorted(repo_groups.items(), key=lambda item: -len(item[1])):
            self.logger.info("Evaluating %s: %d instances", repo, len(instance_ids))
            try:
                swebench_eval(
                    dataset_name=dataset_name,
                    split=registry_split,
                    instance_ids=instance_ids,
                    predictions_path=str(predictions_path),
                    max_workers=max_workers,
                    force_rebuild=False,
                    cache_level="env",
                    clean=False,
                    open_file_limit=4096,
                    run_id=run_id,
                    timeout=get_config().docker.timeout,
                    namespace=None,
                    rewrite_reports=False,
                    modal=False,
                    report_dir=str(self.run_dir),
                )
            except Exception as exc:
                self.logger.error("Official evaluation failed for %s: %s", repo, exc)
            finally:
                if cleanup_images:
                    self.cleanup_docker_images(except_base=True)
        return self.summarize_official_results(run_id=run_id)

    def summarize_official_results(self, run_id: Optional[str] = None) -> dict:
        """Read the official harness report tree and separate infra from patch errors."""
        run_id = run_id or self.run_id
        candidates = [Path("logs/run_evaluation") / run_id / self.model_name]
        try:
            from swebench.harness.constants import RUN_EVALUATION_LOG_DIR

            candidates.insert(0, RUN_EVALUATION_LOG_DIR / run_id / self.model_name)
        except Exception:
            pass
        eval_dir = next((path for path in candidates if path.exists()), candidates[-1])

        resolved: List[str] = []
        unresolved: List[str] = []
        infra_errors: List[str] = []
        patch_errors: List[str] = []
        other_errors: List[str] = []
        if eval_dir.exists():
            for instance_dir in sorted(eval_dir.iterdir()):
                if not instance_dir.is_dir():
                    continue
                self._classify_official_instance(
                    instance_dir, resolved, unresolved, infra_errors, patch_errors, other_errors
                )

        total = len(resolved) + len(unresolved) + len(infra_errors) + len(patch_errors) + len(other_errors)
        return {
            "backend": "official-docker-harness",
            "resolved": resolved,
            "unresolved": unresolved,
            "infra_errors": infra_errors,
            "patch_errors": patch_errors,
            "other_errors": other_errors,
            "total_evaluated": total,
            "resolved_count": len(resolved),
            "unresolved_count": len(unresolved),
            "infra_error_count": len(infra_errors),
            "patch_error_count": len(patch_errors),
            "other_error_count": len(other_errors),
            "resolve_rate": len(resolved) / max(1, total),
            "resolve_rate_excluding_infra": len(resolved) / max(1, total - len(infra_errors)),
        }

    @staticmethod
    def _classify_official_instance(instance_dir, resolved, unresolved, infra, patch, other) -> None:
        instance_id = instance_dir.name
        report_file = instance_dir / "report.json"
        if report_file.exists():
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
                (resolved if data.get(instance_id, {}).get("resolved") else unresolved).append(instance_id)
                return
            except Exception:
                pass
        log_file = instance_dir / "run_instance.log"
        content = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
        lower = content.lower()
        if "toomanyrequests" in lower or "rate limit" in lower:
            infra.append(instance_id)
        elif "malformed patch" in lower or "corrupt patch" in lower or "patch does not apply" in lower:
            patch.append(instance_id)
        elif content:
            other.append(instance_id)
        else:
            unresolved.append(instance_id)

    # -------------------------------------------------------------- phased

    def run_phased_benchmark(
        self,
        phase: str = "generate",
        num_instances: Optional[int] = None,
        split: str = "test",
        predictions_path: Optional[Path] = None,
        run_id: Optional[str] = None,
        eval_workers: int = 2,
        resume: bool = True,
        cleanup_images: bool = True,
    ) -> dict:
        predictions_path = predictions_path or self.run_dir / "predictions.json"
        result: dict = {"predictions_path": str(predictions_path), "stage": self.experiment.stage}

        if phase in {"generate", "both"}:
            self.generate_predictions(
                num_instances=num_instances,
                split=split,
                predictions_path=predictions_path,
                resume=resume,
            )
            result["predictions"] = self.summarize_predictions(predictions_path)

        if phase in {"evaluate", "both"}:
            result.setdefault("predictions", self.summarize_predictions(predictions_path))
            result["evaluation"] = self.evaluate_predictions(
                predictions_path=predictions_path,
                split=split,
                run_id=run_id,
                max_workers=eval_workers,
                cleanup_images=cleanup_images,
            )
        return result

    def summarize_predictions(self, predictions_path: Path) -> dict:
        predictions = self._load_predictions(predictions_path)
        total = len(predictions)
        empty = sum(1 for p in predictions.values() if not p.get("model_patch", "").strip())
        return {"total": total, "non_empty": total - empty, "empty": empty}

    # ------------------------------------------------------------- eval cache

    def _cache_eval(self, instance_id: str, eval_outcome: Any) -> None:
        if eval_outcome is None:
            return
        cache = self._load_eval_cache()
        cache[instance_id] = _eval_outcome_dict(eval_outcome)
        self._save_eval_cache(cache)

    def _load_eval_cache(self) -> Dict[str, dict]:
        if self._eval_cache is None:
            path = self.run_dir / "eval_outcomes.json"
            self._eval_cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return self._eval_cache

    def _save_eval_cache(self, cache: Dict[str, dict]) -> None:
        self._eval_cache = cache
        (self.run_dir / "eval_outcomes.json").write_text(json.dumps(cache, indent=2), encoding="utf-8")

    def _write_final_summary_from_cache(self, predictions_path: Path) -> None:
        """Write resolved/unresolved from in-loop outcomes so generate alone yields numbers."""
        cache = self._load_eval_cache()
        if not cache:
            return
        predictions = self._load_predictions(predictions_path)
        resolved = [iid for iid in predictions if cache.get(iid, {}).get("resolved")]
        graded = [iid for iid in predictions if iid in cache]
        summary = {
            "backend": self.experiment.test_backend,
            "source": "in-loop",
            "resolved": resolved,
            "resolved_count": len(resolved),
            "graded_count": len(graded),
            "prediction_count": len(predictions),
            "resolve_rate": len(resolved) / max(1, len(graded)),
        }
        (self.run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- helpers

    def _prediction_dict(self, instance_id: str, patch: str) -> dict:
        return {
            "instance_id": instance_id,
            "model_name_or_path": self.model_name,
            "model_patch": patch,
        }

    @staticmethod
    def _load_predictions(predictions_path: Path) -> Dict[str, dict]:
        if not predictions_path.exists():
            return {}
        with predictions_path.open(encoding="utf-8") as f:
            return {item["instance_id"]: item for item in json.load(f)}

    @staticmethod
    def _save_predictions(predictions_path: Path, predictions: Dict[str, dict]) -> None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as f:
            json.dump(list(predictions.values()), f, indent=2)

    def _write_manifest(self) -> None:
        manifest = {
            "benchmark": self.name,
            "dataset": self.dataset,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "experiment": self.experiment.to_dict(),
        }
        (self.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @staticmethod
    def cleanup_docker_images(except_base: bool = True) -> None:
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True, text=True,
            )
            targets = [
                line for line in result.stdout.strip().splitlines()
                if "sweb.eval." in line or "sweb.env." in line
                or (not except_base and "sweb.base." in line)
            ]
            if targets:
                subprocess.run(["docker", "rmi", "-f", *targets], capture_output=True, text=True)
            subprocess.run(["docker", "image", "prune", "-f"], capture_output=True, text=True)
        except Exception as exc:
            logger.warning("Docker cleanup error: %s", exc)


def _eval_outcome_dict(eval_outcome: Any) -> dict:
    getter = (
        eval_outcome.get if isinstance(eval_outcome, dict)
        else lambda key, default=0: getattr(eval_outcome, key, default)
    )
    return {
        "f2p_passed": int(getter("f2p_passed") or 0),
        "f2p_total": int(getter("f2p_total") or 0),
        "p2p_passed": int(getter("p2p_passed") or 0),
        "p2p_total": int(getter("p2p_total") or 0),
        "resolved": bool(getter("resolved") or False),
    }


def read_id_file(path: Path) -> set:
    """Read instance ids from a file: one id per line (# comments) or a JSON list."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return {str(item) for item in json.loads(text)}
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def create_swebench_runner(
    dataset: str = "lite",
    output_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    model_name: str = "self-evolver",
    run_id: str = "self-evolver",
    experiment: Optional[ExperimentConfig] = None,
    reward_config: Optional[Path] = None,
    train_ids_path: Optional[Path] = None,
) -> SWEBenchRunner:
    """Create the runner matching ``dataset`` (lite/verified/full -> SWE-bench)."""
    return SWEBenchRunner(
        dataset=dataset,
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        model_name=model_name,
        run_id=run_id,
        experiment=experiment,
        reward_config=reward_config,
        train_ids_path=train_ids_path,
    )
