"""
SWE-bench Runner - Integration with SWE-bench benchmark.

This module provides integration with SWE-bench Lite/Verified datasets.
Requires Docker and the swebench package for full functionality.

Note: Full implementation requires server with Docker support.
"""

import json
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from src.benchmark.base import BenchmarkRunner, InstanceResult
from src.config import get_config
from src.controller.controller_client import ControllerClient
from src.controller.schema import ControllerSignal
from src.critic.judge import CriticJudge
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.memory.memory_retriever import MemoryRetriever
from src.reward.reward_model import RewardModel
from src.rl.online_rollout_runner import build_targeted_test_cmd, run_online_rollout
from src.rl.rollout_writer import RolloutWriter
from src.skills.skill_evolver import SkillEvolver
from src.skills.skill_selector import SkillSelector

logger = logging.getLogger(__name__)


class SWEBenchRunner(BenchmarkRunner):
    """
    SWE-bench benchmark runner.
    
    Supports SWE-bench Lite (300 instances) and SWE-bench Verified.
    Requires Docker for isolated execution environments.
    """
    
    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        output_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
        model_name: str = "self-evolver",
        run_id: str = "self-evolver",
        controller_mode: str = "off",
        controller_stage: str = "eval",
        rollout_jsonl: Optional[Path] = None,
        reward_config: Optional[Path] = None,
    ):
        """
        Initialize SWE-bench runner.
        
        Args:
            dataset_name: HuggingFace dataset name.
            output_dir: Directory for results.
            workspace_dir: Directory for cloning repos.
        """
        super().__init__(name="swebench", output_dir=output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.workspace_dir = workspace_dir or Path(tempfile.mkdtemp(prefix="swebench_"))
        self.model_name = model_name
        self.run_id = run_id
        self.judge = CriticJudge()
        self._dataset = None
        self.controller_mode = controller_mode
        self.controller_stage = controller_stage
        self.controller_client = (
            ControllerClient(mode=controller_mode)
            if controller_mode != "off"
            else None
        )
        self.skill_selector = SkillSelector()
        self.skill_evolver = SkillEvolver() if controller_mode != "off" else None
        self.reward_model = RewardModel.from_config_file(reward_config)
        if rollout_jsonl:
            self.rollout_writer = RolloutWriter(rollout_jsonl)
        elif controller_mode != "off":
            self.rollout_writer = RolloutWriter(self.output_dir / "rollouts.jsonl")
        else:
            self.rollout_writer = None
    
    def _load_dataset(self, split: str = "test"):
        """Load the SWE-bench dataset from HuggingFace."""
        if self._dataset is not None:
            return self._dataset
        
        try:
            from datasets import load_dataset
            self._dataset = load_dataset(self.dataset_name, split=split)
            self.logger.info(f"Loaded {len(self._dataset)} instances from {self.dataset_name}")
            return self._dataset
        except ImportError:
            raise ImportError(
                "datasets package required. Install with: pip install datasets"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def load_instances(self, split: str = "test") -> List[Issue]:
        """Load SWE-bench instances as Issue objects."""
        dataset = self._load_dataset(split)
        
        issues = []
        for item in dataset:
            issue = Issue(
                id=item["instance_id"],
                description=item["problem_statement"],
                repo_name=item["repo"],
                base_commit=item["base_commit"],
                hints=item.get("hints_text"),
                test_patch=item.get("test_patch"),
                metadata={
                    "version": item.get("version"),
                    "fail_to_pass": item.get("FAIL_TO_PASS"),
                    "pass_to_pass": item.get("PASS_TO_PASS"),
                },
            )
            issues.append(issue)
        
        return issues
    
    def setup_instance_environment(self, issue: Issue) -> ProjectEnvironment:
        """
        Set up the execution environment for an instance.
        
        This involves:
        1. Cloning the repository (if not already cloned)
        2. Checking out the base commit
        3. Setting up any required dependencies
        
        Note: Full Docker-based setup should be implemented for production.
        """
        repo_dir = self.workspace_dir / issue.id.replace("/", "_")
        repo_url = f"https://github.com/{issue.repo_name}.git"

        if not repo_dir.exists() or not any(repo_dir.iterdir()):
            self.logger.info(f"Cloning {issue.repo_name} into {repo_dir}...")
            repo_dir.mkdir(parents=True, exist_ok=True)
            env = ProjectEnvironment(repo_dir)
            if not env.clone_repo(repo_url):
                raise RuntimeError(f"Failed to clone repository: {repo_url}")
        else:
            self.logger.info(f"Repository already exists at {repo_dir}, reusing.")
            env = ProjectEnvironment(repo_dir)

        # Checkout base commit
        if issue.base_commit:
            if not env.checkout_commit(issue.base_commit):
                raise RuntimeError(f"Failed to checkout commit: {issue.base_commit}")

        return env
    
    @staticmethod
    def _build_test_cmd(issue: Issue) -> Optional[str]:
        """
        Build a targeted pytest command from the FAIL_TO_PASS test list.

        Running only the specific tests that must change from FAIL to PASS
        is faster and matches SWE-bench evaluation semantics.

        Returns None if no fail_to_pass data is available.
        """
        return build_targeted_test_cmd(issue)

    def run_instance(self, issue: Issue) -> InstanceResult:
        """Run a single SWE-bench instance."""
        self.logger.info(f"Running instance: {issue.id}")

        try:
            # Build a targeted test command from FAIL_TO_PASS
            test_cmd = self._build_test_cmd(issue)
            if test_cmd:
                self.logger.info(f"Using targeted test command: {test_cmd}")
            else:
                self.logger.info("No FAIL_TO_PASS found, using default test command")

            # Set up environment (with targeted test_cmd if available)
            env = self.setup_instance_environment(issue)
            if test_cmd:
                env.test_cmd = test_cmd

            controller_signal = self._build_controller_signal(issue)
            rollout = run_online_rollout(
                issue,
                env=env,
                controller_signal=controller_signal,
                max_iterations=get_config().agent.max_iterations,
                reward_model=self.reward_model,
                rollout_writer=self.rollout_writer,
                skill_evolver=self.skill_evolver,
                judge=self.judge,
            )
            result = rollout.execution_result
            evaluation = rollout.evaluation

            return InstanceResult(
                instance_id=issue.id,
                success=result.success,
                execution_result=result,
                evaluation=evaluation,
            )

        except Exception as e:
            self.logger.error(f"Instance {issue.id} failed: {e}")
            return InstanceResult(
                instance_id=issue.id,
                success=False,
                error=str(e),
            )

    @staticmethod
    def _prediction_patch_from_execution(result) -> str:
        """
        Extract the best canonical patch from an execution result.

        The official SWE-bench prediction should use git-generated diffs from
        the verifier, never the LLM's raw diff text.
        """
        if result.final_patch and result.final_patch.content.strip():
            return result.final_patch.content

        for record in reversed(result.iteration_records):
            verification = record.verification_result
            if verification and verification.canonical_patch_content.strip():
                return verification.canonical_patch_content
        return ""

    @staticmethod
    def _load_predictions(predictions_path: Path) -> Dict[str, dict]:
        if not predictions_path.exists():
            return {}
        with predictions_path.open(encoding="utf-8") as f:
            return {
                item["instance_id"]: item
                for item in json.load(f)
            }

    @staticmethod
    def _save_predictions(predictions_path: Path, predictions: Dict[str, dict]) -> None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as f:
            json.dump(list(predictions.values()), f, indent=2)

    def generate_prediction_for_issue(
        self,
        issue: Issue,
        model_name: Optional[str] = None,
        cleanup_repo: bool = True,
    ) -> dict:
        """Generate a canonical SWE-bench prediction for one issue."""
        model_name = model_name or self.model_name
        patch = ""
        repo_dir = self.workspace_dir / issue.id.replace("/", "_")

        try:
            test_cmd = self._build_test_cmd(issue)
            env = self.setup_instance_environment(issue)
            if test_cmd:
                env.test_cmd = test_cmd

            controller_signal = self._build_controller_signal(issue)
            rollout = run_online_rollout(
                issue,
                env=env,
                controller_signal=controller_signal,
                max_iterations=get_config().agent.max_iterations,
                reward_model=self.reward_model,
                rollout_writer=self.rollout_writer,
                skill_evolver=self.skill_evolver,
                judge=self.judge,
            )
            result = rollout.execution_result
            patch = self._prediction_patch_from_execution(result)
            self.logger.info(
                f"Generated prediction for {issue.id}: "
                f"{len(patch)} chars, success={result.success}"
            )
        except Exception as e:
            self.logger.error(f"Failed to generate prediction for {issue.id}: {e}")
        finally:
            if cleanup_repo:
                self._cleanup_path(repo_dir)

        return {
            "instance_id": issue.id,
            "model_name_or_path": model_name,
            "model_patch": patch,
        }

    def _build_controller_signal(self, issue: Issue) -> Optional[ControllerSignal]:
        """Generate optional controller guidance for an issue."""
        if not self.controller_client:
            return None
        hard_cases = self._retrieve_hard_cases(issue)
        memory_query = " ".join(
            str(item.get("failure_type", "")) for item in hard_cases
        )
        selected_skills = self.skill_selector.select_many(
            memory_query=memory_query or issue.description,
            limit=2,
        )
        return self.controller_client.generate(
            issue,
            stage=self.controller_stage,
            skills=selected_skills,
            hard_cases=hard_cases,
        )

    def _retrieve_hard_cases(self, issue: Issue) -> List[dict]:
        """Fetch similar hard cases if a buffer exists."""
        path = get_config().environment.workspace_dir / "hard_cases.jsonl"
        if not path.exists():
            return []
        retriever = MemoryRetriever(path)
        return [
            record.to_dict()
            for record in retriever.retrieve(repo_name=issue.repo_name, limit=3)
        ]

    def generate_predictions(
        self,
        num_instances: Optional[int] = None,
        split: str = "test",
        predictions_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        max_workers: int = 1,
        resume: bool = True,
        cleanup_repo: bool = True,
    ) -> Path:
        """
        Generate or resume SWE-bench predictions.

        Existing non-empty predictions are preserved when resume=True.
        """
        predictions_path = predictions_path or self.output_dir / "predictions.json"
        model_name = model_name or self.model_name

        issues = self.load_instances(split)
        if num_instances:
            issues = issues[:num_instances]

        predictions = self._load_predictions(predictions_path) if resume else {}
        todo = [
            issue for issue in issues
            if not resume
            or not predictions.get(issue.id, {}).get("model_patch", "").strip()
        ]
        self.logger.info(f"Generating predictions: {len(todo)} to run, {len(predictions)} cached")

        if max_workers <= 1:
            for issue in todo:
                predictions[issue.id] = self.generate_prediction_for_issue(
                    issue,
                    model_name=model_name,
                    cleanup_repo=cleanup_repo,
                )
                self._save_predictions(predictions_path, predictions)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        self.generate_prediction_for_issue,
                        issue,
                        model_name,
                        cleanup_repo,
                    ): issue.id
                    for issue in todo
                }
                for future in as_completed(futures):
                    instance_id = futures[future]
                    try:
                        predictions[instance_id] = future.result()
                    except Exception as e:
                        self.logger.error(f"Prediction worker failed for {instance_id}: {e}")
                        predictions[instance_id] = {
                            "instance_id": instance_id,
                            "model_name_or_path": model_name,
                            "model_patch": "",
                        }
                    self._save_predictions(predictions_path, predictions)

        return predictions_path

    def evaluate_predictions(
        self,
        predictions_path: Path,
        split: str = "test",
        run_id: Optional[str] = None,
        max_workers: int = 2,
        cleanup_images: bool = True,
    ) -> dict:
        """Evaluate predictions with the official SWE-bench harness in repo batches."""
        try:
            from swebench.harness.run_evaluation import main as swebench_eval
        except ImportError as e:
            raise ImportError("swebench package required. Install with: pip install swebench") from e

        run_id = run_id or self.run_id
        predictions = self._load_predictions(predictions_path)
        repo_groups: dict[str, list[str]] = defaultdict(list)
        for instance_id in predictions:
            repo = instance_id.rsplit("__", 1)[0]
            repo_groups[repo].append(instance_id)

        for repo, instance_ids in sorted(repo_groups.items(), key=lambda item: -len(item[1])):
            non_empty_ids = [
                instance_id for instance_id in instance_ids
                if predictions.get(instance_id, {}).get("model_patch", "").strip()
            ]
            if not non_empty_ids:
                self.logger.info(f"Skipping {repo}: no non-empty patches")
                continue

            self.logger.info(f"Evaluating {repo}: {len(non_empty_ids)} instances")
            try:
                swebench_eval(
                    dataset_name=self.dataset_name,
                    split=split,
                    instance_ids=non_empty_ids,
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
                    report_dir=str(self.output_dir),
                )
            except Exception as e:
                self.logger.error(f"Evaluation failed for {repo}: {e}")
            finally:
                if cleanup_images:
                    self.cleanup_docker_images(except_base=True)

        summary = self.summarize_official_results(run_id=run_id)
        summary_path = self.output_dir / "final_summary.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def run_phased_benchmark(
        self,
        phase: str = "generate",
        num_instances: Optional[int] = None,
        split: str = "test",
        predictions_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_workers: int = 1,
        eval_workers: int = 2,
        resume: bool = True,
        cleanup_images: bool = True,
        cleanup_repo: bool = True,
    ) -> dict:
        """Unified SWE-bench generation/evaluation entrypoint."""
        predictions_path = predictions_path or self.output_dir / "predictions.json"
        result: dict = {"predictions_path": str(predictions_path)}

        if phase in {"generate", "both"}:
            self.generate_predictions(
                num_instances=num_instances,
                split=split,
                predictions_path=predictions_path,
                model_name=model_name,
                max_workers=agent_workers,
                resume=resume,
                cleanup_repo=cleanup_repo,
            )
            result["predictions"] = self.summarize_predictions(predictions_path)

        if phase in {"evaluate", "both"}:
            if "predictions" not in result:
                result["predictions"] = self.summarize_predictions(predictions_path)
            result["evaluation"] = self.evaluate_predictions(
                predictions_path=predictions_path,
                split=split,
                run_id=run_id,
                max_workers=eval_workers,
                cleanup_images=cleanup_images,
            )

        return result

    def summarize_predictions(self, predictions_path: Path) -> dict:
        """Count empty/non-empty patches in a predictions file."""
        predictions = self._load_predictions(predictions_path)
        total = len(predictions)
        empty = sum(
            1 for prediction in predictions.values()
            if not prediction.get("model_patch", "").strip()
        )
        return {
            "total": total,
            "non_empty": total - empty,
            "empty": empty,
        }

    def summarize_official_results(self, run_id: Optional[str] = None) -> dict:
        """Summarize official SWE-bench logs and separate infra errors."""
        run_id = run_id or self.run_id
        candidates = [Path("logs/run_evaluation") / run_id / self.model_name]
        try:
            from swebench.harness.constants import RUN_EVALUATION_LOG_DIR
            candidates.insert(0, RUN_EVALUATION_LOG_DIR / run_id / self.model_name)
        except Exception:
            pass
        candidates.append(Path("/root/Self-Evolver/logs/run_evaluation") / run_id / self.model_name)
        eval_dir = next((path for path in candidates if path.exists()), candidates[0])

        resolved: list[str] = []
        unresolved: list[str] = []
        infra_errors: list[str] = []
        patch_errors: list[str] = []
        other_errors: list[str] = []

        if not eval_dir.exists():
            return {
                "resolved": resolved,
                "unresolved": unresolved,
                "infra_errors": infra_errors,
                "patch_errors": patch_errors,
                "other_errors": other_errors,
                "total_evaluated": 0,
            }

        for instance_dir in sorted(eval_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            instance_id = instance_dir.name
            report_file = instance_dir / "report.json"
            log_file = instance_dir / "run_instance.log"

            if report_file.exists():
                try:
                    data = json.loads(report_file.read_text(encoding="utf-8"))
                    if data.get(instance_id, {}).get("resolved", False):
                        resolved.append(instance_id)
                    else:
                        unresolved.append(instance_id)
                    continue
                except Exception:
                    pass

            content = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
            lower = content.lower()
            if "toomanyrequests" in lower or "rate limit" in lower:
                infra_errors.append(instance_id)
            elif "malformed patch" in lower or "hunk" in lower or "unexpected end of file" in lower:
                patch_errors.append(instance_id)
            elif content:
                other_errors.append(instance_id)
            else:
                unresolved.append(instance_id)

        total = len(resolved) + len(unresolved) + len(infra_errors) + len(patch_errors) + len(other_errors)
        return {
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
            "resolve_rate_excluding_infra": (
                len(resolved) / max(1, total - len(infra_errors))
            ),
        }

    @staticmethod
    def cleanup_docker_images(except_base: bool = True) -> None:
        """Remove SWE-bench env/eval images while optionally preserving base images."""
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
            )
            images_to_remove = []
            for line in result.stdout.strip().splitlines():
                if "sweb.eval." in line or "sweb.env." in line:
                    images_to_remove.append(line)
                elif not except_base and "sweb.base." in line:
                    images_to_remove.append(line)

            if images_to_remove:
                subprocess.run(
                    ["docker", "rmi", "-f", *images_to_remove],
                    capture_output=True,
                    text=True,
                )
            subprocess.run(["docker", "image", "prune", "-f"], capture_output=True, text=True)
        except Exception as e:
            logger.warning(f"Docker cleanup error: {e}")

    @staticmethod
    def _cleanup_path(path: Path) -> None:
        try:
            if path.exists():
                shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")
    
    def verify_with_swebench(self, instance_id: str, patch: str) -> bool:
        """
        Verify a patch using official SWE-bench evaluation harness.

        Requires the swebench package and a running Docker daemon.

        Args:
            instance_id: The SWE-bench instance ID.
            patch: The generated patch content (unified diff).

        Returns:
            True if the patch is fully resolved according to SWE-bench.
        """
        try:
            import json
            import os
            import tempfile
            from pathlib import Path

            from swebench import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
            from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR
            from swebench.harness.run_evaluation import main as swebench_eval
        except ImportError:
            self.logger.error(
                "swebench package not installed. "
                "Install with: pip install swebench"
            )
            return False

        model_name = self.model_name
        # run_id must be short and filesystem-safe
        safe_id = instance_id.replace("/", "__").replace(".", "_")[:40]
        run_id = f"se-verify-{safe_id}"

        predictions = [
            {
                KEY_INSTANCE_ID: instance_id,
                KEY_MODEL: model_name,
                KEY_PREDICTION: patch,
            }
        ]

        report_dir = tempfile.mkdtemp(prefix="swebench_report_")
        predictions_file = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                json.dump(predictions, f)
                predictions_file = f.name

            self.logger.info(
                f"Running official SWE-bench evaluation for {instance_id} "
                f"(run_id={run_id})..."
            )

            swebench_eval(
                dataset_name=self.dataset_name,
                split="test",
                instance_ids=[instance_id],
                predictions_path=predictions_file,
                max_workers=1,
                force_rebuild=False,
                cache_level="env",
                clean=False,
                open_file_limit=4096,
                run_id=run_id,
                timeout=get_config().docker.timeout,
                namespace=None,
                rewrite_reports=False,
                modal=False,
                report_dir=report_dir,
            )

            # First, check the per-instance report written by the harness
            instance_report = (
                RUN_EVALUATION_LOG_DIR
                / run_id
                / model_name
                / instance_id
                / LOG_REPORT
            )
            if instance_report.exists():
                content = instance_report.read_text().strip()
                if content:
                    data = json.loads(content)
                    resolved = bool(data.get(instance_id, {}).get("resolved", False))
                    self.logger.info(
                        f"SWE-bench result for {instance_id}: resolved={resolved}"
                    )
                    return resolved

            # Fallback: check the run-level summary report
            run_report_path = (
                Path(report_dir) / f"{model_name}.{run_id}.json"
            )
            if run_report_path.exists():
                run_report = json.loads(run_report_path.read_text())
                resolved = instance_id in run_report.get("resolved_ids", [])
                self.logger.info(
                    f"SWE-bench result (run report) for {instance_id}: resolved={resolved}"
                )
                return resolved

            self.logger.warning(
                f"No evaluation report found for {instance_id}. "
                "Docker may not be running or the image build failed."
            )
            return False

        except Exception as e:
            self.logger.error(f"Official SWE-bench verification failed: {e}")
            return False
        finally:
            if predictions_file and os.path.exists(predictions_file):
                try:
                    os.unlink(predictions_file)
                except OSError:
                    pass


def create_swebench_runner(
    dataset: str = "lite",
    output_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    model_name: str = "self-evolver",
    run_id: str = "self-evolver",
    controller_mode: str = "off",
    controller_stage: str = "eval",
    rollout_jsonl: Optional[Path] = None,
    reward_config: Optional[Path] = None,
) -> SWEBenchRunner:
    """
    Factory function to create a SWE-bench runner.
    
    Args:
        dataset: "lite" for SWE-bench Lite, "verified" for Verified.
        output_dir: Output directory for results.
        workspace_dir: Directory for cloned repositories.
        model_name: SWE-bench prediction model name.
        run_id: SWE-bench evaluation run ID.
        
    Returns:
        Configured SWEBenchRunner instance.
    """
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }
    
    dataset_name = dataset_map.get(dataset, dataset)
    return SWEBenchRunner(
        dataset_name=dataset_name,
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        model_name=model_name,
        run_id=run_id,
        controller_mode=controller_mode,
        controller_stage=controller_stage,
        rollout_jsonl=rollout_jsonl,
        reward_config=reward_config,
    )
