"""
SWE-bench Runner - Integration with SWE-bench benchmark.

This module provides integration with SWE-bench Lite/Verified datasets.
Requires Docker and the swebench package for full functionality.

Note: Full implementation requires server with Docker support.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from src.benchmark.base import BenchmarkRunner, InstanceResult
from src.config import get_config
from src.critic.judge import CriticJudge
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.orchestrator.orchestrator import ExecutionOrchestrator

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
    ):
        """
        Initialize SWE-bench runner.
        
        Args:
            dataset_name: HuggingFace dataset name.
            output_dir: Directory for results.
            workspace_dir: Directory for cloning repos.
        """
        super().__init__(name="swebench", output_dir=output_dir)
        self.dataset_name = dataset_name
        self.workspace_dir = workspace_dir or Path(tempfile.mkdtemp(prefix="swebench_"))
        self.judge = CriticJudge()
        self._dataset = None
    
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

        if not repo_dir.exists():
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
        import json as _json
        raw = issue.metadata.get("fail_to_pass")
        if not raw:
            return None
        try:
            tests: list[str] = _json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            return None
        if not tests:
            return None
        # pytest accepts test IDs directly; quote paths with spaces just in case
        test_args = " ".join(f'"{t}"' for t in tests)
        return f"python3 -m pytest {test_args} -x --tb=short -q"

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

            # Run the orchestrator
            orchestrator = ExecutionOrchestrator(
                env=env,
                max_iterations=get_config().agent.max_iterations,
            )

            result = orchestrator.run(issue)

            # Evaluate
            evaluation = self.judge.evaluate(result)

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

        model_name = "self-evolver"
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
) -> SWEBenchRunner:
    """
    Factory function to create a SWE-bench runner.
    
    Args:
        dataset: "lite" for SWE-bench Lite, "verified" for Verified.
        output_dir: Output directory for results.
        
    Returns:
        Configured SWEBenchRunner instance.
    """
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }
    
    dataset_name = dataset_map.get(dataset, dataset)
    return SWEBenchRunner(dataset_name=dataset_name, output_dir=output_dir)
