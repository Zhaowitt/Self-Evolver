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
        1. Cloning the repository
        2. Checking out the base commit
        3. Setting up any required dependencies
        
        Note: Full Docker-based setup should be implemented for production.
        """
        repo_dir = self.workspace_dir / issue.id.replace("/", "_")
        
        # Clone repository
        repo_url = f"https://github.com/{issue.repo_name}.git"
        
        env = ProjectEnvironment(repo_dir)
        
        if not repo_dir.exists():
            self.logger.info(f"Cloning {issue.repo_name}...")
            env.clone_repo(repo_url)
        
        # Checkout base commit
        if issue.base_commit:
            env.checkout_commit(issue.base_commit)
        
        return env
    
    def run_instance(self, issue: Issue) -> InstanceResult:
        """Run a single SWE-bench instance."""
        self.logger.info(f"Running instance: {issue.id}")
        
        try:
            # Set up environment
            env = self.setup_instance_environment(issue)
            
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
        Verify a patch using official SWE-bench evaluation.
        
        Requires the swebench package and Docker.
        
        Args:
            instance_id: The SWE-bench instance ID.
            patch: The generated patch content.
            
        Returns:
            True if the patch passes official evaluation.
        """
        try:
            # This would use the official swebench evaluation harness
            # For now, just a placeholder
            self.logger.warning(
                "Official SWE-bench verification not implemented. "
                "Install swebench package and Docker for full evaluation."
            )
            return False
        except ImportError:
            self.logger.error("swebench package not installed")
            return False


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
