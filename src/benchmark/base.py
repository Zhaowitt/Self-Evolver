"""
Base Benchmark Runner - Abstract interface for benchmark integration.

Provides base classes for running benchmarks like SWE-bench.
To be extended with specific benchmark implementations.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.critic.judge import Evaluation
from src.environment.models import Issue
from src.orchestrator.orchestrator import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class InstanceResult:
    """Result of running a single benchmark instance."""
    
    instance_id: str
    success: bool
    execution_result: Optional[ExecutionResult] = None
    evaluation: Optional[Evaluation] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated result of running a benchmark."""
    
    benchmark_name: str
    total_instances: int
    successful: int = 0
    failed: int = 0
    errors: int = 0
    instance_results: List[InstanceResult] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_instances == 0:
            return 0.0
        return self.successful / self.total_instances
    
    @property
    def summary(self) -> str:
        return (
            f"{self.benchmark_name}: {self.successful}/{self.total_instances} "
            f"({self.success_rate:.1%}) success rate, "
            f"{self.total_tokens} tokens, {self.total_duration_ms:.0f}ms"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "total_instances": self.total_instances,
            "successful": self.successful,
            "failed": self.failed,
            "errors": self.errors,
            "success_rate": self.success_rate,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metadata": self.metadata,
        }
    
    def save(self, path: Path):
        """Save results to JSON file."""
        data = self.to_dict()
        data["instance_results"] = [
            {
                "instance_id": r.instance_id,
                "success": r.success,
                "error": r.error,
            }
            for r in self.instance_results
        ]
        path.write_text(json.dumps(data, indent=2))


class BenchmarkRunner(ABC):
    """
    Abstract base class for benchmark runners.
    
    Subclasses should implement specific benchmark integrations
    (e.g., SWE-bench, HumanEval, etc.)
    """
    
    def __init__(self, name: str, output_dir: Optional[Path] = None):
        """
        Initialize the benchmark runner.
        
        Args:
            name: Name of the benchmark.
            output_dir: Directory for saving results.
        """
        self.name = name
        self.output_dir = output_dir or Path("./benchmark_results")
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def load_instances(self, split: str = "test") -> List[Issue]:
        """
        Load benchmark instances.
        
        Args:
            split: Dataset split to load (train/dev/test).
            
        Returns:
            List of Issue objects representing benchmark tasks.
        """
        pass
    
    @abstractmethod
    def run_instance(self, issue: Issue) -> InstanceResult:
        """
        Run a single benchmark instance.
        
        Args:
            issue: The issue to solve.
            
        Returns:
            InstanceResult with execution details.
        """
        pass
    
    def run_benchmark(
        self,
        num_instances: Optional[int] = None,
        split: str = "test",
    ) -> BenchmarkResult:
        """
        Run the full benchmark.
        
        Args:
            num_instances: Number of instances to run. None for all.
            split: Dataset split to use.
            
        Returns:
            BenchmarkResult with aggregated results.
        """
        self.logger.info(f"Starting benchmark: {self.name}")
        
        # Load instances
        instances = self.load_instances(split)
        if num_instances:
            instances = instances[:num_instances]
        
        result = BenchmarkResult(
            benchmark_name=self.name,
            total_instances=len(instances),
        )
        
        # Run each instance
        for i, issue in enumerate(instances):
            self.logger.info(f"Running instance {i+1}/{len(instances)}: {issue.id}")
            
            try:
                instance_result = self.run_instance(issue)
                result.instance_results.append(instance_result)
                
                if instance_result.success:
                    result.successful += 1
                else:
                    result.failed += 1
                
                if instance_result.execution_result:
                    result.total_tokens += instance_result.execution_result.total_tokens
                    result.total_duration_ms += instance_result.execution_result.total_duration_ms
                    
            except Exception as e:
                self.logger.error(f"Error running instance {issue.id}: {e}")
                result.errors += 1
                result.instance_results.append(InstanceResult(
                    instance_id=issue.id,
                    success=False,
                    error=str(e),
                ))
        
        result.finished_at = datetime.now()
        
        # Save results
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_file = self.output_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.save(result_file)
        self.logger.info(f"Results saved to: {result_file}")
        
        self.logger.info(f"Benchmark complete: {result.summary}")
        return result
