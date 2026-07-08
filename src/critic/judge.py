"""
Critic/Judge - Execution Result Evaluation.

Evaluates the outcome of code repair attempts and provides
structured feedback for analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.environment.models import PatchInfo
from src.orchestrator.orchestrator import ExecutionResult, ExecutionStatus
from src.skills.failure_types import FailureType, failure_type_from_verification_status

logger = logging.getLogger(__name__)


@dataclass
class Evaluation:
    """Comprehensive evaluation of an execution attempt."""
    
    # Core metrics
    success: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    
    # Patch quality metrics
    patch_lines_changed: int = 0
    patch_files_modified: int = 0
    minimal_patch: bool = True  # Whether patch follows minimal change principle
    
    # Efficiency metrics
    iterations_used: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    
    # Failure analysis
    failure_type: FailureType = FailureType.NONE
    failure_tags: List[str] = field(default_factory=list)
    
    # Summary
    summary: str = ""
    reflection: str = ""
    
    # Raw data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (0-1)."""
        if not self.success:
            return 0.0
        
        # Factors: iterations used, tokens used, patch size
        iter_score = 1.0 - (self.iterations_used - 1) / 10  # Penalize more iterations
        token_score = max(0, 1.0 - self.total_tokens / 50000)  # Penalize high token usage
        patch_score = 1.0 if self.minimal_patch else 0.5
        
        return (iter_score + token_score + patch_score) / 3
    
    @property
    def pass_rate(self) -> float:
        """Calculate test pass rate."""
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total


class CriticJudge:
    """
    Evaluates execution results and provides structured feedback.
    
    Primarily uses rule-based evaluation with optional LLM assistance
    for complex failure analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CriticJudge")
    
    def evaluate(self, result: ExecutionResult, eval_outcome: Any = None) -> Evaluation:
        """
        Evaluate an execution result.

        Args:
            result: The execution result to evaluate.
            eval_outcome: Optional ``EvalOutcome`` (official per-test grading of
                the final patch). When absent, the counts fall back to any
                outcome the verifier attached in the loop, then to the last
                run's per-test statuses.

        Returns:
            Evaluation with metrics and analysis.
        """
        self.logger.info(f"Evaluating execution for issue: {result.issue_id}")

        evaluation = Evaluation(
            success=result.success,
            iterations_used=result.iterations_used,
            total_tokens=result.total_tokens,
            total_duration_ms=result.total_duration_ms,
        )

        # Real test counts, so pass_rate reflects actual F2P/P2P evidence.
        self._populate_test_counts(evaluation, result, eval_outcome)

        # Analyze patch if available
        if result.final_patch:
            evaluation.patch_lines_changed = result.final_patch.total_changes
            evaluation.patch_files_modified = len(result.final_patch.modified_files)
            evaluation.minimal_patch = self._is_minimal_patch(result.final_patch)
        
        # Determine failure type if not successful
        if not result.success:
            evaluation.failure_type = self._determine_failure_type(result)
            evaluation.failure_tags = self._extract_failure_tags(result)
        
        # Generate summary
        evaluation.summary = self._generate_summary(result, evaluation)
        evaluation.reflection = self._generate_reflection(result, evaluation)
        
        # Store metadata
        evaluation.metadata = {
            "issue_id": result.issue_id,
            "status": result.status.value,
            "iteration_count": len(result.iteration_records),
        }
        
        return evaluation
    
    def _is_minimal_patch(self, patch: PatchInfo) -> bool:
        """Check if patch follows minimal change principle."""
        # Simple heuristic: less than 50 lines changed
        return patch.total_changes <= 50

    @staticmethod
    def _populate_test_counts(
        evaluation: Evaluation,
        result: ExecutionResult,
        eval_outcome: Any,
    ) -> None:
        """Set tests_passed/tests_total from the best available test evidence."""
        outcome = eval_outcome
        if outcome is None:
            for record in reversed(result.iteration_records):
                verification = record.verification_result
                candidate = getattr(verification, "eval_outcome", None) if verification else None
                if candidate is not None:
                    outcome = candidate
                    break
        if outcome is not None:
            evaluation.tests_passed = outcome.f2p_passed + outcome.p2p_passed
            evaluation.tests_total = outcome.f2p_total + outcome.p2p_total
            return
        for record in reversed(result.iteration_records):
            verification = record.verification_result
            test_result = getattr(verification, "test_result", None) if verification else None
            if test_result is not None:
                evaluation.tests_passed = test_result.passed_tests
                evaluation.tests_total = test_result.total_tests
                return

    def _determine_failure_type(self, result: ExecutionResult) -> FailureType:
        """Determine the primary failure type via the unified taxonomy."""
        if result.status == ExecutionStatus.ERROR:
            return FailureType.UNKNOWN

        if not result.iteration_records:
            return FailureType.UNKNOWN

        last_record = result.iteration_records[-1]

        if last_record.inspection_result is None:
            return FailureType.LOCALIZATION_ERROR

        if last_record.patch_result is None or not last_record.patch_result.patch_content:
            return FailureType.PATCH_GENERATION_ERROR

        if last_record.verification_result:
            return FailureType(
                failure_type_from_verification_status(
                    last_record.verification_result.status.value
                )
            )

        return FailureType.UNKNOWN
    
    def _extract_failure_tags(self, result: ExecutionResult) -> List[str]:
        """Extract tags describing the failure."""
        tags = []
        
        if result.status == ExecutionStatus.MAX_ITERATIONS:
            tags.append("max_iterations_reached")
        
        # Analyze iteration records
        for record in result.iteration_records:
            if record.error:
                if "timeout" in record.error.lower():
                    tags.append("timeout")
                if "parse" in record.error.lower():
                    tags.append("parse_error")
                if "api" in record.error.lower():
                    tags.append("api_error")
            if record.verification_result:
                status = record.verification_result.status.value
                tags.append(status)
                apply_result = record.verification_result.patch_apply_result
                if apply_result and apply_result.strategy:
                    tags.append(f"patch_apply:{apply_result.strategy}")
        
        # Check for repeated failures
        if len(result.iteration_records) > 1:
            inspection_failures = sum(
                1 for r in result.iteration_records if r.inspection_result is None
            )
            if inspection_failures > 1:
                tags.append("repeated_localization_failure")
            
            patch_failures = sum(
                1 for r in result.iteration_records 
                if r.patch_result is None or not r.patch_result.patch_content
            )
            if patch_failures > 1:
                tags.append("repeated_patch_failure")
        
        return list(set(tags))
    
    def _generate_summary(self, result: ExecutionResult, evaluation: Evaluation) -> str:
        """Generate a human-readable summary."""
        if result.success:
            return (
                f"Successfully fixed issue {result.issue_id} in {result.iterations_used} "
                f"iteration(s). Changed {evaluation.patch_lines_changed} lines across "
                f"{evaluation.patch_files_modified} file(s). "
                f"Used {result.total_tokens} tokens in {result.total_duration_ms:.0f}ms."
            )
        
        failure_desc = evaluation.failure_type.value.replace("_", " ")
        return (
            f"Failed to fix issue {result.issue_id} after {result.iterations_used} "
            f"iteration(s). Primary failure: {failure_desc}. "
            f"Tags: {', '.join(evaluation.failure_tags) or 'none'}. "
            f"Used {result.total_tokens} tokens."
        )
    
    def _generate_reflection(self, result: ExecutionResult, evaluation: Evaluation) -> str:
        """Generate reflection for learning."""
        if result.success:
            return (
                "The fix was successful. Key factors: "
                f"accurate localization with {evaluation.iterations_used} attempt(s), "
                f"{'minimal' if evaluation.minimal_patch else 'extensive'} code changes."
            )
        
        reflections = []
        
        if evaluation.failure_type == FailureType.LOCALIZATION_ERROR:
            reflections.append(
                "Failed to correctly identify the bug location. "
                "Consider: more thorough error log analysis, "
                "checking related test files, examining call chains."
            )
        elif evaluation.failure_type == FailureType.PATCH_GENERATION_ERROR:
            reflections.append(
                "Failed to generate a valid patch. "
                "Consider: reviewing patch format, ensuring correct file paths, "
                "providing more code context."
            )
        elif evaluation.failure_type == FailureType.TEST_FAILURE:
            reflections.append(
                "Patch did not pass tests. "
                "Consider: verifying the fix logic, checking edge cases, "
                "ensuring all affected code paths are addressed."
            )
        elif evaluation.failure_type == FailureType.REGRESSION_INTRODUCED:
            reflections.append(
                "Patch introduced new issues. "
                "Consider: more careful impact analysis, smaller scope changes, "
                "running broader test coverage."
            )
        
        if "max_iterations_reached" in evaluation.failure_tags:
            reflections.append(
                "Exhausted all retry attempts. This issue may require: "
                "different approach, more domain knowledge, or human review."
            )
        
        return " ".join(reflections) if reflections else "No specific reflection available."
