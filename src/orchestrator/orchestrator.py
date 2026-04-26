"""
Execution Orchestrator - Static Workflow Coordination.

Manages the execution flow: Inspector -> Patch Generator -> Verifier
with retry logic on failures.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.environment.models import (
    ExecutionContext,
    Issue,
    PatchInfo,
)
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.workers.inspector import Inspector, InspectionResult
from src.workers.llm_judge import JudgeDecision, JudgeRoute, LLMJudge
from src.workers.patch_generator import PatchGenerator, PatchResult
from src.workers.verifier import Verifier, VerificationResult

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Overall execution status."""
    
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations_reached"
    ERROR = "error"


@dataclass
class IterationRecord:
    """Record of a single iteration attempt."""
    
    iteration: int
    inspection_result: Optional[InspectionResult] = None
    patch_result: Optional[PatchResult] = None
    verification_result: Optional[VerificationResult] = None
    judge_decision: Optional[JudgeDecision] = None
    error: Optional[str] = None
    tokens_used: int = 0
    duration_ms: float = 0.0


@dataclass
class ExecutionResult:
    """Final result of the orchestrated execution."""
    
    status: ExecutionStatus
    issue_id: str
    success: bool = False
    iterations_used: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    final_patch: Optional[PatchInfo] = None
    iteration_records: List[IterationRecord] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def summary(self) -> str:
        status_str = "SUCCESS" if self.success else "FAILED"
        return (
            f"[{status_str}] Issue {self.issue_id}: "
            f"{self.iterations_used} iterations, "
            f"{self.total_tokens} tokens, "
            f"{self.total_duration_ms:.0f}ms"
        )


class ExecutionOrchestrator:
    """
    Static workflow orchestrator.
    
    Coordinates the execution flow:
    1. Inspector analyzes the issue
    2. Patch Generator creates a fix
    3. Verifier tests the fix
    4. If failed, retry with feedback (up to max_iterations)
    """
    
    def __init__(
        self,
        env: ProjectEnvironment,
        llm_client: Optional[LLMClient] = None,
        max_iterations: Optional[int] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            env: Project environment for code operations.
            llm_client: Shared LLM client for all workers.
            max_iterations: Maximum retry attempts. Uses config default if None.
        """
        self.env = env
        self.llm_client = llm_client or LLMClient()
        self.max_iterations = max_iterations or get_config().agent.max_iterations
        
        # Initialize workers with shared LLM client
        self.inspector = Inspector(env, self.llm_client)
        self.patch_generator = PatchGenerator(env, self.llm_client)
        self.verifier = Verifier(env, self.llm_client)
        self.llm_judge = LLMJudge(self.llm_client)
        
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")
    
    def run(self, issue: Issue) -> ExecutionResult:
        """
        Execute the full workflow for an issue.
        
        Args:
            issue: The issue to solve.
            
        Returns:
            ExecutionResult with status and details.
        """
        start_time = time.time()
        self.logger.info(f"Starting execution for issue: {issue.id}")
        
        # Reset token counter
        self.llm_client.reset_token_count()
        
        # Initialize context
        repo_state = self.env.get_repo_state()
        context = ExecutionContext(
            issue=issue,
            repo_state=repo_state,
            max_iterations=self.max_iterations,
        )
        
        # Set up environment for the issue
        if not self.env.setup_issue(issue):
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                issue_id=issue.id,
                error_message="Failed to set up issue environment",
            )
        
        iteration_records: List[IterationRecord] = []
        final_patch: Optional[PatchInfo] = None
        current_inspection: Optional[InspectionResult] = None
        next_route = JudgeRoute.REINSPECT
        hard_case = False
        
        for iteration in range(self.max_iterations):
            context.iteration = iteration
            context.metadata["next_route"] = next_route.value
            self.logger.info(f"=== Iteration {iteration + 1}/{self.max_iterations} ===")
            
            iter_start = time.time()
            record = IterationRecord(iteration=iteration)
            
            try:
                # Step 1: Inspection
                if current_inspection is None or next_route == JudgeRoute.REINSPECT:
                    self.logger.info("Step 1: Running Inspector...")
                    inspect_result = self.inspector.execute(context)

                    if not inspect_result.success or not inspect_result.data:
                        record.error = f"Inspection failed: {inspect_result.error}"
                        iteration_records.append(record)
                        context.previous_errors.append(record.error)
                        next_route = JudgeRoute.REINSPECT
                        continue

                    current_inspection = inspect_result.data
                    record.tokens_used += inspect_result.tokens_used
                else:
                    self.logger.info(f"Step 1: Reusing inspection for route {next_route.value}")

                record.inspection_result = current_inspection
                
                # Step 2: Patch Generation
                self.logger.info("Step 2: Running Patch Generator...")
                patch_result = self.patch_generator.execute(context, current_inspection)
                
                if not patch_result.success or not patch_result.data:
                    record.error = f"Patch generation failed: {patch_result.error}"
                    record.duration_ms = (time.time() - iter_start) * 1000
                    iteration_records.append(record)
                    context.previous_errors.append(record.error)
                    next_route = JudgeRoute.EMPTY_PATCH_REPROMPT
                    continue
                
                record.patch_result = patch_result.data
                record.tokens_used += patch_result.tokens_used
                
                if not patch_result.data.patch_content:
                    record.error = "Empty patch generated"
                    self._judge_failed_attempt(context, record)
                    record.duration_ms = (time.time() - iter_start) * 1000
                    iteration_records.append(record)
                    context.previous_errors.append(record.error)
                    next_route = self._route_from_record(record)
                    if next_route == JudgeRoute.REINSPECT:
                        current_inspection = None
                    continue
                
                # Step 3: Verification
                self.logger.info("Step 3: Running Verifier...")
                verify_result = self.verifier.execute(context, patch_result.data)
                
                record.verification_result = verify_result.data
                record.tokens_used += verify_result.tokens_used
                record.duration_ms = (time.time() - iter_start) * 1000
                iteration_records.append(record)
                
                if verify_result.success and verify_result.data and verify_result.data.success:
                    # Success!
                    self.logger.info("Verification passed! Issue fixed.")
                    final_patch = verify_result.data.canonical_patch_info or patch_result.data.patch_info
                    
                    total_duration = (time.time() - start_time) * 1000
                    return ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        issue_id=issue.id,
                        success=True,
                        iterations_used=iteration + 1,
                        total_tokens=self.llm_client.total_tokens_used,
                        total_duration_ms=total_duration,
                        final_patch=final_patch,
                        iteration_records=iteration_records,
                    )
                
                # Verification failed - prepare for retry
                self.logger.info("Verification failed, preparing for retry...")
                
                # Update context with failure information
                if verify_result.data and verify_result.data.canonical_patch_info:
                    context.previous_patches.append(verify_result.data.canonical_patch_info)
                elif patch_result.data.patch_info:
                    context.previous_patches.append(patch_result.data.patch_info)
                
                if verify_result.data and verify_result.data.test_result:
                    context.test_results.append(verify_result.data.test_result)
                if verify_result.data:
                    error_msg = verify_result.data.summary or verify_result.data.error_message
                    context.previous_errors.append(error_msg or "Verification failed")

                self._judge_failed_attempt(context, record)
                next_route = self._route_from_record(record)
                if next_route == JudgeRoute.GIVE_UP_HARD_CASE:
                    hard_case = True
                    break
                if next_route == JudgeRoute.REINSPECT:
                    current_inspection = None
                
            except Exception as e:
                self.logger.error(f"Iteration {iteration + 1} error: {e}")
                record.error = str(e)
                record.duration_ms = (time.time() - iter_start) * 1000
                iteration_records.append(record)
                context.previous_errors.append(str(e))
                next_route = JudgeRoute.REINSPECT
        
        # Max iterations reached
        total_duration = (time.time() - start_time) * 1000
        self.logger.warning(f"Max iterations ({self.max_iterations}) reached without success")
        
        failure_reason = "judge_give_up" if hard_case else "max_iterations"
        self._record_hard_case(issue, iteration_records, failure_reason)

        return ExecutionResult(
            status=ExecutionStatus.FAILED if hard_case else ExecutionStatus.MAX_ITERATIONS,
            issue_id=issue.id,
            success=False,
            iterations_used=len(iteration_records),
            total_tokens=self.llm_client.total_tokens_used,
            total_duration_ms=total_duration,
            iteration_records=iteration_records,
            error_message=(
                "LLM judge routed this issue to hard-case buffer"
                if hard_case else f"Failed after {self.max_iterations} iterations"
            ),
            metadata={"hard_case": hard_case, "last_route": next_route.value},
        )

    def _judge_failed_attempt(
        self,
        context: ExecutionContext,
        record: IterationRecord,
    ) -> None:
        """Run LLM judge after a failed attempt and store retry feedback."""
        judge_result = self.llm_judge.execute(context, record)
        if judge_result.data:
            record.judge_decision = judge_result.data
            context.metadata["judge_feedback"] = judge_result.data.feedback_for_next_attempt
            context.metadata["next_route"] = judge_result.data.route.value
        if judge_result.llm_response:
            record.tokens_used += judge_result.tokens_used

    def _route_from_record(self, record: IterationRecord) -> JudgeRoute:
        """Extract the next route from a judged record, defaulting to reinspect."""
        if record.judge_decision:
            return record.judge_decision.route
        return JudgeRoute.REINSPECT

    def _record_hard_case(
        self,
        issue: Issue,
        records: List[IterationRecord],
        reason: str,
    ) -> None:
        """Append a compact hard-case record for later failure analysis."""
        try:
            output_path = get_config().environment.workspace_dir / "hard_cases.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "issue_id": issue.id,
                "repo_name": issue.repo_name,
                "base_commit": issue.base_commit,
                "reason": reason,
                "created_at": datetime.now().isoformat(),
                "iterations": len(records),
                "routes": [
                    record.judge_decision.route.value
                    for record in records
                    if record.judge_decision
                ],
                "errors": [
                    record.error
                    for record in records
                    if record.error
                ],
                "verification_statuses": [
                    record.verification_result.status.value
                    for record in records
                    if record.verification_result
                ],
                "patch_apply_strategies": [
                    record.verification_result.patch_apply_result.strategy
                    for record in records
                    if record.verification_result and record.verification_result.patch_apply_result
                ],
            }
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write hard-case record: {e}")
    
    def run_single_iteration(
        self,
        context: ExecutionContext,
    ) -> IterationRecord:
        """
        Run a single iteration of the workflow.
        
        Useful for debugging or manual stepping through the process.
        """
        record = IterationRecord(iteration=context.iteration)
        start_time = time.time()
        
        try:
            # Inspection
            inspect_result = self.inspector.execute(context)
            if inspect_result.success and inspect_result.data:
                record.inspection_result = inspect_result.data
                record.tokens_used += inspect_result.tokens_used
            else:
                record.error = f"Inspection failed: {inspect_result.error}"
                return record
            
            # Patch Generation
            patch_result = self.patch_generator.execute(context, inspect_result.data)
            if patch_result.success and patch_result.data:
                record.patch_result = patch_result.data
                record.tokens_used += patch_result.tokens_used
            else:
                record.error = f"Patch generation failed: {patch_result.error}"
                return record
            
            # Verification
            verify_result = self.verifier.execute(context, patch_result.data)
            record.verification_result = verify_result.data
            record.tokens_used += verify_result.tokens_used
            
        except Exception as e:
            record.error = str(e)
        
        record.duration_ms = (time.time() - start_time) * 1000
        return record
