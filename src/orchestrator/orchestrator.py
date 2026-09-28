"""Inspector -> Patch Generator -> Verifier loop with judge-routed retries."""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.controller.schema import controller_signal_from_any
from src.environment.models import (
    ExecutionContext,
    Issue,
    PatchInfo,
)
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.memory.hard_case_buffer import HardCaseBuffer, resolve_hard_cases_path
from src.workers.inspector import Inspector, InspectionResult
from src.workers.llm_judge import JudgeDecision, JudgeRoute, LLMJudge
from src.workers.patch_generator import PatchGenerator, PatchResult
from src.workers.verifier import Verifier, VerificationResult

SYSTEM_MAX_ITERATIONS = 3


class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations_reached"
    ERROR = "error"


@dataclass
class IterationRecord:
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


def final_candidate_record(records: List[IterationRecord]) -> Optional[IterationRecord]:
    """The episode's final candidate: last attempt that produced a real diff."""
    for record in reversed(records):
        verification = record.verification_result
        if verification is not None and verification.canonical_patch_content.strip():
            return record
    return None


class ExecutionOrchestrator:
    def __init__(
        self,
        env: ProjectEnvironment,
        llm_client: Optional[LLMClient] = None,
        max_iterations: Optional[int] = None,
        controller_signal: Optional[Any] = None,
        test_backend: Any = None,
        skill_bank: Any = None,
        stage: Optional[str] = None,
    ):
        self.env = env
        self.llm_client = llm_client or LLMClient()
        requested_iterations = max_iterations or get_config().agent.max_iterations
        self.max_iterations = min(SYSTEM_MAX_ITERATIONS, max(1, int(requested_iterations)))
        self.controller_signal = controller_signal_from_any(controller_signal)
        self.stage = stage or (
            self.controller_signal.mode if self.controller_signal else "train"
        )

        self.inspector = Inspector(env, self.llm_client, skill_bank=skill_bank)
        self.patch_generator = PatchGenerator(env, self.llm_client, skill_bank=skill_bank)
        self.verifier = Verifier(env, self.llm_client, test_backend=test_backend)
        self.llm_judge = LLMJudge(self.llm_client)

        self.logger = logging.getLogger(f"{__name__}.Orchestrator")

    def _effective_max_iterations(self) -> int:
        """Return the system-owned hard cap; Controller output cannot change it."""
        return max(1, self.max_iterations)

    def run(self, issue: Issue) -> ExecutionResult:
        start_time = time.time()
        self.logger.info(f"Starting execution for issue: {issue.id}")

        self.llm_client.reset_token_count()

        max_iterations = self._effective_max_iterations()

        repo_state = self.env.get_repo_state()
        context = ExecutionContext(
            issue=issue,
            repo_state=repo_state,
            max_iterations=max_iterations,
        )
        context.metadata["stage"] = self.stage
        if self.controller_signal:
            context.metadata["controller_signal"] = self.controller_signal.to_dict()

        # With a container grader the hidden tests stay out of the agents'
        # checkout; host grading needs them applied locally.
        if not self.env.setup_issue(issue, apply_test_patch=not self._grades_in_container(issue)):
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

        for iteration in range(max_iterations):
            context.iteration = iteration
            context.metadata["next_route"] = next_route.value
            self.logger.info(f"=== Iteration {iteration + 1}/{max_iterations} ===")

            iter_start = time.time()
            record = IterationRecord(iteration=iteration)

            try:
                if current_inspection is None or next_route == JudgeRoute.REINSPECT:
                    self.logger.info("Step 1: Running Inspector...")
                    inspect_result = self.inspector.execute(context)

                    if not inspect_result.success or not inspect_result.data:
                        record.error = f"Inspection failed: {inspect_result.error}"
                        record.duration_ms = (time.time() - iter_start) * 1000
                        iteration_records.append(record)
                        context.previous_errors.append(record.error)
                        next_route = JudgeRoute.REINSPECT
                        continue

                    current_inspection = inspect_result.data
                    record.tokens_used += inspect_result.tokens_used
                else:
                    self.logger.info(f"Step 1: Reusing inspection for route {next_route.value}")

                record.inspection_result = current_inspection

                self.logger.info("Step 2: Running Patch Generator...")
                patch_result = self.patch_generator.execute(context, current_inspection)

                if not patch_result.success or not patch_result.data:
                    # Worker exceptions go to the judge rather than being treated as an empty patch.
                    record.error = f"Patch generation failed: {patch_result.error}"
                    record.duration_ms = (time.time() - iter_start) * 1000
                    iteration_records.append(record)
                    context.previous_errors.append(record.error)
                    self._judge_failed_attempt(context, record)
                    next_route = self._route_from_record(record)
                    if next_route == JudgeRoute.GIVE_UP_HARD_CASE:
                        hard_case = True
                        break
                    if next_route == JudgeRoute.REINSPECT:
                        current_inspection = None
                    continue

                record.patch_result = patch_result.data
                record.tokens_used += patch_result.tokens_used

                if not patch_result.data.patch_content:
                    record.error = "Empty patch generated"
                    record.duration_ms = (time.time() - iter_start) * 1000
                    iteration_records.append(record)
                    context.previous_errors.append(record.error)
                    self._judge_failed_attempt(context, record)
                    next_route = self._route_from_record(record)
                    if next_route == JudgeRoute.GIVE_UP_HARD_CASE:
                        hard_case = True
                        break
                    if next_route == JudgeRoute.REINSPECT:
                        current_inspection = None
                    continue

                self.logger.info("Step 3: Running Verifier...")
                verify_result = self.verifier.execute(context, patch_result.data)

                record.verification_result = verify_result.data
                record.tokens_used += verify_result.tokens_used
                record.duration_ms = (time.time() - iter_start) * 1000
                iteration_records.append(record)

                if verify_result.success and verify_result.data and verify_result.data.success:
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
                        metadata=self._base_result_metadata(),
                    )

                self.logger.info("Verification failed, preparing for retry...")

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

        # The final candidate needs a full official outcome even if it only ran the
        # subset gate; final reward never reads subset evidence.
        final_record = final_candidate_record(iteration_records)
        if final_record is not None:
            try:
                finalize = getattr(self.verifier, "finalize_full_eval", None)
                if finalize is not None:
                    finalize(context, final_record.verification_result)
            except Exception as e:
                self.logger.error(f"Final full evaluation failed: {e}")
            verification = final_record.verification_result
            if verification is not None and verification.success:
                total_duration = (time.time() - start_time) * 1000
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    issue_id=issue.id,
                    success=True,
                    iterations_used=len(iteration_records),
                    total_tokens=self.llm_client.total_tokens_used,
                    total_duration_ms=total_duration,
                    final_patch=verification.canonical_patch_info,
                    iteration_records=iteration_records,
                    metadata=self._base_result_metadata(),
                )

        total_duration = (time.time() - start_time) * 1000
        self.logger.warning(f"Max iterations ({max_iterations}) reached without success")

        failure_reason = "judge_give_up" if hard_case else "max_iterations"
        self._record_hard_case(issue, iteration_records, failure_reason, max_iterations)

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
                if hard_case else f"Failed after {max_iterations} iterations"
            ),
            metadata={
                **self._base_result_metadata(),
                "hard_case": hard_case,
                "last_route": next_route.value,
            },
        )

    def _grades_in_container(self, issue: Issue) -> bool:
        """True when an isolated official backend (not the host) grades this issue."""
        from src.environment.test_backend import HostTestBackend
        from src.workers.verifier import _fail_to_pass

        backend = getattr(self.verifier, "test_backend", None)
        return (
            backend is not None
            and not isinstance(backend, HostTestBackend)
            and bool(_fail_to_pass(issue))
        )

    def _judge_failed_attempt(
        self,
        context: ExecutionContext,
        record: IterationRecord,
    ) -> None:
        """Run LLM judge after a failed attempt and store retry feedback."""
        if context.iteration + 1 >= context.max_iterations:
            # No retry follows; judging would only cost tokens and overwrite the termination reason.
            return
        judge_result = self.llm_judge.execute(context, record)
        if judge_result.data:
            record.judge_decision = judge_result.data
            context.metadata["judge_feedback"] = judge_result.data.feedback_for_next_attempt
            context.metadata["next_route"] = judge_result.data.route.value
        if judge_result.llm_response:
            record.tokens_used += judge_result.tokens_used

    def _route_from_record(self, record: IterationRecord) -> JudgeRoute:
        if record.judge_decision:
            return record.judge_decision.route
        return JudgeRoute.REINSPECT

    def _record_hard_case(
        self,
        issue: Issue,
        records: List[IterationRecord],
        reason: str,
        budget: int,
    ) -> None:
        """Append a stage-tagged hard-case record; the buffer derives the failure type."""
        try:
            output_path = resolve_hard_cases_path()
            HardCaseBuffer(output_path).append_from_execution(
                issue=issue,
                records=records,
                reason=reason,
                metadata=self._base_result_metadata(),
                stage=self.stage,
                budget=budget,
            )
        except Exception as e:
            self.logger.warning(f"Failed to write hard-case record: {e}")

    def _base_result_metadata(self) -> Dict[str, Any]:
        """Metadata shared by success, failure, hard-case, and rollout paths."""
        payload: Dict[str, Any] = {
            "stage": self.stage,
            "round_id": os.getenv("SELF_EVOLVER_ROUND_ID") or "",
        }
        if self.controller_signal:
            payload["controller_signal"] = self.controller_signal.to_dict()
        return payload
