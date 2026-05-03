"""
LLM Judge Worker - failure diagnosis and retry routing.

The judge is intentionally not allowed to declare success. It only routes
failed attempts based on deterministic verifier output and concise context.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.environment.models import ExecutionContext
from src.llm.client import LLMClient
from src.workers.base import BaseWorker, WorkerResult

logger = logging.getLogger(__name__)


class JudgeRoute(Enum):
    """Allowed retry routes from the LLM judge."""

    REPAIR_PATCH_FORMAT = "repair_patch_format"
    REGENERATE_PATCH_SAME_LOCATION = "regenerate_patch_same_location"
    REINSPECT = "reinspect"
    EMPTY_PATCH_REPROMPT = "empty_patch_reprompt"
    GIVE_UP_HARD_CASE = "give_up_hard_case"


@dataclass
class JudgeDecision:
    """Structured LLM judge decision."""

    failure_category: str
    route: JudgeRoute
    feedback_for_next_attempt: str
    confidence: float = 0.0
    raw_response: str = ""


LLM_JUDGE_SYSTEM_PROMPT = """You are a strict failure diagnosis and retry-routing judge for a code repair agent.

You cannot mark an attempt as successful. Tests and deterministic verifier results are the only success authority.

Given an issue, attempted patch, patch apply result, and test result, decide the next retry route.

Allowed routes:
- repair_patch_format: patch failed to apply, malformed diff, wrong hunk/context, truncated patch, or no canonical diff.
- regenerate_patch_same_location: patch applied but tests still fail and localization still looks plausible.
- reinspect: current localization likely wrong, missing context, wrong files, or failure changed substantially.
- empty_patch_reprompt: patch generator returned no usable patch.
- give_up_hard_case: repeated attempts are not making progress within the budget.

Respond ONLY with a valid JSON object:
{
  "failure_category": "short_snake_case_category",
  "route": "one_allowed_route",
  "feedback_for_next_attempt": "specific concise instruction for the next worker",
  "confidence": 0.0
}
"""


class LLMJudge(BaseWorker):
    """LLM-based failure router used after an unsuccessful attempt."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(llm_client=llm_client, name="LLMJudge")

    @property
    def system_prompt(self) -> str:
        return LLM_JUDGE_SYSTEM_PROMPT

    def execute(self, context: ExecutionContext, iteration_record=None) -> WorkerResult[JudgeDecision]:
        """Judge the failed attempt and choose the next retry route."""
        if iteration_record and iteration_record.verification_result:
            status = iteration_record.verification_result.status.value
            if status in {"patch_failed", "patch_context_mismatch", "no_changes", "empty_patch"}:
                decision = self._fallback_decision(iteration_record)
                return WorkerResult(success=True, data=decision)

        try:
            user_message = self._build_prompt(context, iteration_record)
            response = self._call_llm(user_message)
            decision = self._parse_response(response.content)
            return WorkerResult(success=True, data=decision, llm_response=response)
        except Exception as e:
            self.logger.warning(f"LLM judge failed, using fallback route: {e}")
            return WorkerResult(
                success=False,
                data=self._fallback_decision(iteration_record, str(e)),
                error=str(e),
            )

    def _build_prompt(self, context: ExecutionContext, iteration_record) -> str:
        parts = [
            "## Issue",
            context.issue.description[:3000],
            f"\n## Attempt\nIteration: {context.iteration + 1}/{context.max_iterations}",
        ]

        if iteration_record and iteration_record.inspection_result:
            inspection = iteration_record.inspection_result
            parts.extend([
                "\n## Inspection",
                f"Suspected files: {', '.join(inspection.suspected_files[:5])}",
                f"Root cause: {inspection.root_cause_analysis[:1200]}",
                "Fix suggestions:",
                "\n".join(f"- {s}" for s in inspection.fix_suggestions[:5]),
            ])

        if iteration_record and iteration_record.patch_result:
            patch = iteration_record.patch_result.patch_content or ""
            parts.extend([
                "\n## Raw Patch",
                f"```diff\n{patch[:4000]}\n```",
            ])
        else:
            parts.extend(["\n## Raw Patch", "<missing>"])

        if iteration_record and iteration_record.verification_result:
            verification = iteration_record.verification_result
            parts.extend([
                "\n## Verification",
                f"status: {verification.status.value}",
                f"summary: {verification.summary[:1200]}",
                f"error_message: {verification.error_message[:1200]}",
            ])

            if verification.patch_apply_result:
                apply_result = verification.patch_apply_result
                parts.extend([
                    "\n## Patch Apply Result",
                    f"success: {apply_result.success}",
                    f"strategy: {apply_result.strategy}",
                    f"diagnostic:\n```\n{apply_result.diagnostic[:2500]}\n```",
                ])

            if verification.test_result:
                test_result = verification.test_result
                test_logs = test_result.error_logs or test_result.output
                parts.extend([
                    "\n## Test Result",
                    f"passed: {test_result.passed}",
                    f"logs:\n```\n{test_logs[:3000]}\n```",
                ])
        elif iteration_record and iteration_record.error:
            parts.extend(["\n## Attempt Error", iteration_record.error[:1500]])

        if context.previous_errors:
            parts.extend([
                "\n## Previous Error Summaries",
                "\n".join(f"- {err[:500]}" for err in context.previous_errors[-3:]),
            ])

        return "\n".join(parts)

    def _parse_response(self, content: str) -> JudgeDecision:
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                raise ValueError("Judge response did not contain JSON")
            json_str = json_match.group(0)

        data = json.loads(json_str)
        route_value = data.get("route", "")
        try:
            route = JudgeRoute(route_value)
        except ValueError:
            route = JudgeRoute.REINSPECT

        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        return JudgeDecision(
            failure_category=str(data.get("failure_category", "unknown")),
            route=route,
            feedback_for_next_attempt=str(data.get("feedback_for_next_attempt", "")),
            confidence=max(0.0, min(1.0, confidence)),
            raw_response=content,
        )

    def _fallback_decision(self, iteration_record, reason: str = "") -> JudgeDecision:
        """Rule-based fallback used when the LLM judge is unavailable."""
        route = JudgeRoute.REINSPECT
        category = "unknown"
        feedback = reason or "Judge unavailable; retry with broader inspection."

        if not iteration_record:
            return JudgeDecision(category, route, feedback, confidence=0.0)

        if not iteration_record.patch_result or not iteration_record.patch_result.patch_content:
            return JudgeDecision(
                failure_category="empty_patch",
                route=JudgeRoute.EMPTY_PATCH_REPROMPT,
                feedback_for_next_attempt="Generate a non-empty unified diff patch.",
                confidence=0.5,
            )

        verification = iteration_record.verification_result
        if verification:
            status = verification.status.value
            if status in {"patch_failed", "patch_context_mismatch", "no_changes"}:
                return JudgeDecision(
                    failure_category=(
                        "patch_context_mismatch"
                        if status == "patch_context_mismatch"
                        else "patch_apply_failure"
                    ),
                    route=JudgeRoute.REPAIR_PATCH_FORMAT,
                    feedback_for_next_attempt=(
                        "Repair the unified diff format, file paths, and hunk context "
                        "so the old-side context exactly matches the current files. "
                        "Preserve the intended code change."
                    ),
                    confidence=0.6,
                )
            if status == "empty_patch":
                return JudgeDecision(
                    failure_category="empty_patch",
                    route=JudgeRoute.EMPTY_PATCH_REPROMPT,
                    feedback_for_next_attempt="Generate a non-empty patch with valid file headers and hunks.",
                    confidence=0.6,
                )
            if status in {"tests_failed", "new_issues"}:
                return JudgeDecision(
                    failure_category=status,
                    route=JudgeRoute.REGENERATE_PATCH_SAME_LOCATION,
                    feedback_for_next_attempt=(
                        "The patch applied but tests failed. Keep the same likely files, "
                        "read the verifier logs, and correct the repair logic."
                    ),
                    confidence=0.5,
                )

        return JudgeDecision(category, route, feedback, confidence=0.0)
