"""
Verifier Worker - applies a candidate patch and decides whether it fixes the issue.

When a test backend is available and the issue is a SWE-bench instance, the
canonical patch is graded with the OFFICIAL per-test FAIL_TO_PASS / PASS_TO_PASS
semantics (an ``EvalOutcome`` from ``src.environment.test_backend``): a real
PASS_TO_PASS regression maps to NEW_ISSUES and an incomplete FAIL_TO_PASS set
maps to TESTS_FAILED. Without a backend (the local ``fix`` command, unit tests)
the environment's test command runs on the host — a passing run is SUCCESS, a
failing run is TESTS_FAILED. There is no host PASS_TO_PASS baseline, so the host
path never fabricates a regression verdict.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from src.environment.models import (
    ExecutionContext,
    PatchApplyResult,
    PatchInfo,
    TestCase,
    TestResult,
    TestStatus,
)
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.workers.base import BaseWorker, WorkerResult
from src.workers.patch_generator import PatchResult

logger = logging.getLogger(__name__)

# swebench grading status string -> host TestStatus (per-test reconstruction).
_SB_TO_TESTSTATUS = {
    "PASSED": TestStatus.PASSED,
    "FAILED": TestStatus.FAILED,
    "ERROR": TestStatus.ERROR,
    "SKIPPED": TestStatus.SKIPPED,
}


class VerificationStatus(Enum):
    """Status of patch verification."""

    SUCCESS = "success"
    EMPTY_PATCH = "empty_patch"
    PATCH_FAILED = "patch_failed"
    NO_CHANGES = "no_changes"
    TESTS_FAILED = "tests_failed"
    NEW_ISSUES = "new_issues"
    ERROR = "error"


@dataclass
class VerificationResult:
    """Result of patch verification."""

    status: VerificationStatus = VerificationStatus.ERROR
    patch_applied: bool = False
    tests_passed: bool = False
    original_issue_fixed: bool = False
    new_issues_introduced: bool = False
    raw_patch_content: str = ""
    canonical_patch_content: str = ""
    canonical_patch_info: Optional[PatchInfo] = None
    patch_apply_result: Optional[PatchApplyResult] = None
    test_result: Optional[TestResult] = None
    eval_outcome: Optional[Any] = None
    error_message: str = ""
    summary: str = ""

    @property
    def success(self) -> bool:
        return self.status == VerificationStatus.SUCCESS


class Verifier(BaseWorker):
    """Verifier worker for testing patches."""

    def __init__(
        self,
        env: ProjectEnvironment,
        llm_client: Optional[LLMClient] = None,
        test_backend: Any = None,
    ):
        super().__init__(llm_client=llm_client, name="Verifier")
        self.env = env
        self.test_backend = test_backend

    @property
    def system_prompt(self) -> str:
        # Verifier is rule-based; it makes no LLM calls.
        return "You are a test verification assistant."

    def execute(
        self,
        context: ExecutionContext,
        patch_result: Optional[PatchResult] = None,
    ) -> WorkerResult[VerificationResult]:
        """Verify a patch by applying it and running tests."""
        self.logger.info(f"Verifying patch for issue: {context.issue.id}")

        if not patch_result or not patch_result.patch_content:
            return WorkerResult(
                success=False,
                data=VerificationResult(
                    status=VerificationStatus.EMPTY_PATCH,
                    error_message="No patch content provided",
                    summary="Patch generator returned empty patch content.",
                ),
            )

        try:
            # Step 1: Apply the patch to capture the canonical git diff.
            self.logger.info("Applying patch...")
            apply_result = self.env.apply_patch_detailed(patch_result.patch_content)

            if not apply_result.success:
                self.logger.warning("Failed to apply patch")
                self.env.revert_changes()
                return WorkerResult(
                    success=False,
                    data=VerificationResult(
                        status=VerificationStatus.PATCH_FAILED,
                        patch_applied=False,
                        raw_patch_content=patch_result.patch_content,
                        patch_apply_result=apply_result,
                        error_message="Failed to apply patch to repository",
                        summary=(
                            "Patch could not be applied. "
                            f"Strategy={apply_result.strategy}. "
                            f"{apply_result.diagnostic[:500]}"
                        ),
                    ),
                )

            canonical_diff = self.env.get_diff()
            canonical_patch_info = PatchInfo.from_diff(canonical_diff) if canonical_diff else None

            if not canonical_diff.strip():
                self.logger.warning("Patch applied but produced no canonical diff")
                self.env.revert_changes()
                return WorkerResult(
                    success=False,
                    data=VerificationResult(
                        status=VerificationStatus.NO_CHANGES,
                        patch_applied=True,
                        raw_patch_content=patch_result.patch_content,
                        canonical_patch_content="",
                        patch_apply_result=apply_result,
                        error_message="Patch applied but produced no changes",
                        summary="Patch applied cleanly but git diff is empty.",
                    ),
                )

            # Step 2: Grade the patch. The backend re-applies the canonical patch
            # in isolation, so revert the host apply first.
            if self._use_backend(context.issue):
                self.env.revert_changes()
                result = self._verify_with_backend(context.issue, canonical_diff)
            else:
                self.logger.info("Running tests...")
                result = self._verify_on_host(self.env.run_tests())
                if not result.success:
                    self.logger.info("Reverting changes due to verification failure")
                    self.env.revert_changes()

            result.patch_applied = True
            result.raw_patch_content = patch_result.patch_content
            result.canonical_patch_content = canonical_diff
            result.canonical_patch_info = canonical_patch_info
            result.patch_apply_result = apply_result

            return WorkerResult(success=result.success, data=result)

        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            try:
                self.env.revert_changes()
            except Exception:
                pass

            return WorkerResult(
                success=False,
                data=VerificationResult(
                    status=VerificationStatus.ERROR,
                    raw_patch_content=patch_result.patch_content if patch_result else "",
                    error_message=str(e),
                    summary=f"Verifier error: {e}",
                ),
            )

    def _use_backend(self, issue) -> bool:
        """Grade with the official backend only for SWE-bench instances."""
        return self.test_backend is not None and bool(_fail_to_pass(issue))

    def _verify_with_backend(self, issue, model_patch: str) -> VerificationResult:
        """Grade the canonical patch with official per-test F2P/P2P semantics."""
        outcome = self.test_backend.run_swebench_eval(_swebench_instance(issue), model_patch)
        if outcome.resolved:
            status = VerificationStatus.SUCCESS
        elif not outcome.p2p_no_regression:
            status = VerificationStatus.NEW_ISSUES
        else:
            status = VerificationStatus.TESTS_FAILED
        summary = (
            f"F2P {outcome.f2p_passed}/{outcome.f2p_total}, "
            f"P2P {outcome.p2p_passed}/{outcome.p2p_total}, resolved={outcome.resolved}"
        )
        return VerificationResult(
            status=status,
            tests_passed=bool(outcome.resolved),
            original_issue_fixed=bool(outcome.resolved),
            new_issues_introduced=not outcome.p2p_no_regression,
            test_result=_test_result_from_outcome(outcome),
            eval_outcome=outcome,
            summary=summary,
        )

    def _verify_on_host(self, test_result: TestResult) -> VerificationResult:
        """Host grading: passing run -> SUCCESS, failing run -> TESTS_FAILED."""
        if test_result.passed:
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                tests_passed=True,
                original_issue_fixed=True,
                test_result=test_result,
                summary="All tests passed. Issue appears to be fixed.",
            )
        error_logs = test_result.error_logs or test_result.output
        return VerificationResult(
            status=VerificationStatus.TESTS_FAILED,
            tests_passed=False,
            test_result=test_result,
            summary=f"Tests failed: {error_logs[:500]}",
        )


def _test_result_from_outcome(outcome) -> TestResult:
    """Rebuild a TestResult from an EvalOutcome's per-test statuses."""
    cases = [
        TestCase(name=name, status=_SB_TO_TESTSTATUS.get(status, TestStatus.ERROR))
        for name, status in outcome.per_test.items()
    ]
    passed = outcome.f2p_passed + outcome.p2p_passed
    total = outcome.f2p_total + outcome.p2p_total
    return TestResult(
        passed=bool(outcome.resolved),
        total_tests=total,
        passed_tests=passed,
        failed_tests=max(0, total - passed),
        test_cases=cases,
        error_logs=outcome.log_tail,
    )


def _fail_to_pass(issue) -> List[str]:
    return _id_list((getattr(issue, "metadata", None) or {}).get("fail_to_pass"))


def _id_list(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _swebench_instance(issue) -> dict:
    """Rebuild the official SWE-bench instance dict a test backend needs."""
    metadata = getattr(issue, "metadata", None) or {}
    instance = {
        "instance_id": issue.id,
        "repo": issue.repo_name or "",
        "base_commit": issue.base_commit or "",
        "problem_statement": issue.description or "",
        "test_patch": issue.test_patch or "",
        "version": metadata.get("version"),
        "FAIL_TO_PASS": metadata.get("fail_to_pass"),
        "PASS_TO_PASS": metadata.get("pass_to_pass"),
    }
    setup_commit = metadata.get("environment_setup_commit")
    if setup_commit:
        instance["environment_setup_commit"] = setup_commit
    return instance
