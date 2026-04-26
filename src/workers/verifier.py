"""
Verifier Worker - Test Verification Agent.

Responsible for applying patches and verifying they fix the issue
without introducing new problems.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from src.environment.models import ExecutionContext, PatchApplyResult, PatchInfo, TestResult
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.workers.base import BaseWorker, WorkerResult
from src.workers.patch_generator import PatchResult

logger = logging.getLogger(__name__)


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
    ):
        super().__init__(llm_client=llm_client, name="Verifier")
        self.env = env
    
    @property
    def system_prompt(self) -> str:
        # Verifier mainly uses rule-based checks, minimal LLM usage
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
            # Step 1: Apply the patch
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
            
            # Step 2: Run tests
            self.logger.info("Running tests...")
            test_result = self.env.run_tests()
            
            # Step 3: Analyze results
            result = self._analyze_test_results(test_result, context)
            result.patch_applied = True
            result.raw_patch_content = patch_result.patch_content
            result.canonical_patch_content = canonical_diff
            result.canonical_patch_info = canonical_patch_info
            result.patch_apply_result = apply_result
            result.test_result = test_result
            
            # Step 4: Revert changes for next iteration if needed
            if not result.success:
                self.logger.info("Reverting changes due to verification failure")
                self.env.revert_changes()
            
            return WorkerResult(success=result.success, data=result)
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            # Try to revert on error
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
    
    def _analyze_test_results(
        self,
        test_result: TestResult,
        context: ExecutionContext,
    ) -> VerificationResult:
        """Analyze test results to determine verification status."""
        if test_result.passed:
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                tests_passed=True,
                original_issue_fixed=True,
                new_issues_introduced=False,
                summary="All tests passed. Issue appears to be fixed.",
            )
        
        # Tests failed - analyze why
        error_logs = test_result.error_logs or test_result.output
        
        # Check if new issues were introduced
        # (simplified: check if error is different from original)
        original_error = ""
        if context.test_results and context.test_results[0].error_logs:
            original_error = context.test_results[0].error_logs
        
        if original_error and error_logs != original_error:
            # Error changed - might be new issues or partial fix
            if self._is_subset_error(error_logs, original_error):
                # Partial fix - some tests now pass
                return VerificationResult(
                    status=VerificationStatus.TESTS_FAILED,
                    tests_passed=False,
                    original_issue_fixed=False,
                    new_issues_introduced=False,
                    summary="Partial fix - some tests still failing.",
                )
            else:
                # Possibly introduced new issues
                return VerificationResult(
                    status=VerificationStatus.NEW_ISSUES,
                    tests_passed=False,
                    original_issue_fixed=False,
                    new_issues_introduced=True,
                    summary="Patch may have introduced new issues.",
                )
        
        return VerificationResult(
            status=VerificationStatus.TESTS_FAILED,
            tests_passed=False,
            original_issue_fixed=False,
            new_issues_introduced=False,
            summary=f"Tests failed: {error_logs[:500]}",
        )
    
    def _is_subset_error(self, new_error: str, original_error: str) -> bool:
        """Check if new error is a subset of original (partial fix)."""
        # Simple heuristic: check if error message is shorter
        return len(new_error) < len(original_error) * 0.8
    
    def verify_with_specific_tests(
        self,
        context: ExecutionContext,
        patch_result: PatchResult,
        test_files: List[str],
    ) -> WorkerResult[VerificationResult]:
        """Verify patch with specific test files."""
        self.logger.info(f"Verifying with specific tests: {test_files}")
        
        if not patch_result.patch_content:
            return WorkerResult(
                success=False,
                data=VerificationResult(
                    status=VerificationStatus.ERROR,
                    error_message="No patch content",
                ),
            )
        
        try:
            # Apply patch
            apply_result = self.env.apply_patch_detailed(patch_result.patch_content)
            if not apply_result.success:
                self.env.revert_changes()
                return WorkerResult(
                    success=False,
                    data=VerificationResult(
                        status=VerificationStatus.PATCH_FAILED,
                        raw_patch_content=patch_result.patch_content,
                        patch_apply_result=apply_result,
                        summary=apply_result.diagnostic,
                    ),
                )

            canonical_diff = self.env.get_diff()
            canonical_patch_info = PatchInfo.from_diff(canonical_diff) if canonical_diff else None
            if not canonical_diff.strip():
                self.env.revert_changes()
                return WorkerResult(
                    success=False,
                    data=VerificationResult(
                        status=VerificationStatus.NO_CHANGES,
                        patch_applied=True,
                        raw_patch_content=patch_result.patch_content,
                        patch_apply_result=apply_result,
                        summary="Patch applied cleanly but git diff is empty.",
                    ),
                )
            
            # Run specific tests
            test_result = self.env.run_specific_tests(test_files)
            result = self._analyze_test_results(test_result, context)
            result.patch_applied = True
            result.raw_patch_content = patch_result.patch_content
            result.canonical_patch_content = canonical_diff
            result.canonical_patch_info = canonical_patch_info
            result.patch_apply_result = apply_result
            result.test_result = test_result
            
            if not result.success:
                self.env.revert_changes()
            
            return WorkerResult(success=result.success, data=result)
            
        except Exception as e:
            try:
                self.env.revert_changes()
            except Exception:
                pass
            return WorkerResult(
                success=False,
                data=VerificationResult(
                    status=VerificationStatus.ERROR,
                    error_message=str(e),
                ),
            )
