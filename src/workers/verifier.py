"""
Verifier Worker - Test Verification Agent.

Responsible for applying patches and verifying they fix the issue
without introducing new problems.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.environment.models import ExecutionContext, TestResult
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.workers.base import BaseWorker, WorkerResult
from src.workers.patch_generator import PatchResult

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of patch verification."""
    
    SUCCESS = "success"
    PATCH_FAILED = "patch_failed"
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
                    status=VerificationStatus.ERROR,
                    error_message="No patch content provided",
                ),
            )
        
        try:
            # Step 1: Apply the patch
            self.logger.info("Applying patch...")
            patch_applied = self.env.apply_patch(patch_result.patch_content)
            
            if not patch_applied:
                self.logger.warning("Failed to apply patch")
                return WorkerResult(
                    success=False,
                    data=VerificationResult(
                        status=VerificationStatus.PATCH_FAILED,
                        patch_applied=False,
                        error_message="Failed to apply patch to repository",
                        summary="Patch could not be applied. Check patch format and file paths.",
                    ),
                )
            
            # Step 2: Run tests
            self.logger.info("Running tests...")
            test_result = self.env.run_tests()
            
            # Step 3: Analyze results
            result = self._analyze_test_results(test_result, context)
            result.patch_applied = True
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
                    error_message=str(e),
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
            if not self.env.apply_patch(patch_result.patch_content):
                return WorkerResult(
                    success=False,
                    data=VerificationResult(
                        status=VerificationStatus.PATCH_FAILED,
                    ),
                )
            
            # Run specific tests
            test_result = self.env.run_specific_tests(test_files)
            result = self._analyze_test_results(test_result, context)
            result.patch_applied = True
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
