"""Verifier: applies a patch and grades it (official F2P/P2P backend, or host tests)."""

import json
import dataclasses
import os
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
from src.environment.f2p_subset import F2PSubsetSelector, parse_test_ids
from src.llm.client import LLMClient
from src.workers.base import BaseWorker, WorkerResult
from src.workers.patch_generator import PatchResult

# swebench grading status string -> host TestStatus.
_SB_TO_TESTSTATUS = {
    "PASSED": TestStatus.PASSED,
    "FAILED": TestStatus.FAILED,
    "ERROR": TestStatus.ERROR,
    "SKIPPED": TestStatus.SKIPPED,
}


class VerificationStatus(Enum):
    SUCCESS = "success"
    EMPTY_PATCH = "empty_patch"
    PATCH_FAILED = "patch_failed"
    NO_CHANGES = "no_changes"
    TESTS_FAILED = "tests_failed"
    NEW_ISSUES = "new_issues"
    ERROR = "error"


@dataclass
class VerificationResult:
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
    subset_eval_outcome: Optional[Any] = None
    full_eval_outcome: Optional[Any] = None
    eval_outcome: Optional[Any] = None
    evaluation_scope: str = "none"
    official_eval_executed: bool = False
    error_message: str = ""
    summary: str = ""

    @property
    def success(self) -> bool:
        return self.status == VerificationStatus.SUCCESS


class Verifier(BaseWorker):
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
        # Rule-based; makes no LLM calls.
        return "You are a test verification assistant."

    def execute(
        self,
        context: ExecutionContext,
        patch_result: Optional[PatchResult] = None,
    ) -> WorkerResult[VerificationResult]:
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
            # Apply on the host to capture the canonical git diff.
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

            # The backend re-applies the canonical patch in isolation, so revert the host apply.
            if self._use_backend(context.issue):
                final_iteration = context.iteration + 1 >= context.max_iterations
                # A patch that breaks Python syntax cannot pass: skip the container run
                # on non-final attempts (the final candidate is always graded).
                syntax_errors = [] if final_iteration else self._syntax_errors(canonical_diff)
                self.env.revert_changes()
                if syntax_errors:
                    result = VerificationResult(
                        status=VerificationStatus.TESTS_FAILED,
                        evaluation_scope="precheck",
                        error_message="\n".join(syntax_errors),
                        summary="Patch introduces a syntax error: " + "; ".join(syntax_errors)[:500],
                    )
                else:
                    result = self._verify_with_backend(context, canonical_diff)
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

    def _syntax_errors(self, diff: str) -> List[str]:
        """Syntax errors introduced in touched .py files; pre-existing ones (version skew) are skipped."""
        errors: List[str] = []
        paths = sorted(
            {
                line[len("+++ b/"):].strip()
                for line in diff.split("\n")
                if line.startswith("+++ b/") and line.strip().endswith(".py")
            }
        )
        for path in paths:
            file_path = self.env.repo_path / path
            try:
                source = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                compile(source, path, "exec", dont_inherit=True)
                continue
            except SyntaxError as exc:
                error = f"{path}:{exc.lineno}: {exc.msg}"
            except (ValueError, TypeError):
                continue
            try:
                original = self.env.repo.git.show(f":{path}")
            except Exception:
                original = None
            if original is not None:
                try:
                    compile(original, path, "exec", dont_inherit=True)
                except (SyntaxError, ValueError, TypeError):
                    continue  # host interpreter cannot judge this file
            errors.append(error)
        return errors

    def _f2p_only_result(
        self,
        context: ExecutionContext,
        model_patch: str,
        subset_outcome: Any,
    ) -> Optional[VerificationResult]:
        """F2P-only grading after a failed subset; None means run the full evaluation."""
        if os.getenv("SELF_EVOLVER_F2P_ONLY_ON_FAILED_SUBSET", "1").strip().lower() in {
            "0", "false", "no", "off",
        }:
            return None
        all_f2p = set(_fail_to_pass(context.issue))
        if all_f2p and set(getattr(subset_outcome, "selected_f2p_ids", []) or []) == all_f2p:
            # The subset already ran every F2P test, so its outcome is the F2P-only outcome.
            outcome = dataclasses.replace(
                subset_outcome,
                evaluation_scope="f2p_only",
                resolved=False,
                # Reuse, not a run: keeps eval_cost from double counting.
                cache_hit=True,
                duration_seconds=0.0,
            )
            return self._f2p_only_verification(outcome, subset_outcome)
        if not hasattr(self.test_backend, "run_f2p_eval"):
            return None
        try:
            outcome = self.test_backend.run_f2p_eval(_swebench_instance(context.issue), model_patch)
        except NotImplementedError as exc:
            self.logger.info(f"F2P-only grading unavailable ({exc}); running full eval")
            return None
        except Exception:
            self.logger.exception("F2P-only grading failed to execute; running full eval")
            return None
        if not getattr(outcome, "all_selected_executed", False):
            self.logger.warning("F2P-only grading inconclusive; running full eval")
            return None
        return self._f2p_only_verification(outcome, subset_outcome)

    @staticmethod
    def _f2p_only_verification(outcome: Any, subset_outcome: Any) -> VerificationResult:
        return VerificationResult(
            status=VerificationStatus.TESTS_FAILED,
            tests_passed=False,
            original_issue_fixed=False,
            test_result=_test_result_from_outcome(outcome),
            subset_eval_outcome=subset_outcome,
            full_eval_outcome=outcome,
            eval_outcome=outcome,
            evaluation_scope="f2p_only",
            official_eval_executed=True,
            summary=(
                f"F2P {outcome.f2p_passed}/{outcome.f2p_total} (subset failed; "
                "P2P skipped, resolved=False)"
            ),
        )

    def _use_backend(self, issue) -> bool:
        """Grade with the official backend only for SWE-bench instances."""
        return self.test_backend is not None and bool(_fail_to_pass(issue))

    def finalize_full_eval(
        self,
        context: ExecutionContext,
        verification: Optional[VerificationResult],
    ) -> bool:
        """Upgrade a subset-only final candidate in place; True when a full eval ran."""
        if (
            verification is None
            or verification.evaluation_scope not in {"subset", "precheck"}
            or not verification.canonical_patch_content.strip()
            or not self._use_backend(context.issue)
        ):
            return False
        subset_outcome = verification.subset_eval_outcome
        full = None
        if _subset_failed(subset_outcome):
            full = self._f2p_only_result(
                context, verification.canonical_patch_content, subset_outcome
            )
        if full is None:
            full = self._verify_with_backend(
                context, verification.canonical_patch_content, force_full=True
            )
        for name in (
            "status",
            "tests_passed",
            "original_issue_fixed",
            "new_issues_introduced",
            "test_result",
            "full_eval_outcome",
            "eval_outcome",
            "evaluation_scope",
            "official_eval_executed",
            "summary",
        ):
            setattr(verification, name, getattr(full, name))
        verification.subset_eval_outcome = subset_outcome
        return True

    def _verify_with_backend(
        self,
        context: ExecutionContext,
        model_patch: str,
        force_full: bool = False,
    ) -> VerificationResult:
        """Subset gate (if enabled), then official grading; subset results never reach ``eval_outcome``."""
        issue = context.issue
        instance = _swebench_instance(issue)
        stage = str(context.metadata.get("stage") or "train").lower()
        final_iteration = context.iteration + 1 >= context.max_iterations
        subset_outcome = None

        # The gate is outcome-preserving (a failed F2P subset implies a failed full
        # F2P set; the final candidate is always graded in full), so eval may enable it.
        on = {"1", "true", "yes", "on"}
        if stage == "train":
            gate_enabled = os.getenv("SELF_EVOLVER_TRAIN_SUBSET_GATE", "1").strip().lower() in on
        else:
            gate_enabled = os.getenv("SELF_EVOLVER_EVAL_SUBSET_GATE", "0").strip().lower() in on
        if not force_full and gate_enabled and hasattr(self.test_backend, "run_subset_eval"):
            selected = _training_subset(issue)
            if selected:
                try:
                    subset_outcome = self.test_backend.run_subset_eval(
                        instance,
                        model_patch,
                        selected,
                    )
                except NotImplementedError as exc:
                    self.logger.info(f"Subset gate unavailable ({exc}); running full eval")
                    subset_outcome = None
                except Exception:
                    # The gate only saves time; its failure must never fail the attempt.
                    self.logger.exception("Subset gate failed to execute; running full eval")
                    subset_outcome = None
            if subset_outcome is not None and not getattr(
                subset_outcome, "all_selected_executed", True
            ):
                self.logger.warning(
                    "Subset gate inconclusive (selected tests missing from log); "
                    "running full eval"
                )
                context.metadata.setdefault("subset_inconclusive", 0)
                context.metadata["subset_inconclusive"] += 1
                subset_outcome = None
            if subset_outcome is not None:
                subset_passed = (
                    subset_outcome.f2p_total > 0
                    and subset_outcome.f2p_passed == subset_outcome.f2p_total
                )
                if not subset_passed and final_iteration:
                    # A failed subset cannot resolve; the reward needs only the F2P fraction.
                    cheap = self._f2p_only_result(context, model_patch, subset_outcome)
                    if cheap is not None:
                        return cheap
                if not subset_passed and not final_iteration:
                    summary = (
                        "Subset gate failed: "
                        f"F2P {subset_outcome.f2p_passed}/{subset_outcome.f2p_total}; "
                        "full official evaluation skipped."
                    )
                    return VerificationResult(
                        status=VerificationStatus.TESTS_FAILED,
                        tests_passed=False,
                        original_issue_fixed=False,
                        test_result=_test_result_from_outcome(subset_outcome),
                        subset_eval_outcome=subset_outcome,
                        eval_outcome=None,
                        evaluation_scope="subset",
                        official_eval_executed=False,
                        summary=summary,
                    )

        if hasattr(self.test_backend, "run_full_eval"):
            outcome = self.test_backend.run_full_eval(instance, model_patch)
        else:
            outcome = self.test_backend.run_swebench_eval(instance, model_patch)
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
            subset_eval_outcome=subset_outcome,
            full_eval_outcome=outcome,
            eval_outcome=outcome,
            evaluation_scope="full",
            official_eval_executed=True,
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


def _subset_failed(outcome: Any) -> bool:
    """A conclusive subset run in which some selected F2P test did not pass."""
    return (
        outcome is not None
        and getattr(outcome, "all_selected_executed", False)
        and getattr(outcome, "f2p_total", 0) > 0
        and getattr(outcome, "f2p_passed", 0) < getattr(outcome, "f2p_total", 0)
    )


def _training_subset(issue) -> List[str]:
    metadata = getattr(issue, "metadata", None) or {}
    frozen = parse_test_ids(
        metadata.get("train_fail_to_pass_subset")
        or metadata.get("TRAIN_FAIL_TO_PASS_SUBSET")
    )
    if frozen:
        return frozen
    round_id = metadata.get("round_id") or os.getenv("SELF_EVOLVER_ROUND_ID") or "0"
    return F2PSubsetSelector().select(
        metadata.get("fail_to_pass"),
        instance_id=issue.id,
        round_id=round_id,
    )


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
