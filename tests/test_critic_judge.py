from src.critic.judge import CriticJudge, FailureType
from src.environment.test_backend import EvalOutcome
from src.orchestrator.orchestrator import ExecutionResult, ExecutionStatus, IterationRecord
from src.skills.failure_types import FailureType as UnifiedFailureType
from src.workers.inspector import InspectionResult
from src.workers.patch_generator import PatchResult
from src.workers.verifier import VerificationResult, VerificationStatus


def _result(record) -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.MAX_ITERATIONS,
        issue_id="pkg__repo-1",
        success=False,
        iterations_used=1,
        iteration_records=[record],
    )


def test_critic_uses_unified_failure_taxonomy():
    assert FailureType is UnifiedFailureType


def test_tests_counts_from_attached_eval_outcome():
    outcome = EvalOutcome(
        f2p_passed=2, f2p_total=3, p2p_passed=5, p2p_total=5,
        resolved=False, per_test={}, log_tail="",
    )
    record = IterationRecord(
        iteration=0,
        verification_result=VerificationResult(
            status=VerificationStatus.TESTS_FAILED, eval_outcome=outcome
        ),
    )

    evaluation = CriticJudge().evaluate(_result(record))

    assert evaluation.tests_passed == 7
    assert evaluation.tests_total == 8
    assert evaluation.pass_rate == 7 / 8


def test_explicit_eval_outcome_takes_precedence():
    attached = EvalOutcome(1, 3, 0, 5, resolved=False, per_test={}, log_tail="")
    explicit = EvalOutcome(3, 3, 5, 5, resolved=True, per_test={}, log_tail="")
    record = IterationRecord(
        iteration=0,
        verification_result=VerificationResult(
            status=VerificationStatus.TESTS_FAILED, eval_outcome=attached
        ),
    )

    evaluation = CriticJudge().evaluate(_result(record), eval_outcome=explicit)

    assert evaluation.tests_passed == 8
    assert evaluation.tests_total == 8


def test_counts_fall_back_to_test_result_when_no_outcome():
    from src.environment.models import TestResult

    test_result = TestResult(passed=False, total_tests=4, passed_tests=1, failed_tests=3)
    record = IterationRecord(
        iteration=0,
        verification_result=VerificationResult(
            status=VerificationStatus.TESTS_FAILED, test_result=test_result
        ),
    )

    evaluation = CriticJudge().evaluate(_result(record))

    assert evaluation.tests_passed == 1
    assert evaluation.tests_total == 4


def test_failure_type_regression_from_new_issues_status():
    record = IterationRecord(
        iteration=0,
        inspection_result=InspectionResult(suspected_files=["calc.py"]),
        patch_result=PatchResult(patch_content="diff"),
        verification_result=VerificationResult(status=VerificationStatus.NEW_ISSUES),
    )

    evaluation = CriticJudge().evaluate(_result(record))

    assert evaluation.failure_type == FailureType.REGRESSION_INTRODUCED
