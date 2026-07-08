from src.workers.llm_judge import JudgeRoute, LLMJudge
from src.orchestrator.orchestrator import IterationRecord
from src.workers.patch_generator import PatchResult
from src.workers.verifier import VerificationResult, VerificationStatus


def test_fallback_routes_patch_worker_crash_to_regenerate():
    # A crashed patch worker records an error but no patch_result at all.
    record = IterationRecord(iteration=1, error="APIError: connection reset")

    decision = LLMJudge()._fallback_decision(record)

    assert decision.route == JudgeRoute.REGENERATE_PATCH_SAME_LOCATION
    assert decision.failure_category == "patch_worker_error"


def test_fallback_still_routes_empty_patch_content_to_reprompt():
    # A patch worker that returned an empty patch (not a crash) still reprompts.
    record = IterationRecord(
        iteration=1,
        error="Empty patch generated",
        patch_result=PatchResult(patch_content=""),
    )

    decision = LLMJudge()._fallback_decision(record)

    assert decision.route == JudgeRoute.EMPTY_PATCH_REPROMPT
    assert decision.failure_category == "empty_patch"


def test_fallback_routes_patch_apply_failure_to_repair_format():
    record = IterationRecord(
        iteration=1,
        patch_result=PatchResult(patch_content="diff"),
        verification_result=VerificationResult(status=VerificationStatus.PATCH_FAILED),
    )

    decision = LLMJudge()._fallback_decision(record)

    assert decision.route == JudgeRoute.REPAIR_PATCH_FORMAT
