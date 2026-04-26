from src.workers.llm_judge import JudgeRoute, LLMJudge


def test_llm_judge_parses_allowed_route_json():
    content = """
    ```json
    {
      "failure_category": "patch_apply_failure",
      "route": "repair_patch_format",
      "feedback_for_next_attempt": "Fix hunk context and keep the intended change.",
      "confidence": 0.75
    }
    ```
    """

    decision = LLMJudge()._parse_response(content)

    assert decision.failure_category == "patch_apply_failure"
    assert decision.route == JudgeRoute.REPAIR_PATCH_FORMAT
    assert decision.confidence == 0.75


def test_llm_judge_fallback_routes_empty_patch():
    class Record:
        patch_result = None
        verification_result = None

    decision = LLMJudge()._fallback_decision(Record())

    assert decision.route == JudgeRoute.EMPTY_PATCH_REPROMPT
    assert decision.failure_category == "empty_patch"
