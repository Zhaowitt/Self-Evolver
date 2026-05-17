"""Rule-based reward components from execution outcomes."""

from __future__ import annotations

from typing import Any, Dict


DEFAULT_MAX_REASONABLE_FILES = 5
DEFAULT_COST_TOKEN_BUDGET = 50000


def compute_reward_components(
    execution_result: Any,
    controller_signal: Any = None,
    max_reasonable_files: int = DEFAULT_MAX_REASONABLE_FILES,
    cost_token_budget: int = DEFAULT_COST_TOKEN_BUDGET,
) -> Dict[str, float]:
    """Compute bounded reward components from an ExecutionResult-like object."""
    final_patch = getattr(execution_result, "final_patch", None)
    records = list(getattr(execution_result, "iteration_records", []) or [])

    patch_apply_success = _any_verification(records, lambda v: bool(getattr(v, "patch_applied", False)))
    patch_apply_error = _any_verification(
        records,
        lambda v: getattr(getattr(v, "status", None), "value", "") == "patch_failed",
    )
    canonical_diff_nonempty = bool(final_patch and getattr(final_patch, "content", "").strip())
    if not canonical_diff_nonempty:
        canonical_diff_nonempty = _any_verification(
            records,
            lambda v: bool(getattr(v, "canonical_patch_content", "").strip()),
        )

    files_modified = len(getattr(final_patch, "modified_files", []) or [])
    if files_modified == 0:
        files_modified = _last_patch_file_count(records)

    timeout = _has_timeout(execution_result, records)
    total_tokens = int(getattr(execution_result, "total_tokens", 0) or 0)
    token_budget = max(1, int(cost_token_budget or DEFAULT_COST_TOKEN_BUDGET))

    return {
        "patch_applies_cleanly": 1.0 if patch_apply_success else 0.0,
        "non_empty_canonical_diff": 1.0 if canonical_diff_nonempty else 0.0,
        "tests_pass": 1.0 if bool(getattr(execution_result, "success", False)) else 0.0,
        "avoids_patch_apply_error": 0.0 if patch_apply_error else 1.0,
        "reasonable_file_count": 1.0 if 0 < files_modified <= max_reasonable_files else 0.0,
        "follows_task_wrapper_constraints": _task_wrapper_component(execution_result, controller_signal),
        "avoids_timeout": 0.0 if timeout else 1.0,
        "cost_efficiency": max(0.0, min(1.0, 1.0 - total_tokens / token_budget)),
    }


def _any_verification(records: list[Any], predicate) -> bool:
    for record in records:
        verification = getattr(record, "verification_result", None)
        if verification and predicate(verification):
            return True
    return False


def _last_patch_file_count(records: list[Any]) -> int:
    for record in reversed(records):
        patch = getattr(record, "patch_result", None)
        if patch:
            files = getattr(patch, "modified_files", None) or []
            if files:
                return len(files)
            patch_info = getattr(patch, "patch_info", None)
            if patch_info:
                return len(getattr(patch_info, "modified_files", []) or [])
    return 0


def _has_timeout(execution_result: Any, records: list[Any]) -> bool:
    text_parts = [str(getattr(execution_result, "error_message", "") or "")]
    for record in records:
        text_parts.append(str(getattr(record, "error", "") or ""))
        verification = getattr(record, "verification_result", None)
        if verification and getattr(verification, "test_result", None):
            test_result = verification.test_result
            text_parts.append(str(getattr(test_result, "error_logs", "") or ""))
            text_parts.append(str(getattr(test_result, "output", "") or ""))
    return "timeout" in "\n".join(text_parts).lower() or "timed out" in "\n".join(text_parts).lower()


def _task_wrapper_component(execution_result: Any, controller_signal: Any) -> float:
    if not controller_signal:
        return 1.0
    task_wrapper = ""
    mode = "train"
    if isinstance(controller_signal, dict):
        task_wrapper = str(controller_signal.get("task_wrapper") or "")
        mode = str(controller_signal.get("mode") or "train")
    else:
        task_wrapper = str(getattr(controller_signal, "task_wrapper", "") or "")
        mode = str(getattr(controller_signal, "mode", "train") or "train")
    if mode != "train" or not task_wrapper:
        return 1.0
    records = list(getattr(execution_result, "iteration_records", []) or [])
    if not records:
        return 0.0
    return 1.0 if any(getattr(record, "inspection_result", None) for record in records) else 0.0
