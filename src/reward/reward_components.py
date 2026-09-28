"""Reward components from final full official evidence only; subset outcomes never become reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_COST_TOKEN_BUDGET = 60000

_EVAL_FIELDS = ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")


@dataclass
class EvalView:
    """Normalized view of a test-execution outcome (mirrors EvalOutcome)."""

    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    resolved: bool = False
    evaluation_scope: str = "full"

    @classmethod
    def from_any(cls, value: Any) -> Optional["EvalView"]:
        """Accept an EvalOutcome-like object or dict; None when not one."""
        if value is None:
            return None
        if isinstance(value, EvalView):
            return value
        getter = value.get if isinstance(value, dict) else lambda k, d=None: getattr(value, k, d)
        if all(getter(field) is None for field in _EVAL_FIELDS):
            return None
        return cls(
            f2p_passed=int(getter("f2p_passed") or 0),
            f2p_total=int(getter("f2p_total") or 0),
            p2p_passed=int(getter("p2p_passed") or 0),
            p2p_total=int(getter("p2p_total") or 0),
            resolved=bool(getter("resolved") or False),
            evaluation_scope=str(getter("evaluation_scope", "full") or "full"),
        )


def compute_reward_components(
    execution_result: Any,
    eval_outcome: Any = None,
    issue: Any = None,
    cost_token_budget: int = DEFAULT_COST_TOKEN_BUDGET,
) -> Dict[str, float]:
    """Compute the execution-utility components for one rollout."""
    view = extract_eval_view(execution_result, eval_outcome=eval_outcome, issue=issue)
    # Explicit subset evidence must never be promoted to an official outcome.
    if view is not None and view.evaluation_scope == "full":
        f2p_fraction = view.f2p_passed / view.f2p_total if view.f2p_total > 0 else 0.0
        p2p_no_regression = view.p2p_passed == view.p2p_total
        # The official grader's resolution bit is authoritative, never inferred from counts.
        resolved = bool(view.resolved)
    elif view is not None and view.evaluation_scope == "f2p_only":
        # Subset already failed, so it cannot be resolved; P2P was not run.
        f2p_fraction = view.f2p_passed / view.f2p_total if view.f2p_total > 0 else 0.0
        p2p_no_regression, resolved = False, False
    else:
        f2p_fraction, p2p_no_regression, resolved = 0.0, False, False

    records = list(getattr(execution_result, "iteration_records", []) or [])
    final_patch = getattr(execution_result, "final_patch", None)
    patch_applied = _any_verification(records, lambda v: bool(getattr(v, "patch_applied", False)))
    non_empty = bool(final_patch and getattr(final_patch, "content", "").strip()) or _any_verification(
        records,
        lambda v: bool(getattr(v, "canonical_patch_content", "").strip()),
    )
    within_budget = (
        getattr(getattr(execution_result, "status", None), "value", "") != "max_iterations_reached"
    )

    total_tokens = int(getattr(execution_result, "total_tokens", 0) or 0)
    budget = max(1, int(cost_token_budget or DEFAULT_COST_TOKEN_BUDGET))

    token_cost = max(0.0, min(1.0, total_tokens / budget))
    official_resolved = 1.0 if resolved else 0.0
    official_f2p_fraction = max(0.0, min(1.0, f2p_fraction))
    return {
        "official_resolved": official_resolved,
        "official_f2p_fraction": official_f2p_fraction,
        "token_cost": token_cost,
        # Diagnostic fields, unweighted by default.
        "resolved": official_resolved,
        "f2p_fraction": official_f2p_fraction,
        "p2p_no_regression": 1.0 if (
            view is not None and view.evaluation_scope == "full" and p2p_no_regression
        ) else 0.0,
        "cost_efficiency": 1.0 - token_cost,
        "process": 1.0 if (patch_applied and non_empty and within_budget) else 0.0,
    }


def extract_eval_view(
    execution_result: Any,
    eval_outcome: Any = None,
    issue: Any = None,
) -> Optional[EvalView]:
    """Explicit ``eval_outcome`` > result metadata > final candidate's verification outcome."""
    view = EvalView.from_any(eval_outcome)
    if view is not None:
        return view
    metadata = getattr(execution_result, "metadata", None) or {}
    view = EvalView.from_any(metadata.get("eval_outcome"))
    if view is not None:
        return view
    # Only the final candidate (last attempt with a real diff) is scored; an
    # older attempt's outcome must not stand in for it.
    records = list(getattr(execution_result, "iteration_records", []) or [])
    for record in reversed(records):
        verification = getattr(record, "verification_result", None)
        if verification is None or not str(
            getattr(verification, "canonical_patch_content", "") or ""
        ).strip():
            continue
        if getattr(verification, "evaluation_scope", "") == "subset":
            return None
        return EvalView.from_any(getattr(verification, "eval_outcome", None))
    return None


def _any_verification(records: List[Any], predicate) -> bool:
    for record in records:
        verification = getattr(record, "verification_result", None)
        if verification and predicate(verification):
            return True
    return False
