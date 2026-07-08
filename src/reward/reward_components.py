"""Execution-utility components computed from real test-execution evidence.

Components (Proposal 2.6), each bounded to [0, 1]:

- ``resolved``: all FAIL_TO_PASS tests pass AND no PASS_TO_PASS regression in
  the evaluation outcome. This is the in-loop resolution predicate; for a
  full-suite outcome it equals official SWE-bench resolution, for a targeted
  subset (e.g. a focused task variant) it can be true without full resolve.
- ``f2p_fraction``: fraction of FAIL_TO_PASS tests passing.
- ``p2p_no_regression``: 1.0 when no observed PASS_TO_PASS test regressed
  (vacuously true when the outcome ran no P2P tests).
- ``cost_efficiency``: 1 - min(1, total_tokens / cost_token_budget).
- ``process``: patch applied AND non-empty canonical diff AND within the
  iteration budget.

Test evidence comes from an ``EvalOutcome`` (``src.environment.test_backend``)
carrying ``f2p_passed/f2p_total/p2p_passed/p2p_total/resolved``, or is derived
from real per-test statuses in a ``TestResult`` plus the instance's
FAIL_TO_PASS/PASS_TO_PASS name lists. No log-substring heuristics.
"""

from __future__ import annotations

import json
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
        )


def compute_reward_components(
    execution_result: Any,
    eval_outcome: Any = None,
    issue: Any = None,
    cost_token_budget: int = DEFAULT_COST_TOKEN_BUDGET,
) -> Dict[str, float]:
    """Compute the execution-utility components for one rollout."""
    view = extract_eval_view(execution_result, eval_outcome=eval_outcome, issue=issue)
    if view is not None:
        f2p_fraction = view.f2p_passed / view.f2p_total if view.f2p_total > 0 else 0.0
        p2p_no_regression = view.p2p_passed == view.p2p_total
        resolved = bool(view.resolved) or (
            view.f2p_total > 0 and view.f2p_passed == view.f2p_total and p2p_no_regression
        )
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

    return {
        "resolved": 1.0 if resolved else 0.0,
        "f2p_fraction": max(0.0, min(1.0, f2p_fraction)),
        "p2p_no_regression": 1.0 if (view is not None and p2p_no_regression) else 0.0,
        "cost_efficiency": max(0.0, min(1.0, 1.0 - total_tokens / budget)),
        "process": 1.0 if (patch_applied and non_empty and within_budget) else 0.0,
    }


def extract_eval_view(
    execution_result: Any,
    eval_outcome: Any = None,
    issue: Any = None,
) -> Optional[EvalView]:
    """Find the best available test-execution evidence for a rollout.

    Priority: explicit ``eval_outcome`` (container eval of the final patch) >
    outcome attached to the execution result by the verifier > per-test
    statuses of the last verification run classified via the issue's
    FAIL_TO_PASS/PASS_TO_PASS lists.
    """
    view = EvalView.from_any(eval_outcome)
    if view is not None:
        return view
    metadata = getattr(execution_result, "metadata", None) or {}
    view = EvalView.from_any(metadata.get("eval_outcome"))
    if view is not None:
        return view
    records = list(getattr(execution_result, "iteration_records", []) or [])
    for record in reversed(records):
        verification = getattr(record, "verification_result", None)
        if verification is None:
            continue
        view = EvalView.from_any(getattr(verification, "eval_outcome", None))
        if view is not None:
            return view
        test_result = getattr(verification, "test_result", None)
        if test_result is not None and issue is not None:
            f2p_names, p2p_names = _test_name_lists(issue)
            view = eval_view_from_test_result(test_result, f2p_names, p2p_names)
            if view is not None:
                return view
    return None


def eval_view_from_test_result(
    test_result: Any,
    f2p_names: List[str],
    p2p_names: List[str],
) -> Optional[EvalView]:
    """Derive an EvalView from real per-test statuses.

    F2P tests missing from the run count as not passed (the agent must make
    them pass); P2P tests are counted only when actually run, so a targeted
    F2P-only run does not fake regression evidence either way.
    """
    cases = {
        case.name: case
        for case in (getattr(test_result, "test_cases", None) or [])
        if getattr(case, "name", "")
    }
    if not cases or not f2p_names:
        return None
    f2p_passed = sum(1 for name in f2p_names if name in cases and cases[name].passed)
    p2p_run = [name for name in p2p_names if name in cases]
    p2p_passed = sum(1 for name in p2p_run if cases[name].passed)
    return EvalView(
        f2p_passed=f2p_passed,
        f2p_total=len(f2p_names),
        p2p_passed=p2p_passed,
        p2p_total=len(p2p_run),
        resolved=False,
    )


def _test_name_lists(issue: Any) -> tuple[List[str], List[str]]:
    metadata = getattr(issue, "metadata", None) or {}
    return (
        _name_list(metadata.get("fail_to_pass")),
        _name_list(metadata.get("pass_to_pass")),
    )


def _name_list(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item)]


def _any_verification(records: List[Any], predicate) -> bool:
    for record in records:
        verification = getattr(record, "verification_result", None)
        if verification and predicate(verification):
            return True
    return False
