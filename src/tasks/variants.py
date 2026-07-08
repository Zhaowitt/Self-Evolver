"""
Focused task variants (Proposal 2.3 failure-driven task generation).

A focused variant keeps the error type but reduces context complexity:
FAIL_TO_PASS is subset to a single test and the problem statement is
augmented with that failing test path.
"""

from __future__ import annotations

import json
from typing import List, Optional

FOCUS_MARKER = "::focus-"


def fail_to_pass(instance: dict) -> List[str]:
    """FAIL_TO_PASS as a list of test paths (handles JSON-string encoding)."""
    raw = instance.get("FAIL_TO_PASS")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [line.strip() for line in raw.splitlines() if line.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(test) for test in raw if str(test).strip()]
    return [str(raw)]


def is_focused_variant(instance: dict) -> bool:
    return FOCUS_MARKER in str(instance.get("instance_id") or "")


def base_instance_id(instance: dict) -> str:
    """The id whose official container image this instance runs in."""
    base = instance.get("base_instance_id")
    if base:
        return str(base)
    return str(instance.get("instance_id") or "").split(FOCUS_MARKER, 1)[0]


def make_focused_variant(instance: dict, n: int = 1) -> Optional[dict]:
    """
    Build the n-th focused variant of a base instance, or None when
    impossible (already a variant, missing id, or fewer than n F2P tests).

    The variant copies every field, keeps the parent's image via
    ``base_instance_id``, subsets FAIL_TO_PASS to the n-th test (preserving
    the parent's list/JSON-string encoding), suffixes the instance id with
    ``::focus-<n>``, and augments the problem statement with the test path.
    """
    instance_id = str(instance.get("instance_id") or "")
    if not instance_id or is_focused_variant(instance):
        return None
    tests = fail_to_pass(instance)
    if n < 1 or n > len(tests):
        return None
    target = tests[n - 1]

    variant = dict(instance)
    variant["instance_id"] = f"{instance_id}{FOCUS_MARKER}{n}"
    variant["base_instance_id"] = instance_id
    if isinstance(instance.get("FAIL_TO_PASS"), str):
        variant["FAIL_TO_PASS"] = json.dumps([target])
    else:
        variant["FAIL_TO_PASS"] = [target]
    statement = str(instance.get("problem_statement") or "").rstrip()
    variant["problem_statement"] = (
        f"{statement}\n\n"
        f"[Focused variant of {instance_id}]\n"
        f"Make this single failing test pass with a minimal change:\n"
        f"    {target}\n"
        f"Other originally failing tests are out of scope for this variant."
    )
    return variant
