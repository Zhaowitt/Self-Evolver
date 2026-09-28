"""Deterministic FAIL_TO_PASS subset selection, independent of patches and outcomes."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Sequence


F2P_SUBSET_POLICY = "stable_hash_v1"
DEFAULT_F2P_SUBSET_SIZE = 2
# sympy's ``bin/test -k`` takes a single keyword, so only one test per run.
REPO_MAX_SUBSET_SIZE = {"sympy__sympy": 1}


def parse_test_ids(value: object) -> List[str]:
    """Normalize a SWE-bench test-id list or its JSON representation."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    result: List[str] = []
    for item in value:
        test_id = str(item).strip()
        if test_id and test_id not in result:
            result.append(test_id)
    return result


def test_scope(test_id: str) -> str:
    """Return a stable file/class-like diversity key for a test id."""
    value = test_id.strip()
    django = re.match(r"^\S+\s+\(([^)]+)\)$", value)
    if django:
        return django.group(1)
    if "::" in value:
        parts = value.split("::")
        return "::".join(parts[:2]) if len(parts) > 1 else parts[0]
    if ".py:" in value:
        return value.split(":", 1)[0]
    dotted = value.split(".")
    # Django/unittest ids end in Class.method; keep module + class as the scope.
    return ".".join(dotted[:-1]) if len(dotted) > 2 else value


@dataclass(frozen=True)
class F2PSubsetSelector:
    subset_size: int = DEFAULT_F2P_SUBSET_SIZE
    policy: str = F2P_SUBSET_POLICY

    def select(
        self,
        fail_to_pass: Sequence[str] | str | object,
        instance_id: str,
        round_id: str | int | None = None,
    ) -> List[str]:
        tests = parse_test_ids(fail_to_pass)
        size = max(1, int(self.subset_size))
        repo_key = str(instance_id).rsplit("-", 1)[0]
        size = min(size, REPO_MAX_SUBSET_SIZE.get(repo_key, size))
        if len(tests) <= size:
            return tests

        seed = f"{self.policy}\0{instance_id}\0{round_id or '0'}"
        ranked = sorted(tests, key=lambda item: (_hash(seed, item), item))
        by_scope: dict[str, List[str]] = defaultdict(list)
        for test_id in ranked:
            by_scope[test_scope(test_id)].append(test_id)

        # First cover distinct files/classes; then fill any remaining slots.
        scope_order = sorted(
            by_scope,
            key=lambda scope: (_hash(seed, f"scope:{scope}"), scope),
        )
        selected = [by_scope[scope][0] for scope in scope_order[:size]]
        if len(selected) < size:
            selected.extend(item for item in ranked if item not in selected)
        return selected[:size]


def _hash(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()
