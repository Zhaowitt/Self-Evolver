"""One failure taxonomy, shared everywhere.

Memory, skills, controller, and critic must all speak the same failure
vocabulary; a second ``FailureType`` enum would silently split the selection
and reflection channels. This guards the unification structurally (exactly one
enum class in the tree) and by identity (every consumer imports the same
object).
"""

from __future__ import annotations

import re
from pathlib import Path

import src.critic.judge as critic_judge
import src.memory.hard_case_buffer as hard_case_buffer
import src.skills as skills_pkg
import src.skills.proposals as proposals
import src.skills.skill_selector as skill_selector
from src.skills.failure_types import (
    FailureType,
    failure_type_from_verification_status,
)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_exactly_one_failure_type_enum_is_defined_in_the_tree():
    definitions = []
    for path in SRC_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^class FailureType\b", text, re.MULTILINE):
            definitions.append(path)
    assert definitions == [SRC_DIR / "skills" / "failure_types.py"], definitions


def test_every_consumer_shares_the_same_enum_object():
    canonical = FailureType
    assert skills_pkg.FailureType is canonical
    assert critic_judge.FailureType is canonical
    assert hard_case_buffer.FailureType is canonical
    assert proposals.FailureType is canonical
    assert skill_selector.FailureType is canonical


def test_verifier_status_mapping_uses_the_shared_taxonomy():
    # The critic re-exports the same mapping function, so both grade identically.
    assert (
        critic_judge.failure_type_from_verification_status
        is failure_type_from_verification_status
    )
    assert failure_type_from_verification_status("new_issues") == FailureType.REGRESSION_INTRODUCED.value
    assert failure_type_from_verification_status("tests_failed") == FailureType.TEST_FAILURE.value
    assert failure_type_from_verification_status("patch_failed") == FailureType.PATCH_APPLICATION_ERROR.value


def test_canonical_vocabulary_is_stable():
    assert {item.value for item in FailureType} == {
        "none",
        "localization_error",
        "patch_generation_error",
        "patch_application_error",
        "test_failure",
        "regression_introduced",
        "timeout",
        "unknown",
        "general",
    }
