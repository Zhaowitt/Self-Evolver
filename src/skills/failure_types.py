"""Unified failure taxonomy shared by memory, skills, controller, and critic."""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable, Optional


class FailureType(str, Enum):
    """Single failure vocabulary used across the whole framework."""

    NONE = "none"
    LOCALIZATION_ERROR = "localization_error"
    PATCH_GENERATION_ERROR = "patch_generation_error"
    PATCH_APPLICATION_ERROR = "patch_application_error"
    TEST_FAILURE = "test_failure"
    REGRESSION_INTRODUCED = "regression_introduced"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    GENERAL = "general"


FAILURE_TYPE_VALUES = frozenset(item.value for item in FailureType)

# Values a skill may declare as its target (ordered; used in the controller prompt).
TARGET_FAILURE_TYPES: tuple[str, ...] = (
    FailureType.LOCALIZATION_ERROR.value,
    FailureType.PATCH_GENERATION_ERROR.value,
    FailureType.PATCH_APPLICATION_ERROR.value,
    FailureType.TEST_FAILURE.value,
    FailureType.REGRESSION_INTRODUCED.value,
    FailureType.UNKNOWN.value,
    FailureType.GENERAL.value,
)

# Verifier status strings -> taxonomy.
_VERIFICATION_STATUS_MAP = {
    "patch_failed": FailureType.PATCH_APPLICATION_ERROR,
    "no_changes": FailureType.PATCH_APPLICATION_ERROR,
    "empty_patch": FailureType.PATCH_GENERATION_ERROR,
    "tests_failed": FailureType.TEST_FAILURE,
    "new_issues": FailureType.REGRESSION_INTRODUCED,
    "timeout": FailureType.TIMEOUT,
}

# Explicit per-skill declaration, e.g. a standalone line "Target failure type: test_failure".
_MARKER_RE = re.compile(r"^\s*target failure type:\s*([a-z_]+)\s*\.?\s*$", re.IGNORECASE | re.MULTILINE)

# Keyword heuristic for skills without an explicit marker (ordered for deterministic ties).
_SKILL_KEYWORDS: tuple[tuple[FailureType, tuple[str, ...]], ...] = (
    (FailureType.LOCALIZATION_ERROR, ("localiz", "root cause", "fault region")),
    (FailureType.REGRESSION_INTRODUCED, ("pattern", "alignment", "regression")),
    (FailureType.PATCH_APPLICATION_ERROR, ("apply diagnostic", "hunk", "malformed")),
    (FailureType.PATCH_GENERATION_ERROR, ("patch", "repair", "minimal diff")),
    (FailureType.TEST_FAILURE, ("failing test", "test failure")),
    (FailureType.GENERAL, ("inspect",)),
)


def normalize_failure_type(value: object, default: str = FailureType.UNKNOWN.value) -> str:
    """Coerce free text into a taxonomy value; empty input yields the default."""
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text if text in FAILURE_TYPE_VALUES else FailureType.UNKNOWN.value


def failure_type_from_verification_status(status: object) -> str:
    """Map a verifier status string to a taxonomy value."""
    text = str(status or "").strip().lower()
    if not text:
        return FailureType.UNKNOWN.value
    mapped = _VERIFICATION_STATUS_MAP.get(text)
    if mapped is not None:
        return mapped.value
    return normalize_failure_type(text)


def failure_type_from_statuses(statuses: Iterable[str]) -> str:
    """Map the last verifier status of an execution to a taxonomy value."""
    status_list = [status for status in statuses if status]
    if not status_list:
        return FailureType.UNKNOWN.value
    return failure_type_from_verification_status(status_list[-1])


def explicit_failure_type(content: str) -> Optional[str]:
    """Read an explicit 'Target failure type: <value>' marker from skill markdown."""
    match = _MARKER_RE.search(content or "")
    if not match:
        return None
    value = match.group(1).lower()
    return value if value in FAILURE_TYPE_VALUES else None


def infer_skill_failure_type(skill_id: str, content: str) -> str:
    """Classify a skill's target failure type from its marker, id, and content."""
    explicit = explicit_failure_type(content)
    if explicit:
        return explicit

    id_text = str(skill_id or "").lower().replace("_", " ")
    content_text = str(content or "").lower()
    best_type = FailureType.GENERAL
    best_score = 0
    for failure_type, keywords in _SKILL_KEYWORDS:
        score = 0
        for keyword in keywords:
            if keyword in id_text:
                score += 3
            if keyword in content_text:
                score += 1
        if score > best_score:
            best_type = failure_type
            best_score = score
    return best_type.value
