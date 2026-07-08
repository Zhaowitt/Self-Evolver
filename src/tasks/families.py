"""
Task family classification (Proposal 2.3).

Instances are grouped into four families from gold patch statistics
(#files changed, paths touching setup/config/dependency files) plus issue
keywords, so the TaskPool can steer sampling per family.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Iterable, List

SINGLE_FILE_BUG_FIX = "single_file_bug_fix"
MULTI_FILE_CONSISTENCY_FIX = "multi_file_consistency_fix"
CONFIG_OR_DEPENDENCY_FIX = "config_or_dependency_fix"
TEST_ALIGNED_BEHAVIOR_FIX = "test_aligned_behavior_fix"

TASK_FAMILIES = (
    SINGLE_FILE_BUG_FIX,
    MULTI_FILE_CONSISTENCY_FIX,
    CONFIG_OR_DEPENDENCY_FIX,
    TEST_ALIGNED_BEHAVIOR_FIX,
)

_DIFF_GIT_RE = re.compile(r"^diff --git a/\S+ b/(?P<path>\S+)", re.MULTILINE)
_PLUS_FILE_RE = re.compile(r"^\+\+\+ b/(?P<path>\S+)", re.MULTILINE)

_CONFIG_BASENAMES = {
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "manifest.in",
    "makefile",
    "pipfile",
    "pipfile.lock",
    "environment.yml",
    "environment.yaml",
}
_CONFIG_SUFFIXES = {".cfg", ".ini", ".toml", ".yaml", ".yml"}
_CONFIG_KEYWORDS = (
    "dependency",
    "dependencies",
    "requirements",
    "installation",
    "pip install",
    "setup.py",
    "install_requires",
    "extras_require",
    "version conflict",
    "importerror",
    "modulenotfounderror",
    "packaging",
)
_TEST_ALIGNED_KEYWORDS = (
    "expected",
    "should return",
    "should be",
    "should not",
    "instead of",
    "failing test",
    "test fails",
    "regression",
    "behavior",
    "behaviour",
    "assert",
    "deprecat",
)
_BUG_KEYWORDS = (
    "traceback",
    "exception",
    "error",
    "raises",
    "crash",
    "segfault",
    "typeerror",
    "valueerror",
    "attributeerror",
    "keyerror",
    "indexerror",
)


def changed_paths(patch: str) -> List[str]:
    """Unique file paths touched by a unified git diff, in patch order."""
    paths: List[str] = []
    matches = _DIFF_GIT_RE.finditer(patch or "")
    found = [match.group("path") for match in matches]
    if not found:
        found = [match.group("path") for match in _PLUS_FILE_RE.finditer(patch or "")]
    for path in found:
        if path not in paths:
            paths.append(path)
    return paths


def _is_test_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1].lower() if parts else ""
    return (
        any(part.lower() in {"test", "tests", "testing"} for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_config_path(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(path).suffix.lower()
    return name in _CONFIG_BASENAMES or suffix in _CONFIG_SUFFIXES or "requirements" in name


def _keyword_hits(text: str, keywords: Iterable[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def classify_family(instance: dict) -> str:
    """
    Classify a SWE-bench-style instance dict into one of TASK_FAMILIES.

    Decision cascade (gold patch stats take priority over issue keywords):
    1. all non-test changed paths are config/dependency files -> config family;
    2. some config path touched and issue mentions config keywords -> config;
    3. more than one non-test file changed -> multi-file consistency fix;
    4. no patch available but issue is strongly config-flavored -> config;
    5. single-file: behavior-expectation wording outweighing error/traceback
       wording -> test-aligned behavior fix, otherwise single-file bug fix.
    """
    text = str(instance.get("problem_statement") or "").lower()
    paths = [path for path in changed_paths(instance.get("patch") or "") if not _is_test_path(path)]
    config_paths = [path for path in paths if _is_config_path(path)]

    if paths and len(config_paths) == len(paths):
        return CONFIG_OR_DEPENDENCY_FIX
    if config_paths and _keyword_hits(text, _CONFIG_KEYWORDS) > 0:
        return CONFIG_OR_DEPENDENCY_FIX
    if len(paths) > 1:
        return MULTI_FILE_CONSISTENCY_FIX
    if not paths and _keyword_hits(text, _CONFIG_KEYWORDS) >= 2:
        return CONFIG_OR_DEPENDENCY_FIX
    if _keyword_hits(text, _TEST_ALIGNED_KEYWORDS) > _keyword_hits(text, _BUG_KEYWORDS):
        return TEST_ALIGNED_BEHAVIOR_FIX
    return SINGLE_FILE_BUG_FIX
