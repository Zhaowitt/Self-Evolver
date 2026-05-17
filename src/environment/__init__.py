"""Project Environment module."""

from src.environment.models import (
    CodeLocation,
    Issue,
    PatchApplyResult,
    RepoState,
    TestResult,
)


def __getattr__(name):
    """Lazily import heavier environment helpers only when requested."""
    if name == "ProjectEnvironment":
        from src.environment.project_env import ProjectEnvironment

        return ProjectEnvironment
    raise AttributeError(name)

__all__ = [
    "ProjectEnvironment",
    "Issue",
    "PatchApplyResult",
    "RepoState",
    "TestResult",
    "CodeLocation",
]
