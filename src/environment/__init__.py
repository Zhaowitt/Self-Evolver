"""Project Environment module."""

from src.environment.models import (
    CodeLocation,
    Issue,
    PatchApplyResult,
    RepoState,
    TestResult,
)


_TEST_BACKEND_NAMES = (
    "EvalOutcome",
    "HostTestBackend",
    "ContainerTestBackend",
    "resolve_backend",
)


def __getattr__(name):
    """Lazily import heavier environment helpers only when requested."""
    if name == "ProjectEnvironment":
        from src.environment.project_env import ProjectEnvironment

        return ProjectEnvironment
    if name in _TEST_BACKEND_NAMES:
        from src.environment import test_backend

        return getattr(test_backend, name)
    raise AttributeError(name)

__all__ = [
    "ProjectEnvironment",
    "Issue",
    "PatchApplyResult",
    "RepoState",
    "TestResult",
    "CodeLocation",
    "EvalOutcome",
    "HostTestBackend",
    "ContainerTestBackend",
    "resolve_backend",
]
