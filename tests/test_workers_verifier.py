import subprocess
from pathlib import Path

from src.environment.models import ExecutionContext, Issue
from src.environment.project_env import ProjectEnvironment
from src.environment.test_backend import EvalOutcome
from src.workers.patch_generator import PatchResult
from src.workers.verifier import Verifier, VerificationStatus


def _run(cmd, cwd: Path):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    _run(["git", "add", "calc.py"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


RAW_PATCH = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)


class FakeBackend:
    """Records the graded patch and returns a scripted EvalOutcome."""

    def __init__(self, outcome: EvalOutcome):
        self.outcome = outcome
        self.calls = []

    def run_swebench_eval(self, instance, model_patch, timeout=None):
        self.calls.append((instance, model_patch))
        return self.outcome


def _swebench_issue() -> Issue:
    return Issue(
        id="pkg__repo-1",
        description="add() should add",
        repo_name="pkg/repo",
        metadata={"fail_to_pass": '["t::test_add"]', "pass_to_pass": '["t::test_keep"]'},
    )


def _verify(tmp_path, outcome: EvalOutcome):
    env = ProjectEnvironment(_init_repo(tmp_path))
    backend = FakeBackend(outcome)
    context = ExecutionContext(issue=_swebench_issue(), repo_state=env.get_repo_state())
    result = Verifier(env, test_backend=backend).execute(context, PatchResult(patch_content=RAW_PATCH))
    return env, backend, result


def test_backend_resolved_maps_to_success(tmp_path):
    outcome = EvalOutcome(
        f2p_passed=1, f2p_total=1, p2p_passed=1, p2p_total=1,
        resolved=True, per_test={"t::test_add": "PASSED", "t::test_keep": "PASSED"},
        log_tail="ok",
    )
    env, backend, result = _verify(tmp_path, outcome)

    assert result.success
    assert result.data.status == VerificationStatus.SUCCESS
    assert result.data.eval_outcome is outcome
    # The backend grades the canonical git diff, not the raw LLM patch.
    graded_patch = backend.calls[0][1]
    assert graded_patch.startswith("diff --git")
    assert result.data.canonical_patch_content.startswith("diff --git")
    # Host apply is reverted before grading, so the repo is left clean.
    assert env.get_diff() == ""


def test_backend_p2p_regression_maps_to_new_issues(tmp_path):
    outcome = EvalOutcome(
        f2p_passed=1, f2p_total=1, p2p_passed=0, p2p_total=1,
        resolved=False, per_test={"t::test_add": "PASSED", "t::test_keep": "FAILED"},
        log_tail="regression",
    )
    _env, _backend, result = _verify(tmp_path, outcome)

    assert not result.success
    assert result.data.status == VerificationStatus.NEW_ISSUES
    assert result.data.new_issues_introduced is True


def test_backend_incomplete_f2p_maps_to_tests_failed(tmp_path):
    outcome = EvalOutcome(
        f2p_passed=0, f2p_total=1, p2p_passed=1, p2p_total=1,
        resolved=False, per_test={"t::test_add": "FAILED", "t::test_keep": "PASSED"},
        log_tail="still failing",
    )
    _env, _backend, result = _verify(tmp_path, outcome)

    assert not result.success
    assert result.data.status == VerificationStatus.TESTS_FAILED
    assert result.data.new_issues_introduced is False


def test_no_backend_falls_back_to_host_and_does_not_fake_regression(tmp_path):
    # No swebench metadata, no backend: a failing host run is TESTS_FAILED, never NEW_ISSUES.
    env = ProjectEnvironment(
        _init_repo(tmp_path),
        test_cmd='python -c "import calc; assert calc.add(1, 2) == 99"',
    )
    context = ExecutionContext(
        issue=Issue(id="local-1", description="add"), repo_state=env.get_repo_state()
    )
    result = Verifier(env).execute(context, PatchResult(patch_content=RAW_PATCH))

    assert not result.success
    assert result.data.status == VerificationStatus.TESTS_FAILED


def test_backend_skipped_when_issue_has_no_fail_to_pass(tmp_path):
    # A backend is present but the issue is not a SWE-bench instance -> host path.
    env = ProjectEnvironment(
        _init_repo(tmp_path),
        test_cmd='python -c "import calc; assert calc.add(1, 2) == 3"',
    )
    backend = FakeBackend(
        EvalOutcome(0, 0, 0, 0, resolved=False, per_test={}, log_tail="")
    )
    context = ExecutionContext(
        issue=Issue(id="local-1", description="add"), repo_state=env.get_repo_state()
    )
    result = Verifier(env, test_backend=backend).execute(
        context, PatchResult(patch_content=RAW_PATCH)
    )

    assert result.success
    assert result.data.status == VerificationStatus.SUCCESS
    assert backend.calls == []  # backend never consulted for a non-swebench issue
