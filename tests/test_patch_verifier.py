import subprocess
import sys
from pathlib import Path

from src.environment.models import ExecutionContext, Issue
from src.environment.project_env import ProjectEnvironment
from src.workers.patch_generator import PatchResult
from src.workers.verifier import Verifier


def _run(cmd, cwd: Path):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_calc.py").write_text("", encoding="utf-8")
    _run(["git", "add", "calc.py", "test_calc.py"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


def test_fix_hunk_counts_recomputes_header():
    patch = (
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -1,99 +1,99 @@\n"
        " def add(a, b):\n"
        "-    return a - b\n"
        "+    return a + b\n"
    )

    fixed = ProjectEnvironment._fix_hunk_counts(patch)

    assert "@@ -1,2 +1,2 @@" in fixed


def test_verifier_returns_canonical_diff_not_raw_patch(tmp_path):
    repo = _init_repo(tmp_path)
    test_cmd = (
        f'"{sys.executable}" -c '
        '"import calc; assert calc.add(1, 2) == 3"'
    )
    env = ProjectEnvironment(repo, test_cmd=test_cmd)
    context = ExecutionContext(issue=Issue(id="local-1", description="Fix add"), repo_state=env.get_repo_state())
    raw_patch = (
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -1,99 +1,99 @@\n"
        " def add(a, b):\n"
        "-    return a - b\n"
        "+    return a + b\n"
    )

    result = Verifier(env).execute(context, PatchResult(patch_content=raw_patch))

    assert result.success
    assert result.data is not None
    assert result.data.canonical_patch_content
    assert result.data.canonical_patch_content != raw_patch
    assert result.data.canonical_patch_content.startswith("diff --git")
    assert result.data.canonical_patch_info is not None


def test_setup_issue_stages_swebench_test_patch_out_of_canonical_diff(tmp_path):
    repo = _init_repo(tmp_path)
    env = ProjectEnvironment(repo)
    test_patch = (
        "--- a/test_calc.py\n"
        "+++ b/test_calc.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+import calc\n"
        "+assert calc.add(1, 2) == 3\n"
    )

    assert env.setup_issue(Issue(id="local-2", description="Add hidden test", test_patch=test_patch))
    assert env.get_diff() == ""
