import subprocess
from pathlib import Path

from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMResponse
from src.orchestrator.orchestrator import ExecutionStatus
from src.workers.single_agent import SingleAgent, run_single_agent


def _run(cmd, cwd: Path):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "unrelated.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


FIX_DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)


class FakeDiffLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0
        self.last_prompt = ""

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.last_prompt = messages[-1].content
        return LLMResponse(
            content=self.content,
            model="fake",
            usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            finish_reason="stop",
        )


def test_single_agent_canonicalizes_diff(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    llm = FakeDiffLLM(f"```diff\n{FIX_DIFF}```")
    issue = Issue(id="local-1", description="add() should add, not subtract")

    result = run_single_agent(issue, env, llm_client=llm)

    assert llm.calls == 1
    assert result.success
    assert result.status == ExecutionStatus.SUCCESS
    assert result.iterations_used == 1
    assert result.total_tokens == 7
    # The reported patch is the canonical git diff, not the raw LLM text.
    assert result.final_patch is not None
    assert result.final_patch.content.startswith("diff --git")
    assert "calc.py" in result.final_patch.modified_files
    # The repo is left clean after canonicalization (no retries, no lingering edits).
    assert env.get_diff() == ""


def test_single_agent_prompt_includes_relevant_file(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    llm = FakeDiffLLM(f"```diff\n{FIX_DIFF}```")
    issue = Issue(id="local-1", description="calc add function returns wrong result")

    run_single_agent(issue, env, llm_client=llm)

    # The lexical-overlap selection surfaces calc.py into the single prompt.
    assert "calc.py" in llm.last_prompt
    assert "def add(a, b)" in llm.last_prompt


def test_single_agent_no_patch_is_failed(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    llm = FakeDiffLLM("I could not determine a fix.")
    issue = Issue(id="local-1", description="mysterious bug")

    result = run_single_agent(issue, env, llm_client=llm)

    assert not result.success
    assert result.status == ExecutionStatus.FAILED
    assert result.final_patch is None


def test_single_agent_top_files_ranks_by_overlap(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    agent = SingleAgent(env, llm_client=FakeDiffLLM(""))

    files = env.list_files("**/*.py")
    top = agent._top_files("the calc add function is broken", files)

    assert top[0] == "calc.py"
