import subprocess
from pathlib import Path

import pytest

from src.environment.models import ExecutionContext, Issue
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMResponse
from src.workers.inspector import Inspector


def _run(cmd, cwd: Path):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "calc.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n",
        encoding="utf-8",
    )
    _run(["git", "add", "pkg/calc.py"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


def test_inspector_repo_tools_are_read_only_and_repo_scoped(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    inspector = Inspector(env, llm_client=None)

    read_output = inspector._tool_read_file("pkg/calc.py", start_line=1, end_line=2)
    listing = inspector._tool_list_dir("pkg")
    matches = inspector._tool_grep_repo("return a - b", path="pkg", glob="*.py")

    assert "FILE: pkg/calc.py lines 1-2" in read_output
    assert "     2:     return a - b" in read_output
    assert "file\tpkg/calc.py" in listing
    assert "pkg/calc.py:2:" in matches
    assert "return a - b" in matches

    for path in ("../outside.py", "/tmp/outside.py", ".git/config"):
        with pytest.raises(ValueError):
            inspector._tool_read_file(path)


class FakeToolLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if self.calls == 1:
            return LLMResponse(
                content="",
                model="fake",
                usage=usage,
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_grep",
                        "type": "function",
                        "function": {
                            "name": "grep_repo",
                            "arguments": '{"pattern":"return a - b","path":"."}',
                        },
                    }
                ],
            )
        if self.calls == 2:
            return LLMResponse(
                content="",
                model="fake",
                usage=usage,
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": (
                                '{"path":"pkg/calc.py","start_line":1,"end_line":2}'
                            ),
                        },
                    }
                ],
            )
        return LLMResponse(
            content=(
                "```json\n"
                "{\n"
                '  "suspected_files": ["pkg/calc.py"],\n'
                '  "suspected_locations": [\n'
                "    {\n"
                '      "file_path": "pkg/calc.py",\n'
                '      "start_line": 2,\n'
                '      "end_line": 2,\n'
                '      "reason": "The implementation subtracts instead of adding."\n'
                "    }\n"
                "  ],\n"
                '  "root_cause_analysis": "add() returns subtraction.",\n'
                '  "fix_suggestions": ["Return a + b."],\n'
                '  "confidence": 0.9\n'
                "}\n"
                "```"
            ),
            model="fake",
            usage=usage,
            finish_reason="stop",
        )


def test_inspector_tool_loop_builds_verified_inspection_result(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    llm = FakeToolLLM()
    inspector = Inspector(env, llm_client=llm)
    context = ExecutionContext(
        issue=Issue(id="local-1", description="add() should add, not subtract"),
        repo_state=env.get_repo_state(),
    )

    result = inspector.execute(context)

    assert result.success
    assert result.data is not None
    assert result.data.suspected_files == ["pkg/calc.py"]
    assert result.data.suspected_locations[0].start_line == 2
    assert result.data.relevant_code_snippets
    assert len(result.metadata["tool_trace"]) == 2
    assert result.llm_response is not None
    assert result.llm_response.total_tokens == 6
