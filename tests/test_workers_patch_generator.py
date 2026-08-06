import json

import pytest

from src.environment.models import ExecutionContext, Issue, RepoState
from src.llm.client import LLMResponse
from src.workers.patch_generator import EditOperation, PatchGenerator, parse_patch_response


def test_parse_patch_from_bare_json_object():
    patch_text = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    content = json.dumps(
        {
            "explanation": "Replace a with b.",
            "modified_files": ["f.py"],
            "patch": patch_text,
        }
    )

    result = parse_patch_response(content)

    assert result.patch_content == patch_text
    assert result.modified_files == ["f.py"]
    assert result.explanation == "Replace a with b."


def test_parse_patch_preserves_literal_backslash_n_in_multiline_diff():
    # A proper multi-line diff whose added code legitimately contains "\n".
    patch_text = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def f():\n"
        "     x = 1\n"
        '+    print("\\n")\n'
    )
    content = "```json\n" + json.dumps({"patch": patch_text, "modified_files": ["f.py"]}) + "\n```"

    result = parse_patch_response(content)

    # The backslash-n inside print("\n") must survive, not become a real newline.
    assert '+    print("\\n")' in result.patch_content
    assert result.patch_content == patch_text
    assert result.modified_files == ["f.py"]


def test_parse_patch_unescapes_single_line_diff():
    # Some models emit the whole diff on one line with literal "\n" separators.
    one_line = "--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-a\\n+b\\n"
    content = "```json\n" + json.dumps({"patch": one_line}) + "\n```"

    result = parse_patch_response(content)

    assert result.patch_content.startswith("--- a/f.py\n")
    assert "\\n" not in result.patch_content
    assert "-a\n+b" in result.patch_content


def test_parse_patch_from_diff_code_block():
    diff = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    content = f"Here is the fix:\n```diff\n{diff}```"

    result = parse_patch_response(content)

    assert result.patch_content.startswith("--- a/f.py")
    assert "f.py" in result.modified_files


def test_parse_patch_returns_empty_when_no_patch():
    result = parse_patch_response("I could not find the bug.")

    assert result.patch_content == ""
    assert "Failed to parse" in result.explanation


def test_parse_structured_edits_and_generate_programmatic_diff(tmp_path):
    source = tmp_path / "f.py"
    source.write_text("def f():\n    return 1\n", encoding="utf-8")

    class Env:
        repo_path = tmp_path

        @staticmethod
        def get_file_content(path):
            return (tmp_path / path).read_text(encoding="utf-8")

    result = parse_patch_response(json.dumps({
        "explanation": "fix return value",
        "edits": [{"path": "f.py", "old_text": "return 1", "new_text": "return 2"}],
    }))
    generator = PatchGenerator(Env(), llm_client=object())
    patch = generator._diff_from_edits(result.edits)

    assert "--- a/f.py" in patch
    assert "+++ b/f.py" in patch
    assert "-    return 1" in patch
    assert "+    return 2" in patch


def test_structured_edit_requires_unique_exact_match(tmp_path):
    (tmp_path / "f.py").write_text("x = 1\nx = 1\n", encoding="utf-8")

    class Env:
        repo_path = tmp_path

        @staticmethod
        def get_file_content(path):
            return (tmp_path / path).read_text(encoding="utf-8")

    generator = PatchGenerator(Env(), llm_client=object())
    with pytest.raises(ValueError, match="found 2"):
        generator._diff_from_edits([EditOperation("f.py", "x = 1", "x = 2")])


def test_structured_edit_accepts_one_whitespace_equivalent_match(tmp_path):
    (tmp_path / "f.py").write_text(
        "def f():\n    if True:\n        return 1  \n", encoding="utf-8"
    )

    class Env:
        repo_path = tmp_path

        @staticmethod
        def get_file_content(path):
            return (tmp_path / path).read_text(encoding="utf-8")

    generator = PatchGenerator(Env(), llm_client=object())
    patch = generator._diff_from_edits([
        EditOperation("f.py", "if True:\n    return 1", "if True:\n    return 2")
    ])

    assert "-        return 1" in patch
    assert "+        return 2" in patch


def test_structured_edit_rejects_ambiguous_whitespace_matches(tmp_path):
    (tmp_path / "f.py").write_text(
        "def f():\n    if True:\n        return 1\n\n"
        "def g():\n    if True:\n        return 1\n",
        encoding="utf-8",
    )

    class Env:
        repo_path = tmp_path

        @staticmethod
        def get_file_content(path):
            return (tmp_path / path).read_text(encoding="utf-8")

    generator = PatchGenerator(Env(), llm_client=object())
    with pytest.raises(ValueError, match="whitespace-equivalent matches"):
        generator._diff_from_edits([
            EditOperation("f.py", "if True:\n    return 1", "if True:\n    return 2")
        ])


def test_structured_edit_no_match_error_contains_repository_context(tmp_path):
    (tmp_path / "f.py").write_text(
        "def calculate(value):\n    return value + 1\n", encoding="utf-8"
    )

    class Env:
        repo_path = tmp_path

        @staticmethod
        def get_file_content(path):
            return (tmp_path / path).read_text(encoding="utf-8")

    generator = PatchGenerator(Env(), llm_client=object())
    with pytest.raises(ValueError, match="Closest repository context") as exc:
        generator._diff_from_edits([
            EditOperation("f.py", "return value - 1", "return value")
        ])

    assert "2:     return value + 1" in str(exc.value)


class ForcedReadPatchLLM:
    """Return too early once, then obey the forced read and emit grounded edits."""

    def __init__(self):
        self.calls = 0
        self.messages = []

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.messages.append(messages)
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if self.calls == 1:
            return LLMResponse(
                content='{"explanation":"guessed","edits":[]}',
                model="fake",
                usage=usage,
                finish_reason="stop",
            )
        if self.calls == 2:
            return LLMResponse(
                content="",
                model="fake",
                usage=usage,
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": (
                            '{"path":"pkg/calc.py","start_line":1,"end_line":2}'
                        ),
                    },
                }],
            )
        return LLMResponse(
            content=json.dumps({
                "explanation": "make add() add",
                "edits": [{
                    "path": "pkg/calc.py",
                    "old_text": "    return a - b",
                    "new_text": "    return a + b",
                }],
            }),
            model="fake",
            usage=usage,
            finish_reason="stop",
        )


def test_patch_generator_forces_read_file_before_grounded_edit(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    class Env:
        repo_path = tmp_path

        @staticmethod
        def get_file_content(path):
            return (tmp_path / path).read_text(encoding="utf-8")

    llm = ForcedReadPatchLLM()
    generator = PatchGenerator(Env(), llm_client=llm)
    context = ExecutionContext(
        issue=Issue(id="local-1", description="add() subtracts"),
        repo_state=RepoState(path=tmp_path),
    )

    result = generator.execute(context)

    assert result.success
    assert result.data is not None
    assert "-    return a - b" in result.data.patch_content
    assert "+    return a + b" in result.data.patch_content
    assert [item["name"] for item in result.metadata["tool_trace"]] == ["read_file"]
    assert result.llm_response is not None
    assert result.llm_response.total_tokens == 6
    assert any(
        "Before finalizing, call read_file" in message.content
        for message in llm.messages[1]
    )
    tool_message = next(
        message for message in llm.messages[-1] if message.role == "tool"
    )
    assert "EXACT CONTENT (copy verbatim):" in tool_message.content
    assert "    return a - b" in tool_message.content
    assert "     2:" not in tool_message.content
