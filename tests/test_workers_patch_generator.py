import json

from src.workers.patch_generator import parse_patch_response


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
