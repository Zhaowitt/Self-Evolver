import json

from src.tasks.families import (
    CONFIG_OR_DEPENDENCY_FIX,
    MULTI_FILE_CONSISTENCY_FIX,
    SINGLE_FILE_BUG_FIX,
    TASK_FAMILIES,
    TEST_ALIGNED_BEHAVIOR_FIX,
    changed_paths,
    classify_family,
)

TRACEBACK_TEXT = (
    "Calling frobnicate() crashes:\n"
    "Traceback (most recent call last):\n"
    "  File \"app.py\", line 3\n"
    "TypeError: unsupported operand"
)
BEHAVIOR_TEXT = (
    "Series.combine should return the fill value instead of NaN. "
    "The expected behavior is documented; assert combine(a, b) == 0."
)
CONFIG_TEXT = (
    "pip install fails with a version conflict in the requirements; "
    "the dependencies in setup.py pin an incompatible packaging release."
)


def _patch(paths):
    return "".join(
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        for path in paths
    )


def make_instance(files, statement, instance_id="org__proj-1"):
    return {
        "instance_id": instance_id,
        "repo": "org/proj",
        "base_commit": "deadbeef",
        "patch": _patch(files),
        "problem_statement": statement,
        "FAIL_TO_PASS": json.dumps(["tests/test_app.py::test_a"]),
    }


def test_single_file_traceback_is_single_file_bug_fix():
    instance = make_instance(["src/app.py"], TRACEBACK_TEXT)
    assert classify_family(instance) == SINGLE_FILE_BUG_FIX


def test_multiple_code_files_is_multi_file_consistency_fix():
    instance = make_instance(["src/app.py", "src/core.py", "src/util.py"], TRACEBACK_TEXT)
    assert classify_family(instance) == MULTI_FILE_CONSISTENCY_FIX


def test_pure_config_patch_is_config_fix():
    instance = make_instance(["setup.py"], TRACEBACK_TEXT)
    assert classify_family(instance) == CONFIG_OR_DEPENDENCY_FIX


def test_mixed_paths_with_config_keywords_is_config_fix():
    instance = make_instance(["requirements.txt", "src/app.py"], CONFIG_TEXT)
    assert classify_family(instance) == CONFIG_OR_DEPENDENCY_FIX


def test_mixed_paths_without_config_keywords_is_multi_file():
    instance = make_instance(["requirements.txt", "src/app.py"], TRACEBACK_TEXT)
    assert classify_family(instance) == MULTI_FILE_CONSISTENCY_FIX


def test_behavior_expectation_wording_is_test_aligned():
    instance = make_instance(["src/series.py"], BEHAVIOR_TEXT)
    assert classify_family(instance) == TEST_ALIGNED_BEHAVIOR_FIX


def test_no_patch_with_strong_config_text_is_config_fix():
    instance = make_instance([], CONFIG_TEXT)
    assert classify_family(instance) == CONFIG_OR_DEPENDENCY_FIX


def test_no_patch_no_keywords_defaults_to_single_file():
    instance = make_instance([], "Something is off in the output format.")
    assert classify_family(instance) == SINGLE_FILE_BUG_FIX


def test_test_paths_are_excluded_from_file_counts():
    instance = make_instance(["src/app.py", "tests/test_app.py"], TRACEBACK_TEXT)
    assert classify_family(instance) == SINGLE_FILE_BUG_FIX


def test_changed_paths_parses_diff_git_headers_uniquely():
    patch = _patch(["a.py", "b.py", "a.py"])
    assert changed_paths(patch) == ["a.py", "b.py"]


def test_changed_paths_falls_back_to_plus_headers():
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    assert changed_paths(patch) == ["x.py"]


def test_every_classification_is_a_known_family():
    for files, text in [
        (["src/app.py"], TRACEBACK_TEXT),
        (["setup.cfg"], ""),
        ([], ""),
        (["a.py", "b.py"], BEHAVIOR_TEXT),
    ]:
        assert classify_family(make_instance(files, text)) in TASK_FAMILIES
