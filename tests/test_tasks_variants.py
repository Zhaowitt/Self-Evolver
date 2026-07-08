import json

from src.tasks.variants import (
    base_instance_id,
    fail_to_pass,
    is_focused_variant,
    make_focused_variant,
)

F2P = ["tests/test_app.py::test_a", "tests/test_app.py::test_b"]


def make_instance(f2p=None, as_json=True, instance_id="org__proj-1"):
    tests = F2P if f2p is None else f2p
    return {
        "instance_id": instance_id,
        "repo": "org/proj",
        "base_commit": "deadbeef",
        "patch": "diff --git a/app.py b/app.py\n",
        "problem_statement": "The app crashes on empty input.",
        "FAIL_TO_PASS": json.dumps(tests) if as_json else list(tests),
        "PASS_TO_PASS": json.dumps(["tests/test_app.py::test_ok"]),
    }


def test_fail_to_pass_parses_json_string_and_list():
    assert fail_to_pass(make_instance()) == F2P
    assert fail_to_pass(make_instance(as_json=False)) == F2P
    assert fail_to_pass(make_instance(f2p=[])) == []
    assert fail_to_pass({"FAIL_TO_PASS": "tests/a\ntests/b"}) == ["tests/a", "tests/b"]
    assert fail_to_pass({}) == []


def test_focused_variant_subsets_f2p_to_one_test():
    parent = make_instance()
    variant = make_focused_variant(parent)

    assert variant is not None
    assert variant["instance_id"] == "org__proj-1::focus-1"
    assert variant["base_instance_id"] == "org__proj-1"
    assert json.loads(variant["FAIL_TO_PASS"]) == [F2P[0]]
    assert variant["PASS_TO_PASS"] == parent["PASS_TO_PASS"]
    assert variant["repo"] == parent["repo"]
    assert is_focused_variant(variant)
    assert base_instance_id(variant) == "org__proj-1"


def test_focused_variant_augments_problem_statement():
    variant = make_focused_variant(make_instance())
    statement = variant["problem_statement"]

    assert statement.startswith("The app crashes on empty input.")
    assert "[Focused variant of org__proj-1]" in statement
    assert F2P[0] in statement


def test_focused_variant_does_not_mutate_parent():
    parent = make_instance()
    before = dict(parent)
    make_focused_variant(parent)
    assert parent == before


def test_focused_variant_second_test_gets_focus_2():
    variant = make_focused_variant(make_instance(), n=2)
    assert variant["instance_id"] == "org__proj-1::focus-2"
    assert json.loads(variant["FAIL_TO_PASS"]) == [F2P[1]]


def test_focused_variant_preserves_list_encoding():
    variant = make_focused_variant(make_instance(as_json=False))
    assert variant["FAIL_TO_PASS"] == [F2P[0]]


def test_focused_variant_impossible_cases_return_none():
    assert make_focused_variant(make_instance(f2p=[])) is None
    assert make_focused_variant(make_instance(), n=0) is None
    assert make_focused_variant(make_instance(), n=3) is None
    assert make_focused_variant({"problem_statement": "no id"}) is None

    already_focused = make_focused_variant(make_instance())
    assert make_focused_variant(already_focused) is None
