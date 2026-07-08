"""Local-first dataset loading, split validation, casing tolerance."""

import pytest

from src.benchmark import datasets


def test_registry_hf_names():
    assert datasets.get_spec("lite").hf_name == "princeton-nlp/SWE-bench_Lite"
    assert datasets.get_spec("verified").hf_name == "princeton-nlp/SWE-bench_Verified"
    assert datasets.get_spec("full").hf_name == "princeton-nlp/SWE-bench"
    assert datasets.get_spec("swebench_live").hf_name == "SWE-bench-Live/SWE-bench-Live"
    assert datasets.get_spec("swebench_pro").hf_name == "ScaleAI/SWE-bench_Pro"


def test_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.get_spec("nope")


def test_available_splits():
    assert datasets.available_splits("verified") == ("test",)
    assert datasets.available_splits("full") == ("train", "dev", "test")
    assert set(datasets.available_splits("swebench_live")) == {"test", "lite", "verified", "full"}


def test_validate_split_lists_available_and_hints_train():
    with pytest.raises(ValueError) as exc:
        datasets.validate_split("verified", "train")
    message = str(exc.value)
    assert "available splits: test" in message
    assert "--dataset full --split train" in message
    with pytest.raises(ValueError, match="available splits: test"):
        datasets.validate_split("swebench_pro", "dev")


def test_load_split_cache_keyed_by_split():
    lite_test = datasets.load_split("lite", "test")
    assert lite_test is datasets.load_split("lite", "test")       # cached
    assert datasets.load_split("lite", "dev") is not lite_test    # keyed by split
    assert len(lite_test) == 300


def test_iter_rows_limit_and_exclude():
    rows = list(datasets.iter_rows("lite", "test", limit=3))
    assert len(rows) == 3
    first_id = rows[0]["instance_id"]
    excluded = list(datasets.iter_rows("lite", "test", limit=3, exclude_ids={first_id}))
    assert first_id not in {row["instance_id"] for row in excluded}


def test_row_to_issue_uppercase_schema_and_hint_gating():
    row = {
        "instance_id": "org__repo-1",
        "problem_statement": "boom",
        "repo": "org/repo",
        "base_commit": "abc",
        "version": "1.2",
        "hints_text": "do this",
        "FAIL_TO_PASS": ["t::a"],
        "PASS_TO_PASS": ["t::b"],
    }
    issue = datasets.row_to_issue(row, use_hints=False)
    assert issue.hints is None
    assert issue.metadata["fail_to_pass"] == ["t::a"]
    assert issue.metadata["version"] == "1.2"
    assert datasets.row_to_issue(row, use_hints=True).hints == "do this"


def test_row_to_issue_lowercase_and_extra_fields():
    row = {
        "instance_id": "instance_x",
        "problem_statement": "boom",
        "repo": "NodeBB/NodeBB",
        "base_commit": "abc",
        "fail_to_pass": '["t | a"]',
        "pass_to_pass": '["t | b"]',
        "dockerhub_tag": "nodebb.tag",
        "test_cmds": ["pytest -rA"],
    }
    issue = datasets.row_to_issue(row)
    assert issue.metadata["fail_to_pass"] == '["t | a"]'
    assert issue.metadata["dockerhub_tag"] == "nodebb.tag"
    assert issue.metadata["test_cmds"] == ["pytest -rA"]


def test_swebench_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SWEBENCH_DATA_DIR", str(tmp_path))
    assert datasets.swebench_data_dir() == tmp_path
    monkeypatch.delenv("SWEBENCH_DATA_DIR")
    assert datasets.swebench_data_dir().name == "benchmarks"
