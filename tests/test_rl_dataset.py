"""EasyR1 dataset builder: split validation, contamination exclusion, prompt contract."""

import json
from collections import Counter

import pytest

from src.rl.easyr1_dataset import (
    DATASET_SPLITS,
    build_easyr1_prompt_record,
    issue_from_swebench_item,
    issues_from_rows,
    iter_swebench_issues,
    load_task_signals,
    read_id_file,
    sample_issues_with_task_pool,
    validate_split,
    write_swebench_prompt_dataset,
)
import src.rl.easyr1_dataset as dataset_module


def _row(instance_id: str) -> dict:
    return {
        "instance_id": instance_id,
        "problem_statement": f"Bug in {instance_id}",
        "repo": "octo/widgets",
        "base_commit": "abc123",
        "version": "1.0",
        "environment_setup_commit": "def456",
        "FAIL_TO_PASS": '["tests/test_x.py::test_a"]',
        "PASS_TO_PASS": '["tests/test_x.py::test_b"]',
    }


def test_lite_has_no_train_split():
    with pytest.raises(ValueError) as excinfo:
        validate_split("lite", "train")
    message = str(excinfo.value)
    assert "dev, test" in message
    assert "--dataset full --split train" in message


def test_verified_only_ships_test_split():
    with pytest.raises(ValueError, match="available splits: test"):
        validate_split("verified", "train")
    validate_split("verified", "test")


def test_full_train_is_valid_and_raw_hf_names_pass_through():
    validate_split("full", "train")
    validate_split("some-org/custom-dataset", "train")  # unknown -> hub decides
    assert set(DATASET_SPLITS) == {"lite", "verified", "full"}


def test_iter_swebench_issues_validates_before_loading():
    # Must raise eagerly, before touching local files or the HF hub.
    with pytest.raises(ValueError, match="no 'train' split"):
        iter_swebench_issues(dataset="lite", split="train")


def test_exclude_ids_filters_before_limit():
    rows = [_row("a__1"), _row("b__2"), _row("c__3")]
    issues = list(issues_from_rows(rows, limit=2, exclude_ids={"a__1"}))
    assert [issue.id for issue in issues] == ["b__2", "c__3"]


def test_include_ids_filters_before_limit():
    rows = [_row("a__1"), _row("b__2"), _row("c__3")]
    issues = list(issues_from_rows(rows, limit=1, include_ids={"b__2", "c__3"}))
    assert [issue.id for issue in issues] == ["b__2"]


def test_read_id_file_line_and_json_formats(tmp_path):
    lines = tmp_path / "ids.txt"
    lines.write_text("# eval ids\na__1\n\nb__2\n", encoding="utf-8")
    assert read_id_file(lines) == {"a__1", "b__2"}

    listing = tmp_path / "ids.json"
    listing.write_text('["a__1", "b__2"]', encoding="utf-8")
    assert read_id_file(listing) == {"a__1", "b__2"}


def test_issue_carries_environment_setup_commit():
    issue = issue_from_swebench_item(_row("a__1"))
    assert issue.metadata["environment_setup_commit"] == "def456"
    assert issue.metadata["fail_to_pass"] == '["tests/test_x.py::test_a"]'


def test_prompt_record_keeps_easyr1_contract():
    issue = issue_from_swebench_item(_row("octo__widgets-7"))
    record = build_easyr1_prompt_record(issue, stage="train", split="train", hard_cases=[])

    assert set(record) == {"prompt", "ground_truth", "extra_info"}
    messages = json.loads(record["prompt"])
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "octo__widgets-7" in messages[1]["content"]

    ground_truth = json.loads(record["ground_truth"])
    assert ground_truth["instance_id"] == "octo__widgets-7"
    assert ground_truth["FAIL_TO_PASS"] == '["tests/test_x.py::test_a"]'
    assert ground_truth["environment_setup_commit"] == "def456"
    assert ground_truth["stage"] == "train"
    assert ground_truth["split"] == "train"

    extra_info = record["extra_info"]
    assert ground_truth["allowed_skill_ids"] == extra_info["allowed_skill_ids"]
    assert len(ground_truth["allowed_skill_ids"]) <= 2
    assert extra_info["stage"] == "train"
    assert extra_info["split"] == "train"
    assert "tests/test_x.py::test_a" in extra_info["test_cmd"]


def test_load_task_signals_from_reflection_report(tmp_path):
    report = tmp_path / "reflection.json"
    report.write_text(
        json.dumps({"reflection": {"task_signals": {
            "instance_boosts": {"a__1": 2}, "family_boosts": {}
        }}}),
        encoding="utf-8",
    )
    assert load_task_signals(report)["instance_boosts"] == {"a__1": 2}


def test_task_pool_materializes_fixed_size_deterministic_round(tmp_path):
    issues = [issue_from_swebench_item(_row(f"repo__{index}")) for index in range(8)]
    signals = {"instance_boosts": {"repo__0": 3}, "family_boosts": {}}
    first = sample_issues_with_task_pool(
        issues, signals, tmp_path / "first.json", seed=7
    )
    second = sample_issues_with_task_pool(
        issues, signals, tmp_path / "second.json", seed=7
    )
    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == len(issues)


def test_task_pool_mixed_sample_is_repo_balanced_and_repeat_bounded(tmp_path):
    issues = []
    for repo in ("alpha", "beta", "gamma", "delta"):
        for index in range(4):
            row = _row(f"{repo}__{index}")
            row["repo"] = f"org/{repo}"
            issues.append(issue_from_swebench_item(row))

    sampled = sample_issues_with_task_pool(
        issues,
        {"instance_boosts": {"alpha__0": 99}, "family_boosts": {}},
        tmp_path / "pool.json",
        seed=11,
        num_samples=24,
        uniform_fraction=0.5,
        max_instance_repeats=2,
    )
    counts = Counter(item.id for item in sampled)
    assert len(sampled) == 24
    assert max(counts.values()) <= 2
    assert {item.repo_name for item in sampled} == {
        "org/alpha", "org/beta", "org/gamma", "org/delta"
    }


def test_task_pool_loads_complete_candidate_split_before_sampling(tmp_path, monkeypatch):
    issues = [issue_from_swebench_item(_row(f"repo__{index}")) for index in range(6)]
    observed_limits = []

    def fake_iter(*, dataset, split, limit, exclude_ids, include_ids):
        observed_limits.append(limit)
        return iter(issues)

    monkeypatch.setattr(dataset_module, "iter_swebench_issues", fake_iter)
    monkeypatch.setattr(
        dataset_module,
        "build_easyr1_prompt_record",
        lambda issue, **kwargs: {"prompt": issue.id, "ground_truth": "{}", "extra_info": {}},
    )
    count = write_swebench_prompt_dataset(
        tmp_path / "train.jsonl",
        num_instances=3,
        task_signals={"instance_boosts": {}, "family_boosts": {}},
        task_pool_state_path=tmp_path / "pool.json",
    )
    assert observed_limits == [None]
    assert count == 3
