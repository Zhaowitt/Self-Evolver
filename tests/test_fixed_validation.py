"""CPU-only regressions for the production Full/dev holdout migration."""

import json
from pathlib import Path

import pytest

from src.rl import fixed_validation as validation


def source_rows():
    return [{"instance_id": f"dev__repo-{i}", "repo": "dev/repo",
             "base_commit": "abc", "problem_statement": f"Issue {i}",
             "test_patch": "diff", "FAIL_TO_PASS": '["test_a"]',
             "PASS_TO_PASS": '["test_b"]'} for i in range(225)]


def records():
    result = []
    for row in source_rows():
        extra = {"instance_id": row["instance_id"], "stage": "eval", "split": "dev"}
        truth = {**extra, "extra_info": extra, "repo_name": row["repo"],
                 **{k: row[k] for k in ("base_commit", "problem_statement", "test_patch",
                                        "FAIL_TO_PASS", "PASS_TO_PASS")}}
        result.append({"prompt": "prompt", "ground_truth": json.dumps(truth),
                       "extra_info": extra})
    return result


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, return_dict):
        assert messages[0]["role"] == "user"
        assert add_generation_prompt
        assert return_dict is False
        return list(range(10))


def test_exactly_all_225_records_required():
    validation.check_records(records(), source_rows())
    with pytest.raises(ValueError, match="exactly all 225"):
        validation.check_records(records()[:18], source_rows())
    duplicate = records()
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="exactly all 225"):
        validation.check_records(duplicate, source_rows())


@pytest.mark.parametrize("location,field,value", [
    ("extra", "stage", "train"), ("truth", "stage", "train"),
    ("nested", "stage", "train"), ("truth", "split", "test"),
    ("truth", "instance_id", "wrong"), ("truth", "base_commit", "wrong"),
    ("truth", "FAIL_TO_PASS", "wrong"),
])
def test_validation_payload_cannot_be_training_or_wrong_task(location, field, value):
    rows = records()
    truth = json.loads(rows[0]["ground_truth"])
    target = {"extra": rows[0]["extra_info"], "truth": truth,
              "nested": truth["extra_info"]}[location]
    target[field] = value
    rows[0]["ground_truth"] = json.dumps(truth)
    with pytest.raises(ValueError):
        validation.check_records(rows, source_rows())


def test_overlong_task_fails_instead_of_silently_dropping():
    assert validation.check_prompt_lengths(records(), Tokenizer(), 10) == 10
    with pytest.raises(ValueError, match="would be filtered"):
        validation.check_prompt_lengths(records(), Tokenizer(), 9)


@pytest.mark.parametrize("relative", [
    "data/controller_train.jsonl", "rounds/r1/controller_train.jsonl",
    "runs/r1/online_rollouts.jsonl", "online_rollouts.jsonl",
])
def test_detects_overlap_in_frozen_data_and_actual_training(tmp_path, relative):
    write_jsonl(tmp_path / relative, [{"instance_id": "dev__repo-0", "stage": "train"}])
    with pytest.raises(ValueError, match="overlaps"):
        validation.check_training({"dev__repo-0"}, tmp_path)


def test_old_eval_records_do_not_count_as_training_overlap(tmp_path):
    write_jsonl(tmp_path / "online_rollouts.jsonl",
                [{"instance_id": "dev__repo-0", "stage": "eval"}])
    validation.check_training({"dev__repo-0"}, tmp_path)


def test_sft_manifest_covers_train_and_validation(tmp_path):
    manifest = tmp_path / "sft.json"
    manifest.write_text(json.dumps({"instance_ids": ["train"], "val_instance_ids": ["val"]}))
    assert validation.sft_ids(manifest) == {"train", "val"}
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="SFT IDs"):
        validation.sft_ids(manifest)


def test_resume_reuses_prompts_without_building_or_touching_training(tmp_path):
    output = tmp_path / "new_validation.jsonl"
    training = tmp_path / "controller_train.jsonl"
    training.write_text("frozen training data")
    original = training.read_bytes()
    write_jsonl(output, records())
    output.with_suffix(".manifest.json").write_text(json.dumps({"sha256": validation.digest(output)}))
    before = output.stat().st_mtime_ns
    # Reuse must not need a live skill bank or rebuild records.
    result = validation.ensure_prompts(output, source_rows(), tmp_path / "absent",
                                       Tokenizer(), 100)
    assert result["reused"]
    assert output.stat().st_mtime_ns == before
    assert training.read_bytes() == original
    output.write_text(output.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        validation.ensure_prompts(output, source_rows(), tmp_path / "absent", Tokenizer(), 100)


def test_prepare_pins_official_source_and_reuses_without_hub(tmp_path, monkeypatch):
    from src.benchmark import datasets

    sft = tmp_path / "sft.json"
    sft.write_text(json.dumps({"instance_ids": ["sft__0"], "val_instance_ids": []}))
    calls = []

    def load(dataset, split):
        calls.append((dataset, split))
        return source_rows() if split == "dev" else [{"instance_id": "train__0"}]

    monkeypatch.setattr(datasets, "load_split", load)
    directory = tmp_path / "source"
    validation.prepare_source(directory, tmp_path / "state", sft)
    assert calls == [("full", "dev"), ("full", "test")]
    calls.clear()
    assert len(validation.prepare_source(directory, tmp_path / "state", sft)) == 225
    assert not calls
    (directory / "instance_ids.txt").write_text("wrong\n")
    with pytest.raises(ValueError, match="changed"):
        validation.load_source(directory)


def test_prepare_rejects_sft_overlap_before_writing(tmp_path, monkeypatch):
    from src.benchmark import datasets

    monkeypatch.setattr(datasets, "load_split", lambda *args: source_rows())
    sft = tmp_path / "sft.json"
    sft.write_text(json.dumps({"instance_ids": ["dev__repo-0"]}))
    directory = tmp_path / "source"
    with pytest.raises(ValueError, match="overlaps SFT"):
        validation.prepare_source(directory, tmp_path, sft)
    assert not directory.exists()


def test_new_validation_keeps_old_18_task_file(tmp_path):
    old = tmp_path / "controller_val.jsonl"
    old.write_text("historical validation")
    root = Path(__file__).resolve().parents[1]
    new = tmp_path / "controller_val_full_dev225.jsonl"
    result = validation.ensure_prompts(new, source_rows(), root / "skills", Tokenizer(), 100)
    assert result["count"] == 225
    assert not result["reused"]
    assert old.read_text() == "historical validation"
    validation.check_records(validation.read_jsonl(new), source_rows())
