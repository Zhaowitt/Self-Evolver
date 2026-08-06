"""GRPO round boundary and train/eval isolation for post-training reflection."""

import json

from src.memory.hard_case_buffer import HardCaseRecord
from src.rl.reflect_after_grpo import read_round_hard_cases


def _line(record: HardCaseRecord) -> str:
    return json.dumps(record.to_dict()) + "\n"


def test_read_round_hard_cases_uses_offset_and_excludes_eval(tmp_path):
    path = tmp_path / "hard_cases.jsonl"
    old = HardCaseRecord(issue_id="old", stage="train")
    path.write_text(_line(old), encoding="utf-8")
    offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_line(HardCaseRecord(issue_id="train-new", stage="train")))
        stream.write(_line(HardCaseRecord(issue_id="eval-new", stage="eval")))

    records = read_round_hard_cases(path, offset)

    assert [record.issue_id for record in records] == ["train-new"]
