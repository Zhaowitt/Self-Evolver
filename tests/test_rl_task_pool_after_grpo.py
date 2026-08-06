"""GRPO round outcomes feed the persistent TaskPool at a frozen boundary."""

import json

from src.rl.update_task_pool_after_grpo import (
    read_round_rollouts,
    update_task_pool_after_grpo,
)
from src.tasks.task_pool import TaskPool


def _instance(instance_id):
    return {
        "instance_id": instance_id,
        "repo": "org/repo",
        "problem_statement": "Traceback: failure",
        "FAIL_TO_PASS": '["tests/test_x.py::test_a"]',
        "PASS_TO_PASS": '[]',
    }


def _record(instance_id, stage="train", resolved=False, reward=0.1):
    return {
        "rollout_id": f"rollout-{instance_id}-{stage}-{reward}",
        "instance_id": instance_id,
        "stage": stage,
        "eval_outcome": {"resolved": resolved},
        "reward": {"total": reward},
    }


def test_read_round_rollouts_uses_offset_and_excludes_eval(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(_record("old")) + "\n", encoding="utf-8")
    offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_record("new")) + "\n")
        stream.write(json.dumps(_record("eval", stage="eval")) + "\n")
    assert [item["instance_id"] for item in read_round_rollouts(path, offset)] == ["new"]


def test_round_outcomes_update_decay_and_preserve_full_base_pool(tmp_path):
    state = tmp_path / "task_pool.json"
    pool = TaskPool.from_instances(
        [_instance("org__a"), _instance("org__b")], state_path=state, verifier=None
    )
    pool.save()
    rollouts = tmp_path / "rollouts.jsonl"
    records = [_record("org__a", reward=0.1 + index / 100) for index in range(3)]
    rollouts.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    # The fallback file may be weighted and omit org__b; persisted base_instances
    # must keep the complete source pool intact.
    train = tmp_path / "train.jsonl"
    train.write_text("", encoding="utf-8")

    result = update_task_pool_after_grpo(state, train, rollouts, 0, "round-1")
    payload = json.loads(state.read_text(encoding="utf-8"))

    assert result["applied_outcomes"] == 3
    assert set(payload["base_instances"]) == {"org__a", "org__b"}
    assert payload["instances"]["org__a"]["attempts"] == 3
    assert payload["instances"]["org__a"]["decay_multiplier"] == 0.3

    repeated = update_task_pool_after_grpo(state, train, rollouts, 0, "round-1")
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert repeated["applied_outcomes"] == 0
    assert repeated["duplicate_outcomes"] == 3
    assert payload["instances"]["org__a"]["attempts"] == 3
