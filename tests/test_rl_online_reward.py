"""Online reward glue: flag parsing, failure scores, rollout record schema."""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from src.reward.online_reward import (
    _as_bool,
    _env_flag,
    _failure_score,
    issue_from_payload,
)
from src.reward.reward_model import DEFAULT_REWARD_WEIGHTS, RewardModel
from src.rl.rollout_writer import SCHEMA_VERSION, RolloutWriter, build_rollout_record
from src.environment.models import Issue

ENV_FLAG = "SELF_EVOLVER_ENABLE_SKILL_EVOLUTION"


@dataclass
class FakeExecution:
    success: bool = False
    total_tokens: int = 123
    iterations_used: int = 1
    status: Any = field(default_factory=lambda: SimpleNamespace(value="success"))
    final_patch: Any = None
    iteration_records: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def test_as_bool():
    assert _as_bool(True) is True
    assert _as_bool(False) is False
    for truthy in ("1", "true", "YES", " on "):
        assert _as_bool(truthy) is True
    for falsy in ("0", "false", "", None, "off"):
        assert _as_bool(falsy) is False


def test_env_flag_kwarg_false_beats_env(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    assert _env_flag(False, ENV_FLAG) is False
    assert _env_flag("false", ENV_FLAG) is False
    assert _env_flag(None, ENV_FLAG) is True
    assert _env_flag(True, ENV_FLAG) is True
    monkeypatch.delenv(ENV_FLAG)
    assert _env_flag(None, ENV_FLAG) is False


def test_failure_score_matches_utility_components():
    score = _failure_score(parse_valid=0.0, worker_executed=0.0)
    assert score["overall"] == 0.0
    assert set(score) == {"overall", "parse_valid", "worker_executed", *DEFAULT_REWARD_WEIGHTS}
    assert all(score[name] == 0.0 for name in DEFAULT_REWARD_WEIGHTS)


def test_issue_from_payload_requires_identity_and_keeps_setup_commit():
    with pytest.raises(ValueError):
        issue_from_payload({"instance_id": "x"})
    issue = issue_from_payload(
        {
            "instance_id": "octo__widgets-7",
            "problem_statement": "Bug",
            "repo_name": "octo/widgets",
            "environment_setup_commit": "def456",
            "FAIL_TO_PASS": '["t::a"]',
        }
    )
    assert issue.metadata["environment_setup_commit"] == "def456"
    assert issue.metadata["fail_to_pass"] == '["t::a"]'


def test_rollout_record_gains_stage_seed_arm_models_and_eval_outcome():
    issue = Issue(id="octo__widgets-7", description="Bug", repo_name="octo/widgets")
    signal = {"mode": "train", "strategy": "focus"}
    reward = RewardModel().score(FakeExecution())
    record = build_rollout_record(
        issue,
        signal,
        FakeExecution(),
        reward=reward,
        eval_outcome={"f2p_passed": 1, "f2p_total": 2, "p2p_passed": 3, "p2p_total": 3,
                      "resolved": False},
        stage="train",
        seed=7,
        experiment="full-method",
        models={"worker": "worker-model", "controller": "controller-model"},
    )
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["stage"] == "train"
    assert record["seed"] == 7
    assert record["experiment"] == "full-method"
    assert record["models"] == {"worker": "worker-model", "controller": "controller-model"}
    assert record["eval_outcome"] == {
        "f2p_passed": 1,
        "f2p_total": 2,
        "p2p_passed": 3,
        "p2p_total": 3,
        "resolved": False,
    }
    assert record["reward"]["evolution_utility"] == 0.0


def test_rollout_record_backward_defaults():
    issue = Issue(id="i-1", description="Bug")
    record = build_rollout_record(issue, {"mode": "eval"}, FakeExecution())
    assert record["stage"] == "eval"  # falls back to the signal mode
    assert record["seed"] is None
    assert record["experiment"] is None
    assert set(record["models"]) == {"worker", "controller"}
    assert record["eval_outcome"] is None
    assert record["skill_updates"] == []


def test_rollout_writer_appends_json_lines(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    writer = RolloutWriter(path)
    issue = Issue(id="i-1", description="Bug")
    for seed in (1, 2):
        writer.append(build_rollout_record(issue, None, FakeExecution(), seed=seed))
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["seed"] for record in records] == [1, 2]
    assert all(record["instance_id"] == "i-1" for record in records)
