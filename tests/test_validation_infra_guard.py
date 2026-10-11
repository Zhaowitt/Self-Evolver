"""Evaluation infrastructure failures must never become policy metrics."""
from types import SimpleNamespace

import pytest

from src.reward.online_reward import _finish_scores, _score_one, ValidationInfrastructureError, compute_score


def inputs(stage, count=2):
    return [{"ground_truth": {"instance_id": "repo__task-1", "stage": stage}}
            for _ in range(count)]


def test_all_eval_candidates_failing_environment_do_not_return_zero():
    with pytest.raises(ValidationInfrastructureError, match="refusing to return zero"):
        _finish_scores(inputs("eval"), [{"overall": 0.0, "infra_error": 1.0}] * 2)


def test_one_failed_eval_candidate_not_imputed_from_other_candidate():
    with pytest.raises(ValidationInfrastructureError):
        _finish_scores(inputs("eval"), [{"overall": 0.5, "infra_error": 0.0},
                                        {"overall": 0.0, "infra_error": 1.0}])


def test_training_group_mean_neutralization_is_unchanged():
    scores = _finish_scores(inputs("train"), [{"overall": -0.5, "infra_error": 0.0},
                                             {"overall": 0.0, "infra_error": 1.0}])
    assert [score["overall"] for score in scores] == [-0.5, -0.5]


def test_normal_eval_scores_are_unchanged():
    scores = [{"overall": 0.7, "infra_error": 0.0}, {"overall": -0.2, "infra_error": 0.0}]
    assert _finish_scores(inputs("eval"), scores) == scores


def test_embedded_easyr1_stage_is_respected():
    rows = [{"ground_truth": '{"instance_id": "task", "extra_info": {"stage": "eval"}}'}]
    with pytest.raises(ValidationInfrastructureError):
        _finish_scores(rows, [{"overall": 0.0, "infra_error": 1.0}])


@pytest.mark.parametrize("workers", [1, 2])
def test_both_compute_score_paths_fail_closed(monkeypatch, workers, tmp_path):
    monkeypatch.setattr("src.reward.online_reward._score_one",
                        lambda *_a, **_k: {"overall": 0.0, "infra_error": 1.0})
    monkeypatch.setattr("src.environment.test_backend.resolve_backend", lambda *_a, **_k: object())
    rows = inputs("eval")
    rows[1]["ground_truth"]["instance_id"] = "repo__task-2"
    with pytest.raises(ValidationInfrastructureError):
        compute_score(rows, parallel_workers=workers, workspace_root=tmp_path)


def test_validation_checks_image_before_worker_or_checkout(monkeypatch):
    calls = []
    signal = SimpleNamespace(parse_error=None)
    monkeypatch.setattr("src.reward.online_reward.parse_controller_response", lambda *_a, **_k: signal)
    monkeypatch.setattr("src.reward.online_reward.ground_controller_skills", lambda value, **_k: value)
    monkeypatch.setattr("src.reward.online_reward.issue_from_payload", lambda *_a: object())
    monkeypatch.setattr("src.reward.online_reward.swebench_instance_from_issue", lambda _issue: {"instance_id": "task"})

    def ensure(_instance):
        calls.append("image")
        raise RuntimeError("Missing environment")

    monkeypatch.setattr("src.reward.online_reward.prepare_environment", lambda *_a, **_k: pytest.fail("No checkout before image readiness"))
    monkeypatch.setattr("src.reward.online_reward.run_online_rollout", lambda *_a, **_k: pytest.fail("No worker API before image readiness"))
    score = _score_one(inputs("eval", 1)[0], reward_model=None, rollout_writer=None,
                       skill_evolver=None, test_backend=SimpleNamespace(ensure_image=ensure))
    assert calls == ["image"]
    assert score["infra_error"] == 1.0
