"""SWEBenchRunner: experiment config, focused-variant grading, freeze, artifacts."""

import json
from dataclasses import dataclass

import pytest

from src.config import reset_config
from src.benchmark.swebench_runner import (
    EpisodeResult,
    ExperimentConfig,
    SWEBenchRunner,
    _ImageBaseAdapter,
    _eval_outcome_dict,
    create_swebench_runner,
    read_id_file,
)


@pytest.fixture(autouse=True)
def _reset_config():
    reset_config()
    yield
    reset_config()


def _runner(tmp_path, **experiment_kwargs) -> SWEBenchRunner:
    experiment = ExperimentConfig(**experiment_kwargs)
    return SWEBenchRunner(
        dataset="lite",
        output_dir=tmp_path / "run",
        workspace_dir=tmp_path / "ws",
        experiment=experiment,
    )


def test_experiment_config_defaults_and_dict():
    experiment = ExperimentConfig()
    assert experiment.stage == "eval" and experiment.agent_mode == "mas"
    assert experiment.test_backend == "auto" and experiment.hints is False
    assert experiment.to_dict()["skills"] == "static"


@dataclass
class _RecordingBackend:
    seen: list

    def run_swebench_eval(self, instance, model_patch, timeout=None):
        self.seen.append(instance["instance_id"])
        return {"resolved": True, "f2p_passed": 1, "f2p_total": 1, "p2p_passed": 0, "p2p_total": 0}


def test_image_base_adapter_remaps_focused_variant():
    seen = []
    adapter = _ImageBaseAdapter(_RecordingBackend(seen))
    adapter.run_swebench_eval({"instance_id": "org__repo-1::focus-1"}, "patch")
    adapter.run_swebench_eval({"instance_id": "org__repo-2"}, "patch")
    assert seen == ["org__repo-1", "org__repo-2"]  # focus id maps to base image


def test_eval_outcome_dict_from_object_and_dict():
    from src.environment.test_backend import EvalOutcome

    outcome = EvalOutcome(f2p_passed=2, f2p_total=3, p2p_passed=1, p2p_total=1, resolved=False)
    as_dict = _eval_outcome_dict(outcome)
    assert as_dict == {"f2p_passed": 2, "f2p_total": 3, "p2p_passed": 1, "p2p_total": 1, "resolved": False}
    assert _eval_outcome_dict({"resolved": True, "f2p_passed": 1})["resolved"] is True


def test_eval_stage_freezes_skill_snapshot(tmp_path):
    runner = _runner(tmp_path, stage="eval", skills="static")
    assert runner._skills_dir == runner.run_dir / "skills_snapshot"
    assert runner._skills_dir.exists()
    # seed skills copied into the snapshot; evolver disabled during eval
    assert list(runner._skills_dir.glob("*.md"))
    assert runner._skill_evolve is False
    assert runner._evolver is None


def test_train_evolve_uses_live_bank_and_evolver(tmp_path):
    runner = _runner(tmp_path, stage="train", skills="evolve")
    assert runner._skills_dir == SWEBenchRunner._repo_skills_dir()
    assert runner._skill_evolve is True
    assert runner._evolver is not None


def test_skills_off_has_no_selector(tmp_path):
    runner = _runner(tmp_path, stage="eval", skills="off")
    assert runner._selector is None
    assert runner._build_controller_signal.__self__ is runner  # bound, callable


def test_manifest_written_with_experiment(tmp_path):
    runner = _runner(tmp_path, stage="train", skills="evolve", task_evolution="on", seed=7)
    manifest = json.loads((runner.run_dir / "run_manifest.json").read_text())
    assert manifest["benchmark"] == "swebench"
    assert manifest["dataset"] == "lite"
    assert manifest["experiment"]["seed"] == 7
    assert manifest["experiment"]["task_evolution"] == "on"


def test_dataset_split_passthrough(tmp_path):
    runner = _runner(tmp_path)
    assert runner._dataset_split("test") == ("lite", "test")


def test_workspace_dir_isolated_to_run(tmp_path):
    from src.config import get_config

    runner = _runner(tmp_path, stage="eval")
    assert get_config().environment.workspace_dir == runner.run_dir


def test_host_backend_returns_none_and_warns(tmp_path):
    runner = _runner(tmp_path, stage="eval", test_backend="host")
    assert runner._grading_backend(env=None) is None
    assert runner._host_warned is True


def test_final_summary_from_eval_cache(tmp_path):
    runner = _runner(tmp_path, stage="eval")
    predictions_path = runner.run_dir / "predictions.json"
    runner._save_predictions(
        predictions_path,
        {
            "a": runner._prediction_dict("a", "diff-a"),
            "b": runner._prediction_dict("b", "diff-b"),
        },
    )
    runner._save_eval_cache({
        "a": {"resolved": True, "f2p_passed": 1, "f2p_total": 1, "p2p_passed": 0, "p2p_total": 0},
        "b": {"resolved": False, "f2p_passed": 0, "f2p_total": 1, "p2p_passed": 0, "p2p_total": 0},
    })
    runner._write_final_summary_from_cache(predictions_path)
    summary = json.loads((runner.run_dir / "final_summary.json").read_text())
    assert summary["resolved"] == ["a"]
    assert summary["resolved_count"] == 1
    assert summary["graded_count"] == 2


def test_contamination_guard_reads_train_ids(tmp_path):
    ids_file = tmp_path / "train_ids.txt"
    ids_file.write_text("org__repo-1\norg__repo-2\n", encoding="utf-8")
    runner = create_swebench_runner(
        dataset="lite",
        output_dir=tmp_path / "run",
        workspace_dir=tmp_path / "ws",
        experiment=ExperimentConfig(stage="eval"),
        train_ids_path=ids_file,
    )
    assert runner._eval_contamination_guard() == {"org__repo-1", "org__repo-2"}


def test_prediction_dict_shape(tmp_path):
    runner = _runner(tmp_path)
    prediction = runner._prediction_dict("org__repo-1", "diff")
    assert prediction == {
        "instance_id": "org__repo-1",
        "model_name_or_path": "self-evolver",
        "model_patch": "diff",
    }


def test_read_id_file_json_and_lines(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text('["x", "y"]', encoding="utf-8")
    assert read_id_file(path) == {"x", "y"}


def test_single_mode_missing_agent_raises(tmp_path):
    runner = _runner(tmp_path, stage="eval", agent_mode="single")
    try:
        import src.workers.single_agent  # noqa: F401
        has_single_agent = True
    except ImportError:
        has_single_agent = False
    if not has_single_agent:
        with pytest.raises(RuntimeError, match="single_agent"):
            runner._single_episode(_stub_issue(), env=None, backend=None, signal=None)


def _stub_issue():
    from src.environment.models import Issue

    return Issue(id="org__repo-1", description="boom", repo_name="org/repo", base_commit="abc")


def test_episode_result_fields():
    result = EpisodeResult(instance_id="a", patch="diff", resolved=True, utility=0.9)
    assert result.resolved is True and result.utility == 0.9
