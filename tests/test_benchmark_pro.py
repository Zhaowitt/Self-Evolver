"""SWE-bench Pro: image naming, generation-only, loud eval failure."""

import pytest

from src.config import reset_config
from src.benchmark.swebench_pro import (
    PRO_IMAGE_NAMESPACE,
    ProEvaluationUnavailable,
    SWEBenchProRunner,
    pro_image_ref,
)
from src.benchmark.swebench_runner import ExperimentConfig


@pytest.fixture(autouse=True)
def _reset_config():
    reset_config()
    yield
    reset_config()


def _runner(tmp_path) -> SWEBenchProRunner:
    return SWEBenchProRunner(
        dataset="test",
        output_dir=tmp_path / "run",
        workspace_dir=tmp_path / "ws",
        experiment=ExperimentConfig(stage="eval"),
    )


def test_pro_image_ref():
    tag = "nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5"
    assert pro_image_ref(tag) == f"{PRO_IMAGE_NAMESPACE}:{tag}"


def test_pro_dataset_split_is_test(tmp_path):
    runner = _runner(tmp_path)
    assert runner.name == "swebench_pro"
    assert runner._dataset_split("anything") == ("swebench_pro", "test")


def test_pro_has_no_in_loop_backend(tmp_path):
    runner = _runner(tmp_path)
    assert runner._container_backend() is None
    assert runner._grading_backend(env=None) is None


def test_pro_eval_fails_loudly_with_instructions(tmp_path):
    runner = _runner(tmp_path)
    predictions_path = runner.run_dir / "predictions.json"
    with pytest.raises(ProEvaluationUnavailable) as exc:
        runner.evaluate_predictions(predictions_path)
    message = str(exc.value)
    assert str(predictions_path) in message
    assert PRO_IMAGE_NAMESPACE in message
    assert "scaleapi/SWE-bench_Pro-os" in message
