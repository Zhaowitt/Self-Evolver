"""Multi-SWE-bench: JSONL loading, row mapping, language globs, loud eval failure."""

import json

import pytest

from src.benchmark import datasets
from src.benchmark.swebench_multi import (
    MultiSWEBenchRunner,
    MultiSWEEvaluationUnavailable,
)
from src.benchmark.swebench_runner import ExperimentConfig
from src.config import reset_config


@pytest.fixture(autouse=True)
def _reset_config():
    reset_config()
    datasets._CACHE.clear()
    yield
    reset_config()
    datasets._CACHE.clear()


def _msb_row(instance_id="cli__cli-10388", org="cli", repo="cli", number=10388):
    return {
        "instance_id": instance_id,
        "org": org,
        "repo": repo,
        "number": number,
        "base": {"sha": "db9dbfa4deadbeef", "ref": "trunk"},
        "title": "Fix the thing",
        "body": "It is broken.",
        "fix_patch": "diff --git a/x.go b/x.go\n",
        "test_patch": "diff --git a/x_test.go b/x_test.go\n",
        "f2p_tests": {"TestA": {}, "TestB": {}},
        "p2p_tests": {"TestC": {}},
    }


def test_source_globs_per_language():
    assert datasets.source_globs("go") == ("**/*.go",)
    assert datasets.source_globs("rust") == ("**/*.rs",)
    assert "**/*.tsx" in datasets.source_globs("typescript")
    assert datasets.source_globs(None) == ("**/*.py",)      # default
    assert datasets.source_globs("klingon") == ("**/*.py",)  # unknown -> default


def test_msb_row_to_issue_maps_all_fields():
    issue = datasets.row_to_issue(_msb_row())
    assert issue.id == "cli__cli-10388"
    assert issue.repo_name == "cli/cli"
    assert issue.base_commit == "db9dbfa4deadbeef"
    assert issue.description.startswith("Fix the thing")
    assert issue.metadata["language"] is None or isinstance(issue.metadata["language"], str)
    assert issue.metadata["fail_to_pass"] == ["TestA", "TestB"]
    assert issue.metadata["pass_to_pass"] == ["TestC"]
    assert issue.metadata["gold_patch"].startswith("diff --git")


def test_jsonl_loader_injects_language_from_dir(tmp_path, monkeypatch):
    root = tmp_path / "multi_swe_bench_full"
    (root / "go").mkdir(parents=True)
    (root / "rust").mkdir(parents=True)
    (root / "go" / "cli__cli_dataset.jsonl").write_text(
        json.dumps(_msb_row()) + "\n", encoding="utf-8")
    (root / "rust" / "x__y_dataset.jsonl").write_text(
        json.dumps(_msb_row("x__y-1", "x", "y", 1)) + "\n", encoding="utf-8")
    monkeypatch.setenv("SWEBENCH_DATA_DIR", str(tmp_path))

    rows = list(datasets.iter_rows("multi_swe_bench", "test"))
    by_lang = {r["language"] for r in rows}
    assert by_lang == {"go", "rust"}
    go_row = next(r for r in rows if r["language"] == "go")
    assert datasets.row_to_issue(go_row).repo_name == "cli/cli"


def test_msb_split_validation():
    datasets.validate_split("multi_swe_bench", "test")  # ok
    with pytest.raises(ValueError, match="no 'train' split"):
        datasets.validate_split("multi_swe_bench", "train")


def _runner(tmp_path, dataset="full") -> MultiSWEBenchRunner:
    return MultiSWEBenchRunner(
        dataset=dataset,
        output_dir=tmp_path / "run",
        workspace_dir=tmp_path / "ws",
        experiment=ExperimentConfig(stage="eval"),
    )


def test_msb_dataset_split_maps_full_and_flash(tmp_path):
    assert _runner(tmp_path, "full")._dataset_split("x") == ("multi_swe_bench", "test")
    assert _runner(tmp_path, "flash")._dataset_split("x") == ("multi_swe_bench_flash", "test")


def test_msb_no_in_loop_backend(tmp_path):
    assert _runner(tmp_path)._container_backend() is None


def test_msb_eval_fails_loudly_and_writes_harness_file(tmp_path, monkeypatch):
    # Point the loader at a synthetic dataset so _write_harness_patch_file can map ids.
    root = tmp_path / "multi_swe_bench_full" / "go"
    root.mkdir(parents=True)
    (root / "cli__cli_dataset.jsonl").write_text(
        json.dumps(_msb_row()) + "\n", encoding="utf-8")
    monkeypatch.setenv("SWEBENCH_DATA_DIR", str(tmp_path))

    runner = _runner(tmp_path, "full")
    predictions_path = runner.run_dir / "predictions.json"
    runner._save_predictions(
        predictions_path,
        {"cli__cli-10388": {"instance_id": "cli__cli-10388", "model_patch": "diff --git a/x.go b/x.go\n"}},
    )
    with pytest.raises(MultiSWEEvaluationUnavailable) as exc:
        runner.evaluate_predictions(predictions_path)
    assert "multi-swe-bench" in str(exc.value)

    harness_file = runner.run_dir / "multi_swe_bench_patches.jsonl"
    assert harness_file.exists()
    record = json.loads(harness_file.read_text().splitlines()[0])
    assert record == {"org": "cli", "repo": "cli", "number": 10388,
                      "fix_patch": "diff --git a/x.go b/x.go\n"}
