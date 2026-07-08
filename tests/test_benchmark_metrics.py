"""Benchmark metric math: pass@k, budget, cost, hard cases, evolution curve."""

import json

import pytest

from src.benchmark import metrics as M


def _record(instance_id, resolved, iters=2, tokens=1000, stage="eval", non_empty=True):
    return {
        "instance_id": instance_id,
        "stage": stage,
        "execution": {
            "iterations_used": iters,
            "total_tokens": tokens,
            "final_patch_non_empty": non_empty,
        },
        "eval_outcome": {"resolved": resolved},
    }


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_pass_at_k_unbiased():
    assert M.pass_at_k(1, 1, 1) == 1.0
    assert M.pass_at_k(1, 0, 1) == 0.0
    assert M.pass_at_k(2, 1, 1) == pytest.approx(0.5)
    assert M.pass_at_k(2, 1, 2) == 1.0        # any resolve => pass@2
    assert M.pass_at_k(3, 0, 2) == 0.0
    with pytest.raises(ValueError):
        M.pass_at_k(1, 1, 2)


def test_record_resolved_falls_back_to_evaluation():
    assert M._record_resolved({"eval_outcome": {"resolved": True}}) is True
    assert M._record_resolved({"eval_outcome": {"resolved": False}}) is False
    assert M._record_resolved({"evaluation": {"success": True}}) is True
    assert M._record_resolved({}) is False


def test_load_rollouts_run_last_record_wins(tmp_path):
    path = tmp_path / "r.jsonl"
    _write_jsonl(path, [_record("a", False), _record("a", True, tokens=222)])
    run = M.load_rollouts_run(path)
    assert run.outcomes["a"].resolved is True
    assert run.outcomes["a"].tokens == 222


def test_report_override_resolved(tmp_path):
    path = tmp_path / "r.jsonl"
    _write_jsonl(path, [_record("a", True), _record("b", True)])
    run = M.load_rollouts_run(path, report_ids={"a"})
    assert run.outcomes["a"].resolved is True
    assert run.outcomes["b"].resolved is False


def test_compute_metrics_full(tmp_path):
    r1 = tmp_path / "r1.jsonl"
    r2 = tmp_path / "r2.jsonl"
    _write_jsonl(r1, [_record("a", True), _record("b", False), _record("c", True, iters=5)])
    _write_jsonl(r2, [_record("a", False), _record("b", True), _record("c", True, iters=1)])
    runs = [M.load_rollouts_run(r1), M.load_rollouts_run(r2)]
    metrics = M.compute_metrics(runs, hard_ids={"c"}, budget=3, price_per_token=2e-6)

    assert metrics["num_runs"] == 2
    assert metrics["pass_at_k"]["1"] == pytest.approx(2 / 3)
    assert metrics["pass_at_k"]["2"] == pytest.approx(1.0)   # every instance resolved by some run
    assert metrics["aggregate"]["resolved_rate"] == pytest.approx(2 / 3)
    assert metrics["aggregate"]["success_under_budget"] == pytest.approx(0.5)
    assert metrics["aggregate"]["hard_case_success_rate"] == pytest.approx(1.0)
    assert metrics["per_run"][0]["cost_to_success_tokens"] == pytest.approx(1000.0)
    assert metrics["per_run"][0]["cost_to_success_usd"] == pytest.approx(1000.0 * 2e-6)


def test_evolution_curve_from_train_records(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(f"i{i}", i % 2 == 0, stage="train") for i in range(30)])
    metrics = M.compute_metrics([M.load_rollouts_run(path)])
    curve = metrics["evolution_curve"]
    assert curve
    assert curve[-1]["cumulative_resolved_rate"] == pytest.approx(0.5, abs=0.05)


def test_predictions_run_with_report(tmp_path):
    predictions = [
        {"instance_id": "a", "model_patch": "diff"},
        {"instance_id": "b", "model_patch": ""},
    ]
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(predictions), encoding="utf-8")
    run = M.load_predictions_run(path, report_ids={"a"})
    assert run.outcomes["a"].resolved is True
    assert run.outcomes["b"].resolved is False
    assert run.outcomes["b"].non_empty is False


def test_read_id_file_line_and_json(tmp_path):
    line_file = tmp_path / "ids.txt"
    line_file.write_text("# comment\norg__repo-1\norg__repo-2\n", encoding="utf-8")
    assert M.read_id_file(line_file) == {"org__repo-1", "org__repo-2"}
    json_file = tmp_path / "ids.json"
    json_file.write_text('["a", "b"]', encoding="utf-8")
    assert M.read_id_file(json_file) == {"a", "b"}


def test_render_markdown_sections(tmp_path):
    path = tmp_path / "r.jsonl"
    _write_jsonl(path, [_record("a", True), _record("b", False)])
    text = M.render_markdown(M.compute_metrics([M.load_rollouts_run(path)]))
    assert "# Benchmark Metrics" in text
    assert "## Aggregate" in text
    assert "pass@k" in text


def test_load_report_resolved_both_shapes(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"resolved": ["x", "y"]}), encoding="utf-8")
    assert M.load_report_resolved(a) == {"x", "y"}
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"resolved_ids": ["z"]}), encoding="utf-8")
    assert M.load_report_resolved(b) == {"z"}
