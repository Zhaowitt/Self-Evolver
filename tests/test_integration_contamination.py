"""Contamination guard: training instance ids never leak into an eval run.

The guard has two moving parts that this exercises end to end without touching
the network: ``read_id_file`` parses a train-id list (JSON or line format), and
the runner excludes those ids from the eval set before the row limit is applied,
via ``datasets.iter_rows(exclude_ids=...)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark import datasets
from src.benchmark.swebench_runner import (
    ExperimentConfig,
    SWEBenchRunner,
    read_id_file,
)


@pytest.fixture
def synthetic_rows():
    """Inject rows into the (lite, test) split cache and restore afterwards, so
    iter_rows serves them without a local dataset or the hub."""
    key = ("lite", "test")
    saved = datasets._CACHE.get(key, "__absent__")
    rows = [{"instance_id": f"org__proj-{i}", "problem_statement": "x"} for i in range(6)]
    datasets._CACHE[key] = rows
    try:
        yield rows
    finally:
        if saved == "__absent__":
            datasets._CACHE.pop(key, None)
        else:
            datasets._CACHE[key] = saved


# -------------------------------------------------------------- read_id_file


def test_read_id_file_parses_line_format(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text(
        "# training ids\norg__proj-0\n\norg__proj-2  \n# comment\norg__proj-4\n",
        encoding="utf-8",
    )
    assert read_id_file(path) == {"org__proj-0", "org__proj-2", "org__proj-4"}


def test_read_id_file_parses_json_list(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["org__proj-1", "org__proj-3"]), encoding="utf-8")
    assert read_id_file(path) == {"org__proj-1", "org__proj-3"}


# ------------------------------------------------------- iter_rows exclusion


def test_iter_rows_excludes_ids_before_applying_the_limit(synthetic_rows):
    excluded = {"org__proj-0", "org__proj-1"}
    kept = list(datasets.iter_rows("lite", "test", limit=2, exclude_ids=excluded))
    ids = [row["instance_id"] for row in kept]
    # Exclusion happens first, so the two returned rows are the first survivors,
    # not proj-0/1 dropped after the limit was already spent.
    assert ids == ["org__proj-2", "org__proj-3"]
    assert excluded.isdisjoint(ids)


# ------------------------------------------------- runner-level round trip


def test_runner_refuses_training_ids_in_the_eval_set(tmp_path, synthetic_rows):
    ids_file = tmp_path / "train_ids.txt"
    ids_file.write_text("org__proj-0\norg__proj-3\n", encoding="utf-8")

    runner = SWEBenchRunner(
        dataset="lite",
        output_dir=tmp_path / "run",
        workspace_dir=tmp_path / "ws",
        experiment=ExperimentConfig(stage="eval", skills="static"),
        train_ids_path=ids_file,
    )

    guard = runner._eval_contamination_guard()
    assert guard == {"org__proj-0", "org__proj-3"}

    rows = runner.load_rows("test", exclude_ids=guard)
    ids = [row["instance_id"] for row in rows]
    assert "org__proj-0" not in ids
    assert "org__proj-3" not in ids
    assert set(ids) == {"org__proj-1", "org__proj-2", "org__proj-4", "org__proj-5"}


def test_runner_without_train_ids_excludes_nothing(tmp_path):
    runner = SWEBenchRunner(
        dataset="lite",
        output_dir=tmp_path / "run",
        workspace_dir=tmp_path / "ws",
        experiment=ExperimentConfig(stage="eval", skills="static"),
    )
    assert runner._eval_contamination_guard() == set()
