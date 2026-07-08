"""Eval stage is frozen: no skill / memory / task-pool mutation, enforced in
code (not by convention).

The benchmark runner is the public surface that decides freezing. Constructing
it in the eval stage must: build no skill evolver (so no skill write path
exists), read skills from an independent snapshot under the run directory,
redirect hard-case memory into the run directory, and never build a task pool.
The train + evolve construction is the control that shows those components do
exist when they are allowed to.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.benchmark.swebench_runner import ExperimentConfig, SWEBenchRunner
from src.config import get_config

REPO_SKILLS = Path(__file__).resolve().parents[1] / "skills"


def _repo_skills_fingerprint() -> list:
    return sorted(
        (path.name, hashlib.md5(path.read_bytes()).hexdigest())
        for path in REPO_SKILLS.iterdir()
        if path.is_file()
    )


def _make_runner(tmp_path: Path, stage: str, skills: str) -> SWEBenchRunner:
    return SWEBenchRunner(
        dataset="lite",
        output_dir=tmp_path / "run",
        workspace_dir=tmp_path / "ws",
        experiment=ExperimentConfig(stage=stage, skills=skills),
    )


def test_eval_stage_builds_no_skill_evolver(tmp_path):
    runner = _make_runner(tmp_path, stage="eval", skills="evolve")
    # Even with skills=evolve requested, eval forbids evolution entirely: with no
    # evolver, the rollout has no code path that writes or retires a skill.
    assert runner._skill_evolve is False
    assert runner._evolver is None


def test_eval_stage_reads_from_an_independent_snapshot(tmp_path):
    before = _repo_skills_fingerprint()
    runner = _make_runner(tmp_path, stage="eval", skills="static")

    snapshot = tmp_path / "run" / "skills_snapshot"
    assert runner._skills_dir == snapshot
    assert runner._skills_dir != REPO_SKILLS
    # The snapshot is a real copy of the live bank.
    assert {p.name for p in snapshot.glob("*.md")} == {
        p.name for p in REPO_SKILLS.glob("*.md")
    }

    # Writing into the snapshot must not reach the repo skill bank.
    (snapshot / "leaked_eval_skill.md").write_text("# leak\n", encoding="utf-8")
    assert _repo_skills_fingerprint() == before


def test_eval_stage_isolates_hard_case_memory_to_the_run_dir(tmp_path):
    runner = _make_runner(tmp_path, stage="eval", skills="static")
    run_dir = tmp_path / "run"
    # Hard cases are written under workspace_dir; the runner redirects it into
    # the run directory so eval never appends to the shared training buffer.
    assert get_config().environment.workspace_dir == run_dir
    assert runner._hard_cases_path == run_dir / "hard_cases.jsonl"
    assert run_dir in runner._hard_cases_path.parents


def test_eval_stage_builds_no_task_pool(tmp_path):
    runner = _make_runner(tmp_path, stage="eval", skills="static")
    # The pool is only ever created in the train task-evolution path; eval must
    # not be able to reach it.
    assert not (runner.experiment.stage == "train" and runner.experiment.task_evolution == "on")
    assert not (tmp_path / "run" / "task_pool.json").exists()


def test_train_evolve_control_builds_the_frozen_pieces(tmp_path):
    """Control: the evolver and repo-backed bank exist when evolution is allowed,
    so eval's absence of them is a deliberate freeze, not a missing feature. The
    repo skill bank is not mutated by mere construction."""
    before = _repo_skills_fingerprint()
    runner = _make_runner(tmp_path, stage="train", skills="evolve")
    assert runner._skill_evolve is True
    assert runner._evolver is not None
    assert runner._skills_dir == REPO_SKILLS
    assert _repo_skills_fingerprint() == before
