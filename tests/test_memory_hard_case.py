"""Hard-case buffer: stage tagging, admission policy, and stage-filtered retrieval."""

from __future__ import annotations

from src.environment.models import Issue
from src.memory.hard_case_buffer import (
    LOW_PROGRESS_MIN_ITERATIONS,
    HardCaseBuffer,
    HardCaseRecord,
    should_admit,
)
from src.memory.memory_retriever import MemoryRetriever


class _Iter:
    """Minimal iteration-record stand-in (all detail attributes absent)."""


def _issue(issue_id="acme__widgets-1", repo="acme/widgets"):
    return Issue(id=issue_id, description="fix the bug", repo_name=repo, base_commit="abc123")


def _record(repo="acme/widgets", failure_type="test_failure", iterations=1, stage="train"):
    return HardCaseRecord(
        repo_name=repo, failure_type=failure_type, iterations=iterations, stage=stage
    )


# --------------------------------------------------------------------- stage field


def test_stage_field_round_trips_through_json():
    record = HardCaseRecord(issue_id="i1", stage="eval", failure_type="test_failure")
    restored = HardCaseRecord.from_dict(record.to_dict())
    assert restored.stage == "eval"


def test_legacy_record_defaults_stage_to_train():
    legacy = {"issue_id": "i1", "verification_statuses": ["tests_failed"]}
    restored = HardCaseRecord.from_dict(legacy)
    assert restored.stage == "train"
    # Failure type comes from the unified taxonomy, not a private duplicate.
    assert restored.failure_type == "test_failure"


# --------------------------------------------------------------------- admission


def test_admit_low_progress_run():
    record = _record(iterations=LOW_PROGRESS_MIN_ITERATIONS)
    admit, reason = should_admit(record, history=[], budget=5)
    assert admit is True
    assert reason == "low_progress"


def test_admit_repeated_similar_failure():
    prior = [_record(repo="acme/widgets", failure_type="test_failure")]
    record = _record(repo="acme/widgets", failure_type="test_failure", iterations=1)
    admit, reason = should_admit(record, history=prior, budget=5)
    assert admit is True
    assert reason == "repeated_similar_failure"


def test_admit_budget_exhausted():
    record = _record(iterations=3)
    admit, reason = should_admit(record, history=[], budget=3)
    # iterations==3 also trips low_progress first; force a low-progress-free case:
    single = _record(iterations=1)
    admit1, reason1 = should_admit(single, history=[], budget=1)
    assert admit1 is True
    assert reason1 == "budget_exhausted"


def test_reject_novel_single_attempt():
    record = _record(iterations=1)
    admit, reason = should_admit(record, history=[], budget=5)
    assert admit is False
    assert reason == "novel_single_attempt"


def test_append_from_execution_respects_admission(tmp_path):
    buffer = HardCaseBuffer(tmp_path / "hc.jsonl")
    # Novel single-attempt failure -> rejected, nothing written.
    admitted = buffer.append_from_execution(
        _issue(), records=[_Iter()], reason="gave up", failure_type="test_failure",
        stage="train", budget=5,
    )
    assert admitted is False
    assert buffer.read() == []

    # Second identical failure is now a repeat -> admitted with stage tagged.
    buffer.append(_record())  # prior similar case in history
    admitted = buffer.append_from_execution(
        _issue(), records=[_Iter()], reason="gave up again", failure_type="test_failure",
        stage="train", budget=5,
    )
    assert admitted is True
    written = buffer.read()[-1]
    assert written.stage == "train"
    assert written.metadata["admission_reason"] == "repeated_similar_failure"


# --------------------------------------------------------------------- retrieval


def test_retriever_filters_by_stage_and_repo(tmp_path):
    buffer = HardCaseBuffer(tmp_path / "hc.jsonl")
    buffer.append(HardCaseRecord(issue_id="t1", repo_name="acme/widgets", stage="train"))
    buffer.append(HardCaseRecord(issue_id="e1", repo_name="acme/widgets", stage="eval"))
    buffer.append(HardCaseRecord(issue_id="t2", repo_name="other/repo", stage="train"))

    retriever = MemoryRetriever(tmp_path / "hc.jsonl")
    train_ids = [r.issue_id for r in retriever.retrieve(stage="train")]
    assert set(train_ids) == {"t1", "t2"}

    scoped = retriever.retrieve(repo_name="acme/widgets", stage="train")
    assert [r.issue_id for r in scoped] == ["t1"]
