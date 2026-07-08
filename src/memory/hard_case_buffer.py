"""Normalized hard-case JSONL buffer with admission policy and legacy-read compatibility."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import get_config
from src.environment.models import Issue
from src.skills.failure_types import FailureType, failure_type_from_statuses
from src.skills.file_lock import file_lock


HARD_CASE_SCHEMA_VERSION = "hard_case_v1"

# Admission policy (Proposal 2.5.1): a failed run is worth keeping when it is a
# repeated failure, shows many interactions with little progress, or exhausted
# the fixed repair budget.
LOW_PROGRESS_MIN_ITERATIONS = 2


@dataclass
class HardCaseRecord:
    schema_version: str = HARD_CASE_SCHEMA_VERSION
    issue_id: str = ""
    repo_name: Optional[str] = None
    base_commit: Optional[str] = None
    reason: str = ""
    failure_type: str = FailureType.UNKNOWN.value
    stage: str = "train"
    created_at: str = ""
    iterations: int = 0
    routes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    verification_statuses: List[str] = field(default_factory=list)
    patch_apply_strategies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not payload["created_at"]:
            payload["created_at"] = datetime.now().isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardCaseRecord":
        if data.get("schema_version") == HARD_CASE_SCHEMA_VERSION:
            return cls(
                schema_version=str(data.get("schema_version", HARD_CASE_SCHEMA_VERSION)),
                issue_id=str(data.get("issue_id", "")),
                repo_name=data.get("repo_name"),
                base_commit=data.get("base_commit"),
                reason=str(data.get("reason", "")),
                failure_type=str(data.get("failure_type", FailureType.UNKNOWN.value)),
                stage=str(data.get("stage", "train")),
                created_at=str(data.get("created_at", "")),
                iterations=int(data.get("iterations", 0) or 0),
                routes=list(data.get("routes") or []),
                errors=list(data.get("errors") or []),
                verification_statuses=list(data.get("verification_statuses") or []),
                patch_apply_strategies=list(data.get("patch_apply_strategies") or []),
                metadata=dict(data.get("metadata") or {}),
            )
        return cls.from_legacy(data)

    @classmethod
    def from_legacy(cls, data: Dict[str, Any]) -> "HardCaseRecord":
        verification_statuses = list(data.get("verification_statuses") or [])
        return cls(
            issue_id=str(data.get("issue_id", "")),
            repo_name=data.get("repo_name"),
            base_commit=data.get("base_commit"),
            reason=str(data.get("reason", "")),
            failure_type=failure_type_from_statuses(verification_statuses),
            stage=str(data.get("stage", "train")),
            created_at=str(data.get("created_at", "")),
            iterations=int(data.get("iterations", 0) or 0),
            routes=list(data.get("routes") or []),
            errors=list(data.get("errors") or []),
            verification_statuses=verification_statuses,
            patch_apply_strategies=list(data.get("patch_apply_strategies") or []),
            metadata={"legacy_schema": True},
        )


class HardCaseBuffer:
    """Append (subject to admission) and read normalized hard-case records."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: HardCaseRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        with file_lock(self.path):
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

    def read(self) -> List[HardCaseRecord]:
        if not self.path.exists():
            return []
        records: List[HardCaseRecord] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(HardCaseRecord.from_dict(data))
                except json.JSONDecodeError:
                    continue
        return records

    def append_from_execution(
        self,
        issue: Issue,
        records: Iterable[Any],
        reason: str,
        failure_type: str = FailureType.UNKNOWN.value,
        metadata: Optional[Dict[str, Any]] = None,
        stage: str = "train",
        budget: Optional[int] = None,
    ) -> bool:
        """Build a hard-case record and admit it only if it meets the policy.

        Returns True when the record was written, False when the admission
        policy rejected it (novel one-shot failure that is neither repeated nor
        low-progress nor budget-exhausting).
        """
        record_list = list(records)
        payload = HardCaseRecord(
            issue_id=issue.id,
            repo_name=issue.repo_name,
            base_commit=issue.base_commit,
            reason=reason,
            failure_type=failure_type,
            stage=stage,
            created_at=datetime.now().isoformat(),
            iterations=len(record_list),
            routes=[
                item.judge_decision.route.value
                for item in record_list
                if getattr(item, "judge_decision", None)
            ],
            errors=[item.error for item in record_list if getattr(item, "error", None)],
            verification_statuses=[
                item.verification_result.status.value
                for item in record_list
                if getattr(item, "verification_result", None)
            ],
            patch_apply_strategies=[
                item.verification_result.patch_apply_result.strategy
                for item in record_list
                if getattr(item, "verification_result", None)
                and getattr(item.verification_result, "patch_apply_result", None)
            ],
            metadata=dict(metadata or {}),
        )
        if payload.failure_type == FailureType.UNKNOWN.value:
            payload.failure_type = failure_type_from_statuses(payload.verification_statuses)

        admit, admit_reason = should_admit(payload, self.read(), budget)
        payload.metadata["admission_reason"] = admit_reason
        if not admit:
            return False
        self.append(payload)
        return True


def should_admit(
    record: HardCaseRecord,
    history: List[HardCaseRecord],
    budget: Optional[int] = None,
) -> tuple[bool, str]:
    """Decide whether a failed run belongs in the hard-case buffer (Proposal 2.5.1)."""
    for prior in history:
        if prior.repo_name == record.repo_name and prior.failure_type == record.failure_type:
            return True, "repeated_similar_failure"
    if record.iterations >= LOW_PROGRESS_MIN_ITERATIONS:
        return True, "low_progress"
    cap = budget if budget is not None else get_config().agent.max_iterations
    if cap and record.iterations >= cap:
        return True, "budget_exhausted"
    return False, "novel_single_attempt"
