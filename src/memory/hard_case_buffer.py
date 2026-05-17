"""Normalized hard-case JSONL buffer with legacy-read compatibility."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.environment.models import Issue


HARD_CASE_SCHEMA_VERSION = "hard_case_v1"


@dataclass
class HardCaseRecord:
    schema_version: str = HARD_CASE_SCHEMA_VERSION
    issue_id: str = ""
    repo_name: Optional[str] = None
    base_commit: Optional[str] = None
    reason: str = ""
    failure_type: str = "unknown"
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
                failure_type=str(data.get("failure_type", "unknown")),
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
        failure_type = _failure_type_from_statuses(verification_statuses)
        return cls(
            issue_id=str(data.get("issue_id", "")),
            repo_name=data.get("repo_name"),
            base_commit=data.get("base_commit"),
            reason=str(data.get("reason", "")),
            failure_type=failure_type,
            created_at=str(data.get("created_at", "")),
            iterations=int(data.get("iterations", 0) or 0),
            routes=list(data.get("routes") or []),
            errors=list(data.get("errors") or []),
            verification_statuses=verification_statuses,
            patch_apply_strategies=list(data.get("patch_apply_strategies") or []),
            metadata={"legacy_schema": True},
        )


class HardCaseBuffer:
    """Append and read normalized hard-case records."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: HardCaseRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

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
        failure_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record_list = list(records)
        payload = HardCaseRecord(
            issue_id=issue.id,
            repo_name=issue.repo_name,
            base_commit=issue.base_commit,
            reason=reason,
            failure_type=failure_type,
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
            metadata=metadata or {},
        )
        if payload.failure_type == "unknown":
            payload.failure_type = _failure_type_from_statuses(payload.verification_statuses)
        self.append(payload)


def _failure_type_from_statuses(statuses: List[str]) -> str:
    if not statuses:
        return "unknown"
    last = statuses[-1]
    if last in {"patch_failed", "no_changes"}:
        return "patch_application_error"
    if last == "empty_patch":
        return "patch_generation_error"
    if last == "tests_failed":
        return "test_failure"
    if last == "new_issues":
        return "regression_introduced"
    return last or "unknown"
