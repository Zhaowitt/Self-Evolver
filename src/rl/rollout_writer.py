"""JSONL rollout logging for controller-guided repair attempts."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.controller.schema import controller_signal_from_any
from src.environment.models import Issue


class RolloutWriter:
    """Append rollout records to JSONL."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def build_rollout_record(
    issue: Issue,
    controller_signal: Any,
    execution_result: Any,
    evaluation: Any = None,
    reward: Any = None,
    skill_evolution: Any = None,
    rollout_id: Optional[str] = None,
) -> Dict[str, Any]:
    signal = controller_signal_from_any(controller_signal)
    final_patch = getattr(execution_result, "final_patch", None)
    reward_payload = reward.to_dict() if hasattr(reward, "to_dict") else reward
    evaluation_payload = _evaluation_payload(evaluation)

    return {
        "rollout_id": rollout_id or str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "instance_id": issue.id,
        "repo_name": issue.repo_name,
        "base_commit": issue.base_commit,
        "mode": signal.mode if signal else "",
        "controller_signal": signal.to_dict() if signal else None,
        "selected_skill_ids": signal.selected_skill_ids if signal else [],
        "skill_updates": [
            proposal.to_dict() for proposal in signal.skill_updates
        ] if signal else [],
        "execution": {
            "success": bool(getattr(execution_result, "success", False)),
            "status": getattr(getattr(execution_result, "status", None), "value", ""),
            "iterations_used": int(getattr(execution_result, "iterations_used", 0) or 0),
            "total_tokens": int(getattr(execution_result, "total_tokens", 0) or 0),
            "total_duration_ms": float(getattr(execution_result, "total_duration_ms", 0.0) or 0.0),
            "final_patch_non_empty": bool(final_patch and getattr(final_patch, "content", "").strip()),
            "final_patch_files": list(getattr(final_patch, "modified_files", []) or []),
        },
        "evaluation": evaluation_payload,
        "reward": reward_payload,
        "skill_evolution": skill_evolution or {
            "events": [],
            "dedup_decisions": [],
            "skill_stats_before": {},
            "skill_stats_after": {},
        },
    }


def _evaluation_payload(evaluation: Any) -> Optional[Dict[str, Any]]:
    if not evaluation:
        return None
    return {
        "success": bool(getattr(evaluation, "success", False)),
        "failure_type": getattr(getattr(evaluation, "failure_type", None), "value", ""),
        "failure_tags": list(getattr(evaluation, "failure_tags", []) or []),
        "summary": getattr(evaluation, "summary", ""),
    }


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value
