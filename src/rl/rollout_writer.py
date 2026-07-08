"""JSONL rollout logging for controller-guided repair attempts.

Record schema (``rollout_record_v2``) — one JSON object per line:

- ``schema_version``: "rollout_record_v2".
- ``rollout_id``: uuid4 string. ``created_at``: ISO timestamp.
- ``instance_id`` / ``repo_name`` / ``base_commit``: task identity.
- ``stage``: "train" | "eval" (falls back to the signal mode).
- ``seed``: int | null — the run's RNG seed.
- ``experiment``: str | null — experiment label (e.g. "full-method").
- ``models``: {"worker": str, "controller": str} — model names in use.
- ``mode``: controller signal mode (kept for v1 readers; equals ``stage``'s
  source signal).
- ``controller_signal``: full signal dict incl. raw_response/parse_error.
- ``selected_skill_ids``: list[str].
- ``skill_updates``: list of proposal dicts (empty now that skill-lifecycle
  proposals are owned by the Reflector; kept for older readers).
- ``execution``: success/status/iterations_used/total_tokens/
  total_duration_ms/final_patch_non_empty/final_patch_files.
- ``eval_outcome``: {f2p_passed, f2p_total, p2p_passed, p2p_total, resolved}
  | null — official-semantics grading of the final patch when a test backend
  ran.
- ``evaluation``: critic summary (success/failure_type/failure_tags/summary).
- ``reward``: RewardResult dict (total/components/weights/evolution_utility/
  baseline). This is execution logging, not the reward source.
- ``skill_evolution``: events/dedup decisions/stats snapshots.

Backward compatibility: v1 records lack ``schema_version``, ``stage``,
``seed``, ``experiment``, ``models``, and ``eval_outcome``; readers must treat those
as absent/null. Writers must only add fields, never repurpose existing ones.
"""

from __future__ import annotations

import fcntl
import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import get_config
from src.controller.schema import controller_signal_from_any
from src.environment.models import Issue

SCHEMA_VERSION = "rollout_record_v2"


class RolloutWriter:
    """Append rollout records to JSONL (flock-guarded for parallel workers)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_json_safe(record), ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def build_rollout_record(
    issue: Issue,
    controller_signal: Any,
    execution_result: Any,
    evaluation: Any = None,
    reward: Any = None,
    skill_evolution: Any = None,
    rollout_id: Optional[str] = None,
    eval_outcome: Any = None,
    stage: Optional[str] = None,
    seed: Optional[int] = None,
    experiment: Optional[str] = None,
    models: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    signal = controller_signal_from_any(controller_signal)
    final_patch = getattr(execution_result, "final_patch", None)
    reward_payload = reward.to_dict() if hasattr(reward, "to_dict") else reward
    evaluation_payload = _evaluation_payload(evaluation)
    skill_updates = getattr(signal, "skill_updates", []) if signal else []

    return {
        "schema_version": SCHEMA_VERSION,
        "rollout_id": rollout_id or str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "instance_id": issue.id,
        "repo_name": issue.repo_name,
        "base_commit": issue.base_commit,
        "stage": stage or (signal.mode if signal else "train"),
        "seed": seed,
        "experiment": experiment,
        "models": models or _default_models(),
        "mode": signal.mode if signal else "",
        "controller_signal": signal.to_dict() if signal else None,
        "selected_skill_ids": signal.selected_skill_ids if signal else [],
        "skill_updates": [
            proposal.to_dict() if hasattr(proposal, "to_dict") else proposal
            for proposal in skill_updates
        ],
        "execution": {
            "success": bool(getattr(execution_result, "success", False)),
            "status": getattr(getattr(execution_result, "status", None), "value", ""),
            "iterations_used": int(getattr(execution_result, "iterations_used", 0) or 0),
            "total_tokens": int(getattr(execution_result, "total_tokens", 0) or 0),
            "total_duration_ms": float(getattr(execution_result, "total_duration_ms", 0.0) or 0.0),
            "final_patch_non_empty": bool(final_patch and getattr(final_patch, "content", "").strip()),
            "final_patch_files": list(getattr(final_patch, "modified_files", []) or []),
        },
        "eval_outcome": _eval_outcome_payload(eval_outcome),
        "evaluation": evaluation_payload,
        "reward": reward_payload,
        "skill_evolution": skill_evolution or {
            "events": [],
            "dedup_decisions": [],
            "skill_stats_before": {},
            "skill_stats_after": {},
        },
    }


def _default_models() -> Dict[str, str]:
    config = get_config()
    return {"worker": config.llm.model, "controller": config.controller.model}


def _eval_outcome_payload(eval_outcome: Any) -> Optional[Dict[str, Any]]:
    if eval_outcome is None:
        return None
    getter = (
        eval_outcome.get
        if isinstance(eval_outcome, dict)
        else lambda k, d=None: getattr(eval_outcome, k, d)
    )
    return {
        "f2p_passed": int(getter("f2p_passed") or 0),
        "f2p_total": int(getter("f2p_total") or 0),
        "p2p_passed": int(getter("p2p_passed") or 0),
        "p2p_total": int(getter("p2p_total") or 0),
        "resolved": bool(getter("resolved") or False),
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
