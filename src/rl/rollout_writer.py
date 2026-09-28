"""JSONL rollout logging (``rollout_record_v2``): add fields, never repurpose existing ones."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import get_config
from src.controller.schema import controller_signal_from_any
from src.environment.models import Issue
from src.reflection.failure_diagnosis import diagnose_failed_trajectory
from src.skills.file_lock import file_lock

SCHEMA_VERSION = "rollout_record_v2"


class RolloutWriter:
    """flock-guarded JSONL appender, safe for parallel workers."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_json_safe(record), ensure_ascii=False) + "\n"
        with file_lock(self.path):
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()


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
    records = list(getattr(execution_result, "iteration_records", []) or [])
    status_value = getattr(getattr(execution_result, "status", None), "value", "")
    failure_diagnosis = (
        None
        if bool(getattr(execution_result, "success", False))
        else diagnose_failed_trajectory(records, status_value or "failed").to_dict()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "rollout_id": rollout_id or str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "instance_id": issue.id,
        "repo_name": issue.repo_name,
        "base_commit": issue.base_commit,
        "stage": stage or (signal.mode if signal else "train"),
        # Lets post-round updates ingest only this round's rollouts.
        "round_id": str(
            issue.metadata.get("round_id") or os.getenv("SELF_EVOLVER_ROUND_ID") or ""
        ),
        "seed": seed,
        "experiment": experiment,
        "models": models or _default_models(),
        "mode": signal.mode if signal else "",
        "controller_signal": signal.to_dict() if signal else None,
        "selected_skill_ids": signal.selected_skill_ids if signal else [],
        "skill_set_key": sorted(signal.selected_skill_ids) if signal else [],
        "observed_failure_label": issue.metadata.get("observed_failure_label", ""),
        "stable_cluster_id": issue.metadata.get("stable_cluster_id", ""),
        "sampling_source": issue.metadata.get("sampling_source", ""),
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
        "verification_evidence": _verification_evidence(records),
        "eval_cost": _eval_cost(records),
        "failure_diagnosis": failure_diagnosis,
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
        "evaluation_scope": str(getter("evaluation_scope", "full") or "full"),
        "selected_f2p_ids": list(getter("selected_f2p_ids", []) or []),
        "duration_seconds": float(getter("duration_seconds", 0.0) or 0.0),
        "cache_hit": bool(getter("cache_hit", False) or False),
        "timed_out": bool(getter("timed_out", False) or False),
    }


def _verification_evidence(records: list[Any]) -> list[Dict[str, Any]]:
    evidence: list[Dict[str, Any]] = []
    for record in records:
        verification = getattr(record, "verification_result", None)
        if not verification:
            continue
        item: Dict[str, Any] = {
            "iteration": int(getattr(record, "iteration", 0) or 0),
            "status": getattr(getattr(verification, "status", None), "value", ""),
            "patch_applied": bool(getattr(verification, "patch_applied", False)),
            "evaluation_scope": getattr(verification, "evaluation_scope", "none"),
            "official_eval_executed": bool(
                getattr(verification, "official_eval_executed", False)
            ),
        }
        subset = getattr(verification, "subset_eval_outcome", None)
        full = getattr(verification, "full_eval_outcome", None)
        if subset is not None:
            item["subset"] = _eval_outcome_payload(subset)
        if full is not None:
            item["full"] = _eval_outcome_payload(full)
        evidence.append(item)
    return evidence


def _eval_cost(records: list[Any]) -> Dict[str, Any]:
    """Container evaluations this rollout triggered, by scope."""
    cost: Dict[str, Any] = {
        "precheck_rejections": 0,
        "subset_runs": 0,
        "f2p_only_runs": 0,
        "full_runs": 0,
        "cache_hits": 0,
        "timeouts": 0,
        "container_seconds": 0.0,
    }
    for record in records:
        verification = getattr(record, "verification_result", None)
        if not verification:
            continue
        if getattr(verification, "evaluation_scope", "") == "precheck":
            cost["precheck_rejections"] += 1
        for key, name in (("subset_runs", "subset_eval_outcome"), ("full_runs", "full_eval_outcome")):
            outcome = getattr(verification, name, None)
            if outcome is None:
                continue
            if getattr(outcome, "cache_hit", False):
                cost["cache_hits"] += 1
            else:
                if key == "full_runs" and getattr(outcome, "evaluation_scope", "") == "f2p_only":
                    key = "f2p_only_runs"
                cost[key] += 1
                cost["timeouts"] += int(bool(getattr(outcome, "timed_out", False)))
                cost["container_seconds"] += float(getattr(outcome, "duration_seconds", 0.0) or 0.0)
    cost["container_seconds"] = round(cost["container_seconds"], 3)
    return cost


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
