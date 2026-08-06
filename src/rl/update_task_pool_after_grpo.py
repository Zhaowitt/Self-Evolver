"""Fold one completed GRPO round's outcomes back into the persistent TaskPool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.tasks.task_pool import TaskPool


def read_round_rollouts(path: Path, start_offset: int) -> list[Dict[str, Any]]:
    """Read complete train rollout records appended after a byte offset."""
    path = Path(path)
    if not path.exists():
        return []
    size = path.stat().st_size
    offset = start_offset if 0 <= start_offset <= size else 0
    records: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        if offset:
            stream.seek(offset - 1)
            if stream.read(1) != "\n":
                stream.readline()
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("stage", "train") == "train":
                records.append(payload)
    return records


def update_task_pool_after_grpo(
    state_path: Path,
    train_jsonl: Path,
    rollouts_path: Path,
    start_offset: int,
    round_id: str,
) -> Dict[str, Any]:
    """Update family EMA and too-hard decay from this round's official outcomes."""
    instances = _state_instances(state_path) or _instances_from_train_jsonl(train_jsonl)
    pool = TaskPool.from_instances(instances, state_path=state_path, verifier=None)
    records = read_round_rollouts(rollouts_path, start_offset)
    ingestion_path = state_path.with_name(state_path.name + ".ingestion.json")
    processed = _load_processed_rollouts(ingestion_path)
    applied = 0
    skipped = 0
    duplicates = 0
    for record in records:
        rollout_id = str(record.get("rollout_id") or "")
        if rollout_id and rollout_id in processed:
            duplicates += 1
            continue
        instance_id = str(record.get("instance_id") or "")
        if instance_id not in pool:
            skipped += 1
            continue
        outcome = record.get("eval_outcome") or {}
        reward = record.get("reward") or {}
        pool.record_outcome(
            instance_id,
            resolved=bool(outcome.get("resolved", False)),
            utility=float(reward.get("total", 0.0) or 0.0),
        )
        applied += 1
        if rollout_id:
            processed.add(rollout_id)
    pool.save()
    _save_processed_rollouts(ingestion_path, processed)
    return {
        "round_id": round_id,
        "rollout_count": len(records),
        "applied_outcomes": applied,
        "skipped_outcomes": skipped,
        "duplicate_outcomes": duplicates,
        "task_pool_state": str(state_path),
    }


def _state_instances(path: Path) -> list[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items = payload.get("base_instances") if isinstance(payload, dict) else None
    return list(items.values()) if isinstance(items, dict) else []


def _instances_from_train_jsonl(path: Path) -> list[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ground_truth = _as_dict(row.get("ground_truth"))
        instance_id = str(ground_truth.get("instance_id") or "")
        if instance_id:
            unique.setdefault(
                instance_id,
                {
                    "instance_id": instance_id,
                    "repo": ground_truth.get("repo_name") or ground_truth.get("repo"),
                    "problem_statement": ground_truth.get("problem_statement", ""),
                    "FAIL_TO_PASS": ground_truth.get("FAIL_TO_PASS"),
                    "PASS_TO_PASS": ground_truth.get("PASS_TO_PASS"),
                },
            )
    return list(unique.values())


def _load_processed_rollouts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return {str(item) for item in payload.get("processed_rollout_ids", [])}


def _save_processed_rollouts(path: Path, rollout_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps({"processed_rollout_ids": sorted(rollout_ids)}, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-pool-state", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--start-offset", type=int, required=True)
    parser.add_argument("--round-id", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = update_task_pool_after_grpo(
        args.task_pool_state,
        args.train_jsonl,
        args.rollouts,
        args.start_offset,
        args.round_id,
    )
    print(
        "GRPO TaskPool update: "
        f"round={result['round_id']} outcomes={result['applied_outcomes']} "
        f"skipped={result['skipped_outcomes']} duplicates={result['duplicate_outcomes']} "
        f"state={result['task_pool_state']}"
    )


if __name__ == "__main__":
    main()
