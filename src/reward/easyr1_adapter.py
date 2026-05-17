"""EasyR1-facing reward adapter for rollout JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from src.reward.reward_model import RewardModel


def score_rollout_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return an EasyR1-style reward payload from a rollout record."""
    reward = record.get("reward") or {}
    components = dict(reward.get("components") or {})
    total = reward.get("total")
    if total is None:
        total = _fallback_reward_from_record(record)
    return {
        "rollout_id": record.get("rollout_id"),
        "instance_id": record.get("instance_id"),
        "reward": float(total),
        "components": components,
    }


def convert_rewards(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in read_jsonl(input_path):
            out.write(json.dumps(score_rollout_record(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _fallback_reward_from_record(record: Dict[str, Any]) -> float:
    execution = record.get("execution") or {}
    if execution.get("success"):
        return 1.0
    if execution.get("final_patch_non_empty"):
        return 0.3
    return 0.0


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert rollout JSONL to EasyR1 reward JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    # Config is accepted for CLI stability; rollout records already contain
    # execution-derived rewards in the v1 workflow.
    if args.config:
        RewardModel.from_config_file(args.config)
    convert_rewards(args.input, args.output)


if __name__ == "__main__":
    main()
