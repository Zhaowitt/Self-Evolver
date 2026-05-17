"""Convert rollout JSONL into a compact EasyR1 dataset shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator


def rollout_to_easyr1_record(record: Dict[str, Any]) -> Dict[str, Any]:
    signal = record.get("controller_signal") or {}
    reward = record.get("reward") or {}
    prompt = {
        "instance_id": record.get("instance_id"),
        "repo_name": record.get("repo_name"),
        "mode": record.get("mode"),
    }
    return {
        "prompt": json.dumps(prompt, ensure_ascii=False),
        "response": json.dumps(signal, ensure_ascii=False),
        "reward": float(reward.get("total", 0.0)),
        "metadata": {
            "rollout_id": record.get("rollout_id"),
            "instance_id": record.get("instance_id"),
            "success": (record.get("execution") or {}).get("success", False),
        },
    }


def convert_rollouts(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in read_jsonl(input_path):
            out.write(json.dumps(rollout_to_easyr1_record(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
