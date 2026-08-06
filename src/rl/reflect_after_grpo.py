"""Run one skill-reflection pass over hard cases produced by a GRPO round."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.memory.hard_case_buffer import HardCaseRecord
from src.reflection.reflector import Reflector


def read_round_hard_cases(path: Path, start_offset: int) -> list[HardCaseRecord]:
    """Read complete train records appended at or after a byte offset."""
    path = Path(path)
    if not path.exists():
        return []
    size = path.stat().st_size
    offset = start_offset if 0 <= start_offset <= size else 0
    records: list[HardCaseRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        # A caller-provided offset may point into a line; discard that fragment.
        if offset and stream.tell() != 0:
            stream.seek(offset - 1)
            if stream.read(1) != "\n":
                stream.readline()
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = HardCaseRecord.from_dict(payload)
            if record.stage == "train":
                records.append(record)
    return records


def reflect_after_grpo(
    hard_cases_path: Path,
    start_offset: int,
    output_path: Path,
    round_id: str,
    min_cluster_size: int = 3,
    reflector: Optional[Reflector] = None,
) -> dict[str, Any]:
    """Reflect once over this GRPO round and persist an auditable report."""
    records = read_round_hard_cases(hard_cases_path, start_offset)
    active_reflector = reflector or Reflector(
        buffer_path=hard_cases_path,
        min_cluster_size=min_cluster_size,
    )
    lifecycle = active_reflector.skill_evolver.finalize_round(round_id)
    result = active_reflector.reflect_records(records)
    payload = {
        "schema_version": "grpo_skill_reflection_v1",
        "round_id": round_id,
        "created_at": datetime.now().isoformat(),
        "hard_cases_path": str(hard_cases_path),
        "start_offset": start_offset,
        "train_hard_case_count": len(records),
        "skill_lifecycle": lifecycle,
        "reflection": result.to_dict(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Reflect over train hard cases appended during one completed GRPO round."
    )
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--start-offset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    args = parser.parse_args(argv)
    payload = reflect_after_grpo(
        hard_cases_path=args.hard_cases,
        start_offset=args.start_offset,
        output_path=args.output,
        round_id=args.round_id,
        min_cluster_size=max(1, args.min_cluster_size),
    )
    reflection = payload["reflection"]
    applied = [
        event
        for event in ((reflection.get("evolution") or {}).get("events") or [])
        if event.get("applied")
    ]
    print(
        "GRPO skill reflection: "
        f"hard_cases={payload['train_hard_case_count']} "
        f"qualifying_clusters={reflection['qualifying_cluster_count']} "
        f"proposals={len(reflection['proposals'])} applied={len(applied)} "
        f"report={args.output}"
    )


if __name__ == "__main__":
    main()
