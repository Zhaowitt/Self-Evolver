#!/usr/bin/env python3
"""Combine frozen prompt JSONL files while rejecting duplicate instance IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    records: list[dict] = []
    seen: set[str] = set()
    for path in args.inputs:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            ground_truth = json.loads(record["ground_truth"])
            instance_id = str(ground_truth["instance_id"])
            if instance_id in seen:
                raise ValueError(f"Duplicate instance_id {instance_id} at {path}:{line_number}")
            seen.add(instance_id)
            records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} unique records to {args.output}")


if __name__ == "__main__":
    main()
