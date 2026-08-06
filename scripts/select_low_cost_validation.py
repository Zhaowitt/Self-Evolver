#!/usr/bin/env python3
"""Select one short, deterministic validation prompt per repository."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def choose_per_repo(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        truth = json.loads(record["ground_truth"])
        grouped[str(truth["repo_name"])].append(record)
    selected = []
    for repo in sorted(grouped):
        # Prefer the shortest prompt to stay safely below EasyR1's token limit;
        # instance id is the deterministic tie-breaker.
        selected.append(
            min(
                grouped[repo],
                key=lambda item: (
                    len(item["prompt"]),
                    json.loads(item["ground_truth"])["instance_id"],
                ),
            )
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indomain", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    selected = choose_per_repo(load(args.indomain)) + choose_per_repo(load(args.ood))
    ids = [json.loads(item["ground_truth"])["instance_id"] for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("low-cost validation contains duplicate instance IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    args.audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection": "shortest prompt per repository",
                "count": len(selected),
                "instance_ids": ids,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} validation records across {len(selected)} repositories")


if __name__ == "__main__":
    main()
