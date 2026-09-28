#!/usr/bin/env python3
"""Pull/convert every eval image a round needs so cold SIF conversions don't stall reward workers."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.test_backend import ContainerTestBackend  # noqa: E402
from src.reward.online_reward import issue_from_payload  # noqa: E402
from src.rl.online_rollout_runner import swebench_instance_from_issue  # noqa: E402


def instances_from_jsonl(paths: list[Path]) -> list[dict]:
    seen: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            truth = row.get("ground_truth")
            truth = json.loads(truth) if isinstance(truth, str) else (truth or {})
            issue = issue_from_payload(truth, row.get("extra_info") or {})
            seen.setdefault(issue.id, swebench_instance_from_issue(issue))
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, action="append", required=True)
    parser.add_argument("--engine", default="apptainer", choices=["apptainer", "docker"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    backend = ContainerTestBackend(engine=args.engine)
    instances = instances_from_jsonl(args.jsonl)
    print(f"Prewarming images for {len(instances)} instances with {args.workers} workers")
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(backend.ensure_image, inst): inst["instance_id"] for inst in instances}
        for done, future in enumerate(as_completed(futures), start=1):
            instance_id = futures[future]
            try:
                image = future.result()
                print(f"[{done}/{len(futures)}] ok {instance_id} ({image})")
            except Exception as exc:  # keep going; report at the end
                failures.append((instance_id, str(exc)[:300]))
                print(f"[{done}/{len(futures)}] FAILED {instance_id}: {exc}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} image(s) failed; those instances will pull lazily.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
