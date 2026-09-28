#!/usr/bin/env python3
"""Write ids whose gold patch evaluates flaky or broken across repeats (for ``EXCLUDE_IDS``)."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.test_backend import ContainerTestBackend, EvalCache  # noqa: E402


def _gold_instances(jsonl_paths: list[Path], dataset: str, split: str) -> list[dict]:
    from src.benchmark.datasets import load_split

    wanted: set[str] = set()
    for path in jsonl_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                truth = json.loads(line).get("ground_truth")
                truth = json.loads(truth) if isinstance(truth, str) else (truth or {})
                wanted.add(str(truth.get("instance_id")))
    return [dict(row) for row in load_split(dataset, split) if row["instance_id"] in wanted]


def classify(backend: ContainerTestBackend, instance: dict, repeats: int) -> dict:
    outcomes = [backend.run_full_eval(instance, instance["patch"]) for _ in range(repeats)]
    resolved = [bool(outcome.resolved) for outcome in outcomes]
    signatures = {
        (outcome.resolved, outcome.f2p_passed, outcome.p2p_passed) for outcome in outcomes
    }
    # Any count variation is flaky: a flaky P2P test can block a correct patch from resolving.
    if len(signatures) > 1:
        status = "flaky"
    elif not any(resolved):
        status = "broken"
    else:
        status = "stable"
    return {
        "instance_id": instance["instance_id"],
        "status": status,
        "resolved_runs": sum(resolved),
        "repeats": repeats,
        "f2p_passed": [outcome.f2p_passed for outcome in outcomes],
        "p2p_passed": [outcome.p2p_passed for outcome in outcomes],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, action="append", required=True)
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--split", default="test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--engine", default="apptainer", choices=["apptainer", "docker"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    backend = ContainerTestBackend(engine=args.engine)
    backend.eval_cache = EvalCache(None)  # every repeat must really run
    instances = _gold_instances(args.jsonl, args.dataset, args.split)
    print(f"Checking {len(instances)} instances x {args.repeats} gold runs")
    report = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(classify, backend, inst, args.repeats) for inst in instances]
        for future in as_completed(futures):
            item = future.result()
            report.append(item)
            if item["status"] != "stable":
                print(f"{item['status']:6s} {item['instance_id']} "
                      f"({item['resolved_runs']}/{item['repeats']} resolved)")
    excluded = sorted(item["instance_id"] for item in report if item["status"] != "stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{iid}\n" for iid in excluded), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(sorted(report, key=lambda item: item["instance_id"]), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{len(excluded)} flaky/broken instance(s) -> {args.output}")


if __name__ == "__main__":
    main()
