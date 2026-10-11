#!/usr/bin/env python3
"""Pull/convert every eval image a round needs so cold SIF conversions don't stall reward workers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.test_backend import ContainerTestBackend  # noqa: E402
from src.reward.online_reward import issue_from_payload  # noqa: E402
from src.rl.online_rollout_runner import swebench_instance_from_issue  # noqa: E402
from src.rl.fixed_validation import atomic_write, load_source  # noqa: E402


class DiskSafetyStop(RuntimeError):
    """Stop new downloads, without removing any completed image."""


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
    parser.add_argument("--jsonl", type=Path, action="append", default=[])
    parser.add_argument("--fixed-validation-source", type=Path,
                        help="SHA-256-checked Full/dev 225-task source directory")
    parser.add_argument("--engine", default="apptainer", choices=["apptainer", "docker"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-free-gib", type=float, default=0,
                        help="Stop scheduling image pulls below this cache filesystem free space")
    parser.add_argument("--dry-run", action="store_true", help="Report cache coverage without downloading")
    parser.add_argument("--report", type=Path, help="Atomic progress report, resumable through the SIF cache")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=30)
    parser.add_argument("--min-tmp-free-gib", type=float, default=0)
    args = parser.parse_args()
    if not args.jsonl and not args.fixed_validation_source:
        parser.error("--jsonl or --fixed-validation-source is required")
    if args.workers < 1 or args.attempts < 1 or args.retry_delay < 0:
        parser.error("workers/attempts must be positive and retry-delay non-negative")

    backend = ContainerTestBackend(engine=args.engine)
    instances = instances_from_jsonl(args.jsonl)
    if args.fixed_validation_source:
        merged = {inst["instance_id"]: inst for inst in instances}
        for inst in load_source(args.fixed_validation_source):
            merged.setdefault(inst["instance_id"], inst)
        instances = list(merged.values())
    total = len(instances)
    if args.engine == "apptainer":
        missing = []
        for inst in instances:
            path = backend.sif_path(backend.image_key(inst))
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(inst)
        print(f"Cache coverage: {len(instances)-len(missing)}/{len(instances)} present; {len(missing)} missing", flush=True)
        instances = missing
    if args.dry_run:
        return
    print(f"Prewarming images for {len(instances)} instances with {args.workers} workers")
    failures = []
    completed = []
    missing_count = len(instances)
    def report(status: str) -> None:
        if args.report:
            atomic_write(args.report, json.dumps({
                "status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
                "target_count": total, "initial_cached": total - missing_count,
                "initial_missing": missing_count, "completed_count": len(completed),
                "completed_ids": completed, "failures": failures,
                "workers": args.workers, "min_free_gib": args.min_free_gib,
            }, indent=2) + "\n")

    def guarded_pull(inst: dict) -> str:
        if args.min_free_gib > 0:
            cache = backend.sif_cache_dir
            cache.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(cache).free / (1024 ** 3)
            if free < args.min_free_gib:
                raise DiskSafetyStop(f"Disk safety stop: {free:.1f} GiB free < {args.min_free_gib:.1f} GiB reserved")
        if args.min_tmp_free_gib > 0:
            import os
            tmp = Path(os.environ.get("APPTAINER_TMPDIR", "/tmp"))
            tmp.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(tmp).free / (1024 ** 3)
            if free < args.min_tmp_free_gib:
                raise DiskSafetyStop(f"Temporary disk safety stop: {free:.1f} GiB free")
        for attempt in range(args.attempts):
            try:
                return backend.ensure_image(inst)
            except Exception:
                if attempt + 1 >= args.attempts:
                    raise
                time.sleep(args.retry_delay)
        raise AssertionError("unreachable")

    report("running")
    # The production CPU job is deliberately sequential: stop immediately
    # on low disk, avoid queuing hundreds of futures, and retain each success.
    if args.workers == 1:
        for done, inst in enumerate(instances, start=1):
            instance_id = inst["instance_id"]
            try:
                image = guarded_pull(inst)
                completed.append(instance_id)
                print(f"[{done}/{missing_count}] ok {instance_id} ({image})", flush=True)
            except DiskSafetyStop as exc:
                failures.append((instance_id, str(exc)))
                report("disk_safety_stop")
                print(str(exc), file=sys.stderr, flush=True)
                sys.exit(2)
            except Exception as exc:
                failures.append((instance_id, str(exc)[:300]))
                print(f"[{done}/{missing_count}] FAILED {instance_id}: {exc}", file=sys.stderr, flush=True)
            report("running")
        report("complete" if not failures else "partial_failure")
        if failures:
            sys.exit(1)
        return

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(guarded_pull, inst): inst["instance_id"] for inst in instances}
        for done, future in enumerate(as_completed(futures), start=1):
            instance_id = futures[future]
            try:
                image = future.result()
                completed.append(instance_id)
                print(f"[{done}/{len(futures)}] ok {instance_id} ({image})")
            except Exception as exc:  # keep going; report at the end
                failures.append((instance_id, str(exc)[:300]))
                print(f"[{done}/{len(futures)}] FAILED {instance_id}: {exc}", file=sys.stderr)
            report("running")
    report("complete" if not failures else "partial_failure")
    if failures:
        print(f"{len(failures)} image(s) failed; those instances will pull lazily.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
