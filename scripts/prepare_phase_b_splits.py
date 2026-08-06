#!/usr/bin/env python3
"""Create deterministic Phase B holdout and contamination-exclusion ID files."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.rl.easyr1_dataset import iter_swebench_issues, read_id_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-repo", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.per_repo < 1:
        parser.error("--per-repo must be at least 1")

    verified = read_id_file(args.verified_ids)
    candidates = list(
        iter_swebench_issues("full", "test", exclude_ids=verified)
    )
    by_repo: dict[str, list[str]] = defaultdict(list)
    for issue in candidates:
        by_repo[issue.repo_name].append(issue.id)

    rng = random.Random(args.seed)
    holdout: list[str] = []
    selected_by_repo: dict[str, list[str]] = {}
    for repo in sorted(by_repo):
        ids = sorted(by_repo[repo])
        selected = sorted(rng.sample(ids, min(args.per_repo, len(ids))))
        selected_by_repo[repo] = selected
        holdout.extend(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = args.output_dir / "in_domain_val_ids.txt"
    exclusion_path = args.output_dir / "train_exclude_ids.txt"
    audit_path = args.output_dir / "split_audit.json"
    holdout_path.write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")
    exclusion_path.write_text(
        "\n".join(sorted(verified | set(holdout))) + "\n", encoding="utf-8"
    )
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": args.seed,
                "per_repo": args.per_repo,
                "full_test_after_verified_exclusion": len(candidates),
                "verified_holdout_count": len(verified),
                "in_domain_holdout_count": len(holdout),
                "repositories": selected_by_repo,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Phase B candidates: {len(candidates)}")
    print(f"In-domain holdout: {len(holdout)} across {len(selected_by_repo)} repos")
    print(f"Training exclusion IDs: {len(verified | set(holdout))}")


if __name__ == "__main__":
    main()
