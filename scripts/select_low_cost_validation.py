#!/usr/bin/env python3
"""Build a fixed, auditable 36-48 task validation set by coverage strata."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path, source: str) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        record["_validation_source"] = source
        records.append(record)
    return records


def choose_stratified(records: list[dict], target_size: int = 48) -> list[dict]:
    """Round-robin across source/repo/F2P-count/problem-length strata."""
    unique: dict[str, dict] = {}
    for record in records:
        truth = _truth(record)
        unique.setdefault(str(truth.get("instance_id") or ""), record)
    unique.pop("", None)
    if len(unique) <= target_size:
        return [unique[key] for key in sorted(unique)]

    strata: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for record in unique.values():
        truth = _truth(record)
        f2p_count = len(_list_value(truth.get("FAIL_TO_PASS")))
        problem_length = len(str(truth.get("problem_statement") or ""))
        key = (
            str(record.get("_validation_source") or "unknown"),
            str(truth.get("repo_name") or truth.get("repo") or "unknown"),
            "f2p_1" if f2p_count <= 1 else ("f2p_2_3" if f2p_count <= 3 else "f2p_4_plus"),
            "short" if problem_length < 1200 else ("medium" if problem_length < 3000 else "long"),
        )
        strata[key].append(record)
    for values in strata.values():
        values.sort(key=lambda item: _stable_rank(str(_truth(item)["instance_id"])))

    selected: list[dict] = []
    keys = sorted(strata)
    while keys and len(selected) < target_size:
        next_keys = []
        for key in keys:
            if strata[key] and len(selected) < target_size:
                selected.append(strata[key].pop(0))
            if strata[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indomain", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=48)
    args = parser.parse_args()
    if not 36 <= args.target_size <= 48:
        raise ValueError("target-size must be in the agreed 36..48 range")

    selected = choose_stratified(
        [*load(args.indomain, "indomain"), *load(args.ood, "ood")],
        target_size=args.target_size,
    )
    ids = [str(_truth(item)["instance_id"]) for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("validation contains duplicate instance IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(_clean(item), ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    audit = {
        "schema_version": 2,
        "selection": "deterministic source/repository/F2P-count/problem-length strata",
        "requested_count": args.target_size,
        "count": len(selected),
        "instance_ids": ids,
        "coverage": _coverage(selected),
    }
    args.audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(selected)} fixed stratified validation records")


def _coverage(records: list[dict]) -> dict[str, Any]:
    truths = [_truth(item) for item in records]
    return {
        "sources": sorted({str(item.get("_validation_source")) for item in records}),
        "repositories": sorted({str(item.get("repo_name") or item.get("repo")) for item in truths}),
        "f2p_counts": sorted({len(_list_value(item.get("FAIL_TO_PASS"))) for item in truths}),
    }


def _truth(record: dict) -> dict:
    value = record.get("ground_truth") or {}
    return json.loads(value) if isinstance(value, str) else dict(value)


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value) if isinstance(value, list) else []


def _stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_validation_")}


if __name__ == "__main__":
    main()
