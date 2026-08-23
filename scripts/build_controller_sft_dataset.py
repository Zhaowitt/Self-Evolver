#!/usr/bin/env python3
"""Join reviewed Controller labels with frozen prompts and audit bank/split IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_sft_dataset(
    prompts_path: Path,
    annotations_path: Path,
    output_path: Path,
    manifest_path: Path,
    skills_dir: Path,
    exclude_ids: set[str] | None = None,
) -> int:
    prompts = {_instance_id(row): row for row in _jsonl(prompts_path)}
    annotations = {_annotation_id(row): row for row in _jsonl(annotations_path)}
    excluded = set(exclude_ids or set())
    rows = []
    rejected = []
    for instance_id in sorted(annotations):
        if instance_id in excluded:
            rejected.append({"instance_id": instance_id, "reason": "contamination exclusion"})
            continue
        prompt = prompts.get(instance_id)
        annotation = annotations[instance_id]
        if not prompt:
            rejected.append({"instance_id": instance_id, "reason": "missing prompt"})
            continue
        truth = _dict_value(prompt.get("ground_truth"))
        allowed = set(truth.get("allowed_skill_ids") or prompt.get("extra_info", {}).get("allowed_skill_ids") or [])
        if not 4 <= len(allowed) <= 6:
            rejected.append(
                {
                    "instance_id": instance_id,
                    "reason": f"candidate skill count must be 4..6, got {len(allowed)}",
                }
            )
            continue
        selected = [str(item) for item in annotation.get("selected_skill_ids") or []]
        if len(selected) > 2 or any(item not in allowed for item in selected):
            rejected.append({"instance_id": instance_id, "reason": "invalid selected skill ids"})
            continue
        target = {"selected_skill_ids": selected}
        strategy = str(annotation.get("strategy") or "").strip()
        if strategy:
            target["strategy"] = strategy
        rows.append(
            {
                "prompt": prompt["prompt"],
                "response": json.dumps(target, ensure_ascii=False),
                "instance_id": instance_id,
                "candidate_skill_ids": sorted(allowed),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "controller_sft_manifest_v1",
        "count": len(rows),
        "instance_ids": [row["instance_id"] for row in rows],
        "skill_bank_hash": _skill_bank_hash(skills_dir),
        "candidate_skill_count_range": [
            min((len(row["candidate_skill_ids"]) for row in rows), default=0),
            max((len(row["candidate_skill_ids"]) for row in rows), default=0),
        ],
        "rejected": rejected,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, default=Path("skills"))
    parser.add_argument("--exclude-ids", type=Path, action="append", default=[])
    args = parser.parse_args()
    exclusions: set[str] = set()
    for path in args.exclude_ids:
        exclusions.update(_read_ids(path))
    count = build_sft_dataset(
        args.prompts,
        args.annotations,
        args.output,
        args.manifest,
        args.skills_dir,
        exclusions,
    )
    print(f"Wrote {count} reviewed Controller SFT records")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _instance_id(row: dict[str, Any]) -> str:
    truth = _dict_value(row.get("ground_truth"))
    return str(truth.get("instance_id") or row.get("instance_id") or "")


def _annotation_id(row: dict[str, Any]) -> str:
    return str(row.get("instance_id") or row.get("id") or "")


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _read_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return {str(item) for item in json.loads(text)}
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def _skill_bank_hash(skills_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(skills_dir.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    main()
