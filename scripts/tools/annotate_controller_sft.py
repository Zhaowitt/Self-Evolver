#!/usr/bin/env python3
"""Cross-model Controller SFT labels on repo-balanced full/train instances."""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.datasets import load_split, read_id_file  # noqa: E402
from src.config import LLMConfig  # noqa: E402
from src.controller.injection import ground_controller_skills  # noqa: E402
from src.controller.parser import parse_controller_response  # noqa: E402
from src.llm.client import LLMClient, Message  # noqa: E402
from src.rl.easyr1_dataset import (  # noqa: E402
    _repo_balanced_ids,
    build_easyr1_prompt_record,
    issue_from_swebench_item,
)


def label(client: LLMClient, record: dict) -> dict:
    allowed = json.loads(record["ground_truth"])["allowed_skill_ids"]
    try:
        response = client.chat([Message(role="user", content=record["prompt"])])
        signal = parse_controller_response(response.content or "")
    except Exception as exc:
        return {"error": str(exc)[:300]}
    if signal.parse_error:
        return {"error": signal.parse_error, "raw": signal.raw_response[:500]}
    signal = ground_controller_skills(signal, allowed_skill_ids=allowed)
    return {"selected_skill_ids": sorted(signal.selected_skill_ids), "strategy": signal.strategy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="two comma-separated annotator models")
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) != 2:
        parser.error("--models needs exactly two annotator models")

    excluded = set().union(*(read_id_file(p) for p in args.exclude_ids))
    issues = [
        issue_from_swebench_item(row)
        for row in load_split(args.dataset, args.split)
        if row["instance_id"] not in excluded
    ]
    chosen = set(_repo_balanced_ids(issues, args.num, random.Random(args.seed)))
    records = [
        build_easyr1_prompt_record(issue, stage="train", split=args.split, hard_cases=[])
        for issue in issues
        if issue.id in chosen
    ]

    clients = [
        LLMClient(LLMConfig(model=model, max_tokens=1024, temperature=0.0)) for model in models
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        answers = [list(pool.map(lambda r, c=c: label(c, r), records)) for c in clients]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    agreed, disputed = [], []
    for record, first, second in zip(records, *answers):
        instance_id = json.loads(record["ground_truth"])["instance_id"]
        if "error" not in first and "error" not in second and (
            first["selected_skill_ids"] == second["selected_skill_ids"]
        ):
            agreed.append({
                "instance_id": instance_id,
                "selected_skill_ids": first["selected_skill_ids"],
                "strategy": first["strategy"],
                "review_status": "cross_model_agreed",
            })
        else:
            disputed.append({"instance_id": instance_id, models[0]: first, models[1]: second})

    def write(name: str, rows: list) -> None:
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    write("prompts.jsonl", records)
    write("annotations.jsonl", agreed)
    write("disputed.jsonl", disputed)
    (out / "ids.txt").write_text("".join(f"{json.loads(r['ground_truth'])['instance_id']}\n" for r in records), encoding="utf-8")
    print(f"{len(agreed)} agreed / {len(disputed)} disputed of {len(records)} -> {out}")
    print("Resolve disputed.jsonl by hand (review_status=human_confirmed) and append to annotations.jsonl.")


if __name__ == "__main__":
    main()
