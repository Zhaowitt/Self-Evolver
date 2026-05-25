"""Build EasyR1 prompt records from SWE-bench instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from src.config import get_config
from src.controller.prompt_builder import ControllerPromptBuilder
from src.environment.models import Issue
from src.memory.memory_retriever import MemoryRetriever
from src.rl.online_rollout_runner import build_targeted_test_cmd
from src.skills.skill_selector import SkillSelector


DATASET_MAP = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def build_easyr1_prompt_record(
    issue: Issue,
    stage: str = "train",
    split: str = "train",
    workspace_root: Optional[str | Path] = None,
    test_cmd: Optional[str] = None,
    prompt_builder: Optional[ControllerPromptBuilder] = None,
    skill_selector: Optional[SkillSelector] = None,
    hard_cases: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build one EasyR1 JSONL record with the real Controller prompt."""
    prompt_builder = prompt_builder or ControllerPromptBuilder()
    candidate_skills = (skill_selector or SkillSelector()).select_many(
        memory_query=issue.description,
        limit=2,
    )
    hard_case_items = list(hard_cases or retrieve_hard_cases(issue))
    skill_payloads = [skill.to_dict() for skill in candidate_skills]
    messages = [
        {"role": "system", "content": prompt_builder.system_prompt},
        {
            "role": "user",
            "content": prompt_builder.build_user_prompt(
                issue,
                mode=stage,
                skills=skill_payloads,
                hard_cases=hard_case_items,
            ),
        },
    ]
    ground_truth = issue_to_ground_truth(issue)
    extra_info = {
        "instance_id": issue.id,
        "repo_name": issue.repo_name,
        "base_commit": issue.base_commit,
        "split": split,
        "stage": stage,
        "workspace_root": str(workspace_root) if workspace_root else "",
        "test_cmd": test_cmd or build_targeted_test_cmd(issue) or "",
    }
    return {
        "prompt": json.dumps(messages, ensure_ascii=False),
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        "extra_info": extra_info,
    }


def issue_to_ground_truth(issue: Issue) -> Dict[str, Any]:
    """Serialize the SWE-bench fields required by the online reward function."""
    return {
        "instance_id": issue.id,
        "repo_name": issue.repo_name,
        "base_commit": issue.base_commit,
        "problem_statement": issue.description,
        "hints_text": issue.hints,
        "test_patch": issue.test_patch,
        "version": issue.metadata.get("version"),
        "FAIL_TO_PASS": issue.metadata.get("fail_to_pass"),
        "PASS_TO_PASS": issue.metadata.get("pass_to_pass"),
    }


def write_swebench_prompt_dataset(
    output_path: Path,
    dataset: str = "lite",
    split: str = "train",
    num_instances: Optional[int] = None,
    stage: str = "train",
    workspace_root: Optional[str | Path] = None,
    test_cmd: Optional[str] = None,
) -> None:
    """Write a JSONL dataset of Controller prompts for EasyR1."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for issue in iter_swebench_issues(dataset=dataset, split=split, limit=num_instances):
            record = build_easyr1_prompt_record(
                issue,
                stage=stage,
                split=split,
                workspace_root=workspace_root,
                test_cmd=test_cmd,
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_swebench_issues(
    dataset: str = "lite",
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[Issue]:
    """Yield SWE-bench instances as Issue objects."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets package required. Install with: pip install datasets") from exc

    dataset_name = DATASET_MAP.get(dataset, dataset)
    loaded = load_dataset(dataset_name, split=split)
    count = 0
    for item in loaded:
        yield issue_from_swebench_item(item)
        count += 1
        if limit is not None and count >= limit:
            break


def issue_from_swebench_item(item: Dict[str, Any]) -> Issue:
    """Convert a raw SWE-bench dataset row to an Issue."""
    return Issue(
        id=item["instance_id"],
        description=item["problem_statement"],
        repo_name=item.get("repo"),
        base_commit=item.get("base_commit"),
        hints=item.get("hints_text"),
        test_patch=item.get("test_patch"),
        metadata={
            "version": item.get("version"),
            "fail_to_pass": item.get("FAIL_TO_PASS"),
            "pass_to_pass": item.get("PASS_TO_PASS"),
        },
    )


def retrieve_hard_cases(issue: Issue, limit: int = 3) -> List[Dict[str, Any]]:
    """Retrieve similar hard cases for Controller prompt context."""
    path = get_config().environment.workspace_dir / "hard_cases.jsonl"
    if not path.exists():
        return []
    retriever = MemoryRetriever(path)
    return [
        record.to_dict()
        for record in retriever.retrieve(repo_name=issue.repo_name, limit=limit)
    ]


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build EasyR1 Controller prompt JSONL from SWE-bench instances."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", default="lite", help="lite, verified, full, or HF dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--stage", choices=["train", "eval"], default="train")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--test-cmd", default=None)
    args = parser.parse_args(argv)

    write_swebench_prompt_dataset(
        output_path=args.output,
        dataset=args.dataset,
        split=args.split,
        num_instances=args.num_instances,
        stage=args.stage,
        workspace_root=args.workspace_root,
        test_cmd=args.test_cmd,
    )


if __name__ == "__main__":
    main()
