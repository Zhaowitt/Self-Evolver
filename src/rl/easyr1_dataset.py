"""Build EasyR1 prompt records from SWE-bench instances."""

from __future__ import annotations

import argparse
import json
import os
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

# HF splits actually shipped per dataset; only `full` has a train split.
DATASET_SPLITS = {
    "lite": ("dev", "test"),
    "verified": ("test",),
    "full": ("train", "dev", "test"),
}

# save_to_disk layouts under $SWEBENCH_DATA_DIR (default <repo>/benchmarks).
# swe_bench_full_train holds only the train split of the full dataset.
LOCAL_DATASET_DIRS = {
    "lite": "swe_bench_lite",
    "verified": "swe_bench_verified",
    "full": "swe_bench_full_train",
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
        "environment_setup_commit": issue.metadata.get("environment_setup_commit"),
        "FAIL_TO_PASS": issue.metadata.get("fail_to_pass"),
        "PASS_TO_PASS": issue.metadata.get("pass_to_pass"),
    }


def write_swebench_prompt_dataset(
    output_path: Path,
    dataset: str = "full",
    split: str = "train",
    num_instances: Optional[int] = None,
    stage: str = "train",
    workspace_root: Optional[str | Path] = None,
    test_cmd: Optional[str] = None,
    exclude_ids: Optional[Iterable[str]] = None,
) -> int:
    """Write a JSONL dataset of Controller prompts for EasyR1; returns row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for issue in iter_swebench_issues(
            dataset=dataset,
            split=split,
            limit=num_instances,
            exclude_ids=exclude_ids,
        ):
            record = build_easyr1_prompt_record(
                issue,
                stage=stage,
                split=split,
                workspace_root=workspace_root,
                test_cmd=test_cmd,
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_split(dataset: str, split: str) -> None:
    """Reject unavailable dataset/split combinations with an actionable error."""
    splits = DATASET_SPLITS.get(dataset)
    if splits is None:
        return  # raw HF dataset name; let the hub resolve its splits
    if split not in splits:
        hint = (
            " For RL training data use --dataset full --split train."
            if split == "train"
            else ""
        )
        raise ValueError(
            f"dataset '{dataset}' ({DATASET_MAP[dataset]}) has no '{split}' split; "
            f"available splits: {', '.join(splits)}.{hint}"
        )


def iter_swebench_issues(
    dataset: str = "full",
    split: str = "train",
    limit: Optional[int] = None,
    exclude_ids: Optional[Iterable[str]] = None,
) -> Iterator[Issue]:
    """Yield SWE-bench instances as Issue objects (validates eagerly)."""
    validate_split(dataset, split)
    return issues_from_rows(
        _load_split(dataset, split),
        limit=limit,
        exclude_ids=exclude_ids,
    )


def issues_from_rows(
    rows: Iterable[Dict[str, Any]],
    limit: Optional[int] = None,
    exclude_ids: Optional[Iterable[str]] = None,
) -> Iterator[Issue]:
    """Convert raw rows to Issues, dropping contaminated ids before the limit."""
    exclude = set(exclude_ids or [])
    count = 0
    for item in rows:
        if item.get("instance_id") in exclude:
            continue
        yield issue_from_swebench_item(item)
        count += 1
        if limit is not None and count >= limit:
            break


def _load_split(dataset: str, split: str):
    """Load a dataset split, preferring local save_to_disk copies."""
    local = _local_split(dataset, split)
    if local is not None:
        return local
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets package required. Install with: pip install datasets") from exc
    return load_dataset(DATASET_MAP.get(dataset, dataset), split=split)


def _local_split(dataset: str, split: str):
    dir_name = LOCAL_DATASET_DIRS.get(dataset)
    if not dir_name:
        return None
    path = swebench_data_dir() / dir_name
    if not path.exists():
        return None
    from datasets import load_from_disk

    loaded = load_from_disk(str(path))
    if hasattr(loaded, "keys"):  # DatasetDict keyed by split
        return loaded[split] if split in loaded else None
    return loaded if split == "train" else None  # single saved split (full train)


def swebench_data_dir() -> Path:
    """Local benchmark data root: $SWEBENCH_DATA_DIR or <repo>/benchmarks."""
    env_dir = os.getenv("SWEBENCH_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[2] / "benchmarks"


def read_id_file(path: Path) -> set[str]:
    """Read instance ids from a file: one id per line (# comments) or a JSON list."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return {str(item) for item in json.loads(text)}
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


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
            "environment_setup_commit": item.get("environment_setup_commit"),
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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build EasyR1 Controller prompt JSONL from SWE-bench instances."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        default="full",
        help="lite, verified, full, or HF dataset name (only 'full' has a train split)",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--stage", choices=["train", "eval"], default="train")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--test-cmd", default=None)
    parser.add_argument(
        "--exclude-ids",
        type=Path,
        default=None,
        help="File of instance ids (one per line or JSON list) to exclude, "
        "e.g. eval ids for contamination control",
    )
    args = parser.parse_args(argv)

    try:
        validate_split(args.dataset, args.split)
    except ValueError as exc:
        parser.error(str(exc))

    exclude_ids = read_id_file(args.exclude_ids) if args.exclude_ids else None
    count = write_swebench_prompt_dataset(
        output_path=args.output,
        dataset=args.dataset,
        split=args.split,
        num_instances=args.num_instances,
        stage=args.stage,
        workspace_root=args.workspace_root,
        test_cmd=args.test_cmd,
        exclude_ids=exclude_ids,
    )
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
