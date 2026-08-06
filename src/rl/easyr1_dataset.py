"""Build EasyR1 prompt records from SWE-bench instances."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from src.config import get_config
from src.controller.prompt_builder import ControllerPromptBuilder
from src.environment.models import Issue
from src.memory.hard_case_buffer import resolve_hard_cases_path
from src.memory.memory_retriever import MemoryRetriever
from src.rl.online_rollout_runner import build_targeted_test_cmd
from src.skills.skill_selector import SkillSelector
from src.tasks.task_pool import TaskPool


DATASET_MAP = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "SWE-bench/SWE-bench",
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
    # EasyR1's batch reward input does not reliably preserve arbitrary
    # ``extra_info`` fields. Keep the lifecycle boundary in ground_truth too so
    # validation can never be mistaken for train-time skill evolution.
    ground_truth["stage"] = stage
    ground_truth["split"] = split
    ground_truth["allowed_skill_ids"] = [skill.id for skill in candidate_skills]
    extra_info = {
        "instance_id": issue.id,
        "repo_name": issue.repo_name,
        "base_commit": issue.base_commit,
        "split": split,
        "stage": stage,
        "workspace_root": str(workspace_root) if workspace_root else "",
        "test_cmd": test_cmd or build_targeted_test_cmd(issue) or "",
        "allowed_skill_ids": [skill.id for skill in candidate_skills],
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
    include_ids: Optional[Iterable[str]] = None,
    task_signals: Optional[Dict[str, Any]] = None,
    task_pool_state_path: Optional[Path] = None,
    sampling_seed: int = 42,
    task_pool_uniform_fraction: float = 0.5,
    max_instance_repeats: int = 3,
) -> int:
    """Write a JSONL dataset of Controller prompts for EasyR1; returns row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # TaskPool must see the complete eligible source split.  Applying
    # ``num_instances`` here used to truncate full/test before sampling; because
    # SWE-bench rows are repository-grouped, NUM_TRAIN=320 silently restricted
    # the pool to Astropy and Django.
    pool_sampling = stage == "train" and task_pool_state_path and task_signals
    issues = list(
        iter_swebench_issues(
            dataset=dataset,
            split=split,
            limit=None if pool_sampling else num_instances,
            exclude_ids=exclude_ids,
            include_ids=include_ids,
        )
    )
    if stage == "train" and task_pool_state_path:
        pool = _task_pool_for_issues(issues, task_pool_state_path)
        if task_signals:
            pool.apply_reflection(task_signals)
            issues = _issues_from_pool_sample(
                issues,
                pool,
                sampling_seed,
                num_samples=num_instances,
                uniform_fraction=task_pool_uniform_fraction,
                max_instance_repeats=max_instance_repeats,
            )
        pool.save()
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for issue in issues:
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


def sample_issues_with_task_pool(
    issues: List[Issue],
    task_signals: Dict[str, Any],
    state_path: Path,
    seed: int,
    num_samples: Optional[int] = None,
    uniform_fraction: float = 0.5,
    max_instance_repeats: int = 3,
) -> List[Issue]:
    """Materialize a balanced, bounded TaskPool sample for a frozen GRPO round."""
    if not issues:
        return []
    pool = _task_pool_for_issues(issues, state_path)
    pool.apply_reflection(task_signals)
    result = _issues_from_pool_sample(
        issues,
        pool,
        seed,
        num_samples=num_samples,
        uniform_fraction=uniform_fraction,
        max_instance_repeats=max_instance_repeats,
    )
    pool.save()
    return result


def _task_pool_for_issues(issues: List[Issue], state_path: Path) -> TaskPool:
    return TaskPool.from_instances(
        [_issue_to_pool_item(issue) for issue in issues],
        state_path=state_path,
        verifier=None,
    )


def _issues_from_pool_sample(
    issues: List[Issue],
    pool: TaskPool,
    seed: int,
    num_samples: Optional[int] = None,
    uniform_fraction: float = 0.5,
    max_instance_repeats: int = 3,
) -> List[Issue]:
    if not 0.0 <= uniform_fraction <= 1.0:
        raise ValueError("uniform_fraction must be between 0 and 1")
    if max_instance_repeats < 1:
        raise ValueError("max_instance_repeats must be at least 1")

    by_id = {issue.id: issue for issue in issues}
    target = len(issues) if num_samples is None else int(num_samples)
    if target < 0:
        raise ValueError("num_samples must be non-negative")
    if target > len(by_id) * max_instance_repeats:
        raise ValueError(
            f"Cannot draw {target} records from {len(by_id)} instances with "
            f"max_instance_repeats={max_instance_repeats}"
        )
    if target == 0:
        return []

    rng = random.Random(seed)
    uniform_target = min(target, round(target * uniform_fraction), len(by_id))
    chosen_ids = _repo_balanced_ids(list(by_id.values()), uniform_target, rng)
    counts = Counter(chosen_ids)

    # Fill the remaining quota using the evolved TaskPool weights.  The hard
    # cap is on materialized records, not merely on the weight multiplier.
    weights_by_id = pool.weights()
    while len(chosen_ids) < target:
        candidates = [
            instance_id
            for instance_id in by_id
            if counts[instance_id] < max_instance_repeats
        ]
        if not candidates:
            raise RuntimeError("TaskPool repeat cap exhausted before sample was full")
        weights = [max(0.0, weights_by_id.get(instance_id, 0.0)) for instance_id in candidates]
        selected = (
            rng.choices(candidates, weights=weights, k=1)[0]
            if sum(weights) > 0.0
            else rng.choice(candidates)
        )
        chosen_ids.append(selected)
        counts[selected] += 1

    rng.shuffle(chosen_ids)
    return [by_id[instance_id] for instance_id in chosen_ids]


def _repo_balanced_ids(issues: List[Issue], n: int, rng: random.Random) -> List[str]:
    """Draw distinct IDs round-robin across repositories."""
    by_repo: Dict[str, List[str]] = defaultdict(list)
    for issue in issues:
        by_repo[issue.repo_name or "unknown"].append(issue.id)
    for ids in by_repo.values():
        rng.shuffle(ids)

    selected: List[str] = []
    active = sorted(by_repo)
    while active and len(selected) < n:
        rng.shuffle(active)
        next_active: List[str] = []
        for repo in active:
            ids = by_repo[repo]
            if ids and len(selected) < n:
                selected.append(ids.pop())
            if ids:
                next_active.append(repo)
        active = next_active
    return selected


def load_task_signals(path: Path) -> Dict[str, Any]:
    """Load either a reflection report or a direct task-signals JSON object."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    signals = (payload.get("reflection") or {}).get("task_signals") or payload.get(
        "task_signals"
    )
    if signals is None and ("instance_boosts" in payload or "family_boosts" in payload):
        signals = payload
    if not isinstance(signals, dict):
        raise ValueError(f"No task_signals object found in {path}")
    return {
        "instance_boosts": dict(signals.get("instance_boosts") or {}),
        "family_boosts": dict(signals.get("family_boosts") or {}),
    }


def _issue_to_pool_item(issue: Issue) -> Dict[str, Any]:
    return {
        "instance_id": issue.id,
        "repo": issue.repo_name,
        "problem_statement": issue.description,
        "FAIL_TO_PASS": issue.metadata.get("fail_to_pass"),
        "PASS_TO_PASS": issue.metadata.get("pass_to_pass"),
    }


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
    include_ids: Optional[Iterable[str]] = None,
) -> Iterator[Issue]:
    """Yield SWE-bench instances as Issue objects (validates eagerly)."""
    validate_split(dataset, split)
    return issues_from_rows(
        _load_split(dataset, split),
        limit=limit,
        exclude_ids=exclude_ids,
        include_ids=include_ids,
    )


def issues_from_rows(
    rows: Iterable[Dict[str, Any]],
    limit: Optional[int] = None,
    exclude_ids: Optional[Iterable[str]] = None,
    include_ids: Optional[Iterable[str]] = None,
) -> Iterator[Issue]:
    """Convert raw rows to Issues, dropping contaminated ids before the limit."""
    exclude = set(exclude_ids or [])
    include = set(include_ids) if include_ids is not None else None
    count = 0
    for item in rows:
        if item.get("instance_id") in exclude:
            continue
        if include is not None and item.get("instance_id") not in include:
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
    path = resolve_hard_cases_path()
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
    parser.add_argument(
        "--task-signals-report",
        type=Path,
        default=None,
        help="Previous-round reflection report used to weight this frozen train dataset.",
    )
    parser.add_argument("--task-pool-state", type=Path, default=None)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--task-pool-uniform-fraction",
        type=float,
        default=0.5,
        help="Fraction drawn without replacement, balanced across repositories.",
    )
    parser.add_argument(
        "--max-instance-repeats",
        type=int,
        default=3,
        help="Maximum materialized copies of one instance in a frozen round.",
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
    parser.add_argument(
        "--include-ids",
        type=Path,
        default=None,
        help="Optional allowlist of instance ids (one per line or JSON list).",
    )
    args = parser.parse_args(argv)

    try:
        validate_split(args.dataset, args.split)
    except ValueError as exc:
        parser.error(str(exc))

    exclude_ids = read_id_file(args.exclude_ids) if args.exclude_ids else None
    include_ids = read_id_file(args.include_ids) if args.include_ids else None
    task_signals = load_task_signals(args.task_signals_report) if args.task_signals_report else None
    count = write_swebench_prompt_dataset(
        output_path=args.output,
        dataset=args.dataset,
        split=args.split,
        num_instances=args.num_instances,
        stage=args.stage,
        workspace_root=args.workspace_root,
        test_cmd=args.test_cmd,
        exclude_ids=exclude_ids,
        include_ids=include_ids,
        task_signals=task_signals,
        task_pool_state_path=args.task_pool_state,
        sampling_seed=args.sampling_seed,
        task_pool_uniform_fraction=args.task_pool_uniform_fraction,
        max_instance_repeats=args.max_instance_repeats,
    )
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
