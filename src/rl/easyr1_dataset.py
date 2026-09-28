"""Build EasyR1 prompt records from SWE-bench instances."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from src.benchmark.datasets import load_split, read_id_file, validate_split
from src.controller.prompt_builder import (
    CONTROLLER_PROMPT_FORMAT,
    MAX_CANDIDATE_SKILLS,
    ControllerPromptBuilder,
)
from src.environment.models import Issue
from src.environment.f2p_subset import F2P_SUBSET_POLICY, F2PSubsetSelector
from src.memory.hard_case_buffer import resolve_hard_cases_path
from src.memory.memory_retriever import MemoryRetriever
from src.rl.online_rollout_runner import build_targeted_test_cmd
from src.skills.skill_selector import SkillSelector
from src.tasks.task_pool import TaskPool


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
    candidate_skills = controller_candidate_skills(issue, skill_selector or SkillSelector())
    hard_case_items = list(hard_cases if hard_cases is not None else retrieve_hard_cases(issue))
    skill_payloads = [skill.to_dict() for skill in candidate_skills]
    prompt_text = prompt_builder.build_prompt_text(
        issue,
        mode=stage,
        skills=skill_payloads,
        hard_cases=hard_case_items,
    )
    ground_truth = issue_to_ground_truth(issue)
    round_id = str(issue.metadata.get("round_id") or os.getenv("SELF_EVOLVER_ROUND_ID") or "0")
    train_subset = F2PSubsetSelector().select(
        issue.metadata.get("fail_to_pass"), issue.id, round_id=round_id
    )
    ground_truth["TRAIN_FAIL_TO_PASS_SUBSET"] = train_subset
    ground_truth["F2P_SUBSET_POLICY"] = F2P_SUBSET_POLICY
    ground_truth["round_id"] = round_id
    # Also top-level in ground_truth: validation must never be taken for training.
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
        "TRAIN_FAIL_TO_PASS_SUBSET": train_subset,
        "F2P_SUBSET_POLICY": F2P_SUBSET_POLICY,
        "round_id": round_id,
        "prompt_format": CONTROLLER_PROMPT_FORMAT,
    }
    # EasyR1 passes only response / response_length / ground_truth to the
    # reward function; extra_info must travel inside ground_truth to arrive.
    ground_truth["extra_info"] = dict(extra_info)
    return {
        # Plain text: EasyR1 wraps it as the single user message itself.
        "prompt": prompt_text,
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        "extra_info": extra_info,
    }


def controller_candidate_skills(issue: Issue, skill_selector: SkillSelector) -> list:
    """Single candidate-set rule shared by GRPO/SFT prompts and served eval."""
    # Failure-replay tasks rank skills targeting their observed failure first.
    return skill_selector.select_candidates(
        memory_query=issue.description,
        limit=MAX_CANDIDATE_SKILLS,
        target_failure_type=str(issue.metadata.get("observed_failure_label") or ""),
    )


def issue_to_ground_truth(issue: Issue) -> Dict[str, Any]:
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
        "observed_failure_label": issue.metadata.get("observed_failure_label", ""),
        "stable_cluster_id": issue.metadata.get("stable_cluster_id", ""),
        "sampling_source": issue.metadata.get("sampling_source", ""),
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
    # TaskPool must see the full eligible split: rows are repository-grouped, so
    # truncating before sampling would restrict the pool to the first repos.
    pool_sampling = stage == "train" and bool(task_pool_state_path)
    issues = list(
        iter_swebench_issues(
            dataset=dataset,
            split=split,
            limit=None if pool_sampling else num_instances,
            exclude_ids=exclude_ids,
            include_ids=include_ids,
        )
    )
    if pool_sampling:
        issues = sample_issues_with_task_pool(
            issues,
            task_signals or {},
            task_pool_state_path,
            sampling_seed,
            num_samples=num_instances,
            uniform_fraction=task_pool_uniform_fraction,
            max_instance_repeats=max_instance_repeats,
        )
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
    if task_signals:
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
    if os.getenv("SELF_EVOLVER_SAMPLING_MODE", "u_f").strip() == "legacy":
        return _legacy_pool_sample(issues, pool, rng, target, uniform_fraction, max_instance_repeats)
    unseen_ids = [instance_id for instance_id in pool.unseen_instance_ids() if instance_id in by_id]
    failure_groups = {
        label: [instance_id for instance_id in ids if instance_id in by_id]
        for label, ids in pool.observed_failure_groups().items()
    }
    failure_groups = {label: ids for label, ids in failure_groups.items() if ids}

    unseen_target = round(target * uniform_fraction)
    unseen_issues = [by_id[instance_id] for instance_id in unseen_ids]
    unseen_chosen = _repo_balanced_ids(unseen_issues, min(unseen_target, len(unseen_ids)), rng)
    chosen: List[tuple[str, str, str]] = [
        (instance_id, "unseen_random", "") for instance_id in unseen_chosen
    ]
    counts = Counter(unseen_chosen)

    failure_target = target - unseen_target
    replay_ids = _failure_balanced_ids(
        failure_groups,
        failure_target,
        counts,
        by_id,
        max_instance_repeats,
        rng,
        gap_counts=pool.failure_gap_counts(),
    )
    for instance_id, label in replay_ids:
        chosen.append((instance_id, "observed_failure_replay", label))
        counts[instance_id] += 1

    # Backfill order: unseen, lowest-attempt unresolved, then anything eligible;
    # resolved tasks are a last resort.
    label_by_id = {
        instance_id: label for label, ids in failure_groups.items() for instance_id in ids
    }
    unresolved = sorted(
        (
            instance_id
            for instance_id in by_id
            if pool.attempts_of(instance_id) > 0 and pool.last_resolved_of(instance_id) is False
        ),
        key=lambda instance_id: (pool.attempts_of(instance_id), instance_id),
    )
    tiers = [
        ("unseen_random_fallback", unseen_ids),
        ("observed_failure_replay_fallback", unresolved),
        ("eligible_pool_fallback", sorted(by_id)),
    ]
    while len(chosen) < target:
        for source, pool_ids in tiers:
            candidates = [
                instance_id for instance_id in pool_ids if counts[instance_id] < max_instance_repeats
            ]
            if candidates:
                break
        else:
            raise RuntimeError("TaskPool repeat cap exhausted before sample was full")
        if source == "observed_failure_replay_fallback":
            # Keep the lowest-attempt preference but randomize among ties.
            lowest = pool.attempts_of(candidates[0])
            candidates = [c for c in candidates if pool.attempts_of(c) == lowest]
        selected = rng.choice(candidates)
        chosen.append((selected, source, label_by_id.get(selected, "")))
        counts[selected] += 1

    rng.shuffle(chosen)
    result: List[Issue] = []
    for instance_id, source, label in chosen:
        issue = by_id[instance_id]
        metadata = {
            **issue.metadata,
            "sampling_source": source,
            "observed_failure_label": label,
            "stable_cluster_id": pool.stable_cluster_id_of(instance_id),
        }
        result.append(replace(issue, metadata=metadata))
    return result


def _legacy_pool_sample(
    issues: List[Issue],
    pool: TaskPool,
    rng: random.Random,
    target: int,
    uniform_fraction: float,
    max_instance_repeats: int,
) -> List[Issue]:
    """Ablation arm: repo-balanced uniform half plus a half weighted by TaskPool family/boost weights."""
    by_id = {issue.id: issue for issue in issues}
    counts: Counter = Counter()
    chosen: List[tuple[str, str]] = []
    uniform_target = round(target * uniform_fraction)
    for instance_id in _repo_balanced_ids(issues, min(uniform_target, len(by_id)), rng):
        chosen.append((instance_id, "legacy_uniform"))
        counts[instance_id] += 1
    while len(chosen) < target:
        draw = [
            item["instance_id"]
            for item in pool.sample(target - len(chosen), rng)
            if item["instance_id"] in by_id and counts[item["instance_id"]] < max_instance_repeats
        ]
        if not draw:
            raise RuntimeError("TaskPool repeat cap exhausted before sample was full")
        for instance_id in draw[: target - len(chosen)]:
            chosen.append((instance_id, "legacy_weighted"))
            counts[instance_id] += 1
    rng.shuffle(chosen)
    return [
        replace(
            by_id[instance_id],
            metadata={
                **by_id[instance_id].metadata,
                "sampling_source": source,
                "observed_failure_label": "",
                "stable_cluster_id": pool.stable_cluster_id_of(instance_id),
            },
        )
        for instance_id, source in chosen
    ]


def _failure_balanced_ids(
    groups: Dict[str, List[str]],
    target: int,
    counts: Counter,
    by_id: Dict[str, Issue],
    repeat_cap: int,
    rng: random.Random,
    label_fraction_cap: float = 0.4,
    gap_counts: Optional[Dict[str, int]] = None,
) -> List[tuple[str, str]]:
    """Gap-proportional capped label quotas, drawn repo-balanced; any shortfall is left to the caller."""
    if target <= 0 or not groups:
        return []
    labels = sorted(groups)
    gaps = {label: max(1, int((gap_counts or {}).get(label, len(groups[label])))) for label in labels}
    cap = max(1, int(target * label_fraction_cap))
    quotas = _proportional_quotas(gaps, target, cap)

    local_counts: Counter = Counter()
    label_counts: Counter = Counter()
    repo_counts: Dict[str, Counter] = {label: Counter() for label in labels}
    chosen: List[tuple[str, str]] = []

    def candidates_for(label: str) -> List[str]:
        return [
            instance_id
            for instance_id in groups[label]
            if counts[instance_id] + local_counts[instance_id] < repeat_cap
        ]

    def draw(label: str) -> bool:
        candidates = candidates_for(label)
        if not candidates:
            return False
        min_repo = min(
            repo_counts[label][by_id[instance_id].repo_name or "unknown"]
            for instance_id in candidates
        )
        candidates = [
            instance_id
            for instance_id in candidates
            if repo_counts[label][by_id[instance_id].repo_name or "unknown"] == min_repo
        ]
        selected = rng.choice(candidates)
        chosen.append((selected, label))
        local_counts[selected] += 1
        label_counts[label] += 1
        repo_counts[label][by_id[selected].repo_name or "unknown"] += 1
        return True

    # 1) each label fills its own quota
    for label in labels:
        while len(chosen) < target and label_counts[label] < quotas[label] and draw(label):
            pass
    # 2) redistribute unused quota, largest gap first; lift the cap only as a last resort
    order = sorted(labels, key=lambda label: (-gaps[label], label))
    for respect_cap in (True, False):
        progressed = True
        while len(chosen) < target and progressed:
            progressed = False
            for label in order:
                if len(chosen) >= target:
                    break
                if respect_cap and label_counts[label] >= cap:
                    continue
                progressed = draw(label) or progressed
    return chosen


def _proportional_quotas(gaps: Dict[str, int], target: int, cap: int) -> Dict[str, int]:
    """Largest-remainder apportionment with a per-label cap and a floor of one."""
    labels = sorted(gaps)
    total = sum(gaps.values())
    floor_one = target >= len(labels)
    raw = {label: target * gaps[label] / total for label in labels}
    quotas = {label: min(cap, int(raw[label])) for label in labels}
    if floor_one:
        quotas = {label: max(1, quota) for label, quota in quotas.items()}
    # The floor of one can overshoot the target: trim the largest quotas.
    while sum(quotas.values()) > target:
        largest = max(labels, key=lambda label: (quotas[label], label))
        if quotas[largest] <= 1:
            break
        quotas[largest] -= 1
    remainder = sorted(labels, key=lambda label: (-(raw[label] - int(raw[label])), label))
    while sum(quotas.values()) < target:
        grew = False
        for label in remainder:
            if sum(quotas.values()) >= target:
                break
            if quotas[label] < cap:
                quotas[label] += 1
                grew = True
        if not grew:
            break
    return quotas


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
        "observed_failure_labels": dict(signals.get("observed_failure_labels") or {}),
        "stable_cluster_ids": dict(signals.get("stable_cluster_ids") or {}),
    }


def _issue_to_pool_item(issue: Issue) -> Dict[str, Any]:
    return {
        "instance_id": issue.id,
        "repo": issue.repo_name,
        "problem_statement": issue.description,
        "FAIL_TO_PASS": issue.metadata.get("fail_to_pass"),
        "PASS_TO_PASS": issue.metadata.get("pass_to_pass"),
    }


def iter_swebench_issues(
    dataset: str = "full",
    split: str = "train",
    limit: Optional[int] = None,
    exclude_ids: Optional[Iterable[str]] = None,
    include_ids: Optional[Iterable[str]] = None,
) -> Iterator[Issue]:
    """Validates eagerly, then yields Issues lazily."""
    return issues_from_rows(
        load_split(dataset, split),
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
    """Excluded ids are dropped before the limit applies."""
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


def issue_from_swebench_item(item: Dict[str, Any]) -> Issue:
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
            "train_fail_to_pass_subset": item.get("TRAIN_FAIL_TO_PASS_SUBSET"),
            "f2p_subset_policy": item.get("F2P_SUBSET_POLICY"),
            "round_id": item.get("round_id"),
        },
    )


def retrieve_hard_cases(issue: Issue, limit: int = 3) -> List[Dict[str, Any]]:
    path = resolve_hard_cases_path()
    if not path.exists():
        return []
    retriever = MemoryRetriever(path)
    return [
        record.to_dict()
        # Only train-stage memory may inform prompts: no eval leakage.
        for record in retriever.retrieve(repo_name=issue.repo_name, stage="train", limit=limit)
    ]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build EasyR1 Controller prompt JSONL from SWE-bench instances."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        default="full",
        help="registered dataset: lite, verified or full",
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
        "--task-pool-unseen-fraction",
        type=float,
        default=0.5,
        help="Fraction drawn from U={attempts==0}, repository-balanced (default 0.5).",
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
        action="append",
        default=[],
        help="File of instance ids (one per line or JSON list) to exclude, "
        "e.g. eval or Controller-SFT ids for contamination control; repeatable",
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

    exclude_ids = set().union(*(read_id_file(path) for path in args.exclude_ids)) or None
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
