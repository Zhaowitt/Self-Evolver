"""Benchmark metrics (Proposal 3.3) from predictions, official reports, and rollouts.

Computes resolved rate, unbiased pass@k across repeated runs, success under a
fixed iteration budget, cost-to-success (tokens and dollars), hard-case success
rate, average tokens, and a per-rollout evolution curve. Emits JSON and a
Markdown table.

CLI::

    python -m src.benchmark.metrics --rollouts run1/rollouts.jsonl run2/rollouts.jsonl \
        --hard-ids hard_ids.txt --budget 3 --price-per-token 0.0 --output metrics.json

Each ``--rollouts`` file is one run (one sample for pass@k). A ``--report`` file
(a ``final_summary.json`` or swebench run report) overrides the resolved set for
the run at the same position. Without ``--rollouts`` a single run is built from
``--predictions`` + ``--report``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 20
MAX_CURVE_POINTS = 60


@dataclass
class InstanceOutcome:
    resolved: bool
    iterations: int = 0
    tokens: int = 0
    non_empty: bool = True


@dataclass
class Run:
    label: str
    outcomes: Dict[str, InstanceOutcome] = field(default_factory=dict)
    train_resolved: List[bool] = field(default_factory=list)  # in rollout order


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: probability that k of n samples includes a resolve."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def read_id_file(path: Path) -> set:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return {str(item) for item in json.loads(text)}
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_report_resolved(path: Path) -> set:
    """Resolved instance ids from a final_summary.json or a swebench run report."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data.get("resolved"), list):
        return {str(i) for i in data["resolved"]}
    if isinstance(data.get("resolved_ids"), list):  # swebench run report
        return {str(i) for i in data["resolved_ids"]}
    return set()


def _record_resolved(record: dict) -> bool:
    eval_outcome = record.get("eval_outcome")
    if isinstance(eval_outcome, dict):
        return bool(eval_outcome.get("resolved"))
    evaluation = record.get("evaluation") or {}
    return bool(evaluation.get("success"))


def load_rollouts_run(path: Path, report_ids: Optional[set] = None, label: str = "") -> Run:
    """Build a run from a rollout JSONL; the last record per instance wins."""
    run = Run(label=label or Path(path).parent.name or Path(path).stem)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        instance_id = record.get("instance_id")
        if not instance_id:
            continue
        execution = record.get("execution") or {}
        resolved = instance_id in report_ids if report_ids is not None else _record_resolved(record)
        run.outcomes[instance_id] = InstanceOutcome(
            resolved=resolved,
            iterations=int(execution.get("iterations_used") or 0),
            tokens=int(execution.get("total_tokens") or 0),
            non_empty=bool(execution.get("final_patch_non_empty")),
        )
        if record.get("stage") == "train":
            run.train_resolved.append(resolved)
    return run


def load_predictions_run(path: Path, report_ids: set, label: str = "") -> Run:
    """Build a run from a predictions file plus a resolved-id set (no token data)."""
    predictions = json.loads(Path(path).read_text(encoding="utf-8"))
    run = Run(label=label or Path(path).parent.name or Path(path).stem)
    for prediction in predictions:
        instance_id = prediction.get("instance_id")
        if not instance_id:
            continue
        run.outcomes[instance_id] = InstanceOutcome(
            resolved=instance_id in report_ids,
            non_empty=bool(prediction.get("model_patch", "").strip()),
        )
    return run


def _run_metrics(
    run: Run,
    hard_ids: Optional[set],
    budget: Optional[int],
    price_per_token: float,
) -> dict:
    outcomes = list(run.outcomes.values())
    total = len(outcomes)
    resolved = [o for o in outcomes if o.resolved]
    tokens = [o.tokens for o in outcomes]
    resolved_tokens = [o.tokens for o in resolved]
    metrics = {
        "label": run.label,
        "total": total,
        "resolved_count": len(resolved),
        "resolved_rate": len(resolved) / total if total else 0.0,
        "non_empty_rate": sum(o.non_empty for o in outcomes) / total if total else 0.0,
        "avg_tokens": sum(tokens) / total if total else 0.0,
        "cost_to_success_tokens": sum(resolved_tokens) / len(resolved) if resolved else 0.0,
        "cost_to_success_usd": (sum(resolved_tokens) / len(resolved) * price_per_token)
        if resolved else 0.0,
    }
    if budget is not None:
        under = [o for o in resolved if o.iterations <= budget]
        metrics["success_under_budget"] = len(under) / total if total else 0.0
        metrics["budget"] = budget
    if hard_ids:
        hard = [o for iid, o in run.outcomes.items() if iid in hard_ids]
        hard_resolved = sum(o.resolved for o in hard)
        metrics["hard_case_count"] = len(hard)
        metrics["hard_case_success_rate"] = hard_resolved / len(hard) if hard else 0.0
    return metrics


def _pass_at_k(runs: List[Run]) -> Dict[str, float]:
    per_instance: Dict[str, List[int]] = {}
    for run in runs:
        for instance_id, outcome in run.outcomes.items():
            counts = per_instance.setdefault(instance_id, [0, 0])
            counts[0] += 1
            counts[1] += int(outcome.resolved)
    result: Dict[str, float] = {}
    for k in range(1, len(runs) + 1):
        eligible = [(n, c) for n, c in per_instance.values() if n >= k]
        if eligible:
            result[str(k)] = sum(pass_at_k(n, c, k) for n, c in eligible) / len(eligible)
    return result


def _evolution_curve(runs: List[Run], window: int = DEFAULT_WINDOW) -> List[dict]:
    events: List[bool] = []
    for run in runs:
        events.extend(run.train_resolved)
    if not events:
        return []
    stride = max(1, len(events) // MAX_CURVE_POINTS)
    curve: List[dict] = []
    for index in range(0, len(events), stride):
        upto = events[: index + 1]
        window_slice = events[max(0, index - window + 1): index + 1]
        curve.append({
            "rollout": index + 1,
            "cumulative_resolved_rate": sum(upto) / len(upto),
            "window_resolved_rate": sum(window_slice) / len(window_slice),
        })
    return curve


def compute_metrics(
    runs: List[Run],
    hard_ids: Optional[set] = None,
    budget: Optional[int] = None,
    price_per_token: float = 0.0,
) -> dict:
    if not runs:
        raise ValueError("no runs to score; pass --rollouts or --predictions + --report")
    per_run = [_run_metrics(run, hard_ids, budget, price_per_token) for run in runs]
    keys = ["resolved_rate", "avg_tokens", "cost_to_success_tokens"]
    if budget is not None:
        keys.append("success_under_budget")
    if hard_ids:
        keys.append("hard_case_success_rate")
    aggregate = {
        key: sum(m.get(key, 0.0) for m in per_run) / len(per_run)
        for key in keys
    }
    return {
        "num_runs": len(runs),
        "aggregate": aggregate,
        "pass_at_k": _pass_at_k(runs),
        "per_run": per_run,
        "evolution_curve": _evolution_curve(runs),
    }


def render_markdown(metrics: dict) -> str:
    lines = ["# Benchmark Metrics", "", f"Runs: {metrics['num_runs']}", "", "## Aggregate", ""]
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key, value in metrics["aggregate"].items():
        lines.append(f"| {key} | {value:.4f} |")
    if metrics["pass_at_k"]:
        lines.extend(["", "## pass@k", "", "| k | pass@k |", "| --- | --- |"])
        for k, value in metrics["pass_at_k"].items():
            lines.append(f"| {k} | {value:.4f} |")
    lines.extend(["", "## Per run", "", "| Run | Resolved | Total | Rate | Avg tokens |", "| --- | --- | --- | --- | --- |"])
    for run in metrics["per_run"]:
        lines.append(
            f"| {run['label']} | {run['resolved_count']} | {run['total']} | "
            f"{run['resolved_rate']:.4f} | {run['avg_tokens']:.0f} |"
        )
    curve = metrics["evolution_curve"]
    if curve:
        last = curve[-1]
        lines.extend([
            "", "## Evolution curve (train)", "",
            f"{len(curve)} points; final cumulative resolved rate "
            f"{last['cumulative_resolved_rate']:.4f}.",
        ])
    return "\n".join(lines) + "\n"


def _build_runs(args) -> List[Run]:
    reports = args.report or []
    if args.rollouts:
        runs = []
        for index, rollouts in enumerate(args.rollouts):
            report_ids = load_report_resolved(Path(reports[index])) if index < len(reports) else None
            runs.append(load_rollouts_run(Path(rollouts), report_ids=report_ids))
        return runs
    if args.predictions and reports:
        report_ids = load_report_resolved(Path(reports[0]))
        return [load_predictions_run(Path(args.predictions[0]), report_ids)]
    raise SystemExit("provide --rollouts, or --predictions with --report")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute SWE-bench benchmark metrics.")
    parser.add_argument("--rollouts", nargs="+", help="rollout JSONL files, one per run")
    parser.add_argument("--report", nargs="+", help="resolved-id reports (final_summary.json), per run")
    parser.add_argument("--predictions", nargs="+", help="predictions files (used with --report when no rollouts)")
    parser.add_argument("--hard-ids", type=Path, default=None, help="file of hard-case instance ids")
    parser.add_argument("--budget", type=int, default=None, help="iteration budget for success_under_budget")
    parser.add_argument("--price-per-token", type=float, default=0.0, help="USD per token for cost_to_success")
    parser.add_argument("--output", type=Path, default=None, help="write metrics JSON here")
    args = parser.parse_args(argv)

    runs = _build_runs(args)
    hard_ids = read_id_file(args.hard_ids) if args.hard_ids else None
    metrics = compute_metrics(
        runs, hard_ids=hard_ids, budget=args.budget, price_per_token=args.price_per_token
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(render_markdown(metrics))


if __name__ == "__main__":
    main()
