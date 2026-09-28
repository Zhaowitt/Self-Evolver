"""Task outcome, token cost, and cluster-relative skill contribution reward."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.reward.reward_components import DEFAULT_COST_TOKEN_BUDGET, compute_reward_components
from src.reward.cluster_baseline import ClusterBaselineStore


DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "official_resolved": 0.3,
    "official_f2p_fraction": 0.7,
}
# token_cost is not weightable here; it is subtracted via token_cost_weight.
REWARD_COMPONENT_KEYS = frozenset({
    "official_resolved", "official_f2p_fraction", "resolved", "f2p_fraction",
    "p2p_no_regression", "cost_efficiency", "process",
})
DEFAULT_TOKEN_COST_WEIGHT = 0.05
# Token use below this budget fraction is free, so small cost gaps cannot dominate all-failed groups.
DEFAULT_TOKEN_COST_DEADBAND = 0.5
DEFAULT_CONTRIBUTION_WEIGHT = 0.1
DEFAULT_CLUSTER_BASELINE_MIN_COUNT = 3
DEFAULT_SKILL_WRITE_GATE = 0.55
# Must stay below the valid floor -(token_cost_weight + contribution_weight) = -0.15.
DEFAULT_FORMAT_FAILURE_REWARD = -0.5


@dataclass
class RewardResult:
    total: float
    components: Dict[str, float]
    weights: Dict[str, float] = field(default_factory=dict)
    evolution_utility: float = 0.0
    baseline: float = 0.0
    task_reward: float = 0.0
    skill_contribution: float = 0.0
    cluster_id: str = ""
    cluster_baseline_count: int = 0
    skill_set_key: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RewardModel:
    """Compute reward against a baseline snapshot frozen for the round."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        cost_token_budget: int = DEFAULT_COST_TOKEN_BUDGET,
        skill_write_gate: float = DEFAULT_SKILL_WRITE_GATE,
        token_cost_weight: float = DEFAULT_TOKEN_COST_WEIGHT,
        contribution_weight: float = DEFAULT_CONTRIBUTION_WEIGHT,
        token_cost_deadband: float = DEFAULT_TOKEN_COST_DEADBAND,
        cluster_baseline_min_count: int = DEFAULT_CLUSTER_BASELINE_MIN_COUNT,
        cluster_baselines: Optional[Dict[str, Dict[str, Any]]] = None,
        cluster_baseline_store: Optional[ClusterBaselineStore] = None,
        format_failure_reward: float = DEFAULT_FORMAT_FAILURE_REWARD,
    ):
        self.weights = dict(DEFAULT_REWARD_WEIGHTS)
        if weights:
            unknown = set(weights) - REWARD_COMPONENT_KEYS
            if unknown:
                raise ValueError(
                    f"unknown reward weight keys: {sorted(unknown)}; "
                    f"expected a subset of {sorted(REWARD_COMPONENT_KEYS)}"
                )
            self.weights.update({key: float(value) for key, value in weights.items()})
        self.cost_token_budget = int(cost_token_budget)
        self.skill_write_gate = float(skill_write_gate)
        self.token_cost_weight = max(0.0, float(token_cost_weight))
        self.contribution_weight = max(0.0, float(contribution_weight))
        self.token_cost_deadband = min(0.99, max(0.0, float(token_cost_deadband)))
        self.cluster_baseline_min_count = max(1, int(cluster_baseline_min_count))
        valid_floor = -(self.token_cost_weight + self.contribution_weight)
        self.format_failure_reward = float(format_failure_reward)
        if self.format_failure_reward >= valid_floor:
            raise ValueError(
                f"format_failure_reward={self.format_failure_reward} must be below the "
                f"lowest valid reward {valid_floor}; otherwise malformed output wins"
            )
        self.cluster_baseline_store = cluster_baseline_store or ClusterBaselineStore()
        self.cluster_baselines = (
            dict(cluster_baselines)
            if cluster_baselines is not None
            else self.cluster_baseline_store.frozen_snapshot()
        )

    @classmethod
    def from_config_file(cls, path: Optional[Path] = None) -> "RewardModel":
        """``path=None`` loads configs/reward_config.yaml (code defaults if absent); an explicit path must exist."""
        if path is None:
            default = default_config_path()
            if not default.exists():
                return cls()
            path = default
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"reward config not found: {path}. Provide an existing file via "
                "--reward-config / SELF_EVOLVER_REWARD_CONFIG, or omit it to use "
                "configs/reward_config.yaml."
            )
        data = _load_config(path)
        return cls(
            weights=dict(data.get("weights") or {}),
            cost_token_budget=int(data.get("cost_token_budget", DEFAULT_COST_TOKEN_BUDGET)),
            skill_write_gate=float(data.get("skill_write_gate", DEFAULT_SKILL_WRITE_GATE)),
            token_cost_weight=float(data.get("token_cost_weight", DEFAULT_TOKEN_COST_WEIGHT)),
            contribution_weight=float(
                data.get("contribution_weight", DEFAULT_CONTRIBUTION_WEIGHT)
            ),
            token_cost_deadband=float(
                data.get("token_cost_deadband", DEFAULT_TOKEN_COST_DEADBAND)
            ),
            cluster_baseline_min_count=int(
                data.get("cluster_baseline_min_count", DEFAULT_CLUSTER_BASELINE_MIN_COUNT)
            ),
            format_failure_reward=float(
                data.get("format_failure_reward", DEFAULT_FORMAT_FAILURE_REWARD)
            ),
        )

    def score(
        self,
        execution_result: Any,
        eval_outcome: Any = None,
        issue: Any = None,
        cluster_id: Optional[str] = None,
        selected_skill_ids: Optional[list[str]] = None,
    ) -> RewardResult:
        """Score one rollout without mutating the frozen cluster baselines."""
        components = compute_reward_components(
            execution_result,
            eval_outcome=eval_outcome,
            issue=issue,
            cost_token_budget=self.cost_token_budget,
        )
        outcome_reward = sum(
            components.get(name, 0.0) * max(0.0, weight)
            for name, weight in self.weights.items()
        )
        deadband = self.token_cost_deadband
        excess_cost = max(0.0, components["token_cost"] - deadband) / (1.0 - deadband)
        task_reward = outcome_reward - self.token_cost_weight * excess_cost

        resolved_cluster = str(
            cluster_id
            or getattr(issue, "metadata", {}).get("stable_cluster_id", "")
            or ""
        ).strip()
        skill_set_key = sorted({str(item) for item in (selected_skill_ids or []) if str(item)})
        baseline_entry = dict(self.cluster_baselines.get(resolved_cluster) or {})
        baseline_count = int(baseline_entry.get("count", 0) or 0)
        baseline = float(baseline_entry.get("ema", 0.0) or 0.0)
        # Independent of the selection, so selecting no skill gains no systematic edge.
        contribution = (
            task_reward - baseline
            if resolved_cluster and baseline_count >= self.cluster_baseline_min_count
            else 0.0
        )
        contribution = max(-1.0, min(1.0, contribution))
        total = task_reward + self.contribution_weight * contribution
        total = round(max(-1.0, min(1.0, total)), 6)
        return RewardResult(
            total=total,
            components={key: round(float(value), 6) for key, value in components.items()},
            weights=dict(self.weights),
            evolution_utility=round(contribution, 6),
            baseline=round(baseline, 6),
            task_reward=round(task_reward, 6),
            skill_contribution=round(contribution, 6),
            cluster_id=resolved_cluster,
            cluster_baseline_count=baseline_count,
            skill_set_key=skill_set_key,
        )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "reward_config.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Strict two-level scalar-mapping YAML subset; lists and deeper nesting are rejected."""
    data: Dict[str, Any] = {}
    section: Optional[str] = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            raise ValueError(f"line {lineno}: list values are not supported in reward config")
        if ":" not in stripped:
            raise ValueError(f"line {lineno}: expected 'key: value', got {stripped!r}")
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if indent == 0:
            if value:
                section = None
                data[key] = _parse_scalar(value)
            else:
                section = key
                data[section] = {}
        else:
            if section is None:
                raise ValueError(f"line {lineno}: indented key {key!r} outside a section")
            if not value:
                raise ValueError(
                    f"line {lineno}: nesting deeper than two levels is not supported in reward config"
                )
            data[section][key] = _parse_scalar(value)
    return data


def _strip_comment(line: str) -> str:
    quote: Optional[str] = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    if len(value) >= 2 and value[0] in "\"'" and value.endswith(value[0]):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
