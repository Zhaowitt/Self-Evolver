"""Weighted rule-based reward model for controller rollouts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.reward.reward_components import compute_reward_components


DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "patch_applies_cleanly": 0.15,
    "non_empty_canonical_diff": 0.15,
    "tests_pass": 0.35,
    "avoids_patch_apply_error": 0.10,
    "reasonable_file_count": 0.10,
    "follows_task_wrapper_constraints": 0.05,
    "avoids_timeout": 0.05,
    "cost_efficiency": 0.05,
}


@dataclass
class RewardResult:
    total: float
    components: Dict[str, float]
    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RewardModel:
    """Aggregate reward components with configurable weights."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        max_reasonable_files: int = 5,
        cost_token_budget: int = 50000,
    ):
        self.weights = dict(DEFAULT_REWARD_WEIGHTS)
        if weights:
            self.weights.update({key: float(value) for key, value in weights.items()})
        self.max_reasonable_files = max_reasonable_files
        self.cost_token_budget = cost_token_budget

    @classmethod
    def from_config_file(cls, path: Optional[Path]) -> "RewardModel":
        if not path or not Path(path).exists():
            return cls()
        data = _load_config(Path(path))
        return cls(
            weights=dict(data.get("weights") or {}),
            max_reasonable_files=int(data.get("max_reasonable_files", 5)),
            cost_token_budget=int(data.get("cost_token_budget", 50000)),
        )

    def score(self, execution_result: Any, controller_signal: Any = None) -> RewardResult:
        components = compute_reward_components(
            execution_result,
            controller_signal=controller_signal,
            max_reasonable_files=self.max_reasonable_files,
            cost_token_budget=self.cost_token_budget,
        )
        total_weight = sum(weight for weight in self.weights.values() if weight > 0)
        if total_weight <= 0:
            total = 0.0
        else:
            total = sum(
                components.get(name, 0.0) * max(0.0, weight)
                for name, weight in self.weights.items()
            ) / total_weight
        return RewardResult(
            total=round(max(0.0, min(1.0, total)), 6),
            components={key: round(float(value), 6) for key, value in components.items()},
            weights=self.weights,
        )


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_section: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            data[current_section] = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        parsed_value: Any = _parse_scalar(value)
        if raw_line.startswith(" ") and current_section:
            data.setdefault(current_section, {})[key] = parsed_value
        else:
            current_section = None
            data[key] = parsed_value
    return data


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
