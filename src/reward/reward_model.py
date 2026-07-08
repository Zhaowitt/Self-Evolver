"""Execution utility and evolution utility for controller rollouts (Proposal 2.6).

Execution utility in [0, 1]:
    0.5*resolved + 0.2*f2p_fraction + 0.1*p2p_no_regression
    + 0.1*cost_efficiency + 0.1*process

``configs/reward_config.yaml`` is the single source of truth for the weights
and knobs; the code defaults below are identical for installs without it.

Evolution utility (advantage-style skill credit) = utility - EMA(utility),
replacing full-reward-by-naming attribution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.reward.reward_components import DEFAULT_COST_TOKEN_BUDGET, compute_reward_components


DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "resolved": 0.5,
    "f2p_fraction": 0.2,
    "p2p_no_regression": 0.1,
    "cost_efficiency": 0.1,
    "process": 0.1,
}
DEFAULT_SKILL_WRITE_GATE = 0.55
DEFAULT_BASELINE_ALPHA = 0.3


@dataclass
class RewardResult:
    total: float
    components: Dict[str, float]
    weights: Dict[str, float] = field(default_factory=dict)
    evolution_utility: float = 0.0
    baseline: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RewardModel:
    """Aggregate execution-utility components and track the EMA baseline."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        cost_token_budget: int = DEFAULT_COST_TOKEN_BUDGET,
        skill_write_gate: float = DEFAULT_SKILL_WRITE_GATE,
        baseline_alpha: float = DEFAULT_BASELINE_ALPHA,
    ):
        self.weights = dict(DEFAULT_REWARD_WEIGHTS)
        if weights:
            unknown = set(weights) - set(DEFAULT_REWARD_WEIGHTS)
            if unknown:
                raise ValueError(
                    f"unknown reward weight keys: {sorted(unknown)}; "
                    f"expected a subset of {sorted(DEFAULT_REWARD_WEIGHTS)}"
                )
            self.weights.update({key: float(value) for key, value in weights.items()})
        self.cost_token_budget = int(cost_token_budget)
        self.skill_write_gate = float(skill_write_gate)
        self.baseline_alpha = float(baseline_alpha)
        self._baseline: Optional[float] = None

    @classmethod
    def from_config_file(cls, path: Optional[Path] = None) -> "RewardModel":
        """Build from a reward config file.

        With ``path=None`` this loads ``configs/reward_config.yaml`` from the
        repo (falling back to identical code defaults if the file is absent,
        e.g. in a bare package install). An explicit path must exist.
        """
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
            baseline_alpha=float(data.get("evolution_baseline_alpha", DEFAULT_BASELINE_ALPHA)),
        )

    def score(
        self,
        execution_result: Any,
        eval_outcome: Any = None,
        issue: Any = None,
    ) -> RewardResult:
        """Score one rollout and update the evolution-utility baseline."""
        components = compute_reward_components(
            execution_result,
            eval_outcome=eval_outcome,
            issue=issue,
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
        total = round(max(0.0, min(1.0, total)), 6)
        advantage = self.evolution_utility(total)
        return RewardResult(
            total=total,
            components={key: round(float(value), 6) for key, value in components.items()},
            weights=dict(self.weights),
            evolution_utility=advantage,
            baseline=round(self._baseline if self._baseline is not None else total, 6),
        )

    def evolution_utility(self, utility: float) -> float:
        """Advantage of ``utility`` over the EMA baseline; updates the baseline.

        The first observation seeds the baseline (advantage 0), so skills are
        credited for improvement over the running level, not for showing up.
        """
        utility = float(utility)
        if self._baseline is None:
            self._baseline = utility
            return 0.0
        advantage = utility - self._baseline
        self._baseline = (1.0 - self.baseline_alpha) * self._baseline + self.baseline_alpha * utility
        return round(advantage, 6)


def default_config_path() -> Path:
    """Repo-relative path of the canonical reward config."""
    return Path(__file__).resolve().parents[2] / "configs" / "reward_config.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the strict two-level mapping subset of YAML used by reward_config.yaml.

    Supports scalar values (str/int/float/bool/null), quoted strings, and
    comments. Rejects lists and deeper nesting loudly instead of misparsing.
    """
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
