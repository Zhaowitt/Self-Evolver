"""
Task evolution configuration (Proposal 2.3).

Tunables live in ``configs/task_evolution.yaml``; the in-code defaults below
are identical to that file so behavior is unchanged when it is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskEvolutionConfig:
    """Tunables for TaskPool sampling and focused-variant generation."""

    ema_alpha: float = 0.3            # EMA step for per-family success rate
    boundary_sigma: float = 0.2       # sigma of exp(-(s-0.5)^2 / (2*sigma^2))
    boost_factor: float = 2.0         # instance boost = 1 + boost_factor * cluster_hits
    decay_factor: float = 0.3         # too-hard weight decay multiplier
    too_hard_attempts: int = 3        # consecutive failures before decay + focus variant
    initial_success_ema: float = 0.5  # neutral prior: family starts at peak weight
    focus_enabled: bool = True        # emit focused variants for too-hard instances
    focus_max_per_instance: int = 1   # max focused variants emitted per base instance

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if self.boundary_sigma <= 0.0:
            raise ValueError(f"boundary_sigma must be > 0, got {self.boundary_sigma}")
        if self.boost_factor < 0.0:
            raise ValueError(f"boost_factor must be >= 0, got {self.boost_factor}")
        if not 0.0 < self.decay_factor <= 1.0:
            raise ValueError(f"decay_factor must be in (0, 1], got {self.decay_factor}")
        if self.too_hard_attempts < 1:
            raise ValueError(f"too_hard_attempts must be >= 1, got {self.too_hard_attempts}")
        if not 0.0 <= self.initial_success_ema <= 1.0:
            raise ValueError(
                f"initial_success_ema must be in [0, 1], got {self.initial_success_ema}"
            )
        if self.focus_max_per_instance < 0:
            raise ValueError(
                f"focus_max_per_instance must be >= 0, got {self.focus_max_per_instance}"
            )


def default_config_path() -> Path:
    """Repo-relative path of configs/task_evolution.yaml (no absolute paths)."""
    return Path(__file__).resolve().parents[2] / "configs" / "task_evolution.yaml"


def load_task_evolution_config(path: Optional[Path] = None) -> TaskEvolutionConfig:
    """
    Load tunables from a YAML file.

    With ``path=None`` the repo default is used and, when that file is absent,
    the in-code defaults (identical values) apply. An explicitly given path
    must exist.
    """
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.exists():
        if path is not None:
            raise FileNotFoundError(f"Task evolution config not found: {config_path}")
        return TaskEvolutionConfig()

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Task evolution config must be a mapping: {config_path}")

    defaults = TaskEvolutionConfig()
    known = {item.name for item in fields(TaskEvolutionConfig)}
    kwargs = {}
    for key, value in data.items():
        if key not in known:
            logger.warning("Ignoring unknown task evolution config key: %s", key)
            continue
        default_value = getattr(defaults, key)
        if isinstance(default_value, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean, got {value!r} in {config_path}")
            kwargs[key] = value
        elif isinstance(default_value, int):
            kwargs[key] = int(value)
        else:
            kwargs[key] = float(value)
    return TaskEvolutionConfig(**kwargs)
