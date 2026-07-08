"""
Weighted task pool for train-stage task evolution (Proposal 2.3).

The pool holds SWE-bench-style instance dicts and maintains a sampling
distribution over them:

  weight(i) = exp(-(s_f - 0.5)^2 / (2 * sigma^2))   # family capability boundary
            * family_boost(f)                        # reflection signal
            * (1 + boost_factor * cluster_hits_i)    # reflection signal
            * decay_multiplier_i                     # too-hard decay

where s_f is the EMA of resolved outcomes for instance i's family. After
``too_hard_attempts`` consecutive failures an instance's weight decays by
``decay_factor`` and a verified focused variant is emitted into the pool.

State is persisted to ``<run_dir>/task_pool.json`` deterministically (no
timestamps, sorted keys), and all sampling randomness comes from the caller's
``random.Random`` — so runs are seed-reproducible.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from src.tasks.config import TaskEvolutionConfig, load_task_evolution_config
from src.tasks.families import TASK_FAMILIES, classify_family
from src.tasks.variants import is_focused_variant, make_focused_variant

logger = logging.getLogger(__name__)

_STATE_SCHEMA_VERSION = 1

Verifier = Callable[[dict], Tuple[bool, str]]


@dataclass
class _FamilyStats:
    success_ema: float
    boost: float = 1.0
    outcomes: int = 0


@dataclass
class _InstanceStats:
    family: str
    attempts: int = 0
    consecutive_failures: int = 0
    resolved_count: int = 0
    utility_ema: Optional[float] = None
    cluster_hits: int = 0
    decay_multiplier: float = 1.0
    focus_emitted: int = 0
    is_variant: bool = False


class TaskPool:
    """Evolving sampling distribution over verified benchmark instances."""

    def __init__(
        self,
        instances: Iterable[dict],
        state_path: Path,
        config: Optional[TaskEvolutionConfig] = None,
        verifier: Optional[Verifier] = None,
    ):
        self.config = config or load_task_evolution_config()
        self.state_path = Path(state_path)
        self.verifier = verifier
        self._instances: Dict[str, dict] = {}
        self._stats: Dict[str, _InstanceStats] = {}
        self._families: Dict[str, _FamilyStats] = {
            family: _FamilyStats(success_ema=self.config.initial_success_ema)
            for family in TASK_FAMILIES
        }
        for item in instances:
            instance = dict(item)
            instance_id = str(instance.get("instance_id") or "")
            if not instance_id:
                raise ValueError("TaskPool instance is missing instance_id")
            if instance_id in self._instances:
                logger.warning("Duplicate instance_id in pool, skipping: %s", instance_id)
                continue
            self._instances[instance_id] = instance
            self._stats[instance_id] = _InstanceStats(
                family=classify_family(instance),
                is_variant=is_focused_variant(instance),
            )
        if self.state_path.exists():
            self._load_state()

    @classmethod
    def from_instances(
        cls,
        instances: Iterable[dict],
        state_path: Path,
        config: Optional[TaskEvolutionConfig] = None,
        verifier: Optional[Verifier] = None,
    ) -> "TaskPool":
        """Build a pool, resuming evolution state from state_path if present."""
        return cls(instances, state_path, config=config, verifier=verifier)

    # ------------------------------------------------------------------ pool

    def __len__(self) -> int:
        return len(self._instances)

    def __contains__(self, instance_id: str) -> bool:
        return instance_id in self._instances

    def get(self, instance_id: str) -> dict:
        return self._instances[instance_id]

    def family_of(self, instance_id: str) -> str:
        return self._stats[instance_id].family

    # --------------------------------------------------------------- weights

    def family_weight(self, family: str) -> float:
        """Capability-boundary weight: peaks at success EMA 0.5."""
        stats = self._families[family]
        sigma = self.config.boundary_sigma
        return math.exp(-((stats.success_ema - 0.5) ** 2) / (2.0 * sigma * sigma))

    def weights(self) -> Dict[str, float]:
        """Current unnormalized sampling weight per instance id."""
        return {instance_id: self._weight(instance_id) for instance_id in self._instances}

    def _weight(self, instance_id: str) -> float:
        stats = self._stats[instance_id]
        family = self._families[stats.family]
        return (
            self.family_weight(stats.family)
            * family.boost
            * (1.0 + self.config.boost_factor * stats.cluster_hits)
            * stats.decay_multiplier
        )

    # -------------------------------------------------------------- sampling

    def sample(self, n: int, rng: random.Random) -> List[dict]:
        """
        Draw up to n distinct instances, weighted, without replacement.

        All randomness comes from ``rng``: identical pool state + seed yields
        identical samples.
        """
        if n <= 0:
            return []
        candidates = list(self._instances)
        weights = [self._weight(instance_id) for instance_id in candidates]
        chosen: List[str] = []
        while candidates and len(chosen) < n:
            total = sum(weights)
            if total <= 0.0:
                index = rng.randrange(len(candidates))
            else:
                threshold = rng.random() * total
                cumulative = 0.0
                index = len(candidates) - 1
                for position, weight in enumerate(weights):
                    cumulative += weight
                    if threshold < cumulative:
                        index = position
                        break
            chosen.append(candidates.pop(index))
            weights.pop(index)
        return [self._instances[instance_id] for instance_id in chosen]

    # -------------------------------------------------------------- outcomes

    def record_outcome(self, instance_id: str, resolved: bool, utility: float) -> Optional[str]:
        """
        Record one rollout outcome and update the sampling distribution.

        Updates the family success EMA and per-instance stats; after
        ``too_hard_attempts`` consecutive failures the instance decays by
        ``decay_factor`` and a verified focused variant is emitted into the
        pool. Returns the emitted variant's instance_id, if any.
        """
        if instance_id not in self._stats:
            raise KeyError(f"record_outcome for unknown instance: {instance_id}")
        alpha = self.config.ema_alpha
        stats = self._stats[instance_id]
        family = self._families[stats.family]

        family.success_ema = (1.0 - alpha) * family.success_ema + alpha * float(bool(resolved))
        family.outcomes += 1

        stats.attempts += 1
        utility = float(utility)
        if stats.utility_ema is None:
            stats.utility_ema = utility
        else:
            stats.utility_ema = (1.0 - alpha) * stats.utility_ema + alpha * utility

        if resolved:
            stats.resolved_count += 1
            stats.consecutive_failures = 0
            return None

        stats.consecutive_failures += 1
        if stats.consecutive_failures % self.config.too_hard_attempts != 0:
            return None
        stats.decay_multiplier *= self.config.decay_factor
        logger.info(
            "Instance %s too hard (%d consecutive failures), weight decayed to x%.3f",
            instance_id,
            stats.consecutive_failures,
            stats.decay_multiplier,
        )
        return self._emit_focused_variant(instance_id)

    def _emit_focused_variant(self, instance_id: str) -> Optional[str]:
        stats = self._stats[instance_id]
        if not self.config.focus_enabled or stats.is_variant:
            return None
        if stats.focus_emitted >= self.config.focus_max_per_instance:
            return None
        variant = make_focused_variant(self._instances[instance_id], n=stats.focus_emitted + 1)
        if variant is None:
            return None
        variant_id = variant["instance_id"]
        if variant_id in self._instances:
            return None
        if self.verifier is None:
            logger.warning(
                "No task verifier configured; focused variant %s not admitted "
                "(Proposal 2.7.1: only verified tasks enter the pool)",
                variant_id,
            )
            return None
        ok, reason = self.verifier(variant)
        if not ok:
            logger.info("Focused variant %s rejected by verification: %s", variant_id, reason)
            return None
        self._instances[variant_id] = variant
        self._stats[variant_id] = _InstanceStats(family=stats.family, is_variant=True)
        stats.focus_emitted += 1
        logger.info("Focused variant %s admitted to pool (%s)", variant_id, reason)
        return variant_id

    # ------------------------------------------------------------ reflection

    def apply_reflection(self, signals: dict) -> None:
        """
        Apply Reflector task-level signals.

        Schema::

            {
              "instance_boosts": {instance_id: cluster_hits_int, ...},
              "family_boosts": {family: multiplier_float, ...},
            }

        cluster_hits and family multipliers are SET (idempotent per
        reflection cycle), not accumulated. Unknown ids/families are logged
        and skipped.
        """
        for instance_id, hits in (signals.get("instance_boosts") or {}).items():
            if instance_id not in self._stats:
                logger.warning("Reflection boost for unknown instance: %s", instance_id)
                continue
            self._stats[instance_id].cluster_hits = max(0, int(hits))
        for family, multiplier in (signals.get("family_boosts") or {}).items():
            if family not in self._families:
                logger.warning("Reflection boost for unknown family: %s", family)
                continue
            self._families[family].boost = max(0.0, float(multiplier))

    # ----------------------------------------------------------- persistence

    def save(self) -> Path:
        """Atomically persist evolution state as deterministic JSON."""
        state = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "families": {name: asdict(stats) for name, stats in self._families.items()},
            "instances": {name: asdict(stats) for name, stats in self._stats.items()},
            "variants": {
                instance_id: instance
                for instance_id, instance in self._instances.items()
                if self._stats[instance_id].is_variant
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self.state_path)
        return self.state_path

    def _load_state(self) -> None:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid task pool state file: {self.state_path}")

        for name, family_state in (data.get("families") or {}).items():
            if name not in self._families:
                logger.warning("Dropping unknown family from state: %s", name)
                continue
            self._families[name] = _FamilyStats(**_known_fields(_FamilyStats, family_state))

        for variant_id, variant in (data.get("variants") or {}).items():
            if variant_id in self._instances:
                continue
            self._instances[variant_id] = dict(variant)
            self._stats[variant_id] = _InstanceStats(
                family=classify_family(variant), is_variant=True
            )

        for instance_id, instance_state in (data.get("instances") or {}).items():
            if instance_id not in self._instances:
                logger.debug("Dropping stale instance state: %s", instance_id)
                continue
            self._stats[instance_id] = _InstanceStats(
                **_known_fields(_InstanceStats, instance_state)
            )
        logger.info("Resumed task pool state from %s (%d instances)", self.state_path, len(self))


def _known_fields(dataclass_type, data: dict) -> dict:
    names = {item.name for item in fields(dataclass_type)}
    return {key: value for key, value in (data or {}).items() if key in names}
