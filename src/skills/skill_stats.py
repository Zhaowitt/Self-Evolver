"""Persistent skill usage and lifecycle metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SkillStats:
    """Runtime metadata for a repair skill."""

    id: str
    usage_count: int = 0
    successes: int = 0
    failures: int = 0
    average_reward: float = 0.0
    average_advantage: float = 0.0
    last_reward: float = 0.0
    last_advantage: float = 0.0
    status: str = "active"
    content_hash: str = ""
    source: str = "seed"
    revision: int = 0
    last_updated_at: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillStats":
        return cls(
            id=str(data.get("id", "")),
            usage_count=int(data.get("usage_count", 0) or 0),
            successes=int(data.get("successes", 0) or 0),
            failures=int(data.get("failures", 0) or 0),
            average_reward=float(data.get("average_reward", 0.0) or 0.0),
            average_advantage=float(data.get("average_advantage", 0.0) or 0.0),
            last_reward=float(data.get("last_reward", 0.0) or 0.0),
            last_advantage=float(data.get("last_advantage", 0.0) or 0.0),
            status=str(data.get("status", "active")),
            content_hash=str(data.get("content_hash", "")),
            source=str(data.get("source", "seed")),
            revision=int(data.get("revision", 0) or 0),
            last_updated_at=str(data.get("last_updated_at", "")),
            history=list(data.get("history") or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def net_success_rate(self) -> float:
        """Retirement contribution signal: (successes - failures) / trials."""
        if self.usage_count <= 0:
            return 0.0
        return (self.successes - self.failures) / self.usage_count

    def record_credit(self, reward: float, advantage: float, success: Optional[bool] = None) -> None:
        """Record one credited trial with advantage-style contribution accounting.

        When `success` is unknown, the sign of the advantage decides the trial
        outcome (zero advantage counts as neither success nor failure).
        """
        previous_count = self.usage_count
        self.usage_count += 1
        self.last_reward = float(reward)
        self.last_advantage = float(advantage)
        if previous_count <= 0:
            self.average_reward = float(reward)
            self.average_advantage = float(advantage)
        else:
            self.average_reward = (
                self.average_reward * previous_count + float(reward)
            ) / self.usage_count
            self.average_advantage = (
                self.average_advantage * previous_count + float(advantage)
            ) / self.usage_count
        if success is None:
            if advantage > 0:
                self.successes += 1
            elif advantage < 0:
                self.failures += 1
        elif success:
            self.successes += 1
        else:
            self.failures += 1
        self.last_updated_at = datetime.now().isoformat()

    def record_event(self, event: str, **payload: Any) -> None:
        self.history.append(
            {
                "event": event,
                "created_at": datetime.now().isoformat(),
                **payload,
            }
        )
        self.history = self.history[-20:]
        self.last_updated_at = datetime.now().isoformat()
