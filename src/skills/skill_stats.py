"""Persistent skill usage and lifecycle metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class SkillStats:
    """Runtime metadata for a repair skill."""

    id: str
    usage_count: int = 0
    average_reward: float = 0.0
    last_reward: float = 0.0
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
            average_reward=float(data.get("average_reward", 0.0) or 0.0),
            last_reward=float(data.get("last_reward", 0.0) or 0.0),
            status=str(data.get("status", "active")),
            content_hash=str(data.get("content_hash", "")),
            source=str(data.get("source", "seed")),
            revision=int(data.get("revision", 0) or 0),
            last_updated_at=str(data.get("last_updated_at", "")),
            history=list(data.get("history") or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def record_reward(self, reward: float) -> None:
        previous_count = self.usage_count
        self.usage_count += 1
        self.last_reward = float(reward)
        if previous_count <= 0:
            self.average_reward = float(reward)
        else:
            self.average_reward = (
                self.average_reward * previous_count + float(reward)
            ) / self.usage_count
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
