"""Reward-gated skill evolution from controller proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.controller.schema import ControllerSignal, SkillUpdateProposal, controller_signal_from_any
from src.skills.embedding_client import EmbeddingClient
from src.skills.skill_dedup import DedupDecision, is_duplicate_skill
from src.skills.skill_store import SkillStore, slug


@dataclass
class SkillEvolutionConfig:
    skill_similarity_threshold: float = 0.88
    skill_write_reward_threshold: float = 0.75
    skill_deprecate_reward_threshold: float = 0.35
    skill_deprecate_min_usage: int = 5
    max_selected_skills: int = 2


@dataclass
class SkillEvolutionEvent:
    action: str
    skill_id: str
    applied: bool
    reason: str = ""
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillEvolver:
    """Update skill stats and materialize reward-gated skill proposals."""

    def __init__(
        self,
        store: Optional[SkillStore] = None,
        config: Optional[SkillEvolutionConfig] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.store = store or SkillStore()
        self.config = config or SkillEvolutionConfig()
        self.embedding_client = embedding_client

    def update_from_rollout(
        self,
        controller_signal: Any,
        reward: Any,
    ) -> Dict[str, Any]:
        signal = controller_signal_from_any(controller_signal)
        reward_total = _reward_total(reward)
        before = self._stats_snapshot()
        events: List[SkillEvolutionEvent] = []
        dedup_decisions: List[Dict[str, Any]] = []

        if not signal:
            return {
                "events": [],
                "dedup_decisions": [],
                "skill_stats_before": before,
                "skill_stats_after": before,
            }

        selected_ids = self._selected_skill_ids(signal)
        if selected_ids:
            self.store.update_skill_stats(selected_ids, reward_total)
            for skill_id in selected_ids:
                events.append(
                    SkillEvolutionEvent(
                        action="update_usage_stats",
                        skill_id=skill_id,
                        applied=True,
                        reward=reward_total,
                    )
                )

        for proposal in signal.skill_updates:
            event, dedup = self._apply_proposal(proposal, reward_total)
            events.append(event)
            if dedup:
                dedup_decisions.append(dedup)

        events.extend(self._auto_deprecate(reward_total))
        after = self._stats_snapshot()
        return {
            "events": [event.to_dict() for event in events],
            "dedup_decisions": dedup_decisions,
            "skill_stats_before": before,
            "skill_stats_after": after,
        }

    def _apply_proposal(
        self,
        proposal: SkillUpdateProposal,
        reward_total: float,
    ) -> tuple[SkillEvolutionEvent, Optional[Dict[str, Any]]]:
        if reward_total < self.config.skill_write_reward_threshold:
            return (
                SkillEvolutionEvent(
                    action=f"proposal_{proposal.operation}",
                    skill_id=proposal.skill_id,
                    applied=False,
                    reason="reward_below_write_threshold",
                    reward=reward_total,
                    metadata={"threshold": self.config.skill_write_reward_threshold},
                ),
                None,
            )

        if proposal.operation == "deprecate":
            self.store.deprecate_skill(proposal.skill_id, reason=proposal.rationale)
            return (
                SkillEvolutionEvent(
                    action="deprecate",
                    skill_id=proposal.skill_id,
                    applied=True,
                    reward=reward_total,
                    metadata={"rationale": proposal.rationale},
                ),
                None,
            )

        dedup = is_duplicate_skill(
            proposal.content,
            self.store.load_skills(),
            threshold=self.config.skill_similarity_threshold,
            embedding_client=self.embedding_client,
        )
        if dedup.duplicate and slug(proposal.skill_id) != dedup.matched_skill_id:
            return (
                SkillEvolutionEvent(
                    action=f"proposal_{proposal.operation}",
                    skill_id=proposal.skill_id,
                    applied=False,
                    reason="duplicate_skill",
                    reward=reward_total,
                    metadata={"matched_skill_id": dedup.matched_skill_id},
                ),
                asdict(dedup),
            )

        self.store.write_skill(
            proposal.skill_id,
            proposal.content,
            source=proposal.source,
            archive_existing=proposal.operation == "update",
        )
        return (
            SkillEvolutionEvent(
                action=proposal.operation,
                skill_id=proposal.skill_id,
                applied=True,
                reward=reward_total,
                metadata={"rationale": proposal.rationale},
            ),
            asdict(dedup) if dedup.duplicate else None,
        )

    def _auto_deprecate(self, reward_total: float) -> List[SkillEvolutionEvent]:
        events: List[SkillEvolutionEvent] = []
        stats = self.store.load_metadata()
        changed = False
        for skill_id, item in stats.items():
            if item.status == "deprecated":
                continue
            if (
                item.usage_count >= self.config.skill_deprecate_min_usage
                and item.average_reward < self.config.skill_deprecate_reward_threshold
            ):
                item.status = "deprecated"
                item.record_event("auto_deprecated", reward=reward_total)
                changed = True
                events.append(
                    SkillEvolutionEvent(
                        action="auto_deprecate",
                        skill_id=skill_id,
                        applied=True,
                        reason="average_reward_below_threshold",
                        reward=reward_total,
                        metadata={
                            "average_reward": item.average_reward,
                            "usage_count": item.usage_count,
                        },
                    )
                )
        if changed:
            self.store.save_metadata(stats)
        return events

    def _selected_skill_ids(self, signal: ControllerSignal) -> List[str]:
        selected = list(signal.selected_skill_ids)
        if not selected:
            selected = [skill.id for skill in signal.skills if skill.id]
        if not selected and signal.skill and signal.skill.id:
            selected = [signal.skill.id]
        normalized: List[str] = []
        for skill_id in selected:
            item = slug(skill_id)
            if item and item not in normalized:
                normalized.append(item)
        return normalized[: self.config.max_selected_skills]

    def _stats_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            skill_id: stats.to_dict()
            for skill_id, stats in self.store.load_metadata().items()
        }


def _reward_total(reward: Any) -> float:
    if reward is None:
        return 0.0
    if isinstance(reward, dict):
        return float(reward.get("total", 0.0) or 0.0)
    return float(getattr(reward, "total", 0.0) or 0.0)
