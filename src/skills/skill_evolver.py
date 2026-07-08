"""Skill lifecycle management: advantage credit, net-success retirement, and proposal application."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.controller.schema import ControllerSignal, controller_signal_from_any
from src.skills.embedding_client import EmbeddingClient
from src.skills.proposals import SkillUpdateProposal
from src.skills.skill_dedup import DedupDecision, is_duplicate_skill
from src.skills.skill_store import SkillStore, normalize_skill_markdown
from src.skills.textnorm import slug

logger = logging.getLogger(__name__)


@dataclass
class SkillEvolutionConfig:
    """Loaded from configs/skill_evolution.yaml; code defaults are identical."""

    skill_similarity_threshold: float = 0.88
    skill_write_utility_threshold: float = 0.55
    max_selected_skills: int = 2
    max_active_skills: int = 12
    retire_min_trials: int = 5
    retire_net_success_threshold: float = -0.2
    baseline_ema_alpha: float = 0.3
    reflect_every_n_rollouts: int = 10

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SkillEvolutionConfig":
        config_path = Path(
            path or Path(__file__).resolve().parents[2] / "configs" / "skill_evolution.yaml"
        )
        if not config_path.exists():
            return cls()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{config_path} must contain a mapping of config keys")
        known = {item.name: item.type for item in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(
                f"unknown skill_evolution config keys in {config_path}: {sorted(unknown)}"
            )
        kwargs: Dict[str, Any] = {}
        for name, value in data.items():
            default = getattr(cls, name)
            kwargs[name] = type(default)(value)
        return cls(**kwargs)


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
    """Credit skill stats per rollout and materialize Reflector proposals."""

    def __init__(
        self,
        store: Optional[SkillStore] = None,
        config: Optional[SkillEvolutionConfig] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.store = store or SkillStore()
        self.config = config or SkillEvolutionConfig.load()
        self.embedding_client = (
            embedding_client if embedding_client is not None else EmbeddingClient.from_env()
        )

    def update_from_rollout(
        self,
        controller_signal: Any,
        reward: Any,
        success: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Credit the selected skills with advantage-style utility and run retirement."""
        signal = controller_signal_from_any(controller_signal)
        utility = _reward_total(reward)
        before = self._stats_snapshot()
        events: List[SkillEvolutionEvent] = []

        if not signal:
            return {
                "events": [],
                "dedup_decisions": [],
                "skill_stats_before": before,
                "skill_stats_after": before,
            }

        selected_ids = self._selected_skill_ids(signal)
        known_ids = {skill.id for skill in self.store.load_skills()}
        credit_ids = [skill_id for skill_id in selected_ids if skill_id in known_ids]
        for skill_id in selected_ids:
            if skill_id not in known_ids:
                events.append(
                    SkillEvolutionEvent(
                        action="credit",
                        skill_id=skill_id,
                        applied=False,
                        reason="unknown_skill_id",
                        reward=utility,
                    )
                )
        if credit_ids:
            credit = self.store.credit_skills(
                credit_ids,
                utility,
                success=success,
                ema_alpha=self.config.baseline_ema_alpha,
            )
            for skill_id in credit_ids:
                events.append(
                    SkillEvolutionEvent(
                        action="credit",
                        skill_id=skill_id,
                        applied=True,
                        reward=utility,
                        metadata={
                            "advantage": credit["advantage"],
                            "baseline_ema": credit["baseline_after"]["ema"],
                        },
                    )
                )

        events.extend(self._retire_unhelpful_skills())
        after = self._stats_snapshot()
        return {
            "events": [event.to_dict() for event in events],
            "dedup_decisions": [],
            "skill_stats_before": before,
            "skill_stats_after": after,
        }

    def apply_proposals(
        self,
        proposals: List[SkillUpdateProposal],
        utility: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply Reflector proposals; when a triggering utility is supplied, gate writes on it."""
        events: List[SkillEvolutionEvent] = []
        dedup_decisions: List[Dict[str, Any]] = []
        for proposal in proposals:
            event, dedup = self._apply_proposal(proposal, utility)
            events.append(event)
            if dedup:
                dedup_decisions.append(dedup)
        events.extend(self._enforce_active_cap())
        return {
            "events": [event.to_dict() for event in events],
            "dedup_decisions": dedup_decisions,
        }

    def _apply_proposal(
        self,
        proposal: SkillUpdateProposal,
        utility: Optional[float],
    ) -> tuple[SkillEvolutionEvent, Optional[Dict[str, Any]]]:
        if utility is not None and utility < self.config.skill_write_utility_threshold:
            return (
                SkillEvolutionEvent(
                    action=f"proposal_{proposal.operation}",
                    skill_id=proposal.skill_id,
                    applied=False,
                    reason="utility_below_write_threshold",
                    reward=utility,
                    metadata={"threshold": self.config.skill_write_utility_threshold},
                ),
                None,
            )
        reward = utility if utility is not None else 0.0

        if proposal.operation == "deprecate":
            self.store.deprecate_skill(proposal.skill_id, reason=proposal.rationale)
            return (
                SkillEvolutionEvent(
                    action="deprecate",
                    skill_id=proposal.skill_id,
                    applied=True,
                    reward=reward,
                    metadata={"rationale": proposal.rationale},
                ),
                None,
            )

        normalized_content = normalize_skill_markdown(
            proposal.content,
            slug(proposal.skill_id),
            proposal.target_failure_type,
        )
        dedup = is_duplicate_skill(
            normalized_content,
            [*self.store.load_skills(), *self.store.load_archived_skills()],
            threshold=self.config.skill_similarity_threshold,
            embedding_client=self.embedding_client,
        )
        if dedup.duplicate and not _is_self_refinement(proposal, dedup):
            return (
                SkillEvolutionEvent(
                    action=f"proposal_{proposal.operation}",
                    skill_id=proposal.skill_id,
                    applied=False,
                    reason="duplicate_skill",
                    reward=reward,
                    metadata={
                        "matched_skill_id": dedup.matched_skill_id,
                        "matched_status": dedup.matched_status,
                    },
                ),
                asdict(dedup),
            )

        self.store.write_skill(
            proposal.skill_id,
            proposal.content,
            source=proposal.source,
            archive_existing=proposal.operation == "update",
            target_failure_type=proposal.target_failure_type,
        )
        return (
            SkillEvolutionEvent(
                action=proposal.operation,
                skill_id=proposal.skill_id,
                applied=True,
                reward=reward,
                metadata={"rationale": proposal.rationale},
            ),
            asdict(dedup) if dedup.duplicate else None,
        )

    def _retire_unhelpful_skills(self) -> List[SkillEvolutionEvent]:
        """Retire skills with trials >= N and net success rate <= threshold."""
        stats = self.store.load_metadata()
        victims: List[str] = []
        events: List[SkillEvolutionEvent] = []
        for skill_id, item in sorted(stats.items()):
            if item.status != "active":
                continue
            if (
                item.usage_count >= self.config.retire_min_trials
                and item.net_success_rate <= self.config.retire_net_success_threshold
            ):
                victims.append(skill_id)
                events.append(
                    SkillEvolutionEvent(
                        action="retire_unhelpful",
                        skill_id=skill_id,
                        applied=True,
                        reason="net_success_rate_below_threshold",
                        metadata={
                            "trials": item.usage_count,
                            "successes": item.successes,
                            "failures": item.failures,
                            "net_success_rate": round(item.net_success_rate, 6),
                        },
                    )
                )
        if victims:
            self.store.retire_skills(victims, reason="net_success_rate_floor")
        return events

    def _enforce_active_cap(self) -> List[SkillEvolutionEvent]:
        """Evict lowest-contribution skills when the active bank exceeds the cap."""
        active = [skill for skill in self.store.load_skills() if skill.status == "active"]
        excess = len(active) - self.config.max_active_skills
        if excess <= 0:
            return []
        stats = self.store.load_metadata()

        def contribution(skill_id: str) -> tuple[float, float, str]:
            item = stats.get(skill_id)
            advantage = item.average_advantage if item and item.usage_count > 0 else 0.0
            reward = item.average_reward if item and item.usage_count > 0 else 0.0
            return (advantage, reward, skill_id)

        victims = sorted((skill.id for skill in active), key=contribution)[:excess]
        self.store.retire_skills(victims, reason="active_cap_evicted")
        return [
            SkillEvolutionEvent(
                action="active_cap_evict",
                skill_id=skill_id,
                applied=True,
                reason="active_skill_cap_exceeded",
                metadata={"max_active_skills": self.config.max_active_skills},
            )
            for skill_id in victims
        ]

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


def _is_self_refinement(proposal: SkillUpdateProposal, dedup: DedupDecision) -> bool:
    """An update may be similar (not identical) to its own live revision."""
    return (
        proposal.operation == "update"
        and dedup.reason == "similarity_threshold"
        and dedup.matched_status == "active"
        and dedup.matched_skill_id == slug(proposal.skill_id)
    )


def _reward_total(reward: Any) -> float:
    if reward is None:
        return 0.0
    if isinstance(reward, dict):
        return float(reward.get("total", 0.0) or 0.0)
    return float(getattr(reward, "total", 0.0) or 0.0)
