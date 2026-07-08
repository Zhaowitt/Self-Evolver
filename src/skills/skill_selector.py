"""Stats-aware skill selection over the unified failure taxonomy."""

from __future__ import annotations

from typing import List, Optional

from src.skills.failure_types import FailureType, normalize_failure_type
from src.skills.skill_bank import SkillBank, SkillMetadata


# Ranking constants: failure-type match dominates, lexical overlap refines,
# accumulated stats reward proven skills and down-weight harmful ones.
TYPE_MATCH_SCORE = 2.0
GENERAL_MATCH_SCORE = 0.5
OVERLAP_TERM_SCORE = 0.25
STATS_MIN_TRIALS = 3
LOW_REWARD_THRESHOLD = 0.35
LOW_REWARD_PENALTY = 2.0
REWARD_BONUS_SCALE = 0.5

_UNTARGETED = {"", FailureType.UNKNOWN.value, FailureType.NONE.value}


class SkillSelector:
    """Select compact repair skills by failure type, query overlap, and reward stats."""

    def __init__(self, skill_bank: Optional[SkillBank] = None):
        self.skill_bank = skill_bank or SkillBank()

    def select_many(
        self,
        target_failure_type: str = "",
        memory_query: str = "",
        limit: int = 2,
    ) -> List[SkillMetadata]:
        skills = self.skill_bank.active()
        if not skills:
            return []

        target = normalize_failure_type(target_failure_type, default="")
        query_terms = _query_terms(memory_query)
        scored = sorted(
            ((_score(skill, target, query_terms), skill) for skill in skills),
            key=lambda item: (-item[0], item[1].id),
        )
        selected = [skill for score, skill in scored if score > 0][: max(1, limit)]
        if selected:
            return selected
        fallback = self.skill_bank.get("inspect_before_editing") or skills[0]
        return [fallback]


def _score(skill: SkillMetadata, target: str, query_terms: set[str]) -> float:
    score = 0.0
    if target not in _UNTARGETED:
        if skill.target_failure_type == target:
            score += TYPE_MATCH_SCORE
        elif skill.target_failure_type == FailureType.GENERAL.value:
            score += GENERAL_MATCH_SCORE
    score += _overlap_count(skill, query_terms) * OVERLAP_TERM_SCORE
    if skill.usage_count >= STATS_MIN_TRIALS:
        if skill.average_reward < LOW_REWARD_THRESHOLD:
            score -= LOW_REWARD_PENALTY
        else:
            score += min(skill.average_reward, 1.0) * REWARD_BONUS_SCALE
    return score


def _overlap_count(skill: SkillMetadata, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    haystack = f"{skill.id} {skill.title} {skill.summary} {skill.target_failure_type}".lower()
    return sum(1 for term in query_terms if term in haystack)


def _query_terms(memory_query: str) -> set[str]:
    query = (memory_query or "").lower()
    return {term for term in query.replace("_", " ").split() if len(term) > 2}
