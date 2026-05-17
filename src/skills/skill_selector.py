"""Simple skill selection for controller guidance."""

from __future__ import annotations

from typing import List, Optional

from src.skills.skill_bank import SkillBank, SkillMetadata


class SkillSelector:
    """Select a compact repair skill by failure type or query text."""

    def __init__(self, skill_bank: Optional[SkillBank] = None):
        self.skill_bank = skill_bank or SkillBank()

    def select(
        self,
        target_failure_type: str = "",
        memory_query: str = "",
    ) -> Optional[SkillMetadata]:
        selected = self.select_many(
            target_failure_type=target_failure_type,
            memory_query=memory_query,
            limit=1,
        )
        return selected[0] if selected else None

    def select_many(
        self,
        target_failure_type: str = "",
        memory_query: str = "",
        limit: int = 2,
    ) -> List[SkillMetadata]:
        skills = self.skill_bank.active()
        if not skills:
            return []

        ranked: List[SkillMetadata] = []
        target = (target_failure_type or "").lower()
        if target:
            ranked.extend(skill for skill in skills if skill.target_failure_type == target)

        query = (memory_query or "").lower()
        if query:
            ranked.extend(
                sorted(
                    [skill for skill in skills if _overlap_score(skill, query) > 0],
                    key=lambda skill: _overlap_score(skill, query),
                    reverse=True,
                )
            )

        fallback = self.skill_bank.get("inspect_before_editing") or skills[0]
        ranked.append(fallback)

        unique: List[SkillMetadata] = []
        seen: set[str] = set()
        for skill in ranked:
            if skill.id in seen:
                continue
            seen.add(skill.id)
            unique.append(skill)
            if len(unique) >= limit:
                break
        return unique


def _overlap_score(skill: SkillMetadata, query: str) -> int:
    haystack = f"{skill.id} {skill.title} {skill.summary} {skill.target_failure_type}".lower()
    query_terms = {term for term in query.replace("_", " ").split() if len(term) > 2}
    return sum(1 for term in query_terms if term in haystack)
