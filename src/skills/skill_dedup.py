"""Layered skill deduplication for skill-bank hygiene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from src.skills.embedding_client import EmbeddingClient, local_text_similarity
from src.skills.skill_bank import SkillMetadata
from src.skills.skill_store import content_hash


@dataclass
class DedupDecision:
    duplicate: bool
    reason: str = ""
    matched_skill_id: str = ""
    similarity: float = 0.0


def deduplicate_by_hash(skills: Iterable[SkillMetadata]) -> List[SkillMetadata]:
    """Return active skills with duplicate content hashes removed."""
    seen: set[str] = set()
    deduped: List[SkillMetadata] = []
    for skill in skills:
        key = skill.content_hash or skill.id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(skill)
    return deduped


def is_duplicate_skill(
    candidate_content: str,
    existing_skills: Iterable[SkillMetadata],
    threshold: float = 0.88,
    embedding_client: Optional[EmbeddingClient] = None,
) -> DedupDecision:
    """Detect exact or near-duplicate skill content."""
    candidate_hash = content_hash(candidate_content)
    for skill in existing_skills:
        if skill.content_hash == candidate_hash:
            return DedupDecision(
                duplicate=True,
                reason="exact_content_hash",
                matched_skill_id=skill.id,
                similarity=1.0,
            )

    for skill in existing_skills:
        existing_text = f"{skill.title}\n{skill.summary}\n{skill.content}"
        similarity = None
        if embedding_client:
            similarity = embedding_client.similarity(candidate_content, existing_text)
        if similarity is None:
            similarity = local_text_similarity(candidate_content, existing_text)
        if similarity >= threshold:
            return DedupDecision(
                duplicate=True,
                reason="similarity_threshold",
                matched_skill_id=skill.id,
                similarity=round(float(similarity), 6),
            )

    return DedupDecision(duplicate=False)
