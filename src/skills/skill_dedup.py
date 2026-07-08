"""Layered skill deduplication for skill-bank hygiene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from src.skills.embedding_client import EmbeddingClient, local_text_similarity
from src.skills.skill_bank import SkillMetadata
from src.skills.textnorm import content_hash


@dataclass
class DedupDecision:
    duplicate: bool
    reason: str = ""
    matched_skill_id: str = ""
    matched_status: str = ""
    similarity: float = 0.0


def is_duplicate_skill(
    candidate_content: str,
    existing_skills: Iterable[SkillMetadata],
    threshold: float = 0.88,
    embedding_client: Optional[EmbeddingClient] = None,
) -> DedupDecision:
    """Detect exact or near-duplicate skill content against live and archived skills.

    Returns the best (highest-similarity) match at or above the threshold so the
    caller can distinguish self-refinements from collisions with other skills.
    """
    skills = list(existing_skills)
    candidate_hash = content_hash(candidate_content)
    for skill in skills:
        if skill.content_hash == candidate_hash:
            return DedupDecision(
                duplicate=True,
                reason="exact_content_hash",
                matched_skill_id=skill.id,
                matched_status=skill.status,
                similarity=1.0,
            )

    best: Optional[DedupDecision] = None
    for skill in skills:
        existing_text = f"{skill.title}\n{skill.summary}\n{skill.content}"
        similarity = None
        if embedding_client:
            similarity = embedding_client.similarity(candidate_content, existing_text)
        if similarity is None:
            similarity = local_text_similarity(candidate_content, existing_text)
        if similarity >= threshold and (best is None or similarity > best.similarity):
            best = DedupDecision(
                duplicate=True,
                reason="similarity_threshold",
                matched_skill_id=skill.id,
                matched_status=skill.status,
                similarity=round(float(similarity), 6),
            )

    return best or DedupDecision(duplicate=False)
