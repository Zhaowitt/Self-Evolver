"""Post-execution Reflector: turn hard-case clusters into skill and task updates.

The Reflector runs periodically (every N completed train rollouts and at the end
of each evolution iteration). It reads recent hard cases, clusters them by
failure pattern, and for each cluster large enough to matter asks the LLM to
propose skill create/update/deprecate actions (the SkillUpdateProposal schema).
Proposals are deduped and applied through the SkillEvolver. It also emits
task-level boost signals consumed by ``TaskPool.apply_reflection``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.controller.parser import extract_json_object
from src.llm.client import LLMClient
from src.memory.memory_retriever import MemoryRetriever
from src.reflection.clustering import (
    FailureCluster,
    cluster_records,
    qualifying_clusters,
)
from src.skills.embedding_client import EmbeddingClient
from src.skills.proposals import SkillUpdateProposal, parse_proposals
from src.skills.skill_bank import SkillBank, SkillMetadata
from src.skills.skill_evolver import SkillEvolver

logger = logging.getLogger(__name__)

MIN_CLUSTER_SIZE = 3
DEFAULT_RETRIEVAL_LIMIT = 200

# validator(proposals, clusters) -> (accepted_proposals, gating_utility)
Validator = Callable[
    [List[SkillUpdateProposal], List[FailureCluster]],
    Tuple[List[SkillUpdateProposal], Optional[float]],
]

REFLECTOR_SYSTEM_PROMPT = """You are a Reflector that evolves a repair skill bank from repeated failures.

You are given clusters of hard cases (recurring failures) and the current skill bank.
Propose a small set of skill changes that would help the worker avoid these failures.

Return one raw JSON object only, no Markdown fences, no commentary:
{
  "skill_updates": [
    {
      "operation": "create|update|deprecate",
      "skill_id": "stable_snake_case_id",
      "title": "Skill Title",
      "summary": "One-sentence summary",
      "target_failure_type": "localization_error|patch_generation_error|patch_application_error|test_failure|regression_introduced|unknown|general",
      "content": "# Skill Title\\n\\n## Description\\n...\\n\\n## How to Apply\\n...\\n",
      "rationale": "Which cluster this addresses and why.",
      "source": "reflector",
      "confidence": 0.0
    }
  ]
}

RULES:
- create/update proposals require title, summary, and content; content must include a '## How to Apply' section.
- Only deprecate a skill that the clusters show is unhelpful.
- Propose at most three changes. Prefer refining an existing skill over creating a near-duplicate.
- Ground every proposal in the provided clusters; do not invent unrelated skills.
"""


@dataclass
class ReflectionResult:
    """Outcome of one reflection pass."""

    cluster_count: int = 0
    qualifying_cluster_count: int = 0
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    rejected_proposals: List[Dict[str, str]] = field(default_factory=list)
    evolution: Optional[Dict[str, Any]] = None
    task_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Reflector:
    """Cluster hard cases and materialize skill/task updates."""

    def __init__(
        self,
        skill_bank: Optional[SkillBank] = None,
        skill_evolver: Optional[SkillEvolver] = None,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        buffer_path: Optional[Path] = None,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
    ):
        from src.config import get_config

        self.skill_bank = skill_bank or SkillBank()
        self.skill_evolver = skill_evolver or SkillEvolver()
        self.llm_client = llm_client
        self.embedding_client = (
            embedding_client if embedding_client is not None else EmbeddingClient.from_env()
        )
        self.buffer_path = Path(
            buffer_path or get_config().environment.workspace_dir / "hard_cases.jsonl"
        )
        self.min_cluster_size = min_cluster_size

    def reflect(
        self,
        stage: str = "train",
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        validator: Optional[Validator] = None,
    ) -> ReflectionResult:
        records = MemoryRetriever(self.buffer_path).retrieve(stage=stage, limit=limit)
        clusters = cluster_records(records, embedding_client=self.embedding_client)
        qualifying = qualifying_clusters(clusters, self.min_cluster_size)

        result = ReflectionResult(
            cluster_count=len(clusters),
            qualifying_cluster_count=len(qualifying),
            task_signals=_task_signals(qualifying),
        )
        if not qualifying:
            return result

        proposals, rejected = self._propose(qualifying)
        result.rejected_proposals = rejected
        if not proposals:
            return result

        accepted, utility = (
            validator(proposals, qualifying) if validator else (proposals, None)
        )
        result.proposals = [proposal.to_dict() for proposal in accepted]
        if accepted:
            result.evolution = self.skill_evolver.apply_proposals(accepted, utility=utility)
        return result

    def _propose(
        self,
        clusters: List[FailureCluster],
    ) -> Tuple[List[SkillUpdateProposal], List[Dict[str, str]]]:
        if self.llm_client is None:
            self.llm_client = LLMClient()
        user_prompt = self._build_user_prompt(clusters)
        response = self.llm_client.chat_with_system(REFLECTOR_SYSTEM_PROMPT, user_prompt)
        try:
            data = json.loads(extract_json_object(response.content or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Reflector response was not valid JSON: %s", exc)
            return [], [{"reason": f"invalid reflector json: {exc}", "proposal": ""}]
        return parse_proposals(data.get("skill_updates"))

    def _build_user_prompt(self, clusters: List[FailureCluster]) -> str:
        parts = ["## Failure Clusters"]
        for index, cluster in enumerate(clusters, start=1):
            parts.append(
                json.dumps(
                    {
                        "cluster": index,
                        "size": cluster.size,
                        "dominant_failure_type": cluster.dominant_failure_type,
                        "instance_ids": cluster.instance_ids[:10],
                        "summaries": [
                            _clip(record.reason or record.failure_type, 200)
                            for record in cluster.members[:5]
                        ],
                    },
                    ensure_ascii=False,
                )
            )

        parts.append("## Current Skill Bank")
        for skill in self.skill_bank.active():
            parts.append(json.dumps(_skill_view(skill), ensure_ascii=False))

        parts.append("## Output")
        parts.append("Return the skill_updates JSON only.")
        return "\n\n".join(parts)


def _task_signals(clusters: List[FailureCluster]) -> Dict[str, Any]:
    """Task-level boosts for TaskPool.apply_reflection.

    An instance that recurs across triggering clusters gets a higher
    ``cluster_hits``; family boosts are left to the task pool's own EMA policy.
    """
    instance_boosts: Dict[str, int] = {}
    for cluster in clusters:
        for instance_id in cluster.instance_ids:
            instance_boosts[instance_id] = instance_boosts.get(instance_id, 0) + 1
    return {"instance_boosts": instance_boosts, "family_boosts": {}}


def _skill_view(skill: SkillMetadata) -> Dict[str, Any]:
    return {
        "id": skill.id,
        "title": skill.title,
        "summary": skill.summary,
        "target_failure_type": skill.target_failure_type,
        "usage_count": skill.usage_count,
        "average_reward": round(skill.average_reward, 4),
    }


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
