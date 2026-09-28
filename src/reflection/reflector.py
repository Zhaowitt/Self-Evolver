"""Batch Reflector over qualifying stable secondary failure clusters."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.controller.parser import extract_json_object
from src.llm.client import LLMClient
from src.memory.hard_case_buffer import HardCaseRecord, resolve_hard_cases_path
from src.memory.memory_retriever import MemoryRetriever
from src.reflection.clustering import (
    FailureCluster,
    StableClusterRegistry,
    cluster_records,
    qualifying_clusters,
)
from src.reflection.failure_diagnosis import adjudicate_ambiguous_diagnoses
from src.skills.embedding_client import EmbeddingClient, local_text_similarity
from src.skills.proposals import SkillUpdateProposal, parse_proposals
from src.skills.skill_bank import SkillBank, SkillMetadata
from src.skills.skill_evolver import SkillEvolver

logger = logging.getLogger(__name__)

MIN_CLUSTER_SIZE = 3
DEFAULT_RETRIEVAL_LIMIT = 200
MATCHED_SKILL_TOP_K = 3

Validator = Callable[
    [List[SkillUpdateProposal], List[FailureCluster]],
    Tuple[List[SkillUpdateProposal], Optional[float]],
]

REFLECTOR_SYSTEM_PROMPT = """You evolve procedural repair skills from stable failure-pattern clusters.

Return one raw JSON object only:
{"cluster_decisions": [{
  "cluster_id": "exact supplied id",
  "operation": "create|refine|no-op",
  "matched_skill_id": "one supplied matched id, or empty",
  "skill_id": "new stable id for create; matched id for refine; empty for no-op",
  "title": "specific procedural title",
  "description": "what concrete failure pattern this handles",
  "purpose": "why the procedure prevents that mechanism",
  "when_to_use": ["observable trigger", "observable trigger"],
  "how_to_apply": ["concrete ordered step", "concrete ordered step"],
  "constraints": ["specific guardrail", "specific guardrail"],
  "action_type": "INSPECT|PATCH|VERIFY|RETRY or a precise combination",
  "applicable_capability_label": "the supplied mechanism",
  "rationale": "decision grounded in shared and representative evidence",
  "evidence_provenance": ["supplied issue id"],
  "confidence": 0.0
}]}

Rules:
- Produce exactly one decision for every supplied cluster_id.
- First compare the pattern with the supplied top-k existing skills.
- refine only a supplied matched skill and preserve useful existing details.
- create only for a reusable new procedure, never merely because a cluster exists.
- no-op when evidence is environmental, weak, non-reusable, or already covered.
- Never deprecate/retire a skill. Long-term contribution controls retirement.
- create/refine must fill every structured section with concrete evidence-grounded content.
"""


@dataclass
class ReflectionResult:
    cluster_count: int = 0
    qualifying_cluster_count: int = 0
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    rejected_proposals: List[Dict[str, str]] = field(default_factory=list)
    evolution: Optional[Dict[str, Any]] = None
    task_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Reflector:
    """Match each stable cluster to existing skills, then create/refine/no-op."""

    def __init__(
        self,
        skill_bank: Optional[SkillBank] = None,
        skill_evolver: Optional[SkillEvolver] = None,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        buffer_path: Optional[Path] = None,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
        registry_path: Optional[Path] = None,
    ):
        self.skill_bank = skill_bank or SkillBank()
        self.skill_evolver = skill_evolver or SkillEvolver()
        self.llm_client = llm_client
        self.embedding_client = (
            embedding_client if embedding_client is not None else EmbeddingClient.from_env()
        )
        self.buffer_path = Path(buffer_path or resolve_hard_cases_path())
        self.registry_path = Path(registry_path or self.buffer_path.with_name("cluster_registry.json"))
        self.min_cluster_size = min_cluster_size

    def reflect(
        self,
        stage: str = "train",
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        validator: Optional[Validator] = None,
    ) -> ReflectionResult:
        records = MemoryRetriever(self.buffer_path).retrieve(stage=stage, limit=limit)
        return self.reflect_records(records, validator=validator)

    def reflect_records(
        self,
        records: List[HardCaseRecord],
        validator: Optional[Validator] = None,
    ) -> ReflectionResult:
        if any(
            record.failure_diagnosis
            and (
                record.failure_diagnosis.get("mechanism") == "unknown"
                or float(record.failure_diagnosis.get("confidence", 0.0) or 0.0) < 0.5
            )
            for record in records
        ):
            if self.llm_client is None:
                self.llm_client = LLMClient()
            records = adjudicate_ambiguous_diagnoses(records, self.llm_client)
        clusters = cluster_records(
            records,
            embedding_client=self.embedding_client,
            registry_path=self.registry_path,
        )
        qualifying = qualifying_clusters(clusters, self.min_cluster_size)
        result = ReflectionResult(
            cluster_count=len(clusters),
            qualifying_cluster_count=len(qualifying),
            task_signals=_task_signals(clusters),
        )
        if not qualifying:
            return result

        matches = {cluster.cluster_id: self._match_skills(cluster) for cluster in qualifying}
        decisions, rejected = self._propose(qualifying, matches)
        decisions, shape_rejections = _one_decision_per_cluster(decisions, qualifying, matches)
        result.rejected_proposals = [*rejected, *shape_rejections]

        write_candidates = [item for item in decisions if item.operation != "no-op"]
        accepted_writes, utility = (
            validator(write_candidates, qualifying)
            if validator and write_candidates
            else (write_candidates, None)
        )
        accepted_by_cluster = {
            item.cluster_id: item for item in decisions if item.operation == "no-op"
        }
        accepted_by_cluster.update({item.cluster_id: item for item in accepted_writes})
        # A validator rejection becomes an auditable no-op, not a dropped decision.
        normalized: List[SkillUpdateProposal] = []
        for cluster in qualifying:
            decision = accepted_by_cluster.get(cluster.cluster_id)
            normalized.append(
                decision
                if decision is not None
                else _no_op(cluster, matches[cluster.cluster_id], "validator rejected write")
            )
        result.proposals = [proposal.to_dict() for proposal in normalized]

        writes = [proposal for proposal in normalized if proposal.operation != "no-op"]
        if writes:
            result.evolution = self.skill_evolver.apply_proposals(writes, utility=utility)

        registry = StableClusterRegistry(self.registry_path)
        for proposal in normalized:
            registry.record_decision(proposal.cluster_id, proposal.to_dict())
        registry.save()
        return result

    def _match_skills(self, cluster: FailureCluster) -> List[SkillMetadata]:
        pattern = json.dumps(cluster.shared_pattern, sort_keys=True, ensure_ascii=False)
        ranked = sorted(
            self.skill_bank.active(),
            key=lambda skill: (
                local_text_similarity(pattern, f"{skill.title} {skill.summary} {skill.content}"),
                skill.id,
            ),
            reverse=True,
        )
        return ranked[:MATCHED_SKILL_TOP_K]

    def _propose(
        self,
        clusters: List[FailureCluster],
        matches: Dict[str, List[SkillMetadata]],
    ) -> Tuple[List[SkillUpdateProposal], List[Dict[str, str]]]:
        if self.llm_client is None:
            self.llm_client = LLMClient()
        response = self.llm_client.chat_with_system(
            REFLECTOR_SYSTEM_PROMPT,
            self._build_user_prompt(clusters, matches),
        )
        try:
            data = json.loads(extract_json_object(response.content or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Reflector response was not valid JSON: %s", exc)
            return [], [{"reason": f"invalid reflector json: {exc}", "proposal": ""}]
        return parse_proposals(data.get("cluster_decisions"))

    def _build_user_prompt(
        self,
        clusters: List[FailureCluster],
        matches: Dict[str, List[SkillMetadata]],
    ) -> str:
        payload = []
        for cluster in clusters:
            payload.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "mechanism": cluster.mechanism,
                    "size": cluster.size,
                    "unique_issue_count": cluster.unique_issue_count,
                    "cohesion": round(cluster.cohesion, 4),
                    "shared_pattern": cluster.shared_pattern,
                    "representative_hard_cases": [
                        {
                            "issue_id": record.issue_id,
                            "repo_name": record.repo_name,
                            "diagnosis": record.failure_diagnosis,
                        }
                        for record in cluster.representatives
                    ],
                    "matched_existing_skills": [
                        {
                            "id": skill.id,
                            "title": skill.title,
                            "summary": skill.summary,
                            "full_content": skill.content,
                        }
                        for skill in matches[cluster.cluster_id]
                    ],
                }
            )
        return json.dumps({"failure_clusters": payload}, ensure_ascii=False, indent=2)


def _one_decision_per_cluster(
    decisions: List[SkillUpdateProposal],
    clusters: List[FailureCluster],
    matches: Dict[str, List[SkillMetadata]],
) -> tuple[List[SkillUpdateProposal], List[Dict[str, str]]]:
    expected = {cluster.cluster_id: cluster for cluster in clusters}
    matched_ids = {
        cluster_id: {skill.id for skill in skills}
        for cluster_id, skills in matches.items()
    }
    accepted: Dict[str, SkillUpdateProposal] = {}
    rejected: List[Dict[str, str]] = []
    for decision in decisions:
        if decision.cluster_id not in expected:
            rejected.append({"reason": "unknown cluster_id", "proposal": str(decision.to_dict())[:300]})
            continue
        if decision.cluster_id in accepted:
            rejected.append({"reason": "duplicate decision for cluster", "proposal": str(decision.to_dict())[:300]})
            continue
        if decision.operation == "refine" and decision.matched_skill_id not in matched_ids[decision.cluster_id]:
            rejected.append({"reason": "refine target was not in matched top-k", "proposal": str(decision.to_dict())[:300]})
            continue
        cluster = expected[decision.cluster_id]
        if decision.operation != "no-op":
            provenance = list(dict.fromkeys(decision.evidence_provenance))
            provenance.extend(
                item.issue_id
                for item in cluster.representatives
                if item.issue_id not in provenance
            )
            decision.evidence_provenance = provenance[:3]
            decision.target_failure_type = cluster.mechanism
        accepted[decision.cluster_id] = decision
    for cluster in clusters:
        accepted.setdefault(
            cluster.cluster_id,
            _no_op(cluster, matches[cluster.cluster_id], "missing or invalid Reflector decision"),
        )
    return [accepted[cluster.cluster_id] for cluster in clusters], rejected


def _no_op(
    cluster: FailureCluster,
    matches: List[SkillMetadata],
    reason: str,
) -> SkillUpdateProposal:
    return SkillUpdateProposal(
        operation="no-op",
        cluster_id=cluster.cluster_id,
        matched_skill_id=matches[0].id if matches else "",
        rationale=reason,
        evidence_provenance=[item.issue_id for item in cluster.representatives],
    )


def _task_signals(clusters: List[FailureCluster]) -> Dict[str, Any]:
    labels: Dict[str, str] = {}
    stable_clusters: Dict[str, str] = {}
    stable_ids = {cluster.cluster_id for cluster in qualifying_clusters(clusters)}
    for cluster in clusters:
        if not cluster.diagnosis_available or cluster.mechanism == "unknown":
            continue
        for instance_id in cluster.instance_ids:
            labels[instance_id] = cluster.mechanism
            if cluster.cluster_id in stable_ids:
                stable_clusters[instance_id] = cluster.cluster_id
    return {
        "observed_failure_labels": labels,
        "stable_cluster_ids": stable_clusters,
        # Always empty; kept for TaskPool readers that expect these keys.
        "instance_boosts": {},
        "family_boosts": {},
    }
