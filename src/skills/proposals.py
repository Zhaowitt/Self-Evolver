"""Structured create/refine/no-op decisions emitted by the Reflector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.skills.skill_schema import (
    lint_skill_markdown,
    render_skill_markdown,
    structured_skill_from_dict,
)
from src.skills.textnorm import slug


ALLOWED_SKILL_UPDATE_OPERATIONS = {"create", "refine", "no-op"}
ALLOWED_SKILL_UPDATE_SOURCES = {"reflector", "hard_case"}


@dataclass
class SkillUpdateProposal:
    """One explicit decision for one qualifying secondary cluster."""

    operation: str
    skill_id: str = ""
    title: str = ""
    summary: str = ""
    target_failure_type: str = "general"
    content: str = ""
    rationale: str = ""
    source: str = "reflector"
    confidence: float = 0.0
    cluster_id: str = ""
    matched_skill_id: str = ""
    evidence_provenance: List[str] = field(default_factory=list)
    action_type: str = ""

    @classmethod
    def from_any(cls, value: Any) -> tuple[Optional["SkillUpdateProposal"], Optional[str]]:
        if not isinstance(value, dict):
            return None, "proposal must be an object"

        operation = _clean_text(value.get("operation") or value.get("decision")).lower()
        if operation == "update":
            operation = "refine"  # synonym the Reflector LLM sometimes emits
        if operation not in ALLOWED_SKILL_UPDATE_OPERATIONS:
            return None, f"unsupported operation: {operation or '<missing>'}"

        cluster_id = _clean_text(value.get("cluster_id"))
        if not cluster_id:
            return None, "cluster_id is required for every Reflector decision"
        matched_skill_id = slug(_clean_text(value.get("matched_skill_id")))
        skill_id = slug(_clean_text(value.get("skill_id")))

        if operation == "no-op":
            return cls(
                operation=operation,
                skill_id=skill_id,
                rationale=_clean_text(value.get("rationale")),
                cluster_id=cluster_id,
                matched_skill_id=matched_skill_id,
                source="reflector",
                confidence=_confidence(value.get("confidence")),
                evidence_provenance=_string_list(value.get("evidence_provenance")),
            ), None

        if not skill_id:
            return None, "skill_id is required for create/refine"
        if operation == "refine" and (not matched_skill_id or matched_skill_id != skill_id):
            return None, "refine must target the matched existing skill_id"

        content = _clean_text(value.get("content"))
        action_type = _clean_text(value.get("action_type"))
        title = _clean_text(value.get("title"))
        applicable = _clean_text(
            value.get("applicable_capability_label")
            or value.get("target_failure_type")
            or "general"
        )
        if not content:
            structured, errors = structured_skill_from_dict(value)
            if not structured:
                return None, "; ".join(errors)
            content = render_skill_markdown(structured)
            title = structured.title
            action_type = structured.action_type
            applicable = structured.applicable_capability_label
        lint_errors = lint_skill_markdown(content)
        if lint_errors:
            return None, "; ".join(lint_errors)

        source = _clean_text(value.get("source")).lower() or "reflector"
        if source not in ALLOWED_SKILL_UPDATE_SOURCES:
            source = "reflector"
        summary = _clean_text(value.get("summary") or value.get("description"))
        if not title or not summary:
            return None, "create/refine require title and description/summary"

        return cls(
            operation=operation,
            skill_id=skill_id,
            title=title,
            summary=summary,
            target_failure_type=applicable,
            content=content,
            rationale=_clean_text(value.get("rationale")),
            source=source,
            confidence=_confidence(value.get("confidence")),
            cluster_id=cluster_id,
            matched_skill_id=matched_skill_id,
            evidence_provenance=_string_list(value.get("evidence_provenance")),
            action_type=action_type,
        ), None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_proposals(value: Any) -> tuple[List[SkillUpdateProposal], List[Dict[str, str]]]:
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], [{"reason": "cluster_decisions must be a list", "proposal": str(value)[:300]}]
    proposals: List[SkillUpdateProposal] = []
    rejected: List[Dict[str, str]] = []
    for item in value:
        proposal, reason = SkillUpdateProposal.from_any(item)
        if proposal:
            proposals.append(proposal)
        else:
            rejected.append({"reason": reason or "invalid proposal", "proposal": str(item)[:300]})
    return proposals, rejected


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()
