"""Skill update proposals emitted by the Reflector and applied by the SkillEvolver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.skills.failure_types import FailureType, normalize_failure_type
from src.skills.textnorm import slug


ALLOWED_SKILL_UPDATE_OPERATIONS = {"create", "update", "deprecate"}
ALLOWED_SKILL_UPDATE_SOURCES = {"reflector", "hard_case"}


@dataclass
class SkillUpdateProposal:
    """A proposed create/update/deprecate operation on the skill bank."""

    operation: str
    skill_id: str
    title: str = ""
    summary: str = ""
    target_failure_type: str = FailureType.GENERAL.value
    content: str = ""
    rationale: str = ""
    source: str = "reflector"
    confidence: float = 0.0

    @classmethod
    def from_any(cls, value: Any) -> tuple[Optional["SkillUpdateProposal"], Optional[str]]:
        """Validate a proposal-like value and return proposal plus rejection reason."""
        if not isinstance(value, dict):
            return None, "proposal must be an object"

        operation = _clean_text(value.get("operation")).lower()
        if operation not in ALLOWED_SKILL_UPDATE_OPERATIONS:
            return None, f"unsupported operation: {operation or '<missing>'}"

        skill_id = slug(_clean_text(value.get("skill_id")))
        if not skill_id:
            return None, "skill_id is required"

        title = _clean_text(value.get("title"))
        summary = _clean_text(value.get("summary"))
        content = _clean_text(value.get("content"))
        if operation in {"create", "update"} and not (title and summary and content):
            return None, "create/update proposals require title, summary, and content"

        source = _clean_text(value.get("source")).lower() or "reflector"
        if source not in ALLOWED_SKILL_UPDATE_SOURCES:
            source = "reflector"

        confidence_raw = value.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        return (
            cls(
                operation=operation,
                skill_id=skill_id,
                title=title,
                summary=summary,
                target_failure_type=normalize_failure_type(
                    value.get("target_failure_type"),
                    default=FailureType.GENERAL.value,
                ),
                content=content,
                rationale=_clean_text(value.get("rationale")),
                source=source,
                confidence=max(0.0, min(1.0, confidence)),
            ),
            None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_proposals(value: Any) -> tuple[List[SkillUpdateProposal], List[Dict[str, str]]]:
    """Validate a list of raw proposal dicts, collecting rejections."""
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], [{"reason": "skill_updates must be a list", "proposal": str(value)[:300]}]
    proposals: List[SkillUpdateProposal] = []
    rejected: List[Dict[str, str]] = []
    for item in value:
        proposal, reason = SkillUpdateProposal.from_any(item)
        if proposal:
            proposals.append(proposal)
        else:
            rejected.append({"reason": reason or "invalid proposal", "proposal": str(item)[:300]})
    return proposals, rejected


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
