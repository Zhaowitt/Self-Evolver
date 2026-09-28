"""Canonical seed-style skill rendering and quality linting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_SECTIONS = (
    "Description",
    "Purpose",
    "When to Use",
    "How to Apply",
    "Constraints",
)


@dataclass
class StructuredSkill:
    title: str
    description: str
    purpose: str
    when_to_use: List[str]
    how_to_apply: List[str]
    constraints: List[str]
    action_type: str
    applicable_capability_label: str = "general"


def render_skill_markdown(skill: StructuredSkill) -> str:
    """Render exactly the canonical structure used by seed skills."""
    return (
        f"# {skill.title.strip()}\n\n"
        f"## Description\n\n{skill.description.strip()}\n\n"
        f"## Purpose\n\n{skill.purpose.strip()}\n\n"
        f"## When to Use\n\n{_bullets(skill.when_to_use)}\n\n"
        f"## How to Apply\n\n{_bullets(skill.how_to_apply)}\n\n"
        f"## Constraints\n\n{_bullets(skill.constraints)}\n\n"
        f"Action type: {skill.action_type.strip()}.\n\n"
        f"Applicable capability label: {skill.applicable_capability_label.strip() or 'general'}\n"
    )


def lint_skill_markdown(content: str) -> List[str]:
    """Return actionable quality errors; an empty list means admissible."""
    errors: List[str] = []
    if not re.search(r"^#\s+\S", content or "", re.MULTILINE):
        errors.append("missing title")
    for section in REQUIRED_SECTIONS:
        body = extract_section(content, section)
        if not body:
            errors.append(f"missing or empty {section} section")
    procedure = _list_items(extract_section(content, "How to Apply"))
    if len(procedure) < 2:
        errors.append("How to Apply needs at least two concrete steps")
    if not re.search(r"^Action type:\s*\S", content or "", re.MULTILINE):
        errors.append("missing Action type")
    vague = {"be careful", "analyze carefully", "fix the issue", "use best practices"}
    normalized_steps = {re.sub(r"\s+", " ", item.lower()).strip(" .") for item in procedure}
    if procedure and normalized_steps.issubset(vague):
        errors.append("How to Apply is generic and non-procedural")
    return errors


def structured_skill_from_dict(value: Dict[str, Any]) -> tuple[Optional[StructuredSkill], List[str]]:
    skill = StructuredSkill(
        title=_text(value.get("title")),
        description=_text(value.get("description")),
        purpose=_text(value.get("purpose")),
        when_to_use=_string_list(value.get("when_to_use")),
        how_to_apply=_string_list(value.get("how_to_apply")),
        constraints=_string_list(value.get("constraints")),
        action_type=_text(value.get("action_type")),
        applicable_capability_label=_text(
            value.get("applicable_capability_label") or value.get("mechanism") or "general"
        ),
    )
    content = render_skill_markdown(skill)
    errors = lint_skill_markdown(content)
    return (skill if not errors else None), errors


def extract_section(content: str, name: str) -> str:
    match = re.search(
        rf"^## {re.escape(name)}\s*(.*?)(?=\n## |\nAction type:|\Z)",
        content or "",
        re.DOTALL | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"  * {str(item).strip()}" for item in items if str(item).strip())


def _list_items(value: str) -> List[str]:
    return [
        match.group(1).strip()
        for line in (value or "").splitlines()
        if (match := re.match(r"\s*(?:[-*]|\d+[.)])\s+(.+)", line))
    ]


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return _list_items(value) or ([value.strip()] if value.strip() else [])
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    return []


def _text(value: Any) -> str:
    return str(value or "").strip()
