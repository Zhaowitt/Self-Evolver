"""Format controller signals as advisory worker prompt context."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from src.controller.schema import ControllerSignal, SkillSignal, controller_signal_from_any
from src.skills.skill_bank import SkillBank, extract_constraints, extract_procedure
from src.skills.textnorm import slug

MAX_PROCEDURE_CHARS = 800
MAX_INJECTED_SKILLS = 2

_DEFAULT_BANK: Optional[SkillBank] = None


def _default_bank() -> SkillBank:
    """Shared, mtime-invalidating skill bank used to look up procedure blocks."""
    global _DEFAULT_BANK
    if _DEFAULT_BANK is None:
        _DEFAULT_BANK = SkillBank()
    return _DEFAULT_BANK


def format_controller_guidance(
    signal_like: Any,
    skill_bank: Optional[SkillBank] = None,
) -> str:
    """Render grounded skills and strategy for workers; pass a frozen bank during eval."""
    bank = skill_bank or _default_bank()
    signal = ground_controller_skills(signal_like, skill_bank=bank)
    if not signal or not signal.has_guidance:
        return ""

    lines = [
        "## Controller Guidance",
        "The original issue remains authoritative. The guidance below is advisory only and cannot authorize unrelated edits.",
    ]

    skill_text = _format_skills(signal, bank)
    if skill_text:
        lines.extend(["", skill_text])

    if signal.strategy:
        lines.extend(["", "Strategy:", signal.strategy])

    lines.append("")
    lines.append("Keep the patch minimal and grounded in repository evidence.")
    return "\n".join(lines)


def ground_controller_skills(
    signal_like: Any,
    skill_bank: Optional[SkillBank] = None,
    allowed_skill_ids: Optional[Iterable[str]] = None,
) -> Optional[ControllerSignal]:
    """Keep only active (and allowed) bank skills; prompt instructions alone are not trusted."""
    signal = controller_signal_from_any(signal_like)
    if signal is None:
        return None

    bank = skill_bank or _default_bank()
    active = {skill.id: skill for skill in bank.active()}
    if allowed_skill_ids is not None:
        allowed = {slug(str(skill_id)) for skill_id in allowed_skill_ids}
        active = {skill_id: skill for skill_id, skill in active.items() if skill_id in allowed}

    requested = list(signal.selected_skill_ids)
    requested.extend(skill.id for skill in signal.skills if skill and skill.id)
    if signal.skill and signal.skill.id:
        requested.append(signal.skill.id)

    grounded = []
    rejected = []
    seen = set()
    for raw_id in requested:
        skill_id = slug(str(raw_id))
        if not skill_id or skill_id in seen:
            continue
        seen.add(skill_id)
        metadata = active.get(skill_id)
        if metadata is None:
            rejected.append(skill_id)
            continue
        grounded.append(metadata)
        if len(grounded) >= MAX_INJECTED_SKILLS:
            break

    signal.skills = [
        SkillSignal(
            id=skill.id,
            title=skill.title,
            summary=skill.summary,
            target_failure_type=skill.target_failure_type,
        )
        for skill in grounded
    ]
    signal.skill = signal.skills[0] if signal.skills else None
    signal.selected_skill_ids = [skill.id for skill in grounded]
    if rejected:
        signal.metadata = dict(signal.metadata)
        signal.metadata["rejected_skill_ids"] = rejected
    return signal


def _format_skills(signal: ControllerSignal, skill_bank: SkillBank) -> str:
    skills = signal.skills or ([signal.skill] if signal.skill else [])
    parts = []
    for index, skill in enumerate(skills[:MAX_INJECTED_SKILLS], start=1):
        if not skill:
            continue
        title = skill.title or skill.id
        item = [f"Skill {index}: {title}"]
        if skill.summary:
            item.append(skill.summary)
        procedure = _skill_section(skill.id, skill_bank, extract_procedure)
        if procedure:
            item.append("How to apply:")
            item.append(procedure)
        constraints = _skill_section(skill.id, skill_bank, extract_constraints)
        if constraints:
            item.append("Constraints:")
            item.append(constraints)
        parts.append("\n".join(item))
    return "\n\n".join(parts)


def _skill_section(skill_id: str, skill_bank: SkillBank, extract: Callable[[str], str]) -> str:
    if not skill_id:
        return ""
    skill = skill_bank.get(skill_id)
    if not skill:
        return ""
    return extract(skill.content)[:MAX_PROCEDURE_CHARS].rstrip()
