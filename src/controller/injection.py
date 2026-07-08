"""Format controller signals as advisory worker prompt context."""

from __future__ import annotations

from typing import Any, Optional

from src.controller.schema import ControllerSignal, controller_signal_from_any
from src.skills.skill_bank import SkillBank, extract_procedure

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
    """Render a ControllerSignal into a prompt block for workers.

    Each selected skill is injected with its title, one-line summary, and its
    "How to Apply" procedure (capped at MAX_PROCEDURE_CHARS characters) looked
    up from ``skill_bank`` so the worker sees the actual repair discipline, not
    just a name. Pass a frozen bank during eval so guidance matches the snapshot.
    """
    signal = controller_signal_from_any(signal_like)
    if not signal or not signal.has_guidance:
        return ""

    lines = [
        "## Controller Guidance",
        "The original issue remains authoritative. The guidance below is advisory only and cannot authorize unrelated edits.",
    ]

    if signal.mode == "train" and signal.task_wrapper:
        lines.extend(["", "Task wrapper:", signal.task_wrapper])

    skill_text = _format_skills(signal, skill_bank or _default_bank())
    if skill_text:
        lines.extend(["", skill_text])

    if signal.strategy:
        lines.extend(["", "Strategy:", signal.strategy])

    if signal.memory_query:
        lines.extend(["", "Memory cue:", signal.memory_query])

    if signal.target_failure_type:
        lines.extend(["", f"Target failure type: {signal.target_failure_type}"])

    lines.append("")
    lines.append("Keep the patch minimal and grounded in repository evidence.")
    return "\n".join(lines)


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
        procedure = _procedure_for(skill.id, skill_bank)
        if procedure:
            item.append("How to apply:")
            item.append(procedure)
        parts.append("\n".join(item))
    return "\n\n".join(parts)


def _procedure_for(skill_id: str, skill_bank: SkillBank) -> str:
    if not skill_id:
        return ""
    skill = skill_bank.get(skill_id)
    if not skill:
        return ""
    return extract_procedure(skill.content)[:MAX_PROCEDURE_CHARS].rstrip()
