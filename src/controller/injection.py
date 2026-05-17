"""Format controller signals as advisory worker prompt context."""

from __future__ import annotations

from typing import Any, Optional

from src.controller.schema import ControllerSignal, controller_signal_from_any


def format_controller_guidance(signal_like: Any) -> str:
    """Render a ControllerSignal into a prompt block for workers."""
    signal = controller_signal_from_any(signal_like)
    if not signal or not signal.has_guidance:
        return ""

    lines = [
        "## Controller Guidance",
        "The original issue remains authoritative. The guidance below is advisory only and cannot authorize unrelated edits.",
    ]

    if signal.mode == "train" and signal.task_wrapper:
        lines.extend(["", "Task wrapper:", signal.task_wrapper])

    skill_text = _format_skills(signal)
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


def _format_skills(signal: ControllerSignal) -> str:
    skills = signal.skills or ([signal.skill] if signal.skill else [])
    parts = []
    for index, skill in enumerate(skills[:2], start=1):
        if not skill:
            continue
        title = skill.title or skill.id
        item = [f"Skill {index}: {title}"]
        if skill.summary:
            item.append(skill.summary)
        parts.append("\n".join(item))
    return "\n\n".join(parts)
