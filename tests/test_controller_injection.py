"""Worker prompt injection: How-to-Apply procedure blocks and prompt contract."""

from __future__ import annotations

from pathlib import Path

from src.controller.injection import (
    MAX_PROCEDURE_CHARS,
    format_controller_guidance,
    ground_controller_skills,
)
from src.controller.prompt_builder import CONTROLLER_SYSTEM_PROMPT
from src.controller.schema import ControllerSignal
from src.skills.skill_bank import SkillBank

SKILL = """# {title}

## Description
{desc}

## How to Apply
{proc}

Target failure type: general
"""


def _write(path: Path, title: str, proc: str, desc: str = "d") -> None:
    path.write_text(SKILL.format(title=title, desc=desc, proc=proc), encoding="utf-8")


def _signal(skill_dicts):
    return ControllerSignal.from_dict(
        {"mode": "eval", "skills": skill_dicts, "strategy": "keep it minimal"}
    )


def test_injection_includes_procedure_block(tmp_path):
    _write(tmp_path / "alpha.md", "Alpha", "Reproduce, then localize, then patch.")
    signal = _signal([{"id": "alpha", "title": "Alpha", "summary": "do alpha"}])
    text = format_controller_guidance(signal, skill_bank=SkillBank(skills_dir=tmp_path))
    assert "Skill 1: Alpha" in text
    assert "\nd\n" in text
    assert "do alpha" not in text
    assert "How to apply:" in text
    assert "Reproduce, then localize, then patch." in text


def test_procedure_block_is_capped(tmp_path):
    long_proc = "step. " * 400  # ~2400 chars, well over the cap
    _write(tmp_path / "alpha.md", "Alpha", long_proc)
    signal = _signal([{"id": "alpha", "title": "Alpha", "summary": "s"}])
    text = format_controller_guidance(signal, skill_bank=SkillBank(skills_dir=tmp_path))
    injected = text.split("How to apply:\n", 1)[1]
    # The procedure segment ends before Strategy; measure just that slice.
    procedure = injected.split("\n\nStrategy:", 1)[0]
    assert len(procedure) <= MAX_PROCEDURE_CHARS


def test_at_most_two_skills_injected(tmp_path):
    for name in ("alpha", "beta", "gamma"):
        _write(tmp_path / f"{name}.md", name.title(), f"{name} steps.")
    signal = _signal(
        [
            {"id": "alpha", "title": "Alpha", "summary": "a"},
            {"id": "beta", "title": "Beta", "summary": "b"},
            {"id": "gamma", "title": "Gamma", "summary": "c"},
        ]
    )
    text = format_controller_guidance(signal, skill_bank=SkillBank(skills_dir=tmp_path))
    assert "Skill 1:" in text and "Skill 2:" in text
    assert "Skill 3:" not in text
    assert "Gamma" not in text


def test_unknown_skill_id_is_not_injected(tmp_path):
    signal = _signal([{"id": "missing", "title": "Missing", "summary": "no file"}])
    text = format_controller_guidance(signal, skill_bank=SkillBank(skills_dir=tmp_path))
    assert "Missing" not in text
    assert "How to apply:" not in text


def test_grounding_uses_candidate_allowlist_and_canonical_metadata(tmp_path):
    _write(tmp_path / "alpha.md", "Canonical Alpha", "Inspect first.", "canonical")
    _write(tmp_path / "beta.md", "Beta", "Patch second.")
    signal = _signal([
        {"id": "alpha", "title": "Spoofed", "summary": "spoofed"},
        {"id": "beta", "title": "Beta", "summary": "beta"},
    ])

    grounded = ground_controller_skills(
        signal,
        skill_bank=SkillBank(skills_dir=tmp_path),
        allowed_skill_ids=["alpha"],
    )

    assert grounded.selected_skill_ids == ["alpha"]
    assert grounded.skill.title == "Canonical Alpha"
    assert [skill.id for skill in grounded.skills] == ["alpha"]
    assert grounded.metadata["rejected_skill_ids"] == ["beta"]


def test_empty_signal_yields_no_block(tmp_path):
    assert format_controller_guidance(None, skill_bank=SkillBank(skills_dir=tmp_path)) == ""


def test_system_prompt_exposes_only_shallow_controller_action():
    assert "skill_updates" not in CONTROLLER_SYSTEM_PROMPT
    assert '"budget"' not in CONTROLLER_SYSTEM_PROMPT
    assert '"target_failure_type"' not in CONTROLLER_SYSTEM_PROMPT
    assert '"memory_query"' not in CONTROLLER_SYSTEM_PROMPT
    assert '"task_wrapper"' not in CONTROLLER_SYSTEM_PROMPT
    assert '"selected_skill_ids"' in CONTROLLER_SYSTEM_PROMPT
    assert "You do not create, update, or deprecate skills" in CONTROLLER_SYSTEM_PROMPT
