"""SkillBank mtime cache invalidation and procedure extraction."""

from __future__ import annotations

from pathlib import Path

from src.skills.skill_bank import SkillBank, extract_procedure

SKILL = """# {title}

## Description
{desc}

## How to Apply
{proc}

Target failure type: {ftype}
"""


def _write(path: Path, title: str, desc: str, proc: str, ftype: str = "general") -> None:
    path.write_text(SKILL.format(title=title, desc=desc, proc=proc, ftype=ftype), encoding="utf-8")


def test_bank_reloads_when_a_skill_file_is_added(tmp_path):
    _write(tmp_path / "alpha.md", "Alpha", "First skill.", "Step A.")
    bank = SkillBank(skills_dir=tmp_path)
    assert {skill.id for skill in bank.load()} == {"alpha"}

    _write(tmp_path / "beta.md", "Beta", "Second skill.", "Step B.")
    assert {skill.id for skill in bank.load()} == {"alpha", "beta"}


def test_bank_reloads_when_a_skill_body_changes(tmp_path):
    path = tmp_path / "alpha.md"
    _write(path, "Alpha", "First skill.", "Old procedure short.")
    bank = SkillBank(skills_dir=tmp_path)
    assert "Old procedure short." in bank.get("alpha").content

    _write(path, "Alpha", "First skill rewritten with more text.", "Brand new much longer procedure body.")
    reloaded = bank.get("alpha")
    assert "Brand new much longer procedure body." in reloaded.content
    assert "Old procedure short." not in reloaded.content


def test_bank_missing_directory_yields_no_skills(tmp_path):
    bank = SkillBank(skills_dir=tmp_path / "does_not_exist")
    assert bank.load() == []
    assert bank.active() == []


def test_extract_procedure_returns_how_to_apply_block():
    content = SKILL.format(
        title="X", desc="d", proc="Line one.\nLine two.", ftype="general"
    )
    procedure = extract_procedure(content)
    assert "Line one." in procedure
    assert "Line two." in procedure
    # The heading and description must not bleed into the procedure block.
    assert "## Description" not in procedure
    assert "How to Apply" not in procedure


def test_extract_procedure_absent_returns_empty():
    assert extract_procedure("# Title\n\n## Description\nNo procedure here.\n") == ""
