"""Stats-aware skill selection over the unified taxonomy."""

from __future__ import annotations

from pathlib import Path

from src.skills.skill_bank import SkillBank
from src.skills.skill_selector import SkillSelector
from src.skills.skill_stats import SkillStats
from src.skills.skill_store import SkillStore

SKILL = """# {title}

## Description
{desc}

## How to Apply
{proc}

Target failure type: {ftype}
"""


def _write(path: Path, title: str, ftype: str, desc: str = "d", proc: str = "p") -> None:
    path.write_text(SKILL.format(title=title, desc=desc, proc=proc, ftype=ftype), encoding="utf-8")


def _bank(tmp_path, stats=None) -> SkillBank:
    _write(tmp_path / "loc.md", "Localize", "localization_error")
    _write(tmp_path / "regr.md", "Align", "regression_introduced")
    _write(tmp_path / "gen.md", "Inspect", "general")
    if stats:
        SkillStore(tmp_path).save_metadata(stats)
    return SkillBank(skills_dir=tmp_path)


def test_type_match_dominates_ranking(tmp_path):
    selector = SkillSelector(_bank(tmp_path))
    selected = selector.select_many(target_failure_type="localization_error", limit=2)
    assert selected[0].id == "loc"
    # 'general' skills are weak matches and rank ahead of unrelated types.
    assert selected[1].id == "gen"


def test_low_reward_skill_is_downweighted_out(tmp_path):
    stats = {"loc": SkillStats(id="loc", usage_count=3, average_reward=0.2)}
    selector = SkillSelector(_bank(tmp_path, stats))
    selected = selector.select_many(target_failure_type="localization_error", limit=2)
    # loc scores 2.0 (type) - 2.0 (low-reward penalty) = 0.0 -> excluded.
    assert "loc" not in {skill.id for skill in selected}
    assert selected[0].id == "gen"


def test_proven_skill_gets_reward_bonus(tmp_path):
    stats = {"gen": SkillStats(id="gen", usage_count=5, average_reward=0.9)}
    selector = SkillSelector(_bank(tmp_path, stats))
    # No explicit target: gen wins purely on its reward bonus (0.45) over untouched peers.
    selected = selector.select_many(target_failure_type="", memory_query="", limit=1)
    assert selected[0].id == "gen"


def test_query_overlap_breaks_ties(tmp_path):
    selector = SkillSelector(_bank(tmp_path))
    selected = selector.select_many(memory_query="align regression pattern", limit=1)
    assert selected[0].id == "regr"


def test_empty_bank_returns_nothing(tmp_path):
    selector = SkillSelector(SkillBank(skills_dir=tmp_path / "empty"))
    assert selector.select_many(target_failure_type="test_failure") == []


def test_fallback_when_no_skill_scores(tmp_path):
    # A general-only bank with an untargeted query and no overlap still yields one skill.
    _write(tmp_path / "gen.md", "Inspect", "general")
    selector = SkillSelector(SkillBank(skills_dir=tmp_path))
    selected = selector.select_many(target_failure_type="", memory_query="", limit=2)
    assert len(selected) == 1
    assert selected[0].id == "gen"
