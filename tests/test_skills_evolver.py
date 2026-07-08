"""SkillEvolver lifecycle: advantage credit, Ratchet retirement, cap, gating, dedup."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.controller.schema import ControllerSignal
from src.skills.proposals import SkillUpdateProposal
from src.skills.skill_dedup import is_duplicate_skill
from src.skills.skill_bank import SkillMetadata
from src.skills.skill_evolver import SkillEvolutionConfig, SkillEvolver
from src.skills.skill_store import SkillStore
from src.skills.textnorm import content_hash

SKILL = """# {title}

## Description
{desc}

## How to Apply
{proc}

Target failure type: {ftype}
"""


def _md(title: str, ftype: str = "general", desc: str = "d", proc: str = "p") -> str:
    return SKILL.format(title=title, desc=desc, proc=proc, ftype=ftype)


def _write(path: Path, title: str, ftype: str = "general", proc: str = "p") -> None:
    path.write_text(_md(title, ftype, proc=proc), encoding="utf-8")


def _evolver(tmp_path, **config) -> SkillEvolver:
    store = SkillStore(skills_dir=tmp_path)
    return SkillEvolver(
        store=store,
        config=SkillEvolutionConfig(**config),
        embedding_client=None,
    )


def _signal(skill_ids):
    return ControllerSignal.from_dict({"mode": "train", "selected_skill_ids": skill_ids})


# --------------------------------------------------------------- advantage credit


def test_advantage_credit_uses_ema_baseline(tmp_path):
    _write(tmp_path / "s.md", "S")
    store = SkillStore(skills_dir=tmp_path)

    first = store.credit_skills(["s"], utility=0.8, ema_alpha=0.3)
    assert first["advantage"] == 0.0  # no baseline on the first credit
    assert first["baseline_after"] == {"ema": pytest.approx(0.8), "count": 1}

    second = store.credit_skills(["s"], utility=0.2, ema_alpha=0.3)
    assert second["advantage"] == pytest.approx(0.2 - 0.8)
    assert second["baseline_after"]["ema"] == pytest.approx(0.7 * 0.8 + 0.3 * 0.2)

    stats = store.load_metadata()["s"]
    assert stats.usage_count == 2
    assert stats.average_advantage == pytest.approx((0.0 + -0.6) / 2)


def test_update_from_rollout_credits_known_skills_only(tmp_path):
    _write(tmp_path / "s.md", "S")
    evolver = _evolver(tmp_path)
    result = evolver.update_from_rollout(_signal(["s", "ghost"]), {"total": 0.9})
    actions = {(e["skill_id"], e["applied"]) for e in result["events"] if e["action"] == "credit"}
    assert ("s", True) in actions
    assert ("ghost", False) in actions
    assert "ghost" not in evolver.store.load_metadata()


# --------------------------------------------------------------- Ratchet retirement


def test_ratchet_retires_persistently_unhelpful_skill(tmp_path):
    _write(tmp_path / "good.md", "Good")
    _write(tmp_path / "bad.md", "Bad")
    store = SkillStore(skills_dir=tmp_path)
    for _ in range(5):
        store.credit_skills(["bad"], utility=0.1, success=False)

    evolver = _evolver(tmp_path, retire_min_trials=5, retire_net_success_threshold=-0.2)
    result = evolver.update_from_rollout(_signal(["good"]), {"total": 0.9})

    retired = {e["skill_id"] for e in result["events"] if e["action"] == "retire_unhelpful"}
    assert retired == {"bad"}
    assert store.load_metadata()["bad"].status == "retired"
    assert store.load_metadata()["good"].status == "active"


def test_ratchet_leaves_untried_skill_alone(tmp_path):
    _write(tmp_path / "fresh.md", "Fresh")
    store = SkillStore(skills_dir=tmp_path)
    for _ in range(4):  # below the 5-trial floor
        store.credit_skills(["fresh"], utility=0.0, success=False)
    evolver = _evolver(tmp_path, retire_min_trials=5)
    evolver.update_from_rollout(_signal([]), {"total": 0.0})
    assert store.load_metadata()["fresh"].status == "active"


# --------------------------------------------------------------- active-skill cap


def test_active_cap_evicts_lowest_contribution(tmp_path):
    for i in range(13):
        _write(tmp_path / f"s{i:02d}.md", f"S{i}")
    store = SkillStore(skills_dir=tmp_path)
    store.credit_skills(["s00"], utility=1.0)          # seeds the baseline high
    store.credit_skills(["s12"], utility=0.0)          # advantage = -1.0 -> worst

    evolver = _evolver(tmp_path, max_active_skills=12)
    result = evolver.apply_proposals([])

    evicted = {e["skill_id"] for e in result["events"] if e["action"] == "active_cap_evict"}
    assert evicted == {"s12"}
    active = [s for s in store.load_skills() if s.status == "active"]
    assert len(active) == 12


# --------------------------------------------------------------- write gating


def _create_proposal(skill_id="new_skill", ftype="test_failure"):
    return SkillUpdateProposal(
        operation="create",
        skill_id=skill_id,
        title="New Skill",
        summary="A new repair skill.",
        target_failure_type=ftype,
        content=_md("New Skill", ftype),
        rationale="cluster evidence",
        source="reflector",
    )


def test_write_gate_blocks_below_threshold(tmp_path):
    evolver = _evolver(tmp_path, skill_write_utility_threshold=0.55)
    result = evolver.apply_proposals([_create_proposal()], utility=0.4)
    event = result["events"][0]
    assert event["applied"] is False
    assert event["reason"] == "utility_below_write_threshold"
    assert not (tmp_path / "new_skill.md").exists()


def test_write_gate_allows_at_threshold(tmp_path):
    evolver = _evolver(tmp_path, skill_write_utility_threshold=0.55)
    result = evolver.apply_proposals([_create_proposal()], utility=0.6)
    event = result["events"][0]
    assert event["applied"] is True
    assert (tmp_path / "new_skill.md").exists()


def test_reflector_write_without_utility_is_ungated(tmp_path):
    evolver = _evolver(tmp_path)
    result = evolver.apply_proposals([_create_proposal()], utility=None)
    assert result["events"][0]["applied"] is True


# --------------------------------------------------------------- dedup vs live AND archive


def _skill(skill_id, content, status="active"):
    return SkillMetadata(
        id=skill_id, title=skill_id, summary="s", content=content,
        file_path=f"{skill_id}.md", status=status, content_hash=content_hash(content),
    )


def test_dedup_detects_archived_duplicate_directly():
    archived_content = _md("Archived", "patch_generation_error")
    existing = [
        _skill("live", _md("Live")),
        _skill("archived", archived_content, status="archived"),
    ]
    decision = is_duplicate_skill(archived_content, existing)
    assert decision.duplicate is True
    assert decision.matched_skill_id == "archived"
    assert decision.matched_status == "archived"


def test_apply_proposal_rejects_create_matching_archived_revision(tmp_path):
    evolver = _evolver(tmp_path)
    # Seed a live skill then update it, which archives the original revision.
    evolver.store.write_skill("recycled", _md("Recycled v1", "test_failure"))
    evolver.store.write_skill(
        "recycled", _md("Recycled v2 with different body text", "test_failure"),
        archive_existing=True,
    )

    duplicate = SkillUpdateProposal(
        operation="create",
        skill_id="recycled_again",
        title="Recycled v1",
        summary="s",
        target_failure_type="test_failure",
        content=_md("Recycled v1", "test_failure"),  # identical to the archived revision
        source="reflector",
    )
    result = evolver.apply_proposals([duplicate])
    event = result["events"][0]
    assert event["applied"] is False
    assert event["reason"] == "duplicate_skill"
    assert event["metadata"]["matched_status"] == "archived"


def test_self_refinement_update_is_not_treated_as_duplicate(tmp_path):
    evolver = _evolver(tmp_path)
    evolver.store.write_skill("evolving", _md("Evolving", "test_failure", proc="Original steps."))
    update = SkillUpdateProposal(
        operation="update",
        skill_id="evolving",
        title="Evolving",
        summary="s",
        target_failure_type="test_failure",
        content=_md("Evolving", "test_failure", proc="Original steps plus one extra refinement line."),
        source="reflector",
    )
    result = evolver.apply_proposals([update])
    assert result["events"][0]["applied"] is True
    assert evolver.store.load_metadata()["evolving"].revision == 2
