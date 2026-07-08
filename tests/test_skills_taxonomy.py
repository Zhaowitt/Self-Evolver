"""Unified failure taxonomy and skill-evolution config loading."""

from __future__ import annotations

import textwrap

import pytest

from src.skills.failure_types import (
    FAILURE_TYPE_VALUES,
    TARGET_FAILURE_TYPES,
    FailureType,
    explicit_failure_type,
    failure_type_from_statuses,
    failure_type_from_verification_status,
    infer_skill_failure_type,
    normalize_failure_type,
)
from src.skills.skill_evolver import SkillEvolutionConfig


def test_normalize_failure_type_coerces_and_defaults():
    assert normalize_failure_type("test_failure") == "test_failure"
    assert normalize_failure_type("TEST_FAILURE") == "test_failure"
    assert normalize_failure_type("", default="general") == "general"
    assert normalize_failure_type("not_a_real_type") == FailureType.UNKNOWN.value


def test_verification_status_mapping_is_exhaustive_over_known_statuses():
    assert failure_type_from_verification_status("patch_failed") == "patch_application_error"
    assert failure_type_from_verification_status("no_changes") == "patch_application_error"
    assert failure_type_from_verification_status("empty_patch") == "patch_generation_error"
    assert failure_type_from_verification_status("tests_failed") == "test_failure"
    assert failure_type_from_verification_status("new_issues") == "regression_introduced"
    assert failure_type_from_verification_status("timeout") == "timeout"
    assert failure_type_from_verification_status("") == FailureType.UNKNOWN.value


def test_failure_type_from_statuses_uses_last_status():
    assert failure_type_from_statuses(["empty_patch", "tests_failed"]) == "test_failure"
    assert failure_type_from_statuses([]) == FailureType.UNKNOWN.value


def test_explicit_marker_beats_keyword_inference():
    content = "# X\n\n## How to Apply\nDo it.\n\nTarget failure type: regression_introduced\n"
    assert explicit_failure_type(content) == "regression_introduced"
    assert infer_skill_failure_type("anything", content) == "regression_introduced"


def test_infer_falls_back_to_keywords_and_general():
    assert infer_skill_failure_type("failure_localization", "localize the root cause") == (
        FailureType.LOCALIZATION_ERROR.value
    )
    assert infer_skill_failure_type("inspect_before_editing", "inspect the failure first") == (
        FailureType.GENERAL.value
    )


def test_seed_skills_resolve_to_valid_taxonomy_values():
    from pathlib import Path

    from src.skills.skill_bank import parse_skill_file

    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    resolved = {
        path.stem: parse_skill_file(path).target_failure_type
        for path in sorted(skills_dir.glob("*.md"))
    }
    assert resolved  # seed skills exist
    for value in resolved.values():
        assert value in FAILURE_TYPE_VALUES
    # Every declared target type is a real taxonomy value.
    for value in TARGET_FAILURE_TYPES:
        assert value in FAILURE_TYPE_VALUES


def test_skill_evolution_config_loads_defaults_from_repo_yaml():
    config = SkillEvolutionConfig.load()
    assert config.skill_write_utility_threshold == pytest.approx(0.55)
    assert config.max_active_skills == 12
    assert config.retire_min_trials == 5
    assert config.retire_net_success_threshold == pytest.approx(-0.2)


def test_skill_evolution_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "cfg.yaml"
    bad.write_text(textwrap.dedent("""
        skill_write_utility_threshold: 0.6
        bogus_key: 1
    """), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown skill_evolution config keys"):
        SkillEvolutionConfig.load(bad)


def test_skill_evolution_config_missing_file_uses_code_defaults(tmp_path):
    config = SkillEvolutionConfig.load(tmp_path / "absent.yaml")
    assert config == SkillEvolutionConfig()
