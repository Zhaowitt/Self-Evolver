"""ControllerSignal schema: shallow skill selection with legacy fields ignored."""

from __future__ import annotations

from src.controller.parser import parse_controller_response
from src.controller.schema import ControllerSignal


def test_budget_is_not_part_of_controller_schema():
    signal = ControllerSignal.from_dict({"budget": 1})
    assert not hasattr(signal, "budget")
    assert "budget" not in signal.to_dict()


def test_signal_has_no_skill_updates_field():
    signal = ControllerSignal.from_dict(
        {"mode": "train", "skill_updates": [{"operation": "create", "skill_id": "x"}]}
    )
    assert not hasattr(signal, "skill_updates")
    assert "skill_updates" not in signal.to_dict()


def test_legacy_task_fields_are_not_part_of_schema():
    signal = ControllerSignal.from_dict(
        {
            "task_wrapper": "do something first",
            "target_failure_type": "test_failure",
            "memory_query": "similar bugs",
        }
    )
    for field in ("task_wrapper", "target_failure_type", "memory_query"):
        assert not hasattr(signal, field)
        assert field not in signal.to_dict()


def test_selected_skill_ids_capped_and_slugged():
    signal = ControllerSignal.from_dict(
        {"selected_skill_ids": ["Inspect Before Editing", "b", "c"]}
    )
    assert signal.selected_skill_ids == ["inspect_before_editing", "b"]


def test_parser_ignores_all_legacy_controller_fields():
    raw = (
        '{"budget": 4, "target_failure_type": "test_failure", '
        '"memory_query": "x", "task_wrapper": "y", '
        '"skill_updates": [{"operation": "create"}], "strategy": "focus"}'
    )
    signal = parse_controller_response(raw, mode="train")
    for field in ("budget", "target_failure_type", "memory_query", "task_wrapper", "skill_updates"):
        assert not hasattr(signal, field)
        assert field not in signal.to_dict()
    assert signal.strategy == "focus"
    assert signal.parse_error == ""
