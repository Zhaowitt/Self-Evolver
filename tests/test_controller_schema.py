"""ControllerSignal schema: integer budget, eval safety, and no skill_updates."""

from __future__ import annotations

import os

import pytest

from src.config import reset_config
from src.controller.parser import parse_controller_response
from src.controller.schema import ControllerSignal


@pytest.fixture
def set_max_iterations():
    saved = os.environ.get("MAX_ITERATIONS")

    def _set(cap: int) -> None:
        os.environ["MAX_ITERATIONS"] = str(cap)
        reset_config()

    yield _set
    if saved is None:
        os.environ.pop("MAX_ITERATIONS", None)
    else:
        os.environ["MAX_ITERATIONS"] = saved
    reset_config()


def test_budget_clamped_to_configured_cap(set_max_iterations):
    set_max_iterations(5)
    assert ControllerSignal.from_dict({"budget": 12}).budget == 5
    assert ControllerSignal.from_dict({"budget": 0}).budget == 1
    assert ControllerSignal.from_dict({"budget": 3}).budget == 3


def test_budget_absent_or_invalid_is_none(set_max_iterations):
    set_max_iterations(3)
    assert ControllerSignal.from_dict({}).budget is None
    assert ControllerSignal.from_dict({"budget": "not-an-int"}).budget is None
    assert ControllerSignal.from_dict({"budget": True}).budget is None


def test_signal_has_no_skill_updates_field():
    signal = ControllerSignal.from_dict(
        {"mode": "train", "skill_updates": [{"operation": "create", "skill_id": "x"}]}
    )
    assert not hasattr(signal, "skill_updates")
    assert "skill_updates" not in signal.to_dict()


def test_eval_mode_forces_task_wrapper_null():
    signal = ControllerSignal.from_dict(
        {"mode": "eval", "task_wrapper": "do something first"}
    )
    assert signal.task_wrapper is None


def test_selected_skill_ids_capped_and_slugged():
    signal = ControllerSignal.from_dict(
        {"selected_skill_ids": ["Inspect Before Editing", "b", "c"]}
    )
    assert signal.selected_skill_ids == ["inspect_before_editing", "b"]


def test_parser_ignores_legacy_skill_updates_and_keeps_budget(set_max_iterations):
    set_max_iterations(4)
    raw = '{"mode": "train", "budget": 4, "skill_updates": [{"operation": "create"}]}'
    signal = parse_controller_response(raw, mode="train")
    assert signal.budget == 4
    assert not hasattr(signal, "skill_updates")
    assert signal.parse_error == ""
