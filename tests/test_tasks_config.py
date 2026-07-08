from pathlib import Path

import pytest

import src.tasks.config as tasks_config
from src.tasks.config import (
    TaskEvolutionConfig,
    default_config_path,
    load_task_evolution_config,
)


def test_repo_config_file_matches_in_code_defaults():
    """configs/task_evolution.yaml is loaded and identical to code defaults."""
    path = default_config_path()
    assert path.exists(), "configs/task_evolution.yaml must ship with the repo"
    assert load_task_evolution_config() == TaskEvolutionConfig()


def test_absent_default_file_falls_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tasks_config, "default_config_path", lambda: tmp_path / "missing.yaml"
    )
    assert load_task_evolution_config() == TaskEvolutionConfig()


def test_explicit_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_task_evolution_config(Path("does/not/exist.yaml"))


def test_overrides_are_loaded_and_coerced(tmp_path):
    path = tmp_path / "task_evolution.yaml"
    path.write_text(
        "ema_alpha: 0.5\ntoo_hard_attempts: 2\nfocus_enabled: false\n",
        encoding="utf-8",
    )
    config = load_task_evolution_config(path)

    assert config.ema_alpha == 0.5
    assert config.too_hard_attempts == 2
    assert config.focus_enabled is False
    assert config.decay_factor == TaskEvolutionConfig().decay_factor


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "task_evolution.yaml"
    path.write_text("not_a_real_knob: 1\n", encoding="utf-8")
    assert load_task_evolution_config(path) == TaskEvolutionConfig()


@pytest.mark.parametrize(
    "content",
    [
        "ema_alpha: 0.0\n",
        "decay_factor: 1.5\n",
        "boundary_sigma: -1\n",
        "too_hard_attempts: 0\n",
        "focus_enabled: 1\n",
    ],
)
def test_invalid_values_raise(tmp_path, content):
    path = tmp_path / "task_evolution.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_task_evolution_config(path)
