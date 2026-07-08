"""CLI surface: the public entrypoints exist and expose every experiment flag.

``python -m src.main benchmark --help`` must advertise the full experiment
matrix so an operator can discover it, and ``config-info`` must run without a
configured API key.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

from src.main import cli

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_FLAGS = [
    "--agent-mode",
    "--skills",
    "--memory",
    "--task-evolution",
    "--controller-mode",
    "--stage",
    "--seed",
    "--test-backend",
]
OTHER_BENCHMARK_FLAGS = [
    "--benchmark",
    "--dataset",
    "--hints",
    "--train-ids",
    "--validate-skills",
]


def test_top_level_help_lists_public_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("fix", "check", "config-info", "benchmark"):
        assert command in result.output


def test_benchmark_help_lists_every_experiment_flag():
    result = CliRunner().invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == 0
    for flag in EXPERIMENT_FLAGS + OTHER_BENCHMARK_FLAGS:
        assert flag in result.output, f"benchmark --help is missing {flag}"


def test_benchmark_help_documents_flag_choices():
    result = CliRunner().invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == 0
    for value in (
        "single", "mas",           # agent-mode
        "off", "static", "evolve",  # skills
        "train", "eval",            # stage
        "auto", "docker", "apptainer", "host",  # test-backend
    ):
        assert value in result.output, f"benchmark --help is missing choice {value}"


def test_config_info_runs_without_api_key():
    result = CliRunner().invoke(cli, ["config-info"])
    assert result.exit_code == 0
    assert result.exception is None
    assert "Model" in result.output


def test_benchmark_help_via_module_entrypoint():
    """The literal ``python -m src.main benchmark --help`` acceptance command
    works and shows the experiment flags (exercises the real entrypoint)."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.main", "benchmark", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in EXPERIMENT_FLAGS:
        assert flag in proc.stdout, f"module entrypoint help is missing {flag}"
