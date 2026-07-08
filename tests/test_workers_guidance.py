import subprocess
from pathlib import Path

from src.controller.schema import ControllerSignal
from src.environment.models import ExecutionContext, Issue
from src.environment.project_env import ProjectEnvironment
from src.orchestrator.orchestrator import ExecutionOrchestrator
from src.workers.inspector import Inspector
from src.workers.patch_generator import PatchGenerator


def _run(cmd, cwd: Path):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    _run(["git", "add", "calc.py"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


class FakeSkill:
    def __init__(self, content):
        self.content = content


class SpyBank:
    """Records get() lookups and returns a skill carrying a procedure block."""

    def __init__(self):
        self.gets = []
        self.skill = FakeSkill(
            "# Reproduce First\n\n## How to Apply\nRun the failing test before editing.\n"
        )

    def get(self, skill_id):
        self.gets.append(skill_id)
        return self.skill


def _signal_metadata():
    return ControllerSignal.from_dict(
        {
            "mode": "train",
            "skills": [{"id": "reproduce_first", "title": "Reproduce First", "summary": "repro"}],
        }
    ).to_dict()


def _context(env):
    context = ExecutionContext(
        issue=Issue(id="x", description="bug in add"), repo_state=env.get_repo_state()
    )
    context.metadata["controller_signal"] = _signal_metadata()
    return context


def test_inspector_injects_procedure_from_provided_bank(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    bank = SpyBank()
    prompt = Inspector(env, llm_client=None, skill_bank=bank)._build_analysis_prompt(_context(env))

    assert "reproduce_first" in bank.gets
    assert "Run the failing test before editing." in prompt


def test_patch_generator_injects_procedure_from_provided_bank(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    bank = SpyBank()
    prompt = PatchGenerator(env, llm_client=None, skill_bank=bank)._build_patch_prompt(
        _context(env), None
    )

    assert "reproduce_first" in bank.gets
    assert "Run the failing test before editing." in prompt


def test_orchestrator_forwards_skill_bank_to_workers(tmp_path):
    env = ProjectEnvironment(_init_repo(tmp_path))
    bank = SpyBank()
    orch = ExecutionOrchestrator(env, llm_client=object(), skill_bank=bank)

    assert orch.inspector.skill_bank is bank
    assert orch.patch_generator.skill_bank is bank
