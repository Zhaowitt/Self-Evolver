import subprocess
from pathlib import Path

from src.config import get_config
from src.environment.models import Issue, PatchInfo
from src.environment.project_env import ProjectEnvironment
from src.orchestrator.orchestrator import ExecutionOrchestrator, ExecutionStatus
from src.workers.base import WorkerResult
from src.workers.inspector import InspectionResult
from src.workers.llm_judge import JudgeDecision, JudgeRoute
from src.workers.patch_generator import PatchResult
from src.workers.verifier import VerificationResult, VerificationStatus


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


CANONICAL_DIFF = (
    "diff --git a/calc.py b/calc.py\n"
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)


class FakeClient:
    def __init__(self):
        self._t = 0

    def reset_token_count(self):
        self._t = 0

    @property
    def total_tokens_used(self):
        return self._t


class FakeInspector:
    def __init__(self):
        self.calls = 0
        self.feedback_seen = []

    def execute(self, context):
        self.calls += 1
        self.feedback_seen.append(context.metadata.get("judge_feedback"))
        return WorkerResult(
            success=True,
            data=InspectionResult(
                suspected_files=["calc.py"],
                root_cause_analysis="subtracts instead of adds",
                fix_suggestions=["return a + b"],
                confidence=0.9,
            ),
        )


class FakePatchGen:
    def __init__(self, patch_content=CANONICAL_DIFF):
        self.patch_content = patch_content

    def execute(self, context, inspection=None):
        info = PatchInfo.from_diff(self.patch_content) if self.patch_content else None
        return WorkerResult(
            success=True,
            data=PatchResult(patch_content=self.patch_content, patch_info=info),
        )


class FakeVerifier:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def execute(self, context, patch_result=None):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        data = VerificationResult(
            status=status,
            patch_applied=True,
            tests_passed=(status == VerificationStatus.SUCCESS),
            canonical_patch_content=CANONICAL_DIFF,
            canonical_patch_info=PatchInfo.from_diff(CANONICAL_DIFF),
            summary=f"verifier {status.value}",
        )
        return WorkerResult(success=data.success, data=data)


class FakeJudge:
    def __init__(self, route):
        self.route = route
        self.calls = 0

    def execute(self, context, record=None):
        self.calls += 1
        return WorkerResult(
            success=True,
            data=JudgeDecision(
                failure_category="scripted",
                route=self.route,
                feedback_for_next_attempt=f"feedback-{self.calls}",
            ),
        )


def _orchestrator(tmp_path, monkeypatch, *, verifier_statuses, judge_route, budget=None, max_iterations=3):
    monkeypatch.setattr(get_config().environment, "workspace_dir", tmp_path / "ws")
    env = ProjectEnvironment(_init_repo(tmp_path))
    signal = {"mode": "train", "budget": budget} if budget is not None else None
    orch = ExecutionOrchestrator(
        env, llm_client=FakeClient(), max_iterations=max_iterations, controller_signal=signal
    )
    orch.inspector = FakeInspector()
    orch.patch_generator = FakePatchGen()
    orch.verifier = FakeVerifier(verifier_statuses)
    orch.llm_judge = FakeJudge(judge_route)
    return orch


def test_controller_budget_caps_iterations(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch,
        verifier_statuses=[VerificationStatus.TESTS_FAILED],
        judge_route=JudgeRoute.REGENERATE_PATCH_SAME_LOCATION,
        budget=2, max_iterations=3,
    )

    result = orch.run(Issue(id="local-1", description="add() should add"))

    assert orch._effective_max_iterations() == 2
    assert len(result.iteration_records) == 2
    assert result.iterations_used == 2
    assert result.status == ExecutionStatus.MAX_ITERATIONS


def test_budget_above_cap_is_clamped_to_config(tmp_path, monkeypatch):
    # The signal clamps budget to [1, config max]; the loop then caps at that.
    orch = _orchestrator(
        tmp_path, monkeypatch,
        verifier_statuses=[VerificationStatus.TESTS_FAILED],
        judge_route=JudgeRoute.REGENERATE_PATCH_SAME_LOCATION,
        budget=999, max_iterations=3,
    )

    result = orch.run(Issue(id="local-1", description="add"))

    assert orch._effective_max_iterations() == min(3, get_config().agent.max_iterations)
    assert len(result.iteration_records) == orch._effective_max_iterations()


def test_reinspect_reruns_inspector_and_propagates_judge_feedback(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch,
        verifier_statuses=[VerificationStatus.TESTS_FAILED],
        judge_route=JudgeRoute.REINSPECT,
        budget=2,
    )

    orch.run(Issue(id="local-1", description="add"))

    # Inspector re-ran on the REINSPECT route and saw the judge's feedback.
    assert orch.inspector.calls == 2
    assert orch.inspector.feedback_seen[0] is None
    assert orch.inspector.feedback_seen[1] == "feedback-1"


def test_regenerate_reuses_cached_inspection(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch,
        verifier_statuses=[VerificationStatus.TESTS_FAILED],
        judge_route=JudgeRoute.REGENERATE_PATCH_SAME_LOCATION,
        budget=2,
    )

    orch.run(Issue(id="local-1", description="add"))

    # Localization is kept across the retry; the inspector runs only once.
    assert orch.inspector.calls == 1


def test_success_exits_immediately(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch,
        verifier_statuses=[VerificationStatus.SUCCESS],
        judge_route=JudgeRoute.REGENERATE_PATCH_SAME_LOCATION,
        budget=3,
    )

    result = orch.run(Issue(id="local-1", description="add"))

    assert result.success
    assert result.status == ExecutionStatus.SUCCESS
    assert result.iterations_used == 1
    assert result.final_patch is not None
    assert orch.verifier.calls == 1


def test_give_up_breaks_loop_and_flags_hard_case(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch,
        verifier_statuses=[VerificationStatus.TESTS_FAILED],
        judge_route=JudgeRoute.GIVE_UP_HARD_CASE,
        budget=3,
    )

    result = orch.run(Issue(id="local-1", description="add"))

    assert result.status == ExecutionStatus.FAILED
    assert result.metadata["hard_case"] is True
    assert len(result.iteration_records) == 1  # broke on the first give-up


def test_effective_max_iterations_without_budget(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch,
        verifier_statuses=[VerificationStatus.SUCCESS],
        judge_route=JudgeRoute.REGENERATE_PATCH_SAME_LOCATION,
        budget=None, max_iterations=3,
    )

    assert orch._effective_max_iterations() == 3
