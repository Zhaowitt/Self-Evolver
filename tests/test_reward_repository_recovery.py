"""Real Git regressions for missing objects, stale locks and sample isolation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import threading

import pytest

from src.environment.models import Issue
from src.reward.online_reward import prepare_environment, compute_score


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "code.py").write_text("original\n")
    git(repo, "add", "code.py")
    git(repo, "commit", "-qm", "original")
    commit = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr("src.reward.online_reward._repo_mirror", lambda *_a, **_k: repo)
    return repo, commit


def prepare(tmp_path, commit, slot=0, extra=None):
    issue = Issue(id="owner__repo-1", description="Bug", repo_name="owner/repo", base_commit=commit)
    return prepare_environment(issue, {}, {"test_cmd": "true", **(extra or {})},
                               default_workspace=tmp_path / "work", slot=slot)


def test_stale_index_lock_is_preserved_and_checkout_rebuilt(tmp_path, source):
    _, commit = source
    env = prepare(tmp_path, commit)
    (env.repo_path / ".git/index.lock").write_text("abandoned lock")
    repaired = prepare(tmp_path, commit)
    assert git(repaired.repo_path, "rev-parse", "HEAD") == commit
    assert not (repaired.repo_path / ".git/index.lock").exists()
    saved = list((tmp_path / "work/_failed_checkouts").iterdir())
    assert len(saved) == 1
    assert (saved[0] / ".git/index.lock").read_text() == "abandoned lock"


def test_missing_exact_commit_is_fetched_into_existing_checkout(tmp_path, source):
    repo, old = source
    env = prepare(tmp_path, old)
    (repo / "code.py").write_text("new\n")
    git(repo, "commit", "-qam", "new")
    new = git(repo, "rev-parse", "HEAD")
    git(repo, "reset", "--hard", old)  # object is outside current branch refs
    repaired = prepare(tmp_path, new)
    assert repaired.repo_path == env.repo_path
    assert git(repaired.repo_path, "rev-parse", "HEAD") == new
    assert (repaired.repo_path / "code.py").read_text() == "new\n"


def test_parallel_sample_uses_mirror_not_modified_base(tmp_path, source):
    _, commit = source
    base = prepare(tmp_path, commit)
    (base.repo_path / "code.py").write_text("candidate zero\n")
    (base.repo_path / ".git/index.lock").write_text("active")
    with ThreadPoolExecutor(max_workers=2) as pool:
        envs = list(pool.map(lambda slot: prepare(tmp_path, commit, slot), [1, 2]))
    assert len({env.repo_path for env in envs}) == 2
    for env in envs:
        assert (env.repo_path / "code.py").read_text() == "original\n"
        assert not (env.repo_path / ".git/index.lock").exists()
    assert (base.repo_path / "code.py").read_text() == "candidate zero\n"


def test_incomplete_checkout_is_quarantined(tmp_path, source):
    _, commit = source
    broken = tmp_path / "work/owner__repo-1"
    broken.mkdir(parents=True)
    (broken / "partial-clone").write_text("preserve")
    env = prepare(tmp_path, commit)
    assert git(env.repo_path, "rev-parse", "HEAD") == commit
    assert len(list((tmp_path / "work/_failed_checkouts").iterdir())) == 1


def test_caller_owned_repo_is_never_rebuilt(tmp_path, source):
    repo, commit = source
    (repo / ".git/index.lock").write_text("active")
    with pytest.raises(RuntimeError, match="caller-owned"):
        prepare(tmp_path, commit, extra={"repo_path": str(repo)})
    assert (repo / ".git/index.lock").read_text() == "active"
    assert not (tmp_path / "work/_failed_checkouts").exists()


def test_overlapping_reward_calls_serialize_same_managed_checkout(tmp_path, monkeypatch):
    active = 0
    peak = 0
    count_lock = threading.Lock()
    entered = threading.Event()
    second_started = threading.Event()
    release = threading.Event()
    def score(*_a, **_k):
        nonlocal active, peak
        with count_lock:
            active += 1
            peak = max(peak, active)
        entered.set()
        assert release.wait(3)
        with count_lock:
            active -= 1
        return {"overall": 0.1}
    monkeypatch.setattr("src.reward.online_reward._score_one", score)
    monkeypatch.setattr("src.environment.test_backend.resolve_backend", lambda *_a, **_k: object())
    row = {"ground_truth": {"instance_id": "same-task", "stage": "train"}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(compute_score, [row], workspace_root=tmp_path)
        assert entered.wait(3)
        def second_call():
            second_started.set()
            return compute_score([row], workspace_root=tmp_path)
        second = pool.submit(second_call)
        # Second call must not enter a simultaneous episode.
        assert second_started.wait(3)
        with pytest.raises(TimeoutError):
            second.result(timeout=0.1)
        release.set()
        assert first.result()[0]["overall"] == second.result()[0]["overall"] == 0.1
    assert peak == 1
