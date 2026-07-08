"""Test-execution backends: eval-script construction, log grading, resolution.

No containers or network here. Grading uses the real swebench TestSpec and log
parsers against captured pytest text; backend resolution is exercised with the
engines actually present on the host.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from swebench.harness.run_evaluation import GIT_APPLY_CMDS as SWEBENCH_GIT_APPLY_CMDS

from src.environment import test_backend as tb
from src.environment.project_env import ProjectEnvironment

# A compact but real SWE-bench instance. repo/version resolve to specs that
# ship with swebench (offline); FAIL_TO_PASS/PASS_TO_PASS are honored verbatim
# by make_test_spec, so short lists keep the fixtures readable.
INSTANCE = {
    "instance_id": "psf__requests-2317",
    "repo": "psf/requests",
    "version": "2.4",
    "base_commit": "091991be0da19de9108dbe5e3752917fea3d7fdc",
    "environment_setup_commit": "091991be0da19de9108dbe5e3752917fea3d7fdc",
    "FAIL_TO_PASS": (
        '["test_requests.py::RequestsTestCase::test_a", '
        '"test_requests.py::RequestsTestCase::test_b"]'
    ),
    "PASS_TO_PASS": '["test_requests.py::RequestsTestCase::test_keep"]',
    "test_patch": "--- a/test_requests.py\n+++ b/test_requests.py\n@@ -1 +1 @@\n-x\n+y\n",
    "patch": "",
    "problem_statement": "bug",
    "hints_text": "",
}

F2P = ["test_requests.py::RequestsTestCase::test_a",
       "test_requests.py::RequestsTestCase::test_b"]
P2P = ["test_requests.py::RequestsTestCase::test_keep"]

AVAILABLE_ENGINE = "apptainer" if shutil.which("apptainer") else (
    "docker" if shutil.which("docker") else None
)


def _spec():
    if AVAILABLE_ENGINE is None:
        pytest.skip("no container engine on PATH")
    return tb.ContainerTestBackend(engine=AVAILABLE_ENGINE).make_spec(INSTANCE)


def _eval_log(body: str) -> str:
    """Wrap pytest body text in the entry+eval markers get_logs_eval expects."""
    return (
        f"{tb.APPLY_PATCH_PASS}\n"
        ">>>>> Start Test Output\n"
        f"{body}\n"
        ">>>>> End Test Output\n"
    )


# --- eval-script / entry-script construction -------------------------------

def test_git_apply_cmds_match_swebench():
    # The entry script must apply patches with the same ladder as the official
    # harness, or grading diverges from official resolution.
    assert tb.GIT_APPLY_CMDS == SWEBENCH_GIT_APPLY_CMDS


def test_entry_script_structure():
    script = tb.make_entry_script()
    assert script.startswith("#!/bin/bash")
    assert "cd /testbed" in script
    for cmd in tb.GIT_APPLY_CMDS:
        assert cmd in script
    assert tb.APPLY_PATCH_PASS in script
    assert tb.APPLY_PATCH_FAIL in script
    assert "/bin/bash /eval.sh" in script
    assert f"{tb.CONTAINER_EVAL_DIR}/patch.diff" in script


def test_eval_script_has_output_markers():
    spec = _spec()
    assert ">>>>> Start Test Output" in spec.eval_script
    assert ">>>>> End Test Output" in spec.eval_script


# --- image naming / sif cache ----------------------------------------------

def test_image_key_uses_swebench_namespace():
    if AVAILABLE_ENGINE is None:
        pytest.skip("no container engine on PATH")
    backend = tb.ContainerTestBackend(engine=AVAILABLE_ENGINE)
    # Verified against swebench TestSpec.instance_image_key (__ -> _1776_).
    assert backend.image_key(INSTANCE) == (
        "swebench/sweb.eval.x86_64.psf_1776_requests-2317:latest"
    )


def test_sif_path_strips_namespace_and_tag(tmp_path):
    if AVAILABLE_ENGINE is None:
        pytest.skip("no container engine on PATH")
    backend = tb.ContainerTestBackend(engine=AVAILABLE_ENGINE, sif_cache_dir=tmp_path)
    sif = backend.sif_path("swebench/sweb.eval.x86_64.psf_1776_requests-2317:latest")
    assert sif == tmp_path / "sweb.eval.x86_64.psf_1776_requests-2317.sif"


def test_default_sif_cache_dir_env_and_fallback(monkeypatch):
    monkeypatch.setenv("SIF_CACHE_DIR", "/custom/sif/dir")
    assert tb.default_sif_cache_dir() == Path("/custom/sif/dir")
    monkeypatch.delenv("SIF_CACHE_DIR", raising=False)
    assert tb.default_sif_cache_dir() == Path.home() / ".cache" / "self_evolver" / "sif"


def test_default_eval_timeout_env(monkeypatch):
    monkeypatch.setenv("SWEBENCH_EVAL_TIMEOUT", "42")
    assert tb.default_eval_timeout() == 42
    monkeypatch.delenv("SWEBENCH_EVAL_TIMEOUT", raising=False)
    assert tb.default_eval_timeout() == 1800


def test_unknown_engine_rejected():
    with pytest.raises(ValueError):
        tb.ContainerTestBackend(engine="podman")


# --- grading via build_outcome (official swebench report logic) ------------

def test_build_outcome_full_resolve():
    status = {t: "PASSED" for t in F2P + P2P}
    out = tb.build_outcome(status, applied=True, f2p=F2P, p2p=P2P,
                           repo="psf/requests", instance_id="psf__requests-2317")
    assert out.resolved is True
    assert out.f2p_passed == 2 and out.f2p_total == 2
    assert out.p2p_passed == 1 and out.p2p_total == 1
    assert out.f2p_pass_fraction == 1.0
    assert out.p2p_no_regression is True


def test_build_outcome_f2p_incomplete_is_unresolved():
    status = {F2P[0]: "PASSED", F2P[1]: "FAILED", P2P[0]: "PASSED"}
    out = tb.build_outcome(status, applied=True, f2p=F2P, p2p=P2P,
                           repo="psf/requests", instance_id="psf__requests-2317")
    assert out.resolved is False
    assert out.f2p_passed == 1 and out.f2p_total == 2
    assert out.f2p_pass_fraction == 0.5


def test_build_outcome_p2p_regression_is_unresolved():
    status = {F2P[0]: "PASSED", F2P[1]: "PASSED", P2P[0]: "FAILED"}
    out = tb.build_outcome(status, applied=True, f2p=F2P, p2p=P2P,
                           repo="psf/requests", instance_id="psf__requests-2317")
    assert out.resolved is False
    assert out.p2p_no_regression is False


def test_build_outcome_patch_not_applied_is_unresolved():
    status = {t: "PASSED" for t in F2P + P2P}
    out = tb.build_outcome(status, applied=False, f2p=F2P, p2p=P2P,
                           repo="psf/requests", instance_id="psf__requests-2317")
    assert out.resolved is False


# --- grading via outcome_from_eval_log (log parsers) -----------------------

def test_eval_log_partial_pass():
    spec = _spec()
    log = _eval_log(
        "PASSED test_requests.py::RequestsTestCase::test_a\n"
        "FAILED test_requests.py::RequestsTestCase::test_b - AssertionError: nope\n"
        "PASSED test_requests.py::RequestsTestCase::test_keep"
    )
    out = tb.outcome_from_eval_log(spec, log)
    assert out.resolved is False
    assert out.f2p_passed == 1 and out.f2p_total == 2
    assert out.p2p_passed == 1 and out.p2p_total == 1
    assert out.per_test["test_requests.py::RequestsTestCase::test_b"] == "FAILED"


def test_eval_log_full_resolve():
    spec = _spec()
    log = _eval_log(
        "PASSED test_requests.py::RequestsTestCase::test_a\n"
        "PASSED test_requests.py::RequestsTestCase::test_b\n"
        "PASSED test_requests.py::RequestsTestCase::test_keep"
    )
    out = tb.outcome_from_eval_log(spec, log)
    assert out.resolved is True
    assert out.f2p_passed == 2 and out.p2p_passed == 1


def test_eval_log_apply_failure_is_unresolved():
    spec = _spec()
    log = f"{tb.APPLY_PATCH_FAIL}\n"
    out = tb.outcome_from_eval_log(spec, log)
    assert out.resolved is False
    assert out.f2p_passed == 0


def test_eval_log_timeout_marker_is_unresolved():
    spec = _spec()
    log = (
        f"{tb.APPLY_PATCH_PASS}\n>>>>> Start Test Output\n"
        "PASSED test_requests.py::RequestsTestCase::test_a\n"
        f"{tb.TESTS_TIMEOUT}: 1800 seconds exceeded."
    )
    out = tb.outcome_from_eval_log(spec, log)
    assert out.resolved is False
    assert out.f2p_passed == 0


def test_eval_log_missing_markers_is_unresolved():
    spec = _spec()
    # Bare pytest output with no Start/End markers must not be graded as a pass.
    out = tb.outcome_from_eval_log(spec, "PASSED test_requests.py::RequestsTestCase::test_a\n")
    assert out.resolved is False
    assert out.f2p_passed == 0


def test_log_tail_is_truncated():
    spec = _spec()
    big = "x" * (tb.LOG_TAIL_CHARS + 500)
    out = tb.outcome_from_eval_log(spec, _eval_log(big))
    assert len(out.log_tail) <= tb.LOG_TAIL_CHARS


# --- backend resolution -----------------------------------------------------

def test_resolve_host_requires_env():
    with pytest.raises(ValueError):
        tb.resolve_backend("host", env=None)


def test_resolve_host_returns_host_backend(tmp_path):
    env = ProjectEnvironment(tmp_path)
    backend = tb.resolve_backend("host", env=env)
    assert isinstance(backend, tb.HostTestBackend)


def test_resolve_unknown_name():
    with pytest.raises(ValueError):
        tb.resolve_backend("nonsense")


def test_resolve_auto_prefers_docker_then_apptainer():
    if AVAILABLE_ENGINE is None:
        with pytest.raises(RuntimeError):
            tb.resolve_backend("auto")
        return
    backend = tb.resolve_backend("auto")
    expected = "docker" if shutil.which("docker") else "apptainer"
    assert isinstance(backend, tb.ContainerTestBackend)
    assert backend.engine == expected


def test_resolve_named_engine_validates_image():
    if AVAILABLE_ENGINE is None:
        pytest.skip("no container engine on PATH")
    backend = tb.resolve_backend(AVAILABLE_ENGINE, instance=INSTANCE)
    assert isinstance(backend, tb.ContainerTestBackend)
    assert backend.engine == AVAILABLE_ENGINE


def test_resolve_missing_engine_raises():
    for engine in ("docker", "apptainer"):
        if shutil.which(engine) is None:
            with pytest.raises(RuntimeError):
                tb.resolve_backend(engine)


# --- host backend eval (real host pytest, real grading) --------------------

def _init_calc_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args):
        subprocess.run(args, cwd=repo, check=True, capture_output=True, text=True)

    run("git", "init")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "T")
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "import calc\n\n\n"
        "def test_add():\n    assert calc.add(1, 2) == 3\n\n\n"
        "def test_keep():\n    assert calc.add(0, 0) == 0\n",
        encoding="utf-8",
    )
    run("git", "add", "-A")
    run("git", "commit", "-m", "init")
    return repo


HOST_INSTANCE = {
    "instance_id": "local__calc-1",
    "repo": "local/calc",
    "FAIL_TO_PASS": ["test_calc.py::test_add"],
    "PASS_TO_PASS": ["test_calc.py::test_keep"],
}

FIX_PATCH = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)


def test_host_eval_resolves_with_real_fix(tmp_path):
    repo = _init_calc_repo(tmp_path)
    backend = tb.HostTestBackend(ProjectEnvironment(repo))
    out = backend.run_swebench_eval(HOST_INSTANCE, FIX_PATCH)
    assert out.resolved is True
    assert out.f2p_passed == 1 and out.f2p_total == 1
    assert out.p2p_passed == 1 and out.p2p_total == 1
    # Patch is reverted after eval (issue state restored).
    assert backend.env.get_diff() == ""


def test_host_eval_empty_patch_is_unresolved(tmp_path):
    repo = _init_calc_repo(tmp_path)
    backend = tb.HostTestBackend(ProjectEnvironment(repo))
    out = backend.run_swebench_eval(HOST_INSTANCE, "")
    assert out.resolved is False
    assert out.f2p_total == 1
