"""
Test execution backends.

ContainerTestBackend runs the OFFICIAL SWE-bench eval semantics inside the
official per-instance container image (docker or apptainer, both via
subprocess — no docker-py). The eval script comes from
swebench.harness.test_spec.make_test_spec and the resulting log is graded
with the swebench.harness.grading parsers, so in-loop verification matches
the official harness exactly.

HostTestBackend runs tests on the host interpreter. It exists for the local
`fix` command (the user's repo, the user's real environment) and — behind an
explicit ``--test-backend host`` with a loud warning — as an approximate
fallback for benchmarks.
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_PASS,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import (
    get_eval_tests_report,
    get_logs_eval,
    get_resolution_status,
)
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from src.environment.models import TestStatus
from src.environment.project_env import ProjectEnvironment

logger = logging.getLogger(__name__)

# DockerHub namespace hosting the official prebuilt SWE-bench images.
DEFAULT_IMAGE_NAMESPACE = "swebench"

# Where the model patch is visible inside the container (host workdir bind).
CONTAINER_EVAL_DIR = "/self_evolver_eval"

# Mirrors swebench.harness.run_evaluation.GIT_APPLY_CMDS (kept literal so the
# production path never imports docker-py; parity is asserted in tests).
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

LOG_TAIL_CHARS = 4000

# Host TestStatus -> swebench grading status string. XFAIL counts as a pass,
# matching swebench.harness.grading.test_passed.
_HOST_TO_SB_STATUS = {
    TestStatus.PASSED: "PASSED",
    TestStatus.FAILED: "FAILED",
    TestStatus.ERROR: "ERROR",
    TestStatus.SKIPPED: "SKIPPED",
    TestStatus.TIMEOUT: "ERROR",
}


def default_eval_timeout() -> int:
    """Per-instance eval timeout in seconds (env: SWEBENCH_EVAL_TIMEOUT)."""
    return int(os.getenv("SWEBENCH_EVAL_TIMEOUT", "1800"))


def default_sif_cache_dir() -> Path:
    """SIF cache directory (env: SIF_CACHE_DIR)."""
    env_dir = os.getenv("SIF_CACHE_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".cache" / "self_evolver" / "sif"


def _test_id_list(value: Union[str, List[str], None]) -> List[str]:
    """FAIL_TO_PASS / PASS_TO_PASS come as JSON strings in raw dataset rows."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _tail(text: str, limit: int = LOG_TAIL_CHARS) -> str:
    return text[-limit:] if len(text) > limit else text


@dataclass
class EvalOutcome:
    """Per-test evaluation outcome, graded with official SWE-bench semantics."""

    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    resolved: bool
    per_test: Dict[str, str] = field(default_factory=dict)
    log_tail: str = ""

    @property
    def f2p_pass_fraction(self) -> float:
        return self.f2p_passed / self.f2p_total if self.f2p_total else 0.0

    @property
    def p2p_no_regression(self) -> bool:
        return self.p2p_passed == self.p2p_total


def build_outcome(
    status_map: Dict[str, str],
    applied: bool,
    f2p: List[str],
    p2p: List[str],
    repo: str,
    instance_id: str,
    log_tail: str = "",
) -> EvalOutcome:
    """
    Grade a test-status map against the instance's F2P/P2P sets using the
    official swebench report logic (FAIL_ONLY repos included).
    """
    eval_ref = {KEY_INSTANCE_ID: instance_id, FAIL_TO_PASS: f2p, PASS_TO_PASS: p2p}
    eval_type = (
        EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    )
    report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
    f2p_passed = len(report[FAIL_TO_PASS]["success"])
    p2p_passed = len(report[PASS_TO_PASS]["success"])
    resolved = (
        applied
        and get_resolution_status(report) == ResolvedStatus.FULL.value
    )
    per_test = {t: status_map[t] for t in [*f2p, *p2p] if t in status_map}
    return EvalOutcome(
        f2p_passed=f2p_passed,
        f2p_total=len(f2p),
        p2p_passed=p2p_passed,
        p2p_total=len(p2p),
        resolved=resolved,
        per_test=per_test,
        log_tail=log_tail,
    )


def outcome_from_eval_log(spec: TestSpec, log_text: str) -> EvalOutcome:
    """
    Grade a raw container eval log (patch-apply markers + eval.sh output)
    with the official swebench log parsers.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write(log_text)
        log_path = f.name
    try:
        status_map, applied = get_logs_eval(spec, log_path)
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass
    return build_outcome(
        status_map=status_map,
        applied=applied,
        f2p=spec.FAIL_TO_PASS,
        p2p=spec.PASS_TO_PASS,
        repo=spec.repo,
        instance_id=spec.instance_id,
        log_tail=_tail(log_text),
    )


def make_entry_script() -> str:
    """
    Bash driver mirroring swebench.harness.run_evaluation.run_instance:
    apply the model patch with the same fallback ladder, then run the
    official eval script. The APPLY_PATCH_* markers make the combined log
    gradable by swebench.harness.grading.get_logs_eval.
    """
    patch = f"{CONTAINER_EVAL_DIR}/patch.diff"
    return (
        "#!/bin/bash\n"
        "cd /testbed\n"
        f"if {GIT_APPLY_CMDS[0]} {patch}; then\n"
        f"    echo '{APPLY_PATCH_PASS}'\n"
        f"elif {GIT_APPLY_CMDS[1]} {patch}; then\n"
        f"    echo '{APPLY_PATCH_PASS}'\n"
        f"elif {GIT_APPLY_CMDS[2]} {patch}; then\n"
        f"    echo '{APPLY_PATCH_PASS}'\n"
        "else\n"
        f"    echo '{APPLY_PATCH_FAIL}'\n"
        "    exit 1\n"
        "fi\n"
        "/bin/bash /eval.sh\n"
    )


class ContainerTestBackend:
    """
    Runs the official SWE-bench eval for one instance inside its official
    per-instance image, via the `docker` or `apptainer` CLI.

    apptainer: `apptainer exec --containall --cleanenv --writable-tmpfs`
    against a cached SIF (pulled from docker://<namespace>/<image> on miss).
    docker: `docker run --rm` (pulls on miss).
    """

    def __init__(
        self,
        engine: str = "apptainer",
        sif_cache_dir: Optional[Union[str, Path]] = None,
        namespace: str = DEFAULT_IMAGE_NAMESPACE,
    ):
        if engine not in ("docker", "apptainer"):
            raise ValueError(f"Unknown container engine: {engine!r}")
        if shutil.which(engine) is None:
            raise RuntimeError(
                f"Container engine '{engine}' not found on PATH. Install it, or "
                "select another backend via --test-backend "
                "(auto|docker|apptainer|host)."
            )
        self.engine = engine
        self.namespace = namespace
        self.sif_cache_dir = (
            Path(sif_cache_dir).expanduser()
            if sif_cache_dir is not None
            else default_sif_cache_dir()
        )

    def make_spec(self, instance: dict) -> TestSpec:
        """Official TestSpec (eval script, image key, F2P/P2P) for an instance."""
        return make_test_spec(instance, namespace=self.namespace)

    def image_key(self, instance: dict) -> str:
        """Registry image reference, e.g. swebench/sweb.eval.x86_64.<id>:latest."""
        return self.make_spec(instance).instance_image_key

    def sif_path(self, image_key: str) -> Path:
        """Cache path for an image's SIF: strip namespace and tag, add .sif."""
        name = image_key.split("/")[-1].rsplit(":", 1)[0]
        return self.sif_cache_dir / f"{name}.sif"

    def _ensure_sif(self, image_key: str, timeout: int = 3600) -> Path:
        """Return the cached SIF for image_key, pulling it on cache miss."""
        sif = self.sif_path(image_key)
        if sif.exists():
            return sif
        self.sif_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = sif.with_name(f".{sif.name}.pull-{os.getpid()}")
        logger.info(f"Pulling docker://{image_key} -> {sif}")
        try:
            result = subprocess.run(
                ["apptainer", "pull", str(tmp), f"docker://{image_key}"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"apptainer pull docker://{image_key} failed:\n"
                    f"{_tail(result.stderr or result.stdout, 2000)}"
                )
            os.replace(tmp, sif)
        finally:
            tmp.unlink(missing_ok=True)
        return sif

    def _container_cmd(self, spec: TestSpec, workdir: Path) -> List[str]:
        entry = f"{CONTAINER_EVAL_DIR}/entry.sh"
        if self.engine == "apptainer":
            sif = self._ensure_sif(spec.instance_image_key)
            return [
                "apptainer", "exec",
                "--containall", "--cleanenv",
                # SIF root is read-only; the eval writes to /testbed (patch
                # apply, pip install -e, pyc). Overlay size is bounded by
                # apptainer.conf `sessiondir max size`.
                "--writable-tmpfs",
                "--bind", f"{workdir}:{CONTAINER_EVAL_DIR}:ro",
                "--bind", f"{workdir / 'eval.sh'}:/eval.sh:ro",
                str(sif),
                "/bin/bash", entry,
            ]
        return [
            "docker", "run", "--rm",
            "-v", f"{workdir}:{CONTAINER_EVAL_DIR}:ro",
            "-v", f"{workdir / 'eval.sh'}:/eval.sh:ro",
            spec.instance_image_key,
            "/bin/bash", entry,
        ]

    def run_swebench_eval(
        self,
        instance: dict,
        model_patch: str,
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        """
        Apply model_patch inside the instance's official container, run the
        official eval script, and grade the log. Timeouts and patch-apply
        failures grade as unresolved, exactly like the official harness.
        """
        timeout = timeout or default_eval_timeout()
        spec = self.make_spec(instance)
        with tempfile.TemporaryDirectory(prefix="sweval_") as td:
            workdir = Path(td)
            (workdir / "patch.diff").write_text(model_patch or "", encoding="utf-8")
            (workdir / "eval.sh").write_text(spec.eval_script, encoding="utf-8")
            (workdir / "entry.sh").write_text(make_entry_script(), encoding="utf-8")
            cmd = self._container_cmd(spec, workdir)
            logger.info(
                f"Evaluating {spec.instance_id} via {self.engine} "
                f"(timeout {timeout}s)"
            )
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                )
                output = result.stdout or ""
            except subprocess.TimeoutExpired as e:
                partial = e.stdout or ""
                if isinstance(partial, bytes):
                    partial = partial.decode("utf-8", errors="replace")
                output = f"{partial}\n\n{TESTS_TIMEOUT}: {timeout} seconds exceeded."
                logger.error(f"Eval timed out for {spec.instance_id} ({timeout}s)")
        outcome = outcome_from_eval_log(spec, output)
        logger.info(
            f"{spec.instance_id}: resolved={outcome.resolved} "
            f"F2P {outcome.f2p_passed}/{outcome.f2p_total} "
            f"P2P {outcome.p2p_passed}/{outcome.p2p_total}"
        )
        return outcome


class HostTestBackend:
    """
    Runs tests on the host via a ProjectEnvironment.

    This is the user's real environment for the `fix` command. For SWE-bench
    instances it is only an approximation of the official container env — the
    caller must opt in explicitly (resolve_backend warns loudly).
    """

    def __init__(self, env: ProjectEnvironment):
        self.env = env

    def run_tests(self, test_cmd: Optional[str] = None, timeout: Optional[int] = None):
        return self.env.run_tests(test_cmd=test_cmd, timeout=timeout)

    def run_swebench_eval(
        self,
        instance: dict,
        model_patch: str,
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        """
        Host approximation of the official eval: the repo must already be at
        issue state (base_commit checked out, test patch staged via
        setup_issue). Applies model_patch, runs the F2P+P2P test ids with
        pytest, grades per-test statuses, then reverts the patch.
        """
        timeout = timeout or default_eval_timeout()
        f2p = _test_id_list(instance.get("FAIL_TO_PASS"))
        p2p = _test_id_list(instance.get("PASS_TO_PASS"))
        repo = instance.get("repo", "")
        instance_id = instance.get("instance_id", "")

        if not (model_patch or "").strip():
            return build_outcome(
                {}, applied=False, f2p=f2p, p2p=p2p, repo=repo,
                instance_id=instance_id, log_tail="Empty model patch",
            )

        apply_result = self.env.apply_patch_detailed(model_patch)
        if not apply_result.success:
            self.env.revert_changes()
            return build_outcome(
                {}, applied=False, f2p=f2p, p2p=p2p, repo=repo,
                instance_id=instance_id,
                log_tail=_tail(f"{APPLY_PATCH_FAIL}\n{apply_result.diagnostic}"),
            )
        try:
            ids = " ".join(shlex.quote(t) for t in [*f2p, *p2p])
            cmd = f"{shlex.quote(sys.executable)} -m pytest -rA --tb=short -q {ids}"
            test_result = self.env.run_tests(test_cmd=cmd, timeout=timeout)
        finally:
            # Restore issue state: staged test patch survives checkout -- .
            self.env.revert_changes()
        status_map = {
            tc.name: _HOST_TO_SB_STATUS[tc.status] for tc in test_result.test_cases
        }
        return build_outcome(
            status_map,
            applied=True,
            f2p=f2p,
            p2p=p2p,
            repo=repo,
            instance_id=instance_id,
            log_tail=_tail(test_result.output + test_result.error_logs),
        )


def resolve_backend(
    name: str = "auto",
    instance: Optional[dict] = None,
    env: Optional[ProjectEnvironment] = None,
) -> Union[ContainerTestBackend, HostTestBackend]:
    """
    Resolve a test backend by name: auto|docker|apptainer|host.

    `auto` picks docker when its CLI is available, else apptainer. `host`
    requires a ProjectEnvironment and warns loudly (it does NOT reproduce the
    official SWE-bench environment). When an instance is given, container
    resolution also validates that its official image reference is derivable.
    """
    if name == "auto":
        if shutil.which("docker"):
            name = "docker"
        elif shutil.which("apptainer"):
            name = "apptainer"
        else:
            raise RuntimeError(
                "No container engine found: install docker or apptainer "
                "(or pass --test-backend host to run on the host, at the "
                "cost of official-eval fidelity)."
            )
    if name in ("docker", "apptainer"):
        backend = ContainerTestBackend(engine=name)
        if instance is not None:
            try:
                image = backend.image_key(instance)
            except KeyError as e:
                raise RuntimeError(
                    f"No official SWE-bench image spec for instance "
                    f"{instance.get('instance_id')!r} (repo/version {e}); "
                    "this instance cannot be evaluated in a container."
                ) from e
            logger.debug(f"Resolved container image: {image}")
        return backend
    if name == "host":
        logger.warning(
            "Host test backend selected: tests run on the HOST interpreter, "
            "NOT the official SWE-bench container environment. Results are "
            "approximate and dependency errors are likely."
        )
        if env is None:
            raise ValueError(
                "host backend requires a ProjectEnvironment (repo checkout)."
            )
        return HostTestBackend(env)
    raise ValueError(
        f"Unknown test backend {name!r}; expected auto|docker|apptainer|host."
    )
