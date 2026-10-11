"""Test backends: official SWE-bench eval in per-instance containers, or an approximate host run."""

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    START_TEST_OUTPUT,
    MAP_REPO_VERSION_TO_SPECS,
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
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from src.environment.models import TestStatus
from src.environment.f2p_subset import parse_test_ids
from src.environment.project_env import ProjectEnvironment

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_NAMESPACE = "swebench"

CONTAINER_EVAL_DIR = "/self_evolver_eval"

# Literal copy of swebench.harness.run_evaluation.GIT_APPLY_CMDS (avoids
# importing docker-py); parity is asserted in tests.
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

LOG_TAIL_CHARS = 4000

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


def default_subset_timeout() -> int:
    """Subset-gate timeout (env: SWEBENCH_SUBSET_TIMEOUT), shorter than the full-eval one."""
    return int(os.getenv("SWEBENCH_SUBSET_TIMEOUT", "600"))


def default_eval_cache_dir() -> Optional[Path]:
    """Evaluation-result cache (env: SELF_EVOLVER_EVAL_CACHE_DIR; "0"/"off" disables)."""
    value = os.getenv("SELF_EVOLVER_EVAL_CACHE_DIR", "").strip()
    if value.lower() in {"0", "off", "false", "none"}:
        return None
    if value:
        return Path(value).expanduser()
    return default_sif_cache_dir().parent / "eval_cache"


class EvalCache:
    """Content-addressed cache of container eval outcomes; timeouts and infra failures are never cached."""

    def __init__(self, root: Optional[Path]):
        self.root = Path(root) if root else None

    @staticmethod
    def key(image: str, eval_script: str, patch: str, scope: str, selected: List[str]) -> str:
        payload = json.dumps(
            {
                "v": 3,
                "image": image,
                "eval_script": hashlib.sha256(eval_script.encode("utf-8")).hexdigest(),
                "patch": hashlib.sha256((patch or "").encode("utf-8")).hexdigest(),
                "scope": scope,
                "selected": sorted(selected),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> "Optional[EvalOutcome]":
        if not self.root:
            return None
        path = self.root / key[:2] / f"{key}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        data.update({"cache_hit": True, "duration_seconds": 0.0})
        known = {item.name for item in fields(EvalOutcome)}
        return EvalOutcome(**{name: value for name, value in data.items() if name in known})

    def put(self, key: str, outcome: "EvalOutcome", log_text: str) -> None:
        if not self.root:
            return
        if TESTS_TIMEOUT in log_text or (
            APPLY_PATCH_PASS not in log_text and APPLY_PATCH_FAIL not in log_text
        ):
            return
        path = self.root / key[:2] / f"{key}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(outcome), handle)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning(f"Could not write eval cache entry {path}: {exc}")


class EvalTimingStore:
    """Adaptive per-(instance, scope) timeout: clamp(factor x slowest observed run, floor, default)."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self.enabled = bool(path) and os.getenv(
            "SELF_EVOLVER_ADAPTIVE_TIMEOUT", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.factor = float(os.getenv("SELF_EVOLVER_ADAPTIVE_TIMEOUT_FACTOR", "4"))
        self.floor = int(os.getenv("SELF_EVOLVER_ADAPTIVE_TIMEOUT_FLOOR", "180"))

    def _load(self) -> Dict[str, float]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, AttributeError):
            return {}

    def timeout(self, instance_id: str, scope: str, default: int) -> int:
        if not self.enabled:
            return default
        slowest = self._load().get(f"{instance_id}\0{scope}")
        if not slowest:
            return default
        return int(min(default, max(self.floor, self.factor * float(slowest))))

    def record(self, instance_id: str, scope: str, seconds: float) -> None:
        if not self.enabled or seconds <= 0:
            return
        from src.skills.file_lock import file_lock

        key = f"{instance_id}\0{scope}"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with file_lock(self.path):
                data = self._load()
                if seconds > float(data.get(key, 0.0)):
                    data[key] = round(seconds, 3)
                    fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(data, handle)
                    os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning(f"Could not record eval timing: {exc}")


def _representative_run(outcome: "EvalOutcome", spec: Any, log_text: str) -> bool:
    """True if the run's duration reflects the real workload (not an apply failure or import crash)."""
    if APPLY_PATCH_PASS not in log_text or END_TEST_OUTPUT not in log_text:
        return False
    expected = len(getattr(spec, "FAIL_TO_PASS", []) or []) + len(
        getattr(spec, "PASS_TO_PASS", []) or []
    )
    return expected > 0 and len(outcome.per_test) >= 0.5 * expected


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
    evaluation_scope: str = "full"
    selected_f2p_ids: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    cache_hit: bool = False
    timed_out: bool = False
    # Selected tests that ran but whose status the official parser could not
    # read (e.g. Django subTest failures); they count as failed.
    ran_without_status: List[str] = field(default_factory=list)

    @property
    def f2p_pass_fraction(self) -> float:
        return self.f2p_passed / self.f2p_total if self.f2p_total else 0.0

    @property
    def p2p_no_regression(self) -> bool:
        return self.p2p_passed == self.p2p_total

    @property
    def all_selected_executed(self) -> bool:
        """False means some selected test never ran, so the subset run is inconclusive."""
        selected = self.selected_f2p_ids or []
        seen = set(self.per_test) | set(self.ran_without_status or [])
        return bool(selected) and all(test_id in seen for test_id in selected)


def build_outcome(
    status_map: Dict[str, str],
    applied: bool,
    f2p: List[str],
    p2p: List[str],
    repo: str,
    instance_id: str,
    log_tail: str = "",
    evaluation_scope: str = "full",
    selected_f2p_ids: Optional[List[str]] = None,
) -> EvalOutcome:
    """Grade a test-status map with the official swebench report logic."""
    if repo == "pydicom/pydicom":
        # The official parser canonicalizes absolute paths in parameter IDs.
        # Full/dev's reference IDs retain the original dataset-machine paths;
        # apply exactly the same canonicalization when matching references.
        from swebench.harness.log_parsers.python import parse_log_pytest_options

        status_map = dict(status_map)
        for test_id in [*f2p, *p2p]:
            canonical = next(iter(parse_log_pytest_options(f"PASSED {test_id}", None)), test_id)
            if test_id not in status_map and canonical in status_map:
                status_map[test_id] = status_map[canonical]
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
        evaluation_scope=evaluation_scope,
        selected_f2p_ids=list(selected_f2p_ids or []),
    )


_SYMPY_FILE_VERDICT = re.compile(r"[ \t]+\[(?:OK|FAIL)\][ \t]*$", re.M)


def outcome_from_eval_log(
    spec: TestSpec,
    log_text: str,
    evaluation_scope: str = "full",
    selected_f2p_ids: Optional[List[str]] = None,
) -> EvalOutcome:
    """Grade a raw container eval log with the official swebench log parsers."""
    if evaluation_scope in {"subset", "f2p_only"} and spec.repo == "sympy/sympy":
        # sympy appends "[OK]"/"[FAIL]" to a file's last test line, which the
        # official parser drops; strip it for subset grading only.
        log_text = _SYMPY_FILE_VERDICT.sub("", log_text)
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
    outcome = build_outcome(
        status_map=status_map,
        applied=applied,
        f2p=spec.FAIL_TO_PASS,
        p2p=spec.PASS_TO_PASS,
        repo=spec.repo,
        instance_id=spec.instance_id,
        log_tail=_tail(log_text),
        evaluation_scope=evaluation_scope,
        selected_f2p_ids=selected_f2p_ids,
    )
    if evaluation_scope in {"subset", "f2p_only"} and applied:
        outcome.ran_without_status = _ran_without_status(
            log_text, [t for t in (selected_f2p_ids or []) if t not in outcome.per_test]
        )
    return outcome


def _ran_without_status(log_text: str, test_ids: List[str]) -> List[str]:
    """Selected ids whose name appears in the test output, i.e. the runner executed them."""
    if not test_ids:
        return []
    section = log_text
    if START_TEST_OUTPUT in section:
        section = section.split(START_TEST_OUTPUT, 1)[1]
    if END_TEST_OUTPUT in section:
        section = section.split(END_TEST_OUTPUT, 1)[0]
    output = "\n".join(
        line
        for line in section.split("\n")
        # Echoed commands and collection errors ("ERROR: not found: ...::x")
        # name tests that never ran.
        if not line.startswith("+ ") and "not found" not in line and not line.startswith("ERROR")
    )
    ran = []
    for test_id in test_ids:
        marker = test_id.split("::")[-1] if "::" in test_id else test_id
        # Whole-name match: test_foo must not match test_foo_bar or test_foo[x].
        if marker and re.search(rf"(?<![\w.]){re.escape(marker)}(?![\w\[])", output):
            ran.append(test_id)
    return ran


class RepositoryTestCommandBuilder:
    """Build a test command for selected ids; unsupported selections raise rather than run the full suite."""

    def build(self, instance: dict, selected_f2p_ids: List[str], strict: bool = True) -> str:
        """``strict`` refuses unvalidated selectors; ``strict=False`` gives the permissive host command."""
        tests = parse_test_ids(selected_f2p_ids)
        if not tests:
            raise ValueError("subset evaluation requires at least one F2P test id")
        repo = str(instance.get("repo") or "").lower()
        if not strict:
            return self._permissive(repo, tests)
        if repo == "sympy/sympy":
            # bin/test treats positional args as file substrings; select with -k.
            specs = MAP_REPO_VERSION_TO_SPECS.get(instance.get("repo"), {}).get(
                str(instance.get("version") or ""), {}
            )
            test_cmd = specs.get("test_cmd")
            files = get_test_directives(instance) if instance.get("test_patch") else []
            if not isinstance(test_cmd, str) or not files:
                raise NotImplementedError("sympy subset needs the official test_cmd and files")
            if any(not re.fullmatch(r"test_\w+", test_id) for test_id in tests):
                raise NotImplementedError("sympy subset ids must be plain test names")
            # ``-k`` accepts exactly one keyword (see REPO_MAX_SUBSET_SIZE).
            if len(tests) != 1:
                raise NotImplementedError("sympy bin/test can select only one test per run")
            return f"{test_cmd} {' '.join(files)} -k {shlex.quote(tests[0])}"
        if repo == "django/django":
            directives = [_django_test_directive(test_id) for test_id in tests]
            if any(item is None for item in directives):
                raise NotImplementedError("Django test id has no runnable label")
            args = " ".join(shlex.quote(item) for item in directives)
            return f"./tests/runtests.py --verbosity 2 {args}"
        # SWE-bench truncates some pytest ids at whitespace (e.g. "...[a"); an
        # uncollectable id makes pytest abort the whole run.
        runnable = [test_id for test_id in tests if _pytest_id_runnable(test_id)]
        if len(runnable) != len(tests):
            raise NotImplementedError("pytest subset contains truncated/unrunnable ids")
        args = " ".join(shlex.quote(test_id) for test_id in runnable)
        return f"python -m pytest -rA --tb=short -q {args}"

    @staticmethod
    def _permissive(repo: str, tests: List[str]) -> str:
        if repo == "django/django":
            labels = [_django_test_directive(t) or t for t in tests]
            return "./tests/runtests.py --verbosity 2 " + " ".join(shlex.quote(t) for t in labels)
        args = " ".join(shlex.quote(t) for t in tests)
        if repo == "sympy/sympy":
            return f"bin/test {args}"
        return f"python -m pytest -rA --tb=short -q {args}"

    def eval_script(
        self,
        instance: dict,
        selected_f2p_ids: List[str],
        official_eval_script: str,
        model_patch: str = "",
    ) -> str:
        """Swap only the official script's test command; drop the reinstall step when an editable install covers a .py-only change."""
        command = self.build(instance, selected_f2p_ids)
        install = _repo_install_command(instance)
        if install and _reinstall_unneeded(install, model_patch, instance.get("test_patch") or ""):
            official_eval_script = "\n".join(
                line for line in official_eval_script.split("\n") if line.strip() != install
            )
        # split("\n"), not splitlines(): splitlines() would also break on \r,
        # \x0c, ... and corrupt the test patch embedded in a heredoc.
        lines = official_eval_script.split("\n")
        starts = [i for i, line in enumerate(lines) if "Start Test Output" in line]
        ends = [i for i, line in enumerate(lines) if "End Test Output" in line]
        if not starts or not ends:
            raise RuntimeError("official eval script has no test-output marker boundary")
        start = starts[-1]
        end = next((index for index in ends if index > start), -1)
        if end < 0:
            raise RuntimeError("official eval script has an incomplete test-output boundary")
        return "\n".join([*lines[: start + 1], command, *lines[end:]]) + "\n"


def _django_test_directive(test_id: str) -> Optional[str]:
    """``test_x (mod.Class)`` -> ``mod.Class.test_x``; None for docstring-style ids."""
    value = test_id.strip()
    match = re.match(r"^(\S+)\s+\(([^)]+)\)$", value)
    if not match:
        return value if re.fullmatch(r"[\w.]+", value) else None
    test_name, dotted_scope = match.groups()
    if dotted_scope.endswith(f".{test_name}"):
        return dotted_scope
    return f"{dotted_scope}.{test_name}"


_BUILD_FILES = {"setup.py", "setup.cfg", "pyproject.toml", "meson.build", "MANIFEST.in"}


def _repo_install_command(instance: dict) -> str:
    specs = MAP_REPO_VERSION_TO_SPECS.get(str(instance.get("repo") or ""), {})
    install = (specs.get(str(instance.get("version") or "")) or {}).get("install", "")
    return install.strip() if isinstance(install, str) else ""


def _patch_paths(patch: str) -> List[str]:
    paths = []
    for line in (patch or "").split("\n"):
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].strip()
            if path != "/dev/null":
                paths.append(path[2:] if path[:2] in {"a/", "b/"} else path)
    return paths


def _reinstall_unneeded(install: str, model_patch: str, test_patch: str) -> bool:
    if " -e " not in f" {install} " and "--editable" not in install:
        return False  # e.g. requests: `pip install .` copies sources
    paths = _patch_paths(model_patch) + _patch_paths(test_patch)
    return bool(paths) and all(
        path.endswith(".py") and Path(path).name not in _BUILD_FILES for path in paths
    )


def _pytest_id_runnable(test_id: str) -> bool:
    """Reject ids SWE-bench truncated at whitespace (unbalanced ``[``)."""
    if not test_id or test_id != test_id.strip():
        return False
    return test_id.count("[") == test_id.count("]")


def run_container_command(cmd: List[str], timeout: int) -> str:
    """Run an eval container and return its combined log."""
    # A timeout kills only the ``docker run`` client, so name the container
    # and force-remove it afterwards.
    cmd = list(cmd)
    name = ""
    if cmd[:2] == ["docker", "run"]:
        name = f"self-evolver-eval-{uuid.uuid4().hex[:12]}"
        cmd[2:2] = ["--name", name]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return result.stdout or ""
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        return f"{partial}\n\n{TESTS_TIMEOUT}: {timeout} seconds exceeded."
    finally:
        if name:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def make_entry_script(repo: Optional[str] = None) -> str:
    """Bash driver mirroring swebench's run_instance: apply the patch, emit APPLY_PATCH_* markers, run eval.sh."""
    patch = f"{CONTAINER_EVAL_DIR}/patch.diff"
    numerical_env = (
        "export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512_SKX,AVX2,FMA3\n"
        if repo == "pvlib/pvlib-python" else ""
    )
    eval_command = "/bin/bash /eval.sh"
    if repo == "pyvista/pyvista":
        eval_command = "PYVISTA_OFF_SCREEN=true xvfb-run -a -s '-screen 0 1024x768x24' /bin/bash /eval.sh"
    return (
        "#!/bin/bash\n"
        # Python 3.6 otherwise inherits Apptainer's ASCII locale and fails
        # legitimate Unicode filename tests. Keep this identical on all nodes.
        "export LANG=C.UTF-8 LC_ALL=C.UTF-8\n"
        # Older pvlib tests have tight trig tolerances. NumPy's dispatched
        # AVX/FMA kernels round differently across CPU and GPU host nodes.
        f"{numerical_env}"
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
        f"{eval_command}\n"
    )


class ContainerTestBackend:
    """Official SWE-bench eval in the per-instance image via the docker or apptainer CLI."""

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
        cache_dir = default_eval_cache_dir()
        self.eval_cache = EvalCache(cache_dir)
        self.timings = EvalTimingStore(cache_dir / "timings.json" if cache_dir else None)

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
        sif = self.sif_path(image_key)
        if sif.is_file() and sif.stat().st_size > 0:
            return sif
        self.sif_cache_dir.mkdir(parents=True, exist_ok=True)
        # Concurrent samples of one instance must pull an image only once.
        from src.skills.file_lock import file_lock

        with file_lock(sif):
            if sif.is_file() and sif.stat().st_size > 0:
                return sif
            tmp = sif.with_name(f".{sif.name}.pull-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            logger.info(f"Pulling docker://{image_key} -> {sif}")
            # OCI `pull` does not forward build-only mksquashfs arguments.
            # Use `build` when compression limits were explicitly requested;
            # keep the same per-image lock and atomic cache publication.
            squashfs_args = os.environ.get("APPTAINER_MKSQUASHFS_ARGS", "").strip()
            if squashfs_args:
                command = ["apptainer", "build", "--mksquashfs-args", squashfs_args,
                           str(tmp), f"docker://{image_key}"]
            else:
                command = ["apptainer", "pull", str(tmp), f"docker://{image_key}"]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"apptainer {command[1]} docker://{image_key} failed:\n"
                        f"{_tail(result.stderr or result.stdout, 2000)}"
                    )
                os.replace(tmp, sif)
            finally:
                tmp.unlink(missing_ok=True)
        return sif

    def ensure_image(self, instance: dict) -> str:
        """Make the instance's image available locally (prewarm); returns its key."""
        image_key = self.image_key(instance)
        if self.engine == "apptainer":
            self._ensure_sif(image_key)
        else:
            subprocess.run(["docker", "pull", image_key], check=True, capture_output=True)
        return image_key

    def _container_cmd(self, spec: TestSpec, workdir: Path) -> List[str]:
        entry = f"{CONTAINER_EVAL_DIR}/entry.sh"
        if self.engine == "apptainer":
            sif = self._ensure_sif(spec.instance_image_key)
            return [
                "apptainer", "exec",
                "--containall", "--cleanenv",
                # SIF root is read-only but the eval writes to /testbed.
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
        """Alias for run_full_eval."""
        return self.run_full_eval(instance, model_patch, timeout=timeout)

    def run_full_eval(
        self,
        instance: dict,
        model_patch: str,
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        spec = self.make_spec(instance)
        return self._run_eval(
            spec,
            model_patch,
            eval_script=spec.eval_script,
            timeout=timeout,
            evaluation_scope="full",
            selected_f2p_ids=[],
        )

    def run_subset_eval(
        self,
        instance: dict,
        model_patch: str,
        selected_f2p_ids: List[str],
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        return self._run_selected_eval(
            instance, model_patch, parse_test_ids(selected_f2p_ids), "subset", timeout
        )

    def run_f2p_eval(
        self,
        instance: dict,
        model_patch: str,
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        """Run all F2P tests, no P2P; only for patches that already failed the subset, so never resolved."""
        outcome = self._run_selected_eval(
            instance,
            model_patch,
            parse_test_ids(instance.get("FAIL_TO_PASS")),
            "f2p_only",
            timeout,
        )
        outcome.resolved = False
        return outcome

    def _run_selected_eval(
        self,
        instance: dict,
        model_patch: str,
        selected: List[str],
        scope: str,
        timeout: Optional[int],
    ) -> EvalOutcome:
        subset_instance = dict(instance)
        subset_instance["FAIL_TO_PASS"] = selected
        subset_instance["PASS_TO_PASS"] = []
        spec = self.make_spec(subset_instance)
        script = RepositoryTestCommandBuilder().eval_script(
            instance,
            selected,
            official_eval_script=spec.eval_script,
            model_patch=model_patch,
        )
        return self._run_eval(
            spec,
            model_patch,
            eval_script=script,
            timeout=timeout,
            evaluation_scope=scope,
            selected_f2p_ids=selected,
        )

    def _run_eval(
        self,
        spec: TestSpec,
        model_patch: str,
        eval_script: str,
        timeout: Optional[int],
        evaluation_scope: str,
        selected_f2p_ids: List[str],
    ) -> EvalOutcome:
        if not timeout:
            default = (
                default_subset_timeout() if evaluation_scope == "subset" else default_eval_timeout()
            )
            timeout = self.timings.timeout(spec.instance_id, evaluation_scope, default)
        image_identity = spec.instance_image_key
        if self.engine == "apptainer":
            # A dependency-repaired SIF keeps the registry key. Its receipt
            # distinguishes the new environment from cached old evaluations.
            receipt = self.sif_path(image_identity).with_suffix(".verified.json")
            try:
                digest = json.loads(receipt.read_text()).get("sif_sha256", "")
                if re.fullmatch(r"[a-f0-9]{64}", digest):
                    image_identity += "@sif:" + digest
            except (OSError, ValueError, TypeError):
                pass
        cache_key = EvalCache.key(
            image_identity, eval_script, model_patch, evaluation_scope, selected_f2p_ids
        )
        cached = self.eval_cache.get(cache_key)
        if cached is not None:
            logger.info(f"{spec.instance_id}: eval cache hit (scope={evaluation_scope})")
            return cached
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="sweval_") as td:
            workdir = Path(td)
            (workdir / "patch.diff").write_text(model_patch or "", encoding="utf-8")
            (workdir / "eval.sh").write_text(eval_script, encoding="utf-8")
            (workdir / "entry.sh").write_text(make_entry_script(spec.repo), encoding="utf-8")
            cmd = self._container_cmd(spec, workdir)
            logger.info(
                f"Evaluating {spec.instance_id} scope={evaluation_scope} "
                f"via {self.engine} (timeout {timeout}s)"
            )
            output = run_container_command(cmd, timeout)
            if TESTS_TIMEOUT in output:
                logger.error(f"Eval timed out for {spec.instance_id} ({timeout}s)")
        outcome = outcome_from_eval_log(
            spec,
            output,
            evaluation_scope=evaluation_scope,
            selected_f2p_ids=selected_f2p_ids,
        )
        outcome.duration_seconds = round(time.monotonic() - started, 3)
        outcome.timed_out = TESTS_TIMEOUT in output
        if outcome.timed_out:
            # Record the limit so the learned timeout never stays below it.
            self.timings.record(spec.instance_id, evaluation_scope, float(timeout))
        elif _representative_run(outcome, spec, output):
            self.timings.record(spec.instance_id, evaluation_scope, outcome.duration_seconds)
        self.eval_cache.put(cache_key, outcome, output)
        logger.info(
            f"{spec.instance_id}: resolved={outcome.resolved} "
            f"F2P {outcome.f2p_passed}/{outcome.f2p_total} "
            f"P2P {outcome.p2p_passed}/{outcome.p2p_total}"
        )
        return outcome


class HostTestBackend:
    """Host test runs for `fix`; only an approximation of the official env for SWE-bench."""

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
        """Alias for run_full_eval."""
        return self.run_full_eval(instance, model_patch, timeout=timeout)

    def run_full_eval(
        self,
        instance: dict,
        model_patch: str,
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        """Apply, test F2P+P2P, grade, revert; the repo must already be at issue state (setup_issue)."""
        f2p = _test_id_list(instance.get("FAIL_TO_PASS"))
        p2p = _test_id_list(instance.get("PASS_TO_PASS"))
        return self._run_selected(
            instance,
            model_patch,
            f2p=f2p,
            p2p=p2p,
            timeout=timeout,
            evaluation_scope="full",
            selected_f2p_ids=[],
        )

    def run_subset_eval(
        self,
        instance: dict,
        model_patch: str,
        selected_f2p_ids: List[str],
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        selected = parse_test_ids(selected_f2p_ids)
        return self._run_selected(
            instance,
            model_patch,
            f2p=selected,
            p2p=[],
            timeout=timeout,
            evaluation_scope="subset",
            selected_f2p_ids=selected,
        )

    def _run_selected(
        self,
        instance: dict,
        model_patch: str,
        f2p: List[str],
        p2p: List[str],
        timeout: Optional[int],
        evaluation_scope: str,
        selected_f2p_ids: List[str],
    ) -> EvalOutcome:
        timeout = timeout or default_eval_timeout()
        repo = instance.get("repo", "")
        instance_id = instance.get("instance_id", "")

        if not (model_patch or "").strip():
            return build_outcome(
                {}, applied=False, f2p=f2p, p2p=p2p, repo=repo,
                instance_id=instance_id, log_tail="Empty model patch",
                evaluation_scope=evaluation_scope,
                selected_f2p_ids=selected_f2p_ids,
            )

        apply_result = self.env.apply_patch_detailed(model_patch)
        if not apply_result.success:
            self.env.revert_changes()
            return build_outcome(
                {}, applied=False, f2p=f2p, p2p=p2p, repo=repo,
                instance_id=instance_id,
                log_tail=_tail(f"{APPLY_PATCH_FAIL}\n{apply_result.diagnostic}"),
                evaluation_scope=evaluation_scope,
                selected_f2p_ids=selected_f2p_ids,
            )
        try:
            selected = [*f2p, *p2p]
            command = RepositoryTestCommandBuilder().build(instance, selected, strict=False)
            if command.startswith("python -m pytest"):
                executable = (
                    subprocess.list2cmdline([sys.executable])
                    if os.name == "nt"
                    else shlex.quote(sys.executable)
                )
                command = command.replace("python", executable, 1)
            test_result = self.env.run_tests(test_cmd=command, timeout=timeout)
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
            evaluation_scope=evaluation_scope,
            selected_f2p_ids=selected_f2p_ids,
        )


def resolve_backend(
    name: str = "auto",
    instance: Optional[dict] = None,
    env: Optional[ProjectEnvironment] = None,
) -> Union[ContainerTestBackend, HostTestBackend]:
    """Resolve auto|docker|apptainer|host; ``auto`` prefers docker, then apptainer."""
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
