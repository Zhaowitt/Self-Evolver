"""SWE-bench-Live runner: ``pytest -rA`` in the official image (its repos have no swebench spec)."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS, TESTS_TIMEOUT

from src.benchmark.swebench_runner import SWEBenchRunner
from src.environment.models import TestResult
from src.environment.test_backend import (
    _HOST_TO_SB_STATUS,
    run_container_command,
    EvalOutcome,
    build_outcome,
    default_eval_timeout,
    default_sif_cache_dir,
)

logger = logging.getLogger(__name__)

LIVE_NAMESPACE = "starryzhang"
LIVE_SPLITS = ("test", "lite", "verified", "full")


def live_image_ref(instance_id: str, namespace: str = LIVE_NAMESPACE) -> str:
    """Mirrors SWE-bench-Live's ``get_default_image_name`` (x86_64)."""
    name = instance_id.replace("__", "_1776_").lower()
    return f"{namespace}/sweb.eval.x86_64.{name}"


def _test_ids(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        import json

        value = value.strip()
        if not value:
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(item) for item in value if str(item).strip()]


def grade_pytest_log(
    instance: dict,
    output: str,
    applied: bool,
    evaluation_scope: str = "full",
    selected_f2p_ids: Optional[List[str]] = None,
) -> EvalOutcome:
    """Grade a captured ``pytest -rA`` log with official swebench report logic."""
    f2p = _test_ids(instance.get("FAIL_TO_PASS"))
    p2p = _test_ids(instance.get("PASS_TO_PASS"))
    parsed = TestResult.from_pytest_output(passed=applied, output=output)
    status_map = {case.name: _HOST_TO_SB_STATUS[case.status] for case in parsed.test_cases}
    return build_outcome(
        status_map,
        applied,
        f2p,
        p2p,
        instance.get("repo", ""),
        instance.get("instance_id", ""),
        log_tail=output[-4000:],
        evaluation_scope=evaluation_scope,
        selected_f2p_ids=selected_f2p_ids,
    )


TEST_PATCH_FAIL = ">>>>> Self-Evolver: test patch failed to apply"


def _test_patch_files(test_patch: str) -> List[str]:
    files = []
    for line in (test_patch or "").split("\n"):
        if line.startswith("diff --git a/"):
            path = line[len("diff --git a/"):].split(" b/", 1)[0]
            if path and path not in files:
                files.append(path)
    return files


def _entry_script(pytest_ids: str, base_commit: str = "", test_files: Optional[List[str]] = None) -> str:
    """Apply the model patch, reset test files to ``base_commit``, apply the test patch, run pytest -rA."""
    # Resetting test files stops a model patch that edits tests from being graded on its own tests.
    patch = "/self_evolver_eval/patch.diff"
    test_patch = "/self_evolver_eval/test.patch"
    reset = ""
    if base_commit and test_files:
        for path in test_files:
            quoted = shlex.quote(path)
            reset += (
                f"git checkout {shlex.quote(base_commit)} -- {quoted} 2>/dev/null "
                f"|| rm -f {quoted}\n"
            )
    return (
        "#!/bin/bash\n"
        "cd /testbed\n"
        f"if git apply --verbose {patch}; then echo '{APPLY_PATCH_PASS}'\n"
        f"elif git apply --verbose --reject {patch}; then echo '{APPLY_PATCH_PASS}'\n"
        f"elif patch --batch --fuzz=5 -p1 -i {patch}; then echo '{APPLY_PATCH_PASS}'\n"
        f"else echo '{APPLY_PATCH_FAIL}'; exit 1; fi\n"
        f"{reset}"
        f"if [ -s {test_patch} ] && ! git apply --verbose {test_patch}; then "
        f"echo '{TEST_PATCH_FAIL}'; exit 1; fi\n"
        "if [ -f /opt/miniconda3/bin/activate ]; then "
        "source /opt/miniconda3/bin/activate testbed 2>/dev/null "
        "|| source /opt/miniconda3/bin/activate 2>/dev/null || true; fi\n"
        f"python -m pytest -rA -p no:cacheprovider {pytest_ids} "
        f"|| pytest -rA -p no:cacheprovider {pytest_ids}\n"
    )


class LiveTestBackend:
    """Grade a SWE-bench-Live instance inside its official image (pytest, spec-free)."""

    def __init__(
        self,
        engine: str = "apptainer",
        sif_cache_dir: Optional[Path] = None,
        namespace: str = LIVE_NAMESPACE,
    ):
        if engine not in ("docker", "apptainer"):
            raise ValueError(f"Unknown container engine: {engine!r}")
        if shutil.which(engine) is None:
            raise RuntimeError(
                f"Container engine {engine!r} not found on PATH; SWE-bench-Live eval "
                "needs docker or apptainer."
            )
        self.engine = engine
        self.namespace = namespace
        self.sif_cache_dir = Path(sif_cache_dir) if sif_cache_dir else default_sif_cache_dir()

    def image_key(self, instance: dict) -> str:
        return live_image_ref(instance["instance_id"], self.namespace)

    def _sif_path(self, image_ref: str) -> Path:
        name = image_ref.split("/")[-1].rsplit(":", 1)[0]
        return self.sif_cache_dir / f"{name}.sif"

    def _ensure_sif(self, image_ref: str, timeout: int = 3600) -> Path:
        sif = self._sif_path(image_ref)
        if sif.exists():
            return sif
        self.sif_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = sif.with_name(f".{sif.name}.pull-{os.getpid()}")
        logger.info("Pulling docker://%s -> %s", image_ref, sif)
        try:
            result = subprocess.run(
                ["apptainer", "pull", str(tmp), f"docker://{image_ref}"],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"apptainer pull docker://{image_ref} failed; the SWE-bench-Live "
                    f"image may not be published:\n{(result.stderr or result.stdout)[-2000:]}"
                )
            os.replace(tmp, sif)
        finally:
            tmp.unlink(missing_ok=True)
        return sif

    def _container_cmd(self, image_ref: str, workdir: Path) -> List[str]:
        entry = "/self_evolver_eval/entry.sh"
        if self.engine == "apptainer":
            sif = self._ensure_sif(image_ref)
            return [
                "apptainer", "exec", "--containall", "--cleanenv", "--writable-tmpfs",
                "--bind", f"{workdir}:/self_evolver_eval:ro",
                str(sif), "/bin/bash", entry,
            ]
        return [
            "docker", "run", "--rm",
            "-v", f"{workdir}:/self_evolver_eval:ro",
            image_ref, "/bin/bash", entry,
        ]

    def run_swebench_eval(
        self,
        instance: dict,
        model_patch: str,
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        return self.run_full_eval(instance, model_patch, timeout=timeout)

    def run_full_eval(
        self,
        instance: dict,
        model_patch: str,
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        return self._run_eval(instance, model_patch, timeout, "full", [])

    def run_subset_eval(
        self,
        instance: dict,
        model_patch: str,
        selected_f2p_ids: List[str],
        timeout: Optional[int] = None,
    ) -> EvalOutcome:
        selected = _test_ids(selected_f2p_ids)
        subset = {**instance, "FAIL_TO_PASS": selected, "PASS_TO_PASS": []}
        return self._run_eval(subset, model_patch, timeout, "subset", selected)

    def _run_eval(
        self,
        instance: dict,
        model_patch: str,
        timeout: Optional[int],
        evaluation_scope: str,
        selected_f2p_ids: List[str],
    ) -> EvalOutcome:
        timeout = timeout or default_eval_timeout()
        f2p = _test_ids(instance.get("FAIL_TO_PASS"))
        p2p = _test_ids(instance.get("PASS_TO_PASS"))
        repo = instance.get("repo", "")
        instance_id = instance.get("instance_id", "")
        if not (model_patch or "").strip():
            return build_outcome(
                {}, False, f2p, p2p, repo, instance_id, "Empty model patch",
                evaluation_scope=evaluation_scope,
                selected_f2p_ids=selected_f2p_ids,
            )

        image_ref = self.image_key(instance)
        ids = " ".join(shlex.quote(test) for test in [*f2p, *p2p])
        with tempfile.TemporaryDirectory(prefix="live_eval_") as td:
            workdir = Path(td)
            (workdir / "patch.diff").write_text(model_patch, encoding="utf-8")
            (workdir / "test.patch").write_text(instance.get("test_patch") or "", encoding="utf-8")
            (workdir / "entry.sh").write_text(
                _entry_script(
                    ids,
                    base_commit=str(instance.get("base_commit") or ""),
                    test_files=_test_patch_files(instance.get("test_patch") or ""),
                ),
                encoding="utf-8",
            )
            cmd = self._container_cmd(image_ref, workdir)
            logger.info("Evaluating %s via %s (SWE-bench-Live)", instance_id, self.engine)
            output = run_container_command(cmd, timeout)

        # A timed-out run or an unapplied test patch is never "resolved".
        applied = (
            APPLY_PATCH_PASS in output
            and APPLY_PATCH_FAIL not in output
            and TEST_PATCH_FAIL not in output
            and TESTS_TIMEOUT not in output
        )
        return grade_pytest_log(
            instance,
            output,
            applied,
            evaluation_scope=evaluation_scope,
            selected_f2p_ids=selected_f2p_ids,
        )


class SWEBenchLiveRunner(SWEBenchRunner):
    """SWE-bench-Live: the ``--dataset`` variant selects the frozen live split."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "swebench_live"

    def _dataset_split(self, split: str) -> tuple[str, str]:
        variant = self.dataset if self.dataset in LIVE_SPLITS else "full"
        return "swebench_live", variant

    def _container_backend(self) -> Any:
        if self._backend_attempted:
            return self._backend
        self._backend_attempted = True
        if self.experiment.test_backend == "host":
            return None
        engine = self.experiment.test_backend
        if engine == "auto":
            engine = "docker" if shutil.which("docker") else "apptainer"
        self._backend = LiveTestBackend(engine=engine)
        return self._backend

    def evaluate_predictions(self, predictions_path, split="test", run_id=None,
                             max_workers=2, cleanup_images=True, official_harness=None):
        # Live repos are not in swebench's registry; grade only with LiveTestBackend.
        return super().evaluate_predictions(
            predictions_path, split=split, run_id=run_id, max_workers=max_workers,
            cleanup_images=cleanup_images, official_harness=False,
        )


def create_swebench_live_runner(dataset: str = "full", **kwargs) -> SWEBenchLiveRunner:
    """Create a SWE-bench-Live runner (dataset in lite|full|verified|test)."""
    return SWEBenchLiveRunner(dataset=dataset, **kwargs)
