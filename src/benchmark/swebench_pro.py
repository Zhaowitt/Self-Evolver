"""SWE-bench Pro runner.

SWE-bench Pro (scaleapi/SWE-bench_Pro-os) is a harder, multi-language OOD set.
Patch generation is benchmark-agnostic, so it runs through the standard rollout
and produces a predictions file. Grading, however, uses Pro's own multi-language
harness (``swe_bench_pro_eval.py``) with per-instance images at
``jefzda/sweap-images:<dockerhub_tag>`` and non-pytest test identifiers; this
framework does not reimplement that grader, so eval fails loudly with the exact
command to run rather than emit unverified numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.benchmark.swebench_runner import SWEBenchRunner

logger = logging.getLogger(__name__)

PRO_IMAGE_NAMESPACE = "jefzda/sweap-images"
PRO_OFFICIAL_REPO = "https://github.com/scaleapi/SWE-bench_Pro-os"


class ProEvaluationUnavailable(RuntimeError):
    """Raised when Pro grading is requested from this framework instead of the official harness."""


def pro_image_ref(dockerhub_tag: str) -> str:
    """Per-instance image reference: ``jefzda/sweap-images:<dockerhub_tag>``."""
    return f"{PRO_IMAGE_NAMESPACE}:{dockerhub_tag}"


class SWEBenchProRunner(SWEBenchRunner):
    """Generate SWE-bench Pro predictions; delegate grading to the official harness."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "swebench_pro"

    def _dataset_split(self, split: str) -> tuple[str, str]:
        return "swebench_pro", "test"

    def _container_backend(self) -> Any:
        # Pro tests are multi-language with custom identifiers; no in-loop
        # container grading. Generation still runs; grading is external.
        return None

    def evaluate_predictions(self, predictions_path: Path, *args, **kwargs) -> dict:
        raise ProEvaluationUnavailable(
            "SWE-bench Pro grades multi-language tests with its own harness and is not "
            "reimplemented here (emitting unverified results would be dishonest).\n"
            f"  1. Predictions: {predictions_path}\n"
            f"  2. Per-instance images: {PRO_IMAGE_NAMESPACE}:<dockerhub_tag> "
            "(the 'dockerhub_tag' field of each instance).\n"
            f"  3. Grade with the official harness: {PRO_OFFICIAL_REPO} "
            "(swe_bench_pro_eval.py), passing the predictions file above."
        )


def create_swebench_pro_runner(dataset: str = "test", **kwargs) -> SWEBenchProRunner:
    """Create a SWE-bench Pro runner (generation only; eval via the official harness)."""
    return SWEBenchProRunner(dataset=dataset, **kwargs)
