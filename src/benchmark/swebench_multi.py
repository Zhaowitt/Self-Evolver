"""Multi-SWE-bench runner.

Multi-SWE-bench (ByteDance-Seed/Multi-SWE-bench) is 1632 issue-resolving tasks
across 7 non-Python languages (C, C++, Go, Java, JavaScript, Rust, TypeScript).
Patch generation is benchmark-agnostic — the same Inspector/PatchGenerator loop
runs, with the Inspector listing source files by the instance's language — so it
produces a predictions file like any other benchmark. Grading, however, needs
Multi-SWE-bench's own Docker harness (``multi_swe_bench.harness.run_evaluation``)
with per-language test runners; this framework does not reimplement that grader,
so eval writes the harness-ready patch file and then fails loudly with the exact
command to run rather than emit unverified numbers.

``--dataset full`` selects all 1632 instances; ``--dataset flash`` selects the
balanced 300-instance subset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.benchmark import datasets
from src.benchmark.swebench_runner import SWEBenchRunner

logger = logging.getLogger(__name__)

MULTI_OFFICIAL_REPO = "https://github.com/multi-swe-bench/multi-swe-bench"


class MultiSWEEvaluationUnavailable(RuntimeError):
    """Raised when Multi-SWE-bench grading is requested instead of the official harness."""


class MultiSWEBenchRunner(SWEBenchRunner):
    """Generate Multi-SWE-bench predictions; delegate grading to the official harness."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "multi_swe_bench"

    def _dataset_split(self, split: str) -> tuple[str, str]:
        key = "multi_swe_bench_flash" if self.dataset == "flash" else "multi_swe_bench"
        return key, "test"

    def _container_backend(self) -> Any:
        # Multi-language tests run in Multi-SWE-bench's own per-language images;
        # there is no in-loop container grading here. Generation still runs.
        return None

    def _write_harness_patch_file(self, predictions: Dict[str, dict]) -> Path:
        """Write the ``{org, repo, number, fix_patch}`` JSONL the official harness reads."""
        key, split = self._dataset_split("test")
        meta = {
            row["instance_id"]: (row["org"], row["repo"], row["number"])
            for row in datasets.iter_rows(key, split)
        }
        out_path = self.run_dir / "multi_swe_bench_patches.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for instance_id, prediction in predictions.items():
                if instance_id not in meta:
                    continue
                org, repo, number = meta[instance_id]
                handle.write(json.dumps({
                    "org": org, "repo": repo, "number": number,
                    "fix_patch": prediction.get("model_patch", ""),
                }) + "\n")
        return out_path

    def evaluate_predictions(self, predictions_path: Path, *args, **kwargs) -> dict:
        predictions = self._load_predictions(Path(predictions_path))
        harness_file = self._write_harness_patch_file(predictions)
        raise MultiSWEEvaluationUnavailable(
            "Multi-SWE-bench grades multi-language tests with its own Docker harness "
            "and is not reimplemented here (emitting unverified results would be "
            "dishonest).\n"
            f"  1. Harness-ready patch file: {harness_file}\n"
            f"  2. Download the per-instance images: bash scripts/download_images.sh "
            f"(from {MULTI_OFFICIAL_REPO}).\n"
            "  3. Grade with the official harness (needs a Docker engine):\n"
            "     python -m multi_swe_bench.harness.run_evaluation --config <config.json>\n"
            f"     with \"patch_files\": [\"{harness_file}\"] in the config."
        )


def create_multi_swe_bench_runner(dataset: str = "full", **kwargs) -> MultiSWEBenchRunner:
    """Create a Multi-SWE-bench runner (generation only; eval via the official harness)."""
    return MultiSWEBenchRunner(dataset=dataset, **kwargs)
