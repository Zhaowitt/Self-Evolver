"""Shared base for benchmark runners.

Concrete runners (SWE-bench, SWE-bench-Live, SWE-bench Pro) generate SWE-bench
predictions and grade them with official-semantics container evaluation. This
base only holds the pieces every runner shares: an output directory and a
namespaced logger.
"""

import logging
from pathlib import Path
from typing import Optional


class BenchmarkRunner:
    """Common state for benchmark runners."""

    def __init__(self, name: str, output_dir: Optional[Path] = None):
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else Path("./benchmark_results")
        self.logger = logging.getLogger(f"{__name__}.{name}")
