"""Shared base for benchmark runners."""

import logging
from pathlib import Path
from typing import Optional


class BenchmarkRunner:
    def __init__(self, name: str, output_dir: Optional[Path] = None):
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else Path("./benchmark_results")
        self.logger = logging.getLogger(f"{__name__}.{name}")
