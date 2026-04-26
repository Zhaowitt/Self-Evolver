"""Compatibility wrapper for the unified SWE-bench CLI.

Prefer:
    python -m src.main benchmark --phase both

This wrapper preserves the old script entrypoint while delegating all logic to
the single maintained benchmark command.
"""

import sys

from src.main import cli


if __name__ == "__main__":
    cli(args=["benchmark", *sys.argv[1:]], obj={})
