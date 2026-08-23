"""Small host-platform compatibility helpers for importing SWE-bench.

SWE-bench 4.1 imports its image-preparation module eagerly.  That module imports
the Unix-only :mod:`resource` module even when callers only need grading or test
spec helpers.  Official evaluation images are Linux and use the real module;
on Windows we provide the tiny API surface touched by that optional command so
the portable parts of the harness remain importable.
"""

from __future__ import annotations

import sys
import types


def ensure_swebench_importable() -> None:
    if sys.platform != "win32" or "resource" in sys.modules:
        return

    resource = types.ModuleType("resource")
    resource.RLIMIT_NOFILE = 7
    resource.getrlimit = lambda _kind: (-1, -1)
    resource.setrlimit = lambda _kind, _limits: None
    sys.modules["resource"] = resource

