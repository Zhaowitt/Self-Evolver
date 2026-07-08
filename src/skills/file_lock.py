"""Advisory fcntl file locking for shared JSON/JSONL state files."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock guarding read-modify-write cycles on `path`.

    A sidecar `<name>.lock` file is used so the guarded file can be atomically
    replaced (tmp + os.replace) without invalidating the locked descriptor.
    Do not nest acquisitions of the same lock: flock blocks between file
    descriptors even within one process.
    """
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
