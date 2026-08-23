"""Advisory fcntl file locking for shared JSON/JSONL state files."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # POSIX training hosts
    import fcntl  # type: ignore
except ImportError:  # Windows development/test hosts
    fcntl = None
    import msvcrt  # type: ignore


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
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        else:
            handle.write("0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
