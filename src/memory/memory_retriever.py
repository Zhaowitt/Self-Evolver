"""Simple retrieval over normalized hard-case records."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.memory.hard_case_buffer import HardCaseBuffer, HardCaseRecord


class MemoryRetriever:
    """Retrieve hard cases by failure type, repo, or judge route."""

    def __init__(self, buffer_path: Path):
        self.buffer = HardCaseBuffer(buffer_path)

    def retrieve(
        self,
        failure_type: str = "",
        repo_name: Optional[str] = None,
        route: str = "",
        limit: int = 5,
    ) -> List[HardCaseRecord]:
        matches: List[HardCaseRecord] = []
        for record in reversed(self.buffer.read()):
            if failure_type and record.failure_type != failure_type:
                continue
            if repo_name and record.repo_name != repo_name:
                continue
            if route and route not in record.routes:
                continue
            matches.append(record)
            if len(matches) >= limit:
                break
        return matches
