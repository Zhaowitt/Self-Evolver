"""Simple retrieval over normalized hard-case records."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.memory.hard_case_buffer import HardCaseBuffer, HardCaseRecord


class MemoryRetriever:
    """Retrieve recent hard cases, newest first, filtered by repo and stage."""

    def __init__(self, buffer_path: Path):
        self.buffer = HardCaseBuffer(buffer_path)

    def retrieve(
        self,
        repo_name: Optional[str] = None,
        stage: str = "",
        limit: int = 5,
    ) -> List[HardCaseRecord]:
        matches: List[HardCaseRecord] = []
        for record in reversed(self.buffer.read()):
            if repo_name and record.repo_name != repo_name:
                continue
            if stage and record.stage != stage:
                continue
            matches.append(record)
            if len(matches) >= limit:
                break
        return matches
