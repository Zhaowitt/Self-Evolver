"""Load Markdown repair skills and expose compact metadata with a self-invalidating cache."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.controller.schema import SkillSignal
from src.skills.failure_types import infer_skill_failure_type
from src.skills.textnorm import content_hash, slug


@dataclass
class SkillMetadata:
    id: str
    title: str
    summary: str
    content: str
    file_path: str
    target_failure_type: str = "general"
    usage_count: int = 0
    average_reward: float = 0.0
    status: str = "active"
    content_hash: str = ""
    source: str = "seed"
    revision: int = 0
    last_reward: float = 0.0
    last_updated_at: str = ""

    def to_dict(self, include_content: bool = False) -> Dict[str, object]:
        payload = asdict(self)
        if not include_content:
            payload.pop("content", None)
        return payload

    def to_skill_signal(self) -> SkillSignal:
        return SkillSignal(
            id=self.id,
            title=self.title,
            summary=self.summary,
            target_failure_type=self.target_failure_type,
        )


def default_skills_dir() -> Path:
    """$SELF_EVOLVER_SKILLS_DIR, else the read-only seed bank <repo>/skills."""
    configured = os.getenv("SELF_EVOLVER_SKILLS_DIR", "").strip()
    if not configured:
        return Path(__file__).resolve().parents[2] / "skills"
    path = Path(configured).expanduser()
    if not path.is_dir():
        # A mistyped bank path would otherwise train silently with zero candidates.
        raise FileNotFoundError(f"SELF_EVOLVER_SKILLS_DIR={path} does not exist (see use_skill_bank)")
    return path


class SkillBank:
    """Read Markdown skills, re-loading whenever the skills directory changes on disk."""

    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = Path(skills_dir or default_skills_dir())
        self._skills: Optional[List[SkillMetadata]] = None
        self._signature: Optional[tuple] = None

    def load(self) -> List[SkillMetadata]:
        signature = self._dir_signature()
        if self._skills is None or signature != self._signature:
            from src.skills.skill_store import SkillStore

            self._skills = SkillStore(self.skills_dir).load_skills()
            self._signature = signature
        return list(self._skills)

    def active(self) -> List[SkillMetadata]:
        return [skill for skill in self.load() if skill.status == "active"]

    def get(self, skill_id: str) -> Optional[SkillMetadata]:
        for skill in self.load():
            if skill.id == skill_id:
                return skill
        return None

    def _dir_signature(self) -> Optional[tuple]:
        if not self.skills_dir.exists():
            return None
        paths = sorted(self.skills_dir.glob("*.md"))
        metadata_path = self.skills_dir / "metadata.json"
        if metadata_path.exists():
            paths.append(metadata_path)
        entries = tuple(
            (path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in paths
        )
        return (self.skills_dir.stat().st_mtime_ns, entries)


def parse_skill_file(path: Path) -> SkillMetadata:
    return parse_skill_content(path, path.read_text(encoding="utf-8"))


def parse_skill_content(path: Path, content: str) -> SkillMetadata:
    title = _extract_title(content) or path.stem.replace("_", " ").title()
    summary = _extract_description(content) or title
    skill_id = slug(path.stem)
    return SkillMetadata(
        id=skill_id,
        title=title,
        summary=summary,
        content=content,
        file_path=str(path),
        target_failure_type=infer_skill_failure_type(skill_id, content),
        content_hash=content_hash(content),
    )


def extract_procedure(content: str) -> str:
    match = re.search(r"^## How to Apply\s*(.*?)(?:\n## |\Z)", content, re.DOTALL | re.MULTILINE)
    if not match:
        return ""
    lines = [line.rstrip() for line in match.group(1).splitlines() if line.strip()]
    return "\n".join(lines)


def extract_constraints(content: str) -> str:
    match = re.search(r"^## Constraints\s*(.*?)(?:\n## |\nAction type:|\Z)", content, re.DOTALL | re.MULTILINE)
    if not match:
        return ""
    lines = [line.rstrip() for line in match.group(1).splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_title(content: str) -> str:
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_description(content: str) -> str:
    match = re.search(r"^## Description\s*(.*?)(?:\n## |\Z)", content, re.DOTALL | re.MULTILINE)
    if not match:
        return ""
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    return " ".join(lines)[:500]
