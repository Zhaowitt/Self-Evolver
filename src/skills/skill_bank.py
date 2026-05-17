"""Load seed repair skills and expose compact metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.controller.schema import SkillSignal


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


class SkillBank:
    """Read root-level Markdown seed skills."""

    def __init__(self, skills_dir: Optional[Path] = None):
        repo_root = Path(__file__).resolve().parents[2]
        self.skills_dir = skills_dir or repo_root / "skills"
        self._skills: Optional[List[SkillMetadata]] = None

    def load(self) -> List[SkillMetadata]:
        if self._skills is not None:
            return list(self._skills)
        from src.skills.skill_store import SkillStore

        self._skills = SkillStore(self.skills_dir).load_skills()
        return list(self._skills)

    def active(self) -> List[SkillMetadata]:
        return [skill for skill in self.load() if skill.status == "active"]

    def get(self, skill_id: str) -> Optional[SkillMetadata]:
        for skill in self.load():
            if skill.id == skill_id:
                return skill
        return None

    def _parse_skill(self, path: Path, content: str) -> SkillMetadata:
        return parse_skill_content(path, content)


def parse_skill_file(path: Path) -> SkillMetadata:
    return parse_skill_content(path, path.read_text(encoding="utf-8"))


def parse_skill_content(path: Path, content: str) -> SkillMetadata:
    title = _extract_title(content) or path.stem.replace("_", " ").title()
    summary = _extract_description(content) or title
    skill_id = _slug(path.stem)
    return SkillMetadata(
        id=skill_id,
        title=title,
        summary=summary,
        content=content,
        file_path=str(path),
        target_failure_type=_infer_failure_type(skill_id, content),
        content_hash=hashlib.sha256(_normalize_content(content).encode("utf-8")).hexdigest(),
    )


def _extract_title(content: str) -> str:
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_description(content: str) -> str:
    match = re.search(r"^## Description\s*(.*?)(?:\n## |\Z)", content, re.DOTALL | re.MULTILINE)
    if not match:
        return ""
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    return " ".join(lines)[:500]


def _infer_failure_type(skill_id: str, content: str) -> str:
    text = f"{skill_id} {content}".lower()
    if "localiz" in text:
        return "localization_error"
    if "patch" in text or "repair" in text:
        return "patch_generation_error"
    if "pattern" in text or "alignment" in text:
        return "regression_introduced"
    if "inspect" in text:
        return "general"
    return "general"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalize_content(content: str) -> str:
    return "\n".join(line.strip() for line in content.lower().splitlines() if line.strip())
