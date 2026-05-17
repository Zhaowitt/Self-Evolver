"""Persistent Markdown skill store with metadata and atomic writes."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.skills.skill_bank import SkillMetadata
from src.skills.skill_stats import SkillStats


class SkillStore:
    """Load, update, archive, and persist skills in the skills directory."""

    def __init__(self, skills_dir: Optional[Path] = None, metadata_path: Optional[Path] = None):
        repo_root = Path(__file__).resolve().parents[2]
        self.skills_dir = Path(skills_dir or repo_root / "skills")
        self.metadata_path = Path(metadata_path or self.skills_dir / "metadata.json")
        self.archive_dir = self.skills_dir / "_archive"

    def load_metadata(self) -> Dict[str, SkillStats]:
        if not self.metadata_path.exists():
            return {}
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if isinstance(data, list):
            return {
                str(item.get("id", "")): SkillStats.from_dict(item)
                for item in data
                if item.get("id")
            }
        if isinstance(data, dict):
            raw_skills = data.get("skills", data)
            return {
                str(skill_id): SkillStats.from_dict({"id": skill_id, **stats})
                for skill_id, stats in raw_skills.items()
                if isinstance(stats, dict)
            }
        return {}

    def save_metadata(self, stats: Dict[str, SkillStats]) -> None:
        payload = {
            "schema_version": "skill_metadata_v1",
            "updated_at": datetime.now().isoformat(),
            "skills": {
                skill_id: item.to_dict()
                for skill_id, item in sorted(stats.items())
            },
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.metadata_path, json.dumps(payload, indent=2, ensure_ascii=False))

    def load_skills(self) -> List[SkillMetadata]:
        from src.skills.skill_bank import parse_skill_file

        stats = self.load_metadata()
        skills: List[SkillMetadata] = []
        if not self.skills_dir.exists():
            return []
        for path in sorted(self.skills_dir.glob("*.md")):
            skill = parse_skill_file(path)
            stat = stats.get(skill.id)
            if stat:
                skill.usage_count = stat.usage_count
                skill.average_reward = stat.average_reward
                skill.status = stat.status
                skill.content_hash = skill.content_hash or stat.content_hash
                skill.source = stat.source
                skill.revision = stat.revision
                skill.last_reward = stat.last_reward
                skill.last_updated_at = stat.last_updated_at
            skills.append(skill)
        return skills

    def get_skill_path(self, skill_id: str) -> Path:
        return self.skills_dir / f"{slug(skill_id)}.md"

    def update_skill_stats(self, skill_ids: Iterable[str], reward: float) -> Dict[str, SkillStats]:
        stats = self.load_metadata()
        current_hashes = {skill.id: skill.content_hash for skill in self.load_skills()}
        for skill_id in skill_ids:
            normalized_id = slug(skill_id)
            if not normalized_id:
                continue
            item = stats.get(normalized_id) or SkillStats(id=normalized_id)
            item.content_hash = current_hashes.get(normalized_id, item.content_hash)
            item.record_reward(reward)
            stats[normalized_id] = item
        self.save_metadata(stats)
        return stats

    def write_skill(
        self,
        skill_id: str,
        content: str,
        source: str = "controller",
        archive_existing: bool = True,
    ) -> SkillStats:
        normalized_id = slug(skill_id)
        path = self.get_skill_path(normalized_id)
        stats = self.load_metadata()
        existing = stats.get(normalized_id) or SkillStats(id=normalized_id, source=source)

        if archive_existing and path.exists():
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            archive_name = f"{normalized_id}_rev{existing.revision}_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
            shutil.copy2(path, self.archive_dir / archive_name)
            existing.record_event("archived_before_update", archive=archive_name)

        normalized_content = normalize_skill_markdown(content, normalized_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, normalized_content)

        existing.status = "active"
        existing.source = source
        existing.revision += 1
        existing.content_hash = content_hash(normalized_content)
        existing.record_event("skill_written", path=str(path))
        stats[normalized_id] = existing
        self.save_metadata(stats)
        return existing

    def deprecate_skill(self, skill_id: str, reason: str = "") -> Optional[SkillStats]:
        normalized_id = slug(skill_id)
        stats = self.load_metadata()
        item = stats.get(normalized_id) or SkillStats(id=normalized_id)
        item.status = "deprecated"
        item.record_event("deprecated", reason=reason)
        stats[normalized_id] = item
        self.save_metadata(stats)
        return item

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)


def normalize_skill_markdown(content: str, skill_id: str) -> str:
    text = content.strip()
    if not text.startswith("# "):
        title = skill_id.replace("_", " ").title()
        text = f"# {title}\n\n{text}"
    return text + "\n"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def content_hash(content: str) -> str:
    import hashlib

    normalized = "\n".join(line.strip() for line in content.lower().splitlines() if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
