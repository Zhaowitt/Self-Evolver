"""Persistent Markdown skill store with metadata, fcntl locking, and atomic writes."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.skills.failure_types import explicit_failure_type, normalize_failure_type
from src.skills.file_lock import file_lock
from src.skills.skill_bank import SkillMetadata, parse_skill_file
from src.skills.skill_stats import SkillStats
from src.skills.textnorm import content_hash, slug


_ARCHIVE_REV_RE = re.compile(r"_rev\d+_\d{8,14}$")


class SkillStore:
    """Load, update, archive, and persist skills in the skills directory.

    Public mutators (`credit_skills`, `write_skill`, `deprecate_skill`,
    `retire_skills`) serialize concurrent read-modify-write cycles through an
    fcntl lock on `metadata.json`; read helpers and `save_metadata` are
    lock-free and must only be combined by those mutators.
    """

    def __init__(self, skills_dir: Optional[Path] = None, metadata_path: Optional[Path] = None):
        repo_root = Path(__file__).resolve().parents[2]
        self.skills_dir = Path(skills_dir or repo_root / "skills")
        self.metadata_path = Path(metadata_path or self.skills_dir / "metadata.json")
        self.archive_dir = self.skills_dir / "_archive"

    def _read_payload(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if isinstance(data, list):
            return {"skills": {str(item.get("id", "")): item for item in data if item.get("id")}}
        if isinstance(data, dict):
            return data
        return {}

    def load_metadata(self) -> Dict[str, SkillStats]:
        payload = self._read_payload()
        raw_skills = payload.get("skills", payload)
        if not isinstance(raw_skills, dict):
            return {}
        return {
            str(skill_id): SkillStats.from_dict({"id": skill_id, **stats})
            for skill_id, stats in raw_skills.items()
            if isinstance(stats, dict)
        }

    def load_baseline(self) -> Dict[str, float]:
        """EMA baseline over rollout utilities used for advantage-style credit."""
        baseline = self._read_payload().get("baseline")
        if not isinstance(baseline, dict):
            return {"ema": 0.0, "count": 0}
        return {
            "ema": float(baseline.get("ema", 0.0) or 0.0),
            "count": int(baseline.get("count", 0) or 0),
        }

    def save_metadata(
        self,
        stats: Dict[str, SkillStats],
        baseline: Optional[Dict[str, float]] = None,
    ) -> None:
        payload = {
            "schema_version": "skill_metadata_v1",
            "updated_at": datetime.now().isoformat(),
            "baseline": baseline if baseline is not None else self.load_baseline(),
            "skills": {
                skill_id: item.to_dict()
                for skill_id, item in sorted(stats.items())
            },
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.metadata_path, json.dumps(payload, indent=2, ensure_ascii=False))

    def load_skills(self) -> List[SkillMetadata]:
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

    def load_archived_skills(self) -> List[SkillMetadata]:
        """Parse archived skill revisions, mapping filenames back to base skill ids."""
        if not self.archive_dir.exists():
            return []
        skills: List[SkillMetadata] = []
        for path in sorted(self.archive_dir.glob("*.md")):
            skill = parse_skill_file(path)
            base_id = _ARCHIVE_REV_RE.sub("", slug(path.stem))
            skill.id = base_id or skill.id
            skill.status = "archived"
            skills.append(skill)
        return skills

    def get_skill_path(self, skill_id: str) -> Path:
        return self.skills_dir / f"{slug(skill_id)}.md"

    def credit_skills(
        self,
        skill_ids: Iterable[str],
        utility: float,
        success: Optional[bool] = None,
        ema_alpha: float = 0.3,
    ) -> Dict[str, Any]:
        """Credit skills with advantage = utility - EMA baseline, then update the baseline."""
        normalized_ids = [slug(skill_id) for skill_id in skill_ids if slug(skill_id)]
        with file_lock(self.metadata_path):
            stats = self.load_metadata()
            baseline = self.load_baseline()
            has_baseline = baseline["count"] > 0
            advantage = float(utility) - baseline["ema"] if has_baseline else 0.0

            current_hashes = {skill.id: skill.content_hash for skill in self.load_skills()}
            for skill_id in normalized_ids:
                item = stats.get(skill_id) or SkillStats(id=skill_id)
                item.content_hash = current_hashes.get(skill_id, item.content_hash)
                item.record_credit(float(utility), advantage, success=success)
                stats[skill_id] = item

            new_ema = (
                float(utility)
                if not has_baseline
                else (1.0 - ema_alpha) * baseline["ema"] + ema_alpha * float(utility)
            )
            new_baseline = {"ema": new_ema, "count": baseline["count"] + 1}
            self.save_metadata(stats, baseline=new_baseline)
        return {
            "advantage": advantage,
            "baseline_before": baseline,
            "baseline_after": new_baseline,
            "credited_skill_ids": normalized_ids,
        }

    def write_skill(
        self,
        skill_id: str,
        content: str,
        source: str = "reflector",
        archive_existing: bool = True,
        target_failure_type: Optional[str] = None,
    ) -> SkillStats:
        normalized_id = slug(skill_id)
        path = self.get_skill_path(normalized_id)
        with file_lock(self.metadata_path):
            stats = self.load_metadata()
            existing = stats.get(normalized_id) or SkillStats(id=normalized_id, source=source)

            if archive_existing and path.exists():
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                archive_name = f"{normalized_id}_rev{existing.revision}_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
                shutil.copy2(path, self.archive_dir / archive_name)
                existing.record_event("archived_before_update", archive=archive_name)

            normalized_content = normalize_skill_markdown(content, normalized_id, target_failure_type)
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

    def deprecate_skill(self, skill_id: str, reason: str = "") -> SkillStats:
        return self._set_status(skill_id, status="deprecated", event="deprecated", reason=reason)

    def retire_skills(self, skill_ids: Iterable[str], reason: str = "") -> List[SkillStats]:
        """Mark skills as retired (net-success retirement / active-cap eviction)."""
        normalized_ids = [slug(skill_id) for skill_id in skill_ids if slug(skill_id)]
        if not normalized_ids:
            return []
        retired: List[SkillStats] = []
        with file_lock(self.metadata_path):
            stats = self.load_metadata()
            for skill_id in normalized_ids:
                item = stats.get(skill_id) or SkillStats(id=skill_id)
                item.status = "retired"
                item.record_event("retired", reason=reason)
                stats[skill_id] = item
                retired.append(item)
            self.save_metadata(stats)
        return retired

    def _set_status(self, skill_id: str, status: str, event: str, reason: str = "") -> SkillStats:
        normalized_id = slug(skill_id)
        with file_lock(self.metadata_path):
            stats = self.load_metadata()
            item = stats.get(normalized_id) or SkillStats(id=normalized_id)
            item.status = status
            item.record_event(event, reason=reason)
            stats[normalized_id] = item
            self.save_metadata(stats)
        return item

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)


def normalize_skill_markdown(
    content: str,
    skill_id: str,
    target_failure_type: Optional[str] = None,
) -> str:
    """Ensure an H1 title and an explicit 'Target failure type' marker line."""
    text = content.strip()
    if not text.startswith("# "):
        title = skill_id.replace("_", " ").title()
        text = f"# {title}\n\n{text}"
    if target_failure_type and explicit_failure_type(text) is None:
        text = f"{text}\n\nTarget failure type: {normalize_failure_type(target_failure_type)}"
    return text + "\n"
