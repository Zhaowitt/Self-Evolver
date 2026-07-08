"""Schema objects for controller-generated guidance signals.

Schema `controller_signal_v1`: mode train|eval, optional train-only task_wrapper,
up to MAX_SELECTED_SKILLS selected skills, strategy/memory cue, a target failure
type from the unified taxonomy, difficulty easy|medium|hard, and an integer
iteration budget clamped to [1, configured max iterations]. Skill lifecycle
proposals are NOT part of this signal; they belong to the Reflector
(src/reflection/reflector.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.skills.failure_types import normalize_failure_type
from src.skills.textnorm import slug


CONTROLLER_SCHEMA_VERSION = "controller_signal_v1"
ALLOWED_MODES = {"train", "eval"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
MAX_SELECTED_SKILLS = 2


@dataclass
class SkillSignal:
    """Compact skill reference injected into worker prompts."""

    id: str = ""
    title: str = ""
    summary: str = ""
    target_failure_type: str = ""

    @classmethod
    def from_any(cls, value: Any) -> Optional["SkillSignal"]:
        """Build a skill signal from a dict, string, object, or None."""
        if value is None:
            return None
        if isinstance(value, SkillSignal):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            return cls(id=text.lower().replace(" ", "_"), title=text, summary=text)
        if hasattr(value, "to_skill_signal"):
            return value.to_skill_signal()
        if isinstance(value, dict):
            return cls(
                id=str(value.get("id", "")),
                title=str(value.get("title", "")),
                summary=str(value.get("summary", "")),
                target_failure_type=str(value.get("target_failure_type", "")),
            )
        return None

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ControllerSignal:
    """Structured natural-language control signal from the controller."""

    schema_version: str = CONTROLLER_SCHEMA_VERSION
    mode: str = "train"
    task_wrapper: Optional[str] = None
    skill: Optional[SkillSignal] = None
    skills: List[SkillSignal] = field(default_factory=list)
    selected_skill_ids: List[str] = field(default_factory=list)
    strategy: str = ""
    memory_query: str = ""
    target_failure_type: str = ""
    difficulty: str = "medium"
    budget: Optional[int] = None
    source: str = ""
    raw_response: str = ""
    parse_error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        mode: str = "train",
        source: str = "",
        raw_response: str = "",
        parse_error: str = "",
    ) -> "ControllerSignal":
        signal = cls(
            mode=mode if mode in ALLOWED_MODES else "train",
            source=source,
            raw_response=raw_response,
            parse_error=parse_error,
        )
        return signal.enforce_mode()

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        mode: Optional[str] = None,
        source: str = "",
        raw_response: str = "",
    ) -> "ControllerSignal":
        selected_mode = mode or str(data.get("mode", "train"))
        if selected_mode not in ALLOWED_MODES:
            selected_mode = "train"

        difficulty = str(data.get("difficulty", "medium")).lower()
        if difficulty not in ALLOWED_DIFFICULTIES:
            difficulty = "medium"

        parsed_skills = _parse_skills(data)
        selected_skill_ids = _parse_selected_skill_ids(data, parsed_skills)
        primary_skill = SkillSignal.from_any(data.get("skill"))
        if primary_skill and not parsed_skills:
            parsed_skills = [primary_skill]
        if not primary_skill and parsed_skills:
            primary_skill = parsed_skills[0]

        signal = cls(
            schema_version=str(data.get("schema_version", CONTROLLER_SCHEMA_VERSION)),
            mode=selected_mode,
            task_wrapper=_clean_optional_text(data.get("task_wrapper")),
            skill=primary_skill,
            skills=parsed_skills[:MAX_SELECTED_SKILLS],
            selected_skill_ids=selected_skill_ids[:MAX_SELECTED_SKILLS],
            strategy=_clean_text(data.get("strategy")),
            memory_query=_clean_text(data.get("memory_query")),
            target_failure_type=normalize_failure_type(
                data.get("target_failure_type"), default=""
            ),
            difficulty=difficulty,
            budget=_parse_budget(data.get("budget")),
            source=source or str(data.get("source", "")),
            raw_response=raw_response,
            parse_error=str(data.get("parse_error", "")),
            metadata=dict(data.get("metadata") or {}),
        )
        return signal.enforce_mode()

    def enforce_mode(self) -> "ControllerSignal":
        """Apply mode-specific safety rules and clamp the budget to [1, cap]."""
        if self.mode not in ALLOWED_MODES:
            self.mode = "train"
        if self.difficulty not in ALLOWED_DIFFICULTIES:
            self.difficulty = "medium"
        if self.mode == "eval":
            self.task_wrapper = None
        if self.budget is not None:
            cap = max(1, get_config().agent.max_iterations)
            self.budget = max(1, min(int(self.budget), cap))
        self.skills = [skill for skill in self.skills if skill][:MAX_SELECTED_SKILLS]
        if not self.skill and self.skills:
            self.skill = self.skills[0]
        if self.skill and not self.skills:
            self.skills = [self.skill]
        self.selected_skill_ids = [
            slug(skill_id)
            for skill_id in self.selected_skill_ids
            if slug(skill_id)
        ][:MAX_SELECTED_SKILLS]
        if not self.selected_skill_ids and self.skills:
            self.selected_skill_ids = [
                skill.id for skill in self.skills if skill.id
            ][:MAX_SELECTED_SKILLS]
        return self

    @property
    def has_guidance(self) -> bool:
        return any(
            [
                self.task_wrapper,
                self.skills or (self.skill and (self.skill.title or self.skill.summary)),
                self.strategy,
                self.memory_query,
                self.target_failure_type,
            ]
        )

    def to_dict(self, include_debug: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "task_wrapper": self.task_wrapper,
            "skill": self.skill.to_dict() if self.skill else None,
            "skills": [skill.to_dict() for skill in self.skills],
            "selected_skill_ids": self.selected_skill_ids,
            "strategy": self.strategy,
            "memory_query": self.memory_query,
            "target_failure_type": self.target_failure_type,
            "difficulty": self.difficulty,
            "budget": self.budget,
            "source": self.source,
            "metadata": self.metadata,
        }
        if include_debug:
            payload["raw_response"] = self.raw_response
            payload["parse_error"] = self.parse_error
        return payload


def controller_signal_from_any(
    value: Any,
    mode: Optional[str] = None,
    source: str = "",
) -> Optional[ControllerSignal]:
    """Normalize a controller signal-like object."""
    if value is None:
        return None
    if isinstance(value, ControllerSignal):
        if mode:
            value.mode = mode
        return value.enforce_mode()
    if isinstance(value, dict):
        return ControllerSignal.from_dict(value, mode=mode, source=source)
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_optional_text(value: Any) -> Optional[str]:
    text = _clean_text(value)
    return text or None


def _parse_budget(value: Any) -> Optional[int]:
    """Parse an integer iteration budget; invalid values degrade to None (use default)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_skills(data: Dict[str, Any]) -> List[SkillSignal]:
    raw_skills = data.get("skills")
    skills: List[SkillSignal] = []
    if isinstance(raw_skills, list):
        for item in raw_skills:
            skill = SkillSignal.from_any(item)
            if skill and (skill.id or skill.title or skill.summary):
                skills.append(skill)
    return skills


def _parse_selected_skill_ids(data: Dict[str, Any], skills: List[SkillSignal]) -> List[str]:
    raw_ids = data.get("selected_skill_ids")
    selected: List[str] = []
    if isinstance(raw_ids, list):
        selected = [slug(_clean_text(item)) for item in raw_ids]
    elif isinstance(raw_ids, str):
        selected = [slug(raw_ids)]
    selected = [item for item in selected if item]
    if not selected and skills:
        selected = [skill.id for skill in skills if skill.id]
    return selected
