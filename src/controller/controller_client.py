"""Controller clients for vLLM/OpenAI-compatible guidance generation."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from src.config import ControllerConfig, LLMConfig, get_config
from src.controller.parser import parse_controller_response
from src.controller.prompt_builder import ControllerPromptBuilder
from src.controller.schema import ControllerSignal, SkillSignal
from src.environment.models import Issue
from src.llm.client import LLMClient
from src.skills.skill_bank import SkillMetadata


class ControllerClient:
    """Generate ControllerSignal objects from an LLM or return an empty signal."""

    def __init__(
        self,
        mode: str = "llm",
        config: Optional[ControllerConfig] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_builder: Optional[ControllerPromptBuilder] = None,
    ):
        self.mode = mode
        self.config = config or get_config().controller
        self.prompt_builder = prompt_builder or ControllerPromptBuilder()
        self.llm_client = llm_client

    def generate(
        self,
        issue: Issue,
        stage: str = "train",
        skill: Optional[SkillMetadata | SkillSignal] = None,
        skills: Optional[Sequence[SkillMetadata | SkillSignal]] = None,
        hard_cases: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> ControllerSignal:
        if self.mode == "off":
            return ControllerSignal.empty(mode=stage, source="off")
        if self.mode == "llm":
            return self._llm_signal(
                issue,
                stage=stage,
                skill=skill,
                skills=skills,
                hard_cases=hard_cases,
            )
        return ControllerSignal.empty(
            mode=stage,
            source=self.mode,
            parse_error=f"unknown controller mode: {self.mode}",
        )

    def _llm_signal(
        self,
        issue: Issue,
        stage: str,
        skill: Optional[SkillMetadata | SkillSignal],
        skills: Optional[Sequence[SkillMetadata | SkillSignal]],
        hard_cases: Optional[Iterable[Mapping[str, object]]],
    ) -> ControllerSignal:
        if self.llm_client is None:
            llm_config = LLMConfig(
                api_key=self.config.api_key,
                model=self.config.model,
                base_url=self.config.base_url,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            self.llm_client = LLMClient(llm_config)

        skill_payloads = [
            skill_signal.to_dict()
            for skill_signal in _skill_list(skill=skill, skills=skills)
        ]

        user_prompt = self.prompt_builder.build_user_prompt(
            issue,
            mode=stage,
            skills=skill_payloads,
            hard_cases=hard_cases,
        )
        try:
            response = self.llm_client.chat_with_system(
                self.prompt_builder.system_prompt,
                user_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.llm_client.chat_with_system(
                self.prompt_builder.system_prompt,
                user_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        return parse_controller_response(response.content, mode=stage, source="llm")


def _skill_to_signal(value: Optional[SkillMetadata | SkillSignal]) -> Optional[SkillSignal]:
    if value is None:
        return None
    if isinstance(value, SkillSignal):
        return value
    if hasattr(value, "to_skill_signal"):
        return value.to_skill_signal()
    return None


def _skill_list(
    skill: Optional[SkillMetadata | SkillSignal] = None,
    skills: Optional[Sequence[SkillMetadata | SkillSignal]] = None,
) -> list[SkillSignal]:
    selected: list[SkillSignal] = []
    seen: set[str] = set()
    for value in [*(skills or []), *([skill] if skill else [])]:
        signal = _skill_to_signal(value)
        if not signal:
            continue
        key = signal.id or signal.title
        if key in seen:
            continue
        seen.add(key)
        selected.append(signal)
        if len(selected) >= 2:
            break
    return selected
