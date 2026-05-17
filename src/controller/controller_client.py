"""Controller clients for mock/template/vLLM-backed guidance generation."""

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
    """Generate ControllerSignal objects from mock, template, or LLM modes."""

    def __init__(
        self,
        mode: str = "mock",
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
        if self.mode == "mock":
            return self._mock_signal(issue, stage=stage, skill=skill, skills=skills)
        if self.mode == "template":
            return self._template_signal(issue, stage=stage, skill=skill, skills=skills)
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

    def _mock_signal(
        self,
        issue: Issue,
        stage: str,
        skill: Optional[SkillMetadata | SkillSignal],
        skills: Optional[Sequence[SkillMetadata | SkillSignal]] = None,
    ) -> ControllerSignal:
        selected_skills = _skill_list(skill=skill, skills=skills)
        if not selected_skills:
            selected_skills = [
                SkillSignal(
                    id="inspect_before_editing",
                    title="Inspect Before Editing",
                    summary="Inspect failure evidence and relevant files before editing.",
                    target_failure_type="general",
                )
            ]
        selected_skill = selected_skills[0]
        task_wrapper = (
            "Ask the worker to inspect the failing test or failure evidence before generating a patch."
            if stage == "train"
            else None
        )
        signal = ControllerSignal(
            mode=stage,
            task_wrapper=task_wrapper,
            skill=selected_skill,
            skills=selected_skills[:2],
            selected_skill_ids=[item.id for item in selected_skills[:2] if item.id],
            strategy="Inspect the focused failure first, then generate the smallest patch that addresses the observed behavior.",
            memory_query=f"{issue.repo_name or issue.id} minimal patch failure evidence",
            target_failure_type=selected_skill.target_failure_type or "general",
            difficulty="medium",
            source="mock",
        )
        return signal.enforce_mode()

    def _template_signal(
        self,
        issue: Issue,
        stage: str,
        skill: Optional[SkillMetadata | SkillSignal],
        skills: Optional[Sequence[SkillMetadata | SkillSignal]] = None,
    ) -> ControllerSignal:
        signal = self._mock_signal(issue, stage=stage, skill=skill, skills=skills)
        tests = issue.metadata.get("fail_to_pass")
        if tests:
            signal.strategy = (
                "Inspect the listed FAIL_TO_PASS test target before editing, then keep the fix scoped to the failing behavior."
            )
            if stage == "train":
                signal.task_wrapper = "Require the worker to inspect the focused failing test before patching."
        signal.source = "template"
        return signal.enforce_mode()

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
