"""Controller prompt construction for grounded guidance generation."""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Optional

from src.environment.models import Issue


CONTROLLER_SYSTEM_PROMPT = """You are a Controller for a repository-level code repair worker.

ROLE BOUNDARIES:
- You do not write patches.
- You do not invent a new repository-level task.
- You do not rewrite the benchmark issue.
- You do not create, update, or deprecate skills; skill evolution is handled elsewhere.
- You emit grounded natural-language control signals that help the existing worker solve the given issue.

OUTPUT FORMAT:
Return one raw JSON object only. Do not wrap it in Markdown fences. Do not add commentary before or after it.

Exact JSON schema:
{
  "schema_version": "controller_signal_v1",
  "mode": "train|eval",
  "task_wrapper": "string or null",
  "selected_skill_ids": ["skill_id", "up to 2 ids"],
  "skill": {
    "id": "primary_skill_id",
    "title": "Primary Skill Title",
    "summary": "One-sentence skill summary",
    "target_failure_type": "failure_type"
  },
  "skills": [
    {
      "id": "skill_id",
      "title": "Skill Title",
      "summary": "One-sentence skill summary",
      "target_failure_type": "failure_type"
    }
  ],
  "strategy": "Concrete process guidance for the worker.",
  "memory_query": "Short search query for related hard cases.",
  "target_failure_type": "localization_error|patch_generation_error|patch_application_error|test_failure|regression_introduced|unknown|general",
  "difficulty": "easy|medium|hard",
  "budget": 3
}

VALIDATION RULES:
- In train mode, task_wrapper may add process constraints such as inspect failing tests first.
- In eval mode, task_wrapper must be null.
- selected_skill_ids must contain at most 2 ids.
- skills must describe selected skills only; do not include long Markdown skill bodies there.
- budget is the integer maximum number of repair iterations to allow, from 1 up to the configured cap; spend fewer on easy tasks and more on hard ones.
- Do not propose broad refactoring, unrelated edits, or test-only hacks.

FORBIDDEN OUTPUT:
```json
{"task_wrapper": "Solve a new bug unrelated to the issue"}
```
This is forbidden because it invents a new task and uses Markdown fences.

FEW-SHOT TRAIN EXAMPLE:
{
  "schema_version": "controller_signal_v1",
  "mode": "train",
  "task_wrapper": "Before patching, inspect the focused failing test or failure evidence and state the expected behavior.",
  "selected_skill_ids": ["inspect_before_editing", "failure_localization"],
  "skill": {
    "id": "inspect_before_editing",
    "title": "Inspect Before Editing",
    "summary": "Inspect failure evidence and relevant files before editing.",
    "target_failure_type": "general"
  },
  "skills": [
    {
      "id": "inspect_before_editing",
      "title": "Inspect Before Editing",
      "summary": "Inspect failure evidence and relevant files before editing.",
      "target_failure_type": "general"
    },
    {
      "id": "failure_localization",
      "title": "Failure Localization",
      "summary": "Trace the failure to the smallest plausible root-cause region.",
      "target_failure_type": "localization_error"
    }
  ],
  "strategy": "Use the failing test or traceback to localize the root cause, then generate the smallest patch.",
  "memory_query": "test_failure localization focused failing test minimal patch",
  "target_failure_type": "test_failure",
  "difficulty": "medium",
  "budget": 3
}

FEW-SHOT EVAL EXAMPLE:
{
  "schema_version": "controller_signal_v1",
  "mode": "eval",
  "task_wrapper": null,
  "selected_skill_ids": ["existing_pattern_alignment"],
  "skill": {
    "id": "existing_pattern_alignment",
    "title": "Existing Pattern Alignment",
    "summary": "Match the repository's existing implementation and test patterns.",
    "target_failure_type": "regression_introduced"
  },
  "skills": [
    {
      "id": "existing_pattern_alignment",
      "title": "Existing Pattern Alignment",
      "summary": "Match the repository's existing implementation and test patterns.",
      "target_failure_type": "regression_introduced"
    }
  ],
  "strategy": "Do not change the task. Inspect nearby patterns and keep the patch compatible with existing behavior.",
  "memory_query": "regression existing pattern minimal patch",
  "target_failure_type": "regression_introduced",
  "difficulty": "hard",
  "budget": 2
}
"""


class ControllerPromptBuilder:
    """Build system/user prompts for the controller model."""

    @property
    def system_prompt(self) -> str:
        return CONTROLLER_SYSTEM_PROMPT

    def build_user_prompt(
        self,
        issue: Issue,
        mode: str = "train",
        skills: Optional[Iterable[Mapping[str, object]]] = None,
        hard_cases: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> str:
        parts = [
            f"## Mode\n{mode}",
            "## Benchmark Issue",
            issue.description[:5000],
            "## Instance Metadata",
            json.dumps(
                {
                    "id": issue.id,
                    "repo_name": issue.repo_name,
                    "base_commit": issue.base_commit,
                    "hints_present": bool(issue.hints),
                    "fail_to_pass": issue.metadata.get("fail_to_pass"),
                    "pass_to_pass": issue.metadata.get("pass_to_pass"),
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]

        if issue.hints:
            parts.extend(["## Hints", issue.hints[:1500]])

        skill_items = list(skills or [])
        if skill_items:
            parts.append("## Candidate Skills")
            for skill in skill_items[:5]:
                parts.append(
                    json.dumps(
                        {
                            "id": skill.get("id", ""),
                            "title": skill.get("title", ""),
                            "summary": skill.get("summary", ""),
                            "target_failure_type": skill.get("target_failure_type", ""),
                            "usage_count": skill.get("usage_count", 0),
                            "average_reward": skill.get("average_reward", 0.0),
                            "status": skill.get("status", "active"),
                        },
                        ensure_ascii=False,
                    )
                )

        hard_case_items = list(hard_cases or [])
        if hard_case_items:
            parts.append("## Similar Hard Cases")
            for hard_case in hard_case_items[:3]:
                parts.append(
                    json.dumps(
                        {
                            "issue_id": hard_case.get("issue_id", ""),
                            "repo_name": hard_case.get("repo_name", ""),
                            "failure_type": hard_case.get("failure_type", ""),
                            "routes": hard_case.get("routes", []),
                            "reason": hard_case.get("reason", ""),
                        },
                        ensure_ascii=False,
                    )
                )

        parts.append("## Output")
        parts.append("Return the ControllerSignal JSON only.")
        return "\n\n".join(parts)
