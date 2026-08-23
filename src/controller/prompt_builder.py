"""Controller prompt construction for grounded guidance generation."""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Optional

from src.environment.models import Issue


CONTROLLER_SYSTEM_PROMPT = """You are a Controller for a repository-level code repair worker.

ROLE BOUNDARIES:
- You do not write patches, rewrite the issue, allocate iteration budget, or diagnose the eventual failure.
- You do not create, update, or deprecate skills; skill evolution is handled after execution.
- You select at most two IDs from the provided Candidate Skills and may add one short grounded strategy.

Return one raw JSON object only, with no Markdown fence or commentary:
{
  "selected_skill_ids": ["up_to_two_candidate_skill_ids"],
  "strategy": "optional short process guidance"
}

RULES:
- Every selected ID must appear in Candidate Skills.
- Select no more than two skills. An empty list is allowed when no candidate is useful.
- Keep strategy concise, concrete, and subordinate to the original issue.
- Do not output task_wrapper, memory_query, target_failure_type, difficulty, budget, skill metadata, or skill updates.
- Do not propose broad refactoring, unrelated edits, or test-only hacks.

EXAMPLE:
{
  "selected_skill_ids": ["inspect_before_editing", "failure_localization"],
  "strategy": "Trace the focused failure to the smallest causal region before generating a minimal patch."
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
            for skill in skill_items[:6]:
                parts.append(
                    json.dumps(
                        {
                            "id": skill.get("id", ""),
                            "title": skill.get("title", ""),
                            "summary": skill.get("summary", ""),
                            "applicable_capability_label": skill.get(
                                "applicable_capability_label",
                                skill.get("target_failure_type", ""),
                            ),
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
