"""Controller prompt construction for grounded guidance generation."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Mapping, Optional

from src.environment.models import Issue

# EasyR1 sends the dataset prompt as one user turn; SFT and served eval must
# render the same single turn or train and eval inputs diverge.
CONTROLLER_PROMPT_FORMAT = "single_user_turn_v1"
MAX_CANDIDATE_SKILLS = 6


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
  "selected_skill_ids": ["<candidate_id_1>", "<candidate_id_2>"],
  "strategy": "Trace the focused failure to the smallest causal region before generating a minimal patch."
}
"""


class ControllerPromptBuilder:
    """Build system/user prompts for the controller model."""

    @property
    def system_prompt(self) -> str:
        return CONTROLLER_SYSTEM_PROMPT

    def build_prompt_text(
        self,
        issue: Issue,
        mode: str = "train",
        skills: Optional[Iterable[Mapping[str, object]]] = None,
        hard_cases: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> str:
        """Render the canonical single-turn Controller prompt (GRPO/SFT/eval)."""
        user_prompt = self.build_user_prompt(
            issue, mode=mode, skills=skills, hard_cases=hard_cases
        )
        return f"{CONTROLLER_SYSTEM_PROMPT.strip()}\n\n{user_prompt}"

    def build_messages(
        self,
        issue: Issue,
        mode: str = "train",
        skills: Optional[Iterable[Mapping[str, object]]] = None,
        hard_cases: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> List[Dict[str, str]]:
        """Chat messages equal to what EasyR1 builds from ``build_prompt_text``."""
        return [
            {
                "role": "user",
                "content": self.build_prompt_text(
                    issue, mode=mode, skills=skills, hard_cases=hard_cases
                ),
            }
        ]

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
                    # Test names are grading labels: exposing them leaks eval
                    # tests and can exceed the prompt length limit.
                    "fail_to_pass_count": len(_test_ids(issue.metadata.get("fail_to_pass"))),
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]

        if issue.hints:
            parts.extend(["## Hints", issue.hints[:1500]])

        skill_items = order_candidate_skills(issue.id, list(skills or [])[:MAX_CANDIDATE_SKILLS])
        if skill_items:
            parts.append("## Candidate Skills")
            for skill in skill_items:
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


def order_candidate_skills(
    instance_id: str,
    skills: List[Mapping[str, object]],
) -> List[Mapping[str, object]]:
    """Stable per-instance order, so the policy cannot learn "pick the top-ranked two"."""
    def key(skill: Mapping[str, object]) -> str:
        seed = f"{instance_id}\0{skill.get('id', '')}".encode("utf-8")
        return hashlib.sha256(seed).hexdigest()

    return sorted(skills, key=key)


def _test_ids(raw: object) -> List[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return [line for line in raw.splitlines() if line.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item).strip()]
    return []
