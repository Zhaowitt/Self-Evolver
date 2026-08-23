"""Reflector: cluster hard cases, propose skills via a scripted LLM, emit task signals."""

from __future__ import annotations

import json

from src.llm.client import LLMResponse
from src.memory.hard_case_buffer import HardCaseBuffer, HardCaseRecord
from src.reflection.reflector import Reflector
from src.skills.skill_bank import SkillBank
from src.skills.skill_evolver import SkillEvolutionConfig, SkillEvolver
from src.skills.skill_store import SkillStore

PROPOSAL_JSON = json.dumps(
    {
        "cluster_decisions": [
            {
                "operation": "create",
                "cluster_id": "__CLUSTER_ID__",
                "skill_id": "diagnose_apply_errors",
                "title": "Diagnose Apply Errors",
                "description": "Repair malformed diffs from apply diagnostics.",
                "purpose": "Prevent retries of a structurally invalid patch.",
                "when_to_use": ["The canonical diff fails to apply."],
                "how_to_apply": [
                    "Read the first rejected hunk and compare its context with the current file.",
                    "Regenerate only that hunk and rerun the apply check."
                ],
                "constraints": ["Do not change code semantics until the diff applies cleanly."],
                "action_type": "PATCH|RETRY",
                "applicable_capability_label": "patch_application_failure",
                "rationale": "Recurring apply failures in the cluster.",
                "evidence_provenance": ["acme__widgets-1", "acme__widgets-2"],
                "source": "reflector",
                "confidence": 0.7,
            }
        ]
    }
)


class ScriptedLLM:
    """Deterministic LLM double following the tests/test_inspector_tools.py pattern."""

    def __init__(self, content: str):
        self._content = content
        self.calls = 0

    def chat_with_system(self, system_prompt: str, user_message: str, **kwargs) -> LLMResponse:
        self.calls += 1
        self.last_user = user_message
        content = self._content
        if "__CLUSTER_ID__" in content:
            cluster_id = json.loads(user_message)["failure_clusters"][0]["cluster_id"]
            content = content.replace("__CLUSTER_ID__", cluster_id)
        return LLMResponse(content=content, model="scripted")


def _seed_buffer(path, reason, ids):
    buffer = HardCaseBuffer(path)
    for issue_id in ids:
        buffer.append(
            HardCaseRecord(
                issue_id=issue_id,
                repo_name="acme/widgets",
                failure_type="patch_application_failure",
                reason=reason,
                stage="train",
                failure_diagnosis={
                    "mechanism": "patch_application_failure",
                    "summary": reason,
                    "error_signatures": ["git apply: malformed hunk"],
                    "evidence": {"patch_applied": [False]},
                    "confidence": 0.9,
                },
            )
        )


def _reflector(tmp_path, llm):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return Reflector(
        skill_bank=SkillBank(skills_dir=skills_dir),
        skill_evolver=SkillEvolver(
            store=SkillStore(skills_dir=skills_dir),
            config=SkillEvolutionConfig(),
            embedding_client=None,
        ),
        llm_client=llm,
        embedding_client=None,
        buffer_path=tmp_path / "hard_cases.jsonl",
        min_cluster_size=3,
    )


def test_reflect_applies_proposals_from_qualifying_cluster(tmp_path):
    _seed_buffer(
        tmp_path / "hard_cases.jsonl",
        "git apply failed malformed hunk context",
        ["acme__widgets-1", "acme__widgets-1", "acme__widgets-2"],
    )
    llm = ScriptedLLM(PROPOSAL_JSON)
    reflector = _reflector(tmp_path, llm)

    result = reflector.reflect(stage="train")

    assert llm.calls == 1
    assert result.qualifying_cluster_count == 1
    assert result.proposals and result.proposals[0]["skill_id"] == "diagnose_apply_errors"
    assert (tmp_path / "skills" / "diagnose_apply_errors.md").exists()
    applied = [e for e in result.evolution["events"] if e["applied"]]
    assert any(e["skill_id"] == "diagnose_apply_errors" for e in applied)


def test_reflect_emits_observed_labels_and_stable_cluster_ids(tmp_path):
    _seed_buffer(
        tmp_path / "hard_cases.jsonl",
        "git apply failed malformed hunk context",
        ["acme__widgets-1", "acme__widgets-1", "acme__widgets-2"],
    )
    reflector = _reflector(tmp_path, ScriptedLLM(PROPOSAL_JSON))
    result = reflector.reflect(stage="train")
    assert result.task_signals["observed_failure_labels"] == {
        "acme__widgets-1": "patch_application_failure",
        "acme__widgets-2": "patch_application_failure",
    }
    assert set(result.task_signals["stable_cluster_ids"]) == {
        "acme__widgets-1", "acme__widgets-2"
    }
    assert result.task_signals["instance_boosts"] == {}
    assert result.task_signals["family_boosts"] == {}


def test_reflect_without_qualifying_cluster_skips_llm(tmp_path):
    _seed_buffer(
        tmp_path / "hard_cases.jsonl",
        "git apply failed malformed hunk context",
        ["acme__widgets-1", "acme__widgets-2"],  # only two -> below min_cluster_size
    )
    llm = ScriptedLLM(PROPOSAL_JSON)
    reflector = _reflector(tmp_path, llm)
    result = reflector.reflect(stage="train")
    assert llm.calls == 0
    assert result.qualifying_cluster_count == 0
    assert result.proposals == []
    assert result.evolution is None


def test_reflect_records_only_uses_explicit_round_slice(tmp_path):
    llm = ScriptedLLM(PROPOSAL_JSON)
    reflector = _reflector(tmp_path, llm)
    records = [
        HardCaseRecord(
            issue_id=f"round__issue-{index}",
            repo_name="acme/widgets",
            failure_type="patch_application_failure",
            reason="git apply failed malformed hunk context",
            stage="train",
            failure_diagnosis={
                "mechanism": "patch_application_failure",
                "summary": "git apply failed malformed hunk context",
                "error_signatures": ["git apply: malformed hunk"],
                "evidence": {"patch_applied": [False]},
                "confidence": 0.9,
            },
        )
        for index in range(3)
    ]

    result = reflector.reflect_records(records)

    assert llm.calls == 1
    assert result.qualifying_cluster_count == 1
    assert (tmp_path / "skills" / "diagnose_apply_errors.md").exists()


def test_reflect_ignores_other_stage_records(tmp_path):
    buffer = HardCaseBuffer(tmp_path / "hard_cases.jsonl")
    for issue_id in ("e1", "e2", "e3"):
        buffer.append(
            HardCaseRecord(
                issue_id=issue_id,
                repo_name="acme/widgets",
                failure_type="patch_application_failure",
                reason="git apply failed malformed hunk context",
                stage="eval",
                failure_diagnosis={
                    "mechanism": "patch_application_failure",
                    "summary": "git apply failed malformed hunk context",
                    "error_signatures": ["git apply: malformed hunk"],
                    "evidence": {"patch_applied": [False]},
                    "confidence": 0.9,
                },
            )
        )
    llm = ScriptedLLM(PROPOSAL_JSON)
    result = _reflector(tmp_path, llm).reflect(stage="train")
    assert llm.calls == 0
    assert result.qualifying_cluster_count == 0


def test_reflect_survives_non_json_llm_output(tmp_path):
    _seed_buffer(
        tmp_path / "hard_cases.jsonl",
        "git apply failed malformed hunk context",
        ["acme__widgets-1", "acme__widgets-2", "acme__widgets-3"],
    )
    reflector = _reflector(tmp_path, ScriptedLLM("sorry, I cannot help with that"))
    result = reflector.reflect(stage="train")
    assert len(result.proposals) == 1
    assert result.proposals[0]["operation"] == "no-op"
    assert result.evolution is None
    assert result.rejected_proposals  # the parse failure is recorded, not raised


def test_reflect_validator_gates_and_selects(tmp_path):
    _seed_buffer(
        tmp_path / "hard_cases.jsonl",
        "git apply failed malformed hunk context",
        ["acme__widgets-1", "acme__widgets-2", "acme__widgets-3"],
    )
    reflector = _reflector(tmp_path, ScriptedLLM(PROPOSAL_JSON))

    def reject_all(proposals, clusters):
        return [], 0.0

    result = reflector.reflect(stage="train", validator=reject_all)
    assert len(result.proposals) == 1
    assert result.proposals[0]["operation"] == "no-op"
    assert result.proposals[0]["rationale"] == "validator rejected write"
    assert result.evolution is None
    assert not (tmp_path / "skills" / "diagnose_apply_errors.md").exists()
