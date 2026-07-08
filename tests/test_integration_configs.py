"""Every shipped ``configs/*.yaml`` is real config, not decoration.

For each file that code loads we prove two things: the repo file is actually
read (its values reach the object), and changing a value in a tmp copy changes
observable behavior. The one non-code-loaded file (the external EasyR1 training
example) is proven to reference live code, so nothing in ``configs/`` is a
file that no one reads.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.reward.reward_model import (
    DEFAULT_REWARD_WEIGHTS,
    RewardModel,
    default_config_path as reward_default_path,
)
from src.skills.proposals import SkillUpdateProposal
from src.skills.skill_evolver import SkillEvolutionConfig, SkillEvolver
from src.skills.skill_store import SkillStore
from src.tasks.config import (
    TaskEvolutionConfig,
    default_config_path as task_default_path,
    load_task_evolution_config,
)
from src.tasks.task_pool import TaskPool

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Files whose values reach the code, each proven below.
CODE_LOADED = {"reward_config.yaml", "skill_evolution.yaml", "task_evolution.yaml"}
# The external EasyR1 trainer reads this; our code does not load it. Proven live
# (references compute_score + reward_config) rather than decorative.
EXTERNAL_EXAMPLE = {"easyr1_online_grpo_example.yaml"}


def test_no_unaccounted_config_files():
    """A new configs/*.yaml must be code-loaded (add a loader + proof) or a
    documented external example; a decorative config trips this guard."""
    shipped = {path.name for path in CONFIG_DIR.glob("*.yaml")}
    assert shipped == CODE_LOADED | EXTERNAL_EXAMPLE, (
        "unaccounted config files: "
        f"{sorted(shipped - (CODE_LOADED | EXTERNAL_EXAMPLE))}"
    )


# ------------------------------------------------------------ reward_config.yaml


def _execution(tokens: int):
    """Minimal rollout evidence: only the token count matters for cost."""
    return SimpleNamespace(
        total_tokens=tokens,
        iteration_records=[],
        final_patch=None,
        status=SimpleNamespace(value="success"),
        metadata={},
    )


def test_reward_config_is_loaded_from_the_repo_file():
    assert reward_default_path() == CONFIG_DIR / "reward_config.yaml"
    model = RewardModel.from_config_file()  # default path = the repo file
    assert model.weights == DEFAULT_REWARD_WEIGHTS
    assert model.skill_write_gate == 0.55
    assert model.cost_token_budget == 60000


def test_reward_config_value_change_changes_scoring(tmp_path):
    """cost_token_budget from the file drives the cost-efficiency component."""
    tight = tmp_path / "tight.yaml"
    tight.write_text("cost_token_budget: 10000\n", encoding="utf-8")
    loose = tmp_path / "loose.yaml"
    loose.write_text("cost_token_budget: 40000\n", encoding="utf-8")

    tight_reward = RewardModel.from_config_file(tight).score(_execution(20000))
    loose_reward = RewardModel.from_config_file(loose).score(_execution(20000))

    assert tight_reward.components["cost_efficiency"] == 0.0   # 20k/10k clamps to 1
    assert loose_reward.components["cost_efficiency"] == 0.5   # 1 - 20k/40k
    assert loose_reward.total > tight_reward.total


# --------------------------------------------------------- skill_evolution.yaml


_SKILL_MD = (
    "# New Skill\n\n## Description\nd\n\n## How to Apply\np\n\n"
    "Target failure type: test_failure\n"
)


def _create_proposal() -> SkillUpdateProposal:
    return SkillUpdateProposal(
        operation="create",
        skill_id="new_skill",
        title="New Skill",
        summary="a repair skill",
        target_failure_type="test_failure",
        content=_SKILL_MD,
        rationale="cluster evidence",
        source="reflector",
    )


def _evolver_from_yaml(config_yaml: Path, skills_dir: Path) -> SkillEvolver:
    config = SkillEvolutionConfig.load(config_yaml)
    return SkillEvolver(
        store=SkillStore(skills_dir=skills_dir),
        config=config,
        embedding_client=None,
    )


def test_skill_evolution_config_is_loaded_from_the_repo_file():
    config = SkillEvolutionConfig.load()  # default path = the repo file
    assert config == SkillEvolutionConfig()  # values identical to code defaults
    assert config.reflect_every_n_rollouts == 10
    assert config.skill_write_utility_threshold == 0.55


def test_skill_write_threshold_value_change_gates_writes(tmp_path):
    """skill_write_utility_threshold from the file decides whether a proposal
    at a fixed utility is written to disk."""
    high = tmp_path / "high.yaml"
    high.write_text("skill_write_utility_threshold: 0.9\n", encoding="utf-8")
    low = tmp_path / "low.yaml"
    low.write_text("skill_write_utility_threshold: 0.1\n", encoding="utf-8")

    high_dir = tmp_path / "skills_high"
    high_dir.mkdir()
    high_result = _evolver_from_yaml(high, high_dir).apply_proposals(
        [_create_proposal()], utility=0.5
    )
    assert high_result["events"][0]["applied"] is False
    assert high_result["events"][0]["reason"] == "utility_below_write_threshold"
    assert not (high_dir / "new_skill.md").exists()

    low_dir = tmp_path / "skills_low"
    low_dir.mkdir()
    low_result = _evolver_from_yaml(low, low_dir).apply_proposals(
        [_create_proposal()], utility=0.5
    )
    assert low_result["events"][0]["applied"] is True
    assert (low_dir / "new_skill.md").exists()


def test_skill_evolution_config_rejects_unknown_keys(tmp_path):
    """Unknown keys raise, proving the file is parsed against the dataclass and
    not silently ignored."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_real_knob: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown skill_evolution config keys"):
        SkillEvolutionConfig.load(bad)


# ----------------------------------------------------------- task_evolution.yaml


import json


def _instance(instance_id: str) -> dict:
    return {
        "instance_id": instance_id,
        "repo": "org/proj",
        "base_commit": "deadbeef",
        "patch": (
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n+++ b/src/app.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        ),
        "problem_statement": "Traceback (most recent call last):\nTypeError: bad operand",
        "FAIL_TO_PASS": json.dumps(["tests/test_app.py::test_a", "tests/test_app.py::test_b"]),
        "PASS_TO_PASS": json.dumps(["tests/test_app.py::test_ok"]),
    }


def _weight_after_two_failures(config_yaml: Path, state_path: Path) -> float:
    pool = TaskPool.from_instances(
        [_instance("org__hard-1")],
        state_path,
        config=load_task_evolution_config(config_yaml),
        verifier=lambda instance: (True, "ok"),
    )
    pool.record_outcome("org__hard-1", resolved=False, utility=0.0)
    pool.record_outcome("org__hard-1", resolved=False, utility=0.0)
    return pool.weights()["org__hard-1"]


def test_task_evolution_config_is_loaded_from_the_repo_file():
    assert task_default_path() == CONFIG_DIR / "task_evolution.yaml"
    assert load_task_evolution_config() == TaskEvolutionConfig()


def test_too_hard_attempts_value_change_changes_decay(tmp_path):
    """too_hard_attempts from the file decides when a struggling instance's
    sampling weight decays. Both pools see the same two failures (identical
    family EMA); only the one whose threshold is reached decays by the factor."""
    quick = tmp_path / "quick.yaml"
    quick.write_text("too_hard_attempts: 2\n", encoding="utf-8")
    patient = tmp_path / "patient.yaml"
    patient.write_text("too_hard_attempts: 4\n", encoding="utf-8")

    quick_weight = _weight_after_two_failures(quick, tmp_path / "quick.json")
    patient_weight = _weight_after_two_failures(patient, tmp_path / "patient.json")

    assert quick_weight < patient_weight
    # decay_factor default is 0.3: the only difference is the decay multiplier.
    assert quick_weight == pytest.approx(patient_weight * 0.3)


# ------------------------------------------------ easyr1_online_grpo_example.yaml


def test_easyr1_example_references_live_reward_code():
    """The external training example is not code-loaded, so prove it points at
    real entrypoints (compute_score + the reward config) instead of being dead."""
    data = yaml.safe_load((CONFIG_DIR / "easyr1_online_grpo_example.yaml").read_text())
    assert isinstance(data, dict)

    reward_function = data["worker"]["reward"]["reward_function"]
    module_path, function_name = reward_function.split(":")
    assert (Path(__file__).resolve().parents[1] / module_path).exists()
    module = importlib.import_module(module_path.replace("/", ".")[: -len(".py")])
    assert callable(getattr(module, function_name))

    reward_config = data["self_evolver"]["reward_config"]
    assert (Path(__file__).resolve().parents[1] / reward_config).exists()
