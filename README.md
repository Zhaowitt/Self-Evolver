# Self-Evolver

Self-Evolver is a multi-agent code repair framework for repository-level
software engineering tasks. The current implementation is focused on SWE-bench
style repair loops: inspect the issue, generate a patch, verify it in a real
repository, judge failures, and retry with structured feedback.

The broader research goal is a self-evolving coding agent that learns from hard
cases through task evolution and skill evolution. This repository currently
implements the static multi-agent baseline plus a failure-routing judge,
SWE-bench evaluation workflow, and an optional upstream Controller that can
provide task wrappers, strategy cues, memory cues, and skill-selection signals.

## What Is Included

- `ProjectEnvironment`: Git repository interaction, patch application, test
  execution, canonical diff capture, and SWE-bench issue setup.
- `Inspector`: LLM-based fault localization and root-cause analysis.
- `PatchGenerator`: LLM-based unified diff patch generation.
- `Verifier`: tool-based patch application and test verification. Successful
  predictions use `git diff` canonical output, not raw LLM diff text.
- `LLMJudge`: failure diagnosis and retry routing. It can route retries, but
  cannot declare success.
- `CriticJudge`: rule-based execution evaluation and failure summary.
- `SWEBenchRunner`: prediction generation, official SWE-bench evaluation,
  Docker cleanup, and infra/patch error separation.
- `Controller`: optional upstream guidance generator for worker prompts. It
  runs in `off` or OpenAI-compatible `llm` mode.
- `SkillBank` / `SkillEvolver`: seed skill loading, skill selection, reward
  tracking, and reward-gated skill create/update/deprecate proposals.
- `RewardModel` / online reward: execution-derived reward components, JSONL
  rollout logging, and an EasyR1 callable reward function.

## Repository Layout

```text
src/
  benchmark/       SWE-bench runner and benchmark result models
  controller/      Controller schema, prompt builder, parser, and client
  critic/          rule-based execution evaluator
  environment/     Git, patch, test, and issue environment APIs
  llm/             OpenAI-compatible LLM client
  memory/          Hard-case buffer and retrieval helpers
  orchestrator/    repair loop coordination and retry routing
  reward/          Reward components and online EasyR1 reward function
  rl/              Online rollout runner, rollout writer, and EasyR1 prompt dataset
  skills/          Skill bank, selector, dedup, stats, and evolution logic
  workers/         Inspector, PatchGenerator, Verifier, LLMJudge
configs/           Controller, reward, skill evolution, and EasyR1 examples
skills/            Markdown seed skills used by the skill bank
run_swebench_v2.py compatibility wrapper around the unified CLI
Proposal.md        research proposal and design motivation
```

## Setup

Python 3.10 or newer is required.

```bash
pip install -e ".[dev]"
```

For SWE-bench evaluation:

```bash
pip install -e ".[swebench]"
```

Create a `.env` file or export environment variables:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1
MAX_ITERATIONS=3
WORKSPACE_DIR=./workspace
DOCKER_TIMEOUT=600
```

For an OpenAI-compatible Controller endpoint such as vLLM:

```bash
CONTROLLER_API_KEY=token-or-empty
CONTROLLER_MODEL=Qwen/Qwen2.5-1.5B-Instruct
CONTROLLER_BASE_URL=http://<gpu-host>:8000/v1
CONTROLLER_MAX_TOKENS=1024
CONTROLLER_TEMPERATURE=0.2
```

## CLI Usage

Check local configuration and repository test execution:

```bash
python -m src.main check --repo /path/to/repo --test-cmd "pytest"
python -m src.main config-info
```

Run a repair task against a local repository:

```bash
python -m src.main fix \
  --repo /path/to/repo \
  --issue "Describe the bug or requested behavior" \
  --test-cmd "pytest"
```

## SWE-bench Workflow

Generate predictions only:

```bash
python -m src.main benchmark \
  --phase generate \
  --dataset lite \
  --predictions-path swebench_results/predictions.json
```

Evaluate existing predictions only:

```bash
python -m src.main benchmark \
  --phase evaluate \
  --dataset lite \
  --predictions-path swebench_results/predictions.json \
  --run-id se-lite-v2
```

Generate and evaluate in one run:

```bash
python -m src.main benchmark \
  --phase both \
  --dataset lite \
  --agent-workers 4 \
  --eval-workers 2 \
  --predictions-path swebench_results/predictions.json \
  --run-id se-lite-v2
```

The old `run_swebench_v2.py` entrypoint is now only a compatibility wrapper:

```bash
python run_swebench_v2.py --phase both --dataset lite
```

## Controller, Rollouts, and Skill Evolution

The Controller is disabled by default. When enabled, it runs before the repair
loop and writes a structured `ControllerSignal` into the execution context. The
existing worker flow is preserved:

```text
ControllerSignal -> Inspector -> PatchGenerator -> Verifier -> LLMJudge
```

Training/evolution mode may use a `task_wrapper` to guide how the worker solves
an existing task. Evaluation mode keeps the benchmark issue fixed and forces
`task_wrapper` to `null`; only skill, strategy, and memory cues are allowed.

Use a vLLM-served Controller:

```bash
python -m src.main benchmark \
  --dataset lite \
  --split train \
  --num-instances 1 \
  --controller-mode llm \
  --controller-stage train
```

Relevant benchmark options:

- `--controller-mode off|llm`
- `--controller-stage train|eval`
- `--rollout-jsonl PATH`
- `--reward-config PATH`

Controller outputs can include `skill_updates` proposals. These are not passed
to the downstream workers. They are materialized only after execution reward
meets the configured write threshold. Existing skill files are updated
atomically and previous content is archived under `skills/_archive/`. Runtime
skill stats are stored in `skills/metadata.json`.

Key defaults are in:

- `configs/controller_schema.yaml`
- `configs/reward_config.yaml`
- `configs/skill_evolution.yaml`
- `configs/easyr1_online_grpo_example.yaml`

## EasyR1 Integration

This repo does not embed EasyR1 training. It exposes full Controller prompt
datasets and an online reward function that EasyR1 can call during GRPO.
Rollout JSONL files are execution logs only; they are not the reward source.

Build a SWE-bench train prompt dataset:

```bash
python -m src.rl.easyr1_dataset \
  --dataset lite \
  --split train \
  --num-instances 20 \
  --stage train \
  --workspace-root /path/to/reward_workspace \
  --output benchmark_results/controller_train.jsonl
```

Each JSONL row contains:

- `prompt`: JSON-encoded chat messages with the exact Controller system and
  user prompt.
- `ground_truth`: serialized SWE-bench instance metadata used by the reward
  function.
- `extra_info`: runtime metadata such as stage, split, workspace root, and
  targeted test command.

Point EasyR1 to the online reward function:

```yaml
data:
  train_files: /path/to/controller_train.jsonl
  prompt_key: prompt
  answer_key: ground_truth
worker:
  reward:
    reward_function: /path/to/Self_Evolver/src/reward/online_reward.py:compute_score
```

The online reward path is:

```text
EasyR1 policy response -> Controller JSON parser -> ControllerSignal injection
-> Inspector -> PatchGenerator -> Verifier/Judge -> RewardModel -> score
```

Useful environment variables for reward execution:

```bash
SELF_EVOLVER_REWARD_WORKSPACE=/path/to/reward_workspace
SELF_EVOLVER_ROLLOUT_JSONL=benchmark_results/online_rollouts.jsonl
SELF_EVOLVER_REWARD_CONFIG=configs/reward_config.yaml
SELF_EVOLVER_ENABLE_SKILL_EVOLUTION=0
SELF_EVOLVER_MAX_ITERATIONS=3
```

## Patch Verification Design

LLMs often produce raw unified diffs with malformed hunks, missing context
prefixes, or stale line counts. Self-Evolver now treats raw LLM diff text as an
intermediate artifact:

```text
raw LLM diff -> local apply/fix/fuzzy apply -> verifier tests -> git diff canonical patch
```

Only the canonical `git diff` output is used as the final SWE-bench prediction.
This reduces malformed patch, hunk mismatch, and truncated patch failures in
official evaluation.

SWE-bench `test_patch` changes are staged during local issue setup so they are
available for targeted verification but excluded from final predictions.

## Failure Routing

The LLM judge emits one of these routes after failed attempts:

- `repair_patch_format`
- `regenerate_patch_same_location`
- `reinspect`
- `empty_patch_reprompt`
- `give_up_hard_case`

Success is still determined only by deterministic verifier results. If an issue
exhausts the retry budget or is routed to hard-case handling, a normalized JSONL
record is written to `WORKSPACE_DIR/hard_cases.jsonl`. These hard cases can be
retrieved as Controller context for curriculum sampling and skill proposals.

## Validation

Run the focused tests:

```bash
pytest tests -q
```

Useful lightweight checks:

```bash
python -m compileall -q src tests run_swebench_v2.py
git diff --check
```
