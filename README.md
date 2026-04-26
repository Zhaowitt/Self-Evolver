# Self-Evolver

Self-Evolver is a multi-agent code repair framework for repository-level
software engineering tasks. The current implementation is focused on SWE-bench
style repair loops: inspect the issue, generate a patch, verify it in a real
repository, judge failures, and retry with structured feedback.

The broader research goal is a self-evolving coding agent that learns from hard
cases through task evolution and skill evolution. This repository currently
implements the static multi-agent baseline plus a failure-routing judge and
SWE-bench evaluation workflow.

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

## Repository Layout

```text
src/
  benchmark/       SWE-bench runner and benchmark result models
  critic/          rule-based execution evaluator
  environment/     Git, patch, test, and issue environment APIs
  llm/             OpenAI-compatible LLM client
  orchestrator/    repair loop coordination and retry routing
  workers/         Inspector, PatchGenerator, Verifier, LLMJudge
tests/             focused unit/integration tests for patch and judge behavior
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
exhausts the retry budget or is routed to hard-case handling, a compact JSONL
record is written to `WORKSPACE_DIR/hard_cases.jsonl`.

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

