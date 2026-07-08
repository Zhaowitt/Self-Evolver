# Self-Evolver

A self-evolving multi-agent coding agent for repository-level issue resolving.
The Controller drives an outer loop of **task evolution** and **skill evolution**
over an Inspector → PatchGenerator → Verifier repair loop, graded with official
SWE-bench-family semantics.

```bash
pip install -e .
cp .env.example .env      # then fill in OPENAI_API_KEY
```

Container grading uses `apptainer` (or `docker` if present); no other setup.

## Benchmark configuration

Set the data root and container-image cache (both have portable defaults):

```bash
export SWEBENCH_DATA_DIR="$(pwd)/benchmarks"     # default: <repo>/benchmarks
export SIF_CACHE_DIR="$HOME/.cache/self_evolver/sif"
```

Download every dataset (idempotent; `DATASETS=` selects a subset):

```bash
bash scripts/download_benchmarks.sh
```

| `--benchmark`    | `--dataset`          | Role                         | Grading |
| ---------------- | -------------------- | ---------------------------- | ------- |
| `swebench`       | `full` (`--split train`) | RL / evolution training pool | apptainer/docker |
| `swebench`       | `verified`           | primary eval (Python)        | apptainer/docker |
| `swebench`       | `lite`               | fast dev (`dev`/`test`)      | apptainer/docker |
| `swebench_live`  | `lite`\|`full`\|`test` | contamination-free held-out | apptainer/docker |
| `multi_swe_bench`| `full` (1632, 7 langs)\|`flash` (300) | cross-language transfer | official Docker harness |
| `swebench_pro`   | `test`               | harder OOD                   | official Docker harness |

## Training commands

Each experiment is one script (`SEED=`, `TEST_BACKEND=`, `NUM_*=` override defaults).
Baselines to full method:

```bash
bash scripts/run_zero_shot.sh        # base model, single agent, no evolution
bash scripts/run_mas_static.sh       # fixed multi-agent workflow, no evolution
bash scripts/run_task_evolution.sh   # task evolution only
bash scripts/run_skill_evolution.sh  # skill evolution only
bash scripts/run_full_method.sh      # task + skill evolution (main method)
bash scripts/run_rl_controller.sh    # build EasyR1 GRPO data + train the Controller
bash scripts/run_all_experiments.sh  # every experiment x seeds -> metrics tables
```

Equivalent direct CLI (a train run evolves the skill bank, then freezes it):

```bash
python -m src.main benchmark \
  --benchmark swebench --dataset full --split train --stage train --phase generate \
  --agent-mode mas --skills evolve --memory on --task-evolution on \
  --num-instances 100 --seed 0 --run-id full_method-train
```

Flags: `--agent-mode single|mas`, `--skills off|static|evolve`, `--memory on|off`,
`--task-evolution on|off`, `--controller-mode off|llm`, `--stage train|eval`,
`--test-backend auto|apptainer|docker|host`, `--train-ids FILE` (contamination guard).

## Evaluation commands

Frozen held-out eval (skills snapshotted read-only) generates predictions:

```bash
python -m src.main benchmark \
  --benchmark swebench --dataset verified --stage eval --phase generate \
  --skills static --seed 0 --run-id full_method-eval

bash scripts/run_transfer_eval.sh    # frozen eval on SWE-bench-Live + Multi-SWE-bench
```

Grade a predictions file (official Docker harness when docker is present, else the
same swebench grading via apptainer):

```bash
PREDICTIONS=runs/full_method-eval/predictions.json bash scripts/evaluate.sh
```

Aggregate metrics (resolved rate, pass@k, cost-to-success, per-iteration evolution
curve) as JSON + Markdown:

```bash
python -m src.benchmark.metrics --rollouts runs/*/rollouts.jsonl --report
```

Multi-SWE-bench and SWE-bench Pro grade with their own Docker harnesses: the runner
generates predictions and writes a harness-ready patch file, then prints the exact
official command.


