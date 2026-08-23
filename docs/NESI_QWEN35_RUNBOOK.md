# NeSI Qwen3.5 Controller Runbook

## Recommended starting point

Use `Qwen/Qwen3.5-4B` with one A100 80 GB for the first migrated experiment.
The Controller only selects up to two skills and emits a short strategy, so 4B
is a better first trade-off than 9B.  Move to 9B only after the 4B environment,
one-step rollout, and full round have completed without compatibility or memory
errors.

Qwen3.5 is a newer multimodal architecture.  A model download succeeding does
not prove that the installed EasyR1, Transformers, and vLLM combination can
train it.  `scripts/preflight_qwen35.py` checks those registrations without
loading the full weights.  A one-step GPU smoke test is still required because
only it exercises FSDP, LoRA, vLLM weight sharing, and the online reward.

## Cluster prerequisites

The Slurm launcher assumes the following locations by default:

```text
/nesi/project/<project>/$USER/envs/self-evolver   Conda environment
/nesi/project/<project>/$USER/EasyR1             EasyR1 checkout
/nesi/nobackup/<project>/$USER/benchmarks        SWE-bench datasets
/nesi/nobackup/<project>/$USER/sif               official-image SIF files
```

Override `CONDA_ENV`, `EASYR1_DIR`, `SWEBENCH_DATA_DIR`, or `SIF_CACHE_DIR`
at submission time if the actual locations differ.  The active environment
must contain a Qwen3.5-capable EasyR1/verl, Transformers, vLLM, PyTorch, and an
editable install of Self-Evolver.  Keep the exact EasyR1 commit and Python
package versions in the experiment record.

Create a real `.env` in the Self-Evolver checkout and set the worker API:

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

Do not commit `.env`.

## One-step smoke test

Submit this before the 64-step round.  It uses a separate state directory and
disables skill writes/reflection so a failed infrastructure test cannot mutate
the real experiment state.

```bash
cd /path/to/Self-Evolver

sbatch \
  --time=04:00:00 \
  --job-name=se-qwen35-smoke \
  --export=ALL,EASYR1_MAX_STEPS=1,NUM_TRAIN=4,NUM_VAL=2,REBUILD_DATASETS=1,ROUND_ID=qwen35-smoke,ENABLE_SKILL_CREDIT=0,ENABLE_SKILL_REFLECTION=0,STATE_ROOT=/nesi/nobackup/uoa04658/$USER/self_evolver_smoke \
  scripts/run_nesi_qwen35_4b.sl
```

The job must reach and finish `global_step_1`; passing only the preflight is not
enough.  Check the result with:

```bash
squeue --me
tail -F se-qwen35-smoke-<jobid>.out
seff <jobid>
```

## First 64-step round

After the one-step job succeeds:

```bash
sbatch scripts/run_nesi_qwen35_4b.sl
```

The defaults are one A100, 16 CPUs, 64 GB host RAM, 320 training prompts, 18
validation prompts, four online reward workers, and a cumulative target of 64
steps.

For a continuation from `global_step_960`, copy the checkpoint and previous
TaskPool/reflection state to NeSI, then submit with a cumulative target of 1024:

```bash
sbatch \
  --export=ALL,EASYR1_LOAD_CHECKPOINT_PATH=/absolute/path/global_step_960,EASYR1_MAX_STEPS=1024,ROUND_ID=qwen35-round12,TASK_SIGNALS_REPORT=/absolute/path/previous_reflection.json \
  scripts/run_nesi_qwen35_4b.sl
```

Do not load a Qwen2.5 checkpoint into Qwen3.5: their architectures and parameter
shapes differ.  `global_step_960` is usable only when it came from the same base
model architecture.  When changing from Qwen2.5-1.5B to Qwen3.5-4B, start the
Controller weights from Qwen3.5 and treat it as a new experiment namespace;
previous hard cases and task-sampling signals may still be reused as data.

If a job stops after saving a Qwen3.5 checkpoint, resubmit the same experiment
with `REBUILD_DATASETS=0` and keep the cumulative `EASYR1_MAX_STEPS` unchanged.
